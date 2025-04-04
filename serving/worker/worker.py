import logging
import os
import random
import time
from typing import Any, Dict, Optional, Tuple
import shutil


from celery import Celery
from celery.utils.log import get_task_logger

from .run_ai import run_ai, google_storage_file_upload, google_storage_file_download
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = get_task_logger(__name__)

# REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
REDIS_HOST: str = "localhost"
# docker compose
app: Celery = Celery(
    "tasks",
    broker=f"redis://{REDIS_HOST}:6379/0",
    backend=f"redis://{REDIS_HOST}:6379/1",
)
 
# Configure Redis connection pool settings
redis_socket_timeout = 5.0  # seconds
redis_socket_connect_timeout = 5.0  # seconds
broker_pool_limit = 50  # Adjust based on expected number of connections per worker
backend_pool_limit = 50  # Adjust for result backend connections

# Configure Celery
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={
        "visibility_timeout": 3600,  # 1 hour
        "socket_timeout": redis_socket_timeout,
        "socket_connect_timeout": redis_socket_connect_timeout,
    },
    # Redis connection pool settings
    broker_pool_limit=broker_pool_limit,  # Max connections in broker pool
    redis_max_connections=broker_pool_limit
    + backend_pool_limit,  # Total max Redis connections
    broker_connection_retry_on_startup=True,
    # Backend (result store) connection pooling
    result_backend_transport_options={
        "socket_timeout": redis_socket_timeout,
        "socket_connect_timeout": redis_socket_connect_timeout,
        "connection_pool": {
            "max_connections": backend_pool_limit,  # Max connections in result backend pool
        },
    },
    # Task result expiration time (in seconds)
    result_expires=86400,  # 24 hours
)

is_diffusers: bool = False
is_save_to_gcs: bool = False
from .embed_sequence import EmbeddingsModel
EmbeddingsInference = EmbeddingsModel()
# Global batch processor that will be initialized per worker process
global_is_batch_processing_wait_each_get = False
batch_processor = None # BatchProcessor(batch_size=IMAGE_BATCH, max_wait_time=1.0)
def process_input(task_id: str, input_string: str) -> Tuple[str, str, str]:
    """
    Process a request - simplified task with just a sleep.
    """
    global batch_processor
    
    if batch_processor is not None:
        logger.info(f"Task {task_id}: Using batch processor for {input_string}")
        output_fn:Optional[str] = batch_processor.run_task_in_batch(task_id, input_string, EmbeddingsInference)

        if output_fn is not None:
            output_fn_list = [output_fn]
        else:
            # Fallback to original method if batch processing failed
            logger.warning(f"Batch processing failed for {task_id}, using original method")
            output_fn_list = EmbeddingsInference.process_files_batch(input_image_filename_list=[input_string])
    else:
        # Use original method if batch processor is not available
        logger.info(f"Batch processor not available, using original method for {input_string}")
        # comment out for testing of the framework latency 
        # output_fn_list = [input_string]
        output_fn_list = EmbeddingsInference.process_files_batch(input_image_filename_list=[input_string])
        
    ret_dic: dict = {"input_string": input_string, "output_fn": output_fn_list[0]}

    destination_path = output_fn_list[0]
    gcs_link:str = "gcs://none/none"
    ret_dic: dict = {"input_string": input_string} 
    ret_str:str = json.dumps(ret_dic)    

    return destination_path, gcs_link, ret_str

# Single task definition without retries
@app.task(bind=True, name="worker.process_request")
def process_request(self: Celery.Task, input_string: str) -> Dict[str, Any]:
    """
    Process a request - simplified task with just a sleep.

    Args:
        input_string (str): input_string to process

    Returns:
        dict: Result including input parameters
    """
    task_id: str = self.request.id
    logger.info(
        f" ****** Processing request for input_string {input_string} with task {task_id}"
    )

    # Random sleep between 1 and 5 seconds
    time_start = time.time()

    if is_diffusers:
        temp_filename = f"/home/roman/PycharmProjects/comfyui/celery-main/worker/test.png"
        ret_dic: dict = run_ai(prompt=input_string, filename=temp_filename)    
        ret_str:str = json.dumps(ret_dic)
        logger.info(f" ******* Task_id={task_id} completed with {ret_str} for {time.time() - time_start:.2f} seconds")
        destination_dir = "/tmp/"
        new_filename = f"task_id_{task_id}.png" # {str(task_id)}.png",
        destination_path = f"{destination_dir}{new_filename}"
        uploaded_file_name = f"test001/{new_filename}"
    else:
        destination_path, gcs_link, ret_str = process_input(task_id, input_string)
    logger.info(f" ******* Task_id={task_id} input={input_string}: {time.time() - time_start:.2f} seconds")
    if is_save_to_gcs:
        try:
            # Copy the file and rename it
            shutil.copy(temp_filename, destination_path)
            print(f"File copied and renamed successfully to: {destination_path}")
        except FileNotFoundError:
            print(f"Error: Source file not found: {temp_filename}")
        except PermissionError:
            print(f"Error: Permission denied to access {temp_filename} or {destination_dir}")
        except Exception as e:
            print(f"An error occurred: {e}")

        
        gcs_link:str = google_storage_file_upload(up_file=destination_path, uploaded_file_name=uploaded_file_name)
        logger.info(f"File uploaded successfully: gcs_link={gcs_link}, {destination_path}, {uploaded_file_name}")
    
    # Return the input parameters along with the result
    return {
        "gcs_link": gcs_link, # f"https://storage.googleapis.com/bucket/object/{task_id}",
        "local_filename": destination_path,
        "input_string": input_string,
        "ret_json": ret_str,
        "task_id": task_id,
    }


if __name__ == "__main__":
    # This allows running the worker directly with: python worker.py
    app.start()
