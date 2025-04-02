import logging
import os
import random
import time
from typing import Any, Dict, List, Union
import shutil
import threading
import queue
from functools import wraps
from typing import Optional, List, Tuple
import numpy as np 

from celery import Celery
from celery.signals import worker_process_init
from celery.utils.log import get_task_logger

import json
from PIL import Image
import PIL.JpegImagePlugin

from worker.embed_sequence import EmbeddingsModel, convert_file_to_image
import torch
from worker.worker_embed_batch import BatchProcessor


IMAGE_BATCH = 600


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = get_task_logger(__name__)

# REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
REDIS_HOST: str = "localhost"
# docker compose
app_embed: Celery = Celery(
    "tasks",
    broker=f"redis://{REDIS_HOST}:6379/0",
    backend=f"redis://{REDIS_HOST}:6379/1",
)
 
# Configure Redis connection pool settings
redis_socket_timeout = 5.0  # seconds
redis_socket_connect_timeout = 5.0  # seconds
broker_pool_limit = 50  # Adjust based on expected number of connections per worker
backend_pool_limit = 50  # Adjust for result backend connections

def google_storage_file_upload(up_file: str = "/tmp/surfer_wave.png", bucket_name: str = "blog_inference", uploaded_file_name: str = "test_up.png")->str:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload a file to GCP bucket    
    blob = bucket.blob(uploaded_file_name)
    blob.upload_from_filename(up_file)
    #logger.info(f"File uploaded successfully:{bucket_name}, {up_file}, {uploaded_file_name}")
    gcs_link = f"gs://{bucket_name}/{uploaded_file_name}"
    return gcs_link 

# Configure Celery
app_embed.conf.update(
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

# Global batch processor that will be initialized per worker process
global_is_batch_processing_wait_each_get = False

batch_processor = BatchProcessor(batch_size=IMAGE_BATCH, max_wait_time=1.0)

logger.info("Batch processor initialized for worker")

EmbeddingsInference = EmbeddingsModel()

is_save_to_gcs = False

@worker_process_init.connect
def initialize_batch_processor(**kwargs):
    """Initialize the batch processor when a worker process starts."""
    pass


# Single task definition without retries
@app_embed.task(bind=True, name="worker.process_request_embed")
def process_request_embed(self: Celery.Task, input_string: str) -> Dict[str, Any]:
    """
    Process a request - simplified task with just a sleep.

    Args:
        input_string (str): input_string to process

    Returns:
        dict: Result including input parameters
    """
    task_id: str = self.request.id
    time_start = time.time()

    # Use batch processor if available, otherwise fallback to the original method
    global batch_processor
    
    if batch_processor is not None:
        logger.info(f"Task {task_id}: Using batch processor for {input_string}")
        output_fn:Optional[str] = batch_processor.run_task_in_batch(task_id, input_string, EmbeddingsInference)

        if output_fn:
            output_fn_list = [output_fn]
        else:
            # Fallback to original method if batch processing failed
            logger.warning(f"Batch processing failed for {task_id}, using original method")
            output_fn_list = EmbeddingsInference.process_files_batch(input_image_filename_list=[input_string])
            # output_fn_list = [input_string] 
    else:
        # Use original method if batch processor is not available
        logger.info(f"Batch processor not available, using original method for {input_string}")
        output_fn_list = EmbeddingsInference.process_files_batch(input_image_filename_list=[input_string])
        
    ret_dic: dict = {"input_string": input_string, "output_fn": output_fn_list[0]}

    destination_path = output_fn_list[0]
    gcs_link:str = "gcs://none/none"
    ret_dic: dict = {"input_string": input_string} 
    ret_str:str = json.dumps(ret_dic)
    logger.info(f" ******* Task_id={task_id} input={input_string}: {time.time() - time_start:.2f} seconds")
    
    if is_save_to_gcs:
        # upload to gcs
        temp_filename = "/tmp/test.npy" # validate io successed
        new_filename = f"task_id_{task_id}.npy" # {str(task_id)}.png",
        destination_path = f"/tmp/{new_filename}"
        uploaded_file_name = f"test001/{new_filename}"
        try:
            # Copy the file and rename it
            # os.makedirs(destination_dir, exist_ok=True)
            shutil.copy(destination_path, temp_filename)
            print(f"File copied and renamed successfully to: {destination_path} {temp_filename}")
            gcs_link:str = google_storage_file_upload(up_file=destination_path, uploaded_file_name=uploaded_file_name)
        except FileNotFoundError:
            print(f"Error: Source file not found: {temp_filename}")
        except PermissionError:
            print(f"Error: Permission denied to access {temp_filename} or {destination_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        logger.info(f"File uploaded successfully: gcs_link={gcs_link}, {destination_path}, {uploaded_file_name}")
    
    # Return the input parameters along with the result
    return {
        "task_id": task_id,
        "input_string": input_string,
        "local_filename": destination_path,        
        "gcs_link": gcs_link,    
        "ret_json": ret_str,
    }


if __name__ == "__main__":
    # This allows running the worker directly with: python worker.py
    app_embed.start()