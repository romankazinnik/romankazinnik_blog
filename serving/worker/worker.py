import logging
import os
import random
import time
from typing import Any, Dict
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
    temp_filename = f"/home/roman/PycharmProjects/comfyui/celery-main/worker/test.png"
    
    ret_dic: dict = run_ai(prompt=input_string, filename=temp_filename)
    
    ret_str:str = json.dumps(ret_dic)
    logger.info(f" ******* Task_id={task_id} completed with {ret_str} for {time.time() - time_start:.2f} seconds")

    
    # Ensure the destination directory exists. If not, create it.
    # os.makedirs(destination_dir, exist_ok=True)
    #shutil.copy(temp_filename, destination_dir)
    
    destination_dir = "/tmp/"
    new_filename = f"task_id_{task_id}.png" # {str(task_id)}.png",
    destination_path = f"{destination_dir}{new_filename}"
    uploaded_file_name = f"test001/{new_filename}"

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
