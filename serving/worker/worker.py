import logging
import os
import random
import time
from typing import Any, Dict, List
import shutil
import threading
import queue
from functools import wraps
from typing import Optional, List, Tuple

from celery import Celery
from celery.signals import worker_process_init
from celery.utils.log import get_task_logger

# from .run_ai import run_ai, google_storage_file_upload, google_storage_file_download
import json

from worker.embed_sequence import EmbeddingsModel

IMAGE_BATCH = 600

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

# Global batch processor that will be initialized per worker process
batch_processor = None

class BatchProcessor:
    def __init__(self, batch_size: int = IMAGE_BATCH, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = queue.Queue()
        self.results = {}
        self.lock = threading.Lock()
        self.batch_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.batch_thread.start()
        self.EmbeddingsInference = EmbeddingsModel()
        logger.info(f"BatchProcessor initialized with batch_size={batch_size}, max_wait_time={max_wait_time}")
    
    def add_task(self, task_id: str, filename: str) -> queue.Queue:
        """Add a task to the batch queue and return a queue for the result."""
        result_queue = queue.Queue()
        # Minimal lock usage - only when modifying the shared dictionary
        self.results[task_id] = result_queue
        self.queue.put((task_id, filename))
        return result_queue
    
    def get_result(self, task_id: str, timeout: float = None) -> Any:
        """Wait for and return the result for a specific task."""
        # Check if the task exists without a lock
        if task_id not in self.results:
            raise KeyError(f"No task with ID {task_id} found")
        
        # Get the queue reference without a lock
        result_queue = self.results[task_id]
        
        # Wait for the result (this doesn't need a lock)
        result = result_queue.get(timeout=timeout)
        
        # Clean up the dictionary entry with minimal lock
        if task_id in self.results:
            del self.results[task_id]
            
        return result
    
    def _process_batches(self) -> None:
        """Background thread that processes batches when ready."""
        while True:
            batch_filenames = []
            task_ids = []
            start_time = time.time()
            
            if False:
                # Collect up to batch_size tasks or wait until max_wait_time
                while len(batch_filenames) < self.batch_size and time.time() - start_time < self.max_wait_time:
                    try:
                        # Wait for the remaining time or a small timeout
                        remaining_time = max(0, self.max_wait_time - (time.time() - start_time))
                        timeout = min(remaining_time, 0.01)
                        task_id, filename = self.queue.get(timeout=timeout)
                        batch_filenames.append(filename)
                        task_ids.append(task_id)
                    except queue.Empty:
                        # If queue is empty and we already have some tasks, process them
                        if batch_filenames:
                            break 
            else:
            # First wait for at least one task to arrive
                try:
                    task_id, filename = self.queue.get(timeout=self.max_wait_time)
                    batch_filenames.append(filename)
                    task_ids.append(task_id)
                except queue.Empty:
                    # No tasks arrived, start over
                    continue
                
                time.sleep(1)
                # Now pull as many tasks as possible from the queue without blocking
                remaining_capacity = self.batch_size - 1  # We already have one task
                
                # Try to get all available tasks up to remaining capacity
                while remaining_capacity > 0:
                    try:
                        # Non-blocking get - only get what's immediately available
                        task_id, filename = self.queue.get(block=False)
                        batch_filenames.append(filename)
                        task_ids.append(task_id)
                        remaining_capacity -= 1
                    except queue.Empty:
                        # No more tasks available immediately
                        break                
            # If we collected any tasks, process the batch
            if batch_filenames:
                batch_size = len(batch_filenames)
                logger.error(f"Processing batch of {batch_size} files")
                
                start_process_time = time.time()
                
                try:
                    # batch_results = batch_filenames 
                    batch_results = self.EmbeddingsInference.process_files_batch(input_image_filename_list=batch_filenames)
                    
                    # Distribute results back to individual tasks
                    for i, task_id in enumerate(task_ids):
                        # Check if the task still exists in results without a lock
                        if task_id in self.results:
                            # Send the specific result for this task
                            result = batch_results[i] if i < len(batch_results) else None
                            self.results[task_id].put(result)
                    
                    logger.info(f"Batch processed in {time.time() - start_process_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # On error, propagate the exception to all tasks in the batch
                    for task_id in task_ids:
                        # Check if the task still exists in results without a lock
                        if task_id in self.results:
                            self.results[task_id].put(None)  # or propagate the exception if preferred


EmbeddingsInference = EmbeddingsModel()

# batch_processor = BatchProcessor(batch_size=IMAGE_BATCH, max_wait_time=1.0)

logger.info("Batch processor initialized for worker")

@worker_process_init.connect
def initialize_batch_processor(**kwargs):
    """Initialize the batch processor when a worker process starts."""
    global batch_processor
    #batch_processor = BatchProcessor(batch_size=100, max_wait_time=1.0)
    logger.info("Batch processor initialized for worker")
    


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
    #logger.info(f" ****** Processing request input_string={input_string} task={task_id}")

    # Random sleep between 1 and 5 seconds
    time_start = time.time()
    temp_filename = f"/home/roman/PycharmProjects/comfyui/celery-main/worker/test.png"
    
    # AI load
    ret_dic: dict = {"input_string": input_string} # run_ai(prompt=input_string, filename=temp_filename)

    # Use batch processor if available, otherwise fallback to the original method
    global batch_processor
    output_fn = None
    
    if batch_processor is not None:
        logger.info(f"Task {task_id}: Using batch processor for {input_string}")
        result_queue = batch_processor.add_task(task_id, input_string)
        output_fn = batch_processor.get_result(task_id)
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
        # output_fn_list = [input_string] 
        
    ret_dic: dict = {"input_string": input_string, "output_fn": output_fn_list[0]}

    ret_str:str = json.dumps(ret_dic)
    logger.info(f" ******* Task_id={task_id} input={input_string}: {time.time() - time_start:.2f} seconds")

    
    # Ensure the destination directory exists. If not, create it.
    # os.makedirs(destination_dir, exist_ok=True)
    #shutil.copy(temp_filename, destination_dir)
    
    destination_dir = "/tmp/"
    new_filename = f"task_id_{task_id}.png" # {str(task_id)}.png",
    destination_path = f"{destination_dir}{new_filename}"
    uploaded_file_name = f"test001/{new_filename}"
    gcs_link:str = "gcs://none/none"
    if False:
        # gcs
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