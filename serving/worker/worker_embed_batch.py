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

IMAGE_BATCH = 600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = get_task_logger(__name__)

global_is_batch_processing_wait_each_get = False

class BatchProcessor:
    def __init__(self, batch_size: int = IMAGE_BATCH, max_wait_time: float = 1.0, is_batch_processing_full: bool = False):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = queue.Queue()
        self.results = {}
        self.lock = threading.Lock()
        
        self.batch_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.batch_thread.start()
        
        self.EmbeddingsInference = EmbeddingsModel()
        self.is_batch_processing_full = is_batch_processing_full
        self.wait_time_between_batches = 2 # seconds
        logger.info(f"BatchProcessor initialized with batch_size={batch_size}, max_wait_time={max_wait_time}")
    
    def add_task(self, task_id: str, filename: Union[str, np.ndarray]) -> queue.Queue:
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
        global global_is_batch_processing_wait_each_get
        while True:
            batch_filenames = []
            task_ids = []
            start_time = time.time()
            
            if global_is_batch_processing_wait_each_get:
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
                
                time.sleep(self.wait_time_between_batches)
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
                    if self.is_batch_processing_full:
                        # batch_results = batch_filenames 
                        assert(isinstance(batch_filenames[0], str))
                        batch_results = self.EmbeddingsInference.process_files_batch(input_image_filename_list=batch_filenames)
                    else:
                        #logger.error(f"{type(batch_filenames[0])}") # 
                        assert(isinstance(batch_filenames[0], np.ndarray))
                        
                        image_batch_transformed = torch.from_numpy(np.stack(batch_filenames))
                        
                        embeddings = self.EmbeddingsInference.process_batch_model(image_batch_transformed)

                        embeddings_list: List[torch.Tensor] = list(torch.unbind(embeddings,dim=0)) 

                        batch_results: List[np.ndarray] = [embedding.numpy() for embedding in embeddings_list]
                        
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
    
    def run_task_in_batch(self, task_id: str, input_string: str, EmbeddingsInference: EmbeddingsModel) -> str:
    
        if self.is_batch_processing_full:
            # Read files and write embeddings to files
            result_queue = self.add_task(task_id, input_string)
        else:
            # Single image processing and batch GPU processing
            jpeg_image: PIL.JpegImagePlugin.JpegImageFile = convert_file_to_image(input_string) 
            
            image_batch_transformed: np.ndarray = EmbeddingsInference.transformation_chain(jpeg_image).numpy()
            
            result_queue = self.add_task(task_id, image_batch_transformed)

        output_fn = self.get_result(task_id)
        
        if output_fn is not None and isinstance(output_fn, np.ndarray):
            # Save the embedding to a file
            embedding = output_fn
            output_fn = f"{input_string}.npy"
            np.save(output_fn, embedding)
        return output_fn

