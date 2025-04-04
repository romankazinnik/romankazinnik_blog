import logging
import os
import copy
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from celery import Celery, group
from celery.result import GroupResult
from fastapi import FastAPI, HTTPException, status
from .openapi_docs import (
    HEALTH_RESPONSES,
    PROCESS_REQUEST_EXAMPLES,
    PROCESS_RESPONSES,
    STATUS_REQUEST_EXAMPLES,
    STATUS_RESPONSES,
    BATCH_REQUEST_EXAMPLES,
    BATCH_RESPONSES,
)
from pydantic import AnyUrl, BaseModel, Field

# Enhanced FastAPI app with better documentation
app: FastAPI = FastAPI(
    title="Roman - API",
    description="Roman - API for processing a task asynchronously with Celery",
    version="1.0.0",
    docs_url="/uber", # "/docs",
    redoc_url="/reuber", # redoc",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

#REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
REDIS_HOST: str = "localhost"
# docker compose
celery_app: Celery = Celery(
    "tasks",
    broker=f"redis://{REDIS_HOST}:6379/0",
    backend=f"redis://{REDIS_HOST}:6379/1",
)


# Request Models
class ProcessUrlRequest(BaseModel):
    input_string: Union[str, List[str]] = Field(
        ...,
        min_length=1,
        description="The input string(s) to process. Can be a single string or a list of strings.",
    )


class StatusRequest(BaseModel):
    batch_id: str = Field(..., description="The batch ID for the group of tasks")

class batchRequest(BaseModel):
    pass # batch_id: str = Field(..., description="The batch ID for the group of tasks")

# Response Models
class ProcessResponse(BaseModel):
    batch_id: str = Field(..., description="The ID of the batch for tracking status")
    message: str = Field(..., description="Status message about the scheduled tasks")


class TaskResult(BaseModel):
    gcs_link: AnyUrl = Field(
        ..., description="Google Cloud Storage link to the processed result"
    )
    input_string: str = Field(
        ..., description="The original input string that was processed"
    )
    task_id: str = Field(..., description="The unique ID of the task")


class StatusResponse(BaseModel):
    ready: bool = Field(..., description="Whether all tasks in the batch are completed")
    tasks: Dict[str, int] = Field(
        ..., description="Count of tasks by status (e.g. PENDING, SUCCESS, FAILURE)"
    )
    results: List[Optional[TaskResult]] = Field(
        ...,
        description="Results of completed tasks. Will be null for tasks that are not completed.",
    )

class BatchStatusResponse(BaseModel):
    batch_ids: List[Optional[str]] = Field(..., description="all batch ids.")
    batches: Dict[str, str] = Field(..., description="Batches with all the tasks and results.")

# 
class BatchTaskStore:
    def __init__(self) -> None:
        self.m_batches: Dict[str, Optional[Dict[str,Any]]] = {}

glabal_batches = BatchTaskStore()

# create text to image
# uv run celery -A worker.worker  worker --loglevel=error --concurrency=1000 --pool=gevent --autoscale=10,20
global_worker_signature="worker.process_request" 
# creare image embedding 

@app.post(
    "/process",
    response_model=ProcessResponse,
    summary="Process one or more inputs",
    response_description="Returns a batch ID for tracking status",
    responses=PROCESS_RESPONSES,
    openapi_extra=PROCESS_REQUEST_EXAMPLES,
)
async def process_urls(request: ProcessUrlRequest) -> ProcessResponse:
    """
    Process one or more input strings asynchronously using Celery workers.

    - **input_string**: A single string or list of strings to process

    ## Notes:
    - All tasks are grouped together using Celery's group feature
    - The batch_id can be used with the /status endpoint to check progress
    """

    # Convert single input to list for consistent handling
    inputs: List[str] = (
        [request.input_string]
        if isinstance(request.input_string, str)
        else request.input_string
    )

    # Create a group of tasks for all inputs (even single inputs)
    tasks = []
    for input_str in inputs:
        # Create a signature for each input
        task = celery_app.signature(
            global_worker_signature, kwargs={"input_string": input_str}
        )
        tasks.append(task)

    # Execute the group
    job = group(tasks)
    group_result = job.apply_async()
    group_result.save()

    batch_id=group_result.id
    glabal_batches.m_batches[batch_id] = {"started":1}

    logger.info(f" ******* batch_id={batch_id} {str(  [f't.kwargs={json.dumps(t.kwargs,indent=4)}' for t in tasks]  )}")

    return ProcessResponse(
        batch_id=batch_id,
        message=f"Scheduled {len(inputs)} inputs for processing in batch {batch_id}",
    )

@app.post(
    "/batch_status",
    response_model=BatchStatusResponse,
    summary="Get status and results of all the batches",
    responses=BATCH_RESPONSES,
    openapi_extra=BATCH_REQUEST_EXAMPLES,
)
async def get_batch_status(request: batchRequest) -> BatchStatusResponse:
    """
    Get status of all batches and all tasks.

    ## Returns list of batches:
    - **ids**: batch ids
    - **batches**: List of batches with results for completed tasks
    """

    batch_list: List = []
    for batch_id in glabal_batches.m_batches.keys():
        restored_results: GroupResult = GroupResult.restore(
            batch_id, app=celery_app
        )
        if not restored_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch ID {batch_id} not found",
            )
        is_ready: bool = restored_results.ready()

        # Track task status counts
        tasks_by_status: Dict[str, int] = defaultdict(int)
        # Access the child tasks directly through the children property
        for task in restored_results.children:
            # logger.info(f" ******* task={str(task)} {task.state}") 
            tasks_by_status[task.state] += 1

        # Get task results, which will be None for tasks that aren't done
        results: List[Optional[Dict[str, Any]]] = [
            task.result for task in restored_results.children
        ]
        task_local_filename: List[Optional[Any]] = [
            None if task.result is None else task.result['local_filename'] for task in restored_results.children
        ]
        task_gcs_link: List[Optional[Any]] = [
            None if task.result is None else task.result['gcs_link'] for task in restored_results.children
        ]

        glabal_batches.m_batches[batch_id] = {
                'ready': is_ready,
                'tasks': tasks_by_status,
                'results': results,
                'task_gcs_link': task_gcs_link,
                'task_local_filename': task_local_filename
                #'json_str': json.dump(batches),
        } 
    if len(glabal_batches.m_batches) > 0:
        # :Dict[str,str] = {"none":"none"} # {key: json.dumps(value) for key, value in glabal_batches.m_batches.items()},
        #batch_ids: List[Optional[Dict[str, Any]]] = [
        batch_ids: List[Optional[str]] = [
            key for key, value in glabal_batches.m_batches.items()
        ]
        
        #batches_json:Dict[str,str] = {key: json.dumps(value, indent=4) for key, value in glabal_batches.m_batches.items()} 
        batches_json:Dict[str,str] = {key: str(value['task_gcs_link']) for key, value in glabal_batches.m_batches.items()} 
        
    else:
        batch_ids:Dict[str,str] = {"none":"none"}
        batches_json: Dict[str,str] = {"none":"no batches requested"}

    return BatchStatusResponse(
        batch_ids=batch_ids,
        batches=batches_json,
    )

@app.post(
    "/status",
    response_model=StatusResponse,
    summary="Get status and results of a batch",
    responses=STATUS_RESPONSES,
    openapi_extra=STATUS_REQUEST_EXAMPLES,
)
async def get_status(request: StatusRequest) -> StatusResponse:
    """
    Get status of all tasks in a batch by batch_id.

    - **batch_id**: The batch ID returned from the /process endpoint

    ## Returns:
    - **ready**: Whether all tasks in the batch are completed
    - **tasks**: Count of tasks by status (PENDING, SUCCESS, FAILURE, etc.)
    - **results**: List of results for completed tasks
    """

    restored_results: GroupResult = GroupResult.restore(
        request.batch_id, app=celery_app
    )
    if not restored_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch ID {request.batch_id} not found",
        )

    is_ready: bool = restored_results.ready()

    # Track task status counts
    tasks_by_status: Dict[str, int] = defaultdict(int)

    # Access the child tasks directly through the children property
    for task in restored_results.children:
        tasks_by_status[task.state] += 1
        #logger.info(f" ******* task.id={task.id} task.state={task.state}")

    # Get task results, which will be None for tasks that aren't done
    results: List[Optional[Dict[str, Any]]] = [
        task.result for task in restored_results.children
    ]

    return StatusResponse(
        ready=is_ready,
        tasks=tasks_by_status,
        results=results,
    )


@app.get(
    "/health",
    response_model=bool,
    summary="Health check endpoint",
    description="Simple health check endpoint that returns true if the service is running",
    response_description="Returns true if the service is running",
    responses=HEALTH_RESPONSES,
)
def health_check() -> bool:
    """Simple health check endpoint for monitoring purposes"""
    return True


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
