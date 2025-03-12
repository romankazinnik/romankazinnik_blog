import logging
import os
import copy
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Tuple
import requests    

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
    docs_url="/uber_rider",
    redoc_url="/reuber_rider",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)


#REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
#REDIS_HOST: str = "localhost"
#celery_app: Celery = Celery(
#    "tasks",
#    broker=f"redis://{REDIS_HOST}:6379/2",
#    backend=f"redis://{REDIS_HOST}:6379/3",
#)
IP_ADDRESS="localhost"

IP_ADDRESS="192.168.7.230"

BUCKET_NAME="blog_inference"
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

##############################################

def post_request_to_uber(input_strings: Union[str,List[str]]="boy", logger=None)->Optional[str]:
    """
    Example:
        curl -X 'POST' \
        'http://localhost:8000/process' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
          "input_string": [
            "boy riding a bike",
            "girl riding a bike"
            ]
        }'
    """
    url = f"http://{IP_ADDRESS}:8000/process"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = { 'input_string' : f"{str(input_strings)}"}

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    #print(response.status_code)
    #print(response.json())
    #print(f"payload={payload}")
    result = response.json()
    batch_id = None
    if response.status_code == 200:    
        batch_id = result['batch_id']
        if logger is not None:
            for key, value in result.items():
                logger.info(f"*** {key}: {value}")                
    else:
        if logger is not None:
            logger.info(f"Error: response.status_code={response.status_code} result={result}")
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    return batch_id

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

    batch_id = post_request_to_uber(input_strings=inputs[0], logger=logger)
    glabal_batches.m_batches[batch_id] = {"started":1}

    return ProcessResponse(
        batch_id=batch_id,
        message=f"Scheduled {len(inputs)} inputs for processing in batch {batch_id}",
    )


########################################################

def post_request_to_uber_status(local_batch_id_list:Optional[List[str]]=None,logger=None) -> Tuple[List[str], Dict[str, str], Dict[str,List[str]]]:
    """
    Example:
    curl -X 'POST' \
    'http://localhost:8000/batch_status' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{}'"""    
    def convert_string_to_list(input_string):
        """
        Converts a string representation of a list like "['a.png','b.png']" 
        to an actual Python list ['a.png','b.png']
        """
        # Remove the outer brackets
        cleaned_string = input_string.strip()[1:-1]
        
        # Split by commas
        items = cleaned_string.split(',')
        
        # Clean each item (remove quotes and extra spaces)
        result = []
        for item in items:
            # Remove surrounding quotes and whitespace
            cleaned_item = item.strip().strip("'\"")
            result.append(cleaned_item)
        
        return result

    url = f"http://{IP_ADDRESS}:8000/batch_status"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload ={} #  { 'input_string' : f"{str(input_strings)}"}

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    #print(response.status_code)
    #print(response.json())
    #print(f"payload={payload}")
    result = response.json()
    batch_ids:List[str] = []
    gcs_path_dic:Dict[str,List[str]] = None
    batch_list_dic: Dict[str,str] = {}
    if response.status_code == 200:    
        batch_ids = result['batch_ids']
        if logger is not None:
            for key, value in result.items():
                logger.info(f"*** key={key}")           
        
        for key, value in result.items():
            if key == 'batches':
                gcs_path_dic = {}
                batch_list_dic = {}
                for b_id,fn_list_str in value.items():
                    result = convert_string_to_list(fn_list_str)           
                    if logger is not None:
                        logger.info(f"***======  {b_id}: {result}")     
                    gcs_path_dic[b_id] = result
                    batch_list_dic[b_id] = fn_list_str
                
    else:
        if logger is not None:
            logger.info(f"Error: response.status_code={response.status_code} result={result}")
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    if local_batch_id_list is not None:
        # local list of batch ids is a subset of remote GPU batch ids
        batch_ids = [key for key in batch_ids if key in local_batch_id_list]
        batch_list_dic = {key: value for key, value in batch_list_dic.items() if key in local_batch_id_list}
        gcs_path_dic = {key: value for key, value in gcs_path_dic.items() if key in local_batch_id_list}
    return batch_ids,batch_list_dic,gcs_path_dic

##################
def google_storage_file_download(uploaded_file_name: str = "test_up.png", bucket_name: str = "blog_inference", down_file: str = "/tmp/test_down.png"):
    """
     gsutil ls gs://blog_inference
     """
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # Download a file from GCP bucket
    blob = bucket.blob(uploaded_file_name)
    logger.info(f"downloading gs://{bucket_name}/{uploaded_file_name} to {down_file}...")
    blob.download_to_filename(down_file)
    #logger.info(f"File downloaded successfully: {bucket_name}, {uploaded_file_name}, {down_file}")
    return
#####################
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

    local_batch_ids: List[Optional[str]] = [
        key for key, value in glabal_batches.m_batches.items()
    ]
    batch_ids, batches_json, gcs_path_dic = post_request_to_uber_status(local_batch_id_list=local_batch_ids,logger=logger)
    #print(f"batch_ids,batch_list_dic,gcs_path_list=\n{batch_ids},\n,{batch_list_dic},\n,{gcs_path_list}")
    
    # TBD: download from gcs file names in gcs_path_list
    bucket_name=BUCKET_NAME
    batche_id_local_fn_dic: Dict[str,str]={}
    for batch_id in local_batch_ids:
        if batch_id in batches_json:
            glabal_batches.m_batches[batch_id] = {
                    'task_gcs_link': gcs_path_dic[batch_id],
            } 
            local_fn_list = []
            for gcs_fn in gcs_path_dic[batch_id]:
                prefix_str = f"gs://{bucket_name}/"
                uploaded_file_name = gcs_fn[len(prefix_str):]
                if len(uploaded_file_name) > 0 and uploaded_file_name.startswith('test001/'):
                    down_file=f"/tmp/{uploaded_file_name}"
                    google_storage_file_download(uploaded_file_name=uploaded_file_name, down_file=down_file)
                    local_fn_list.append(down_file)
            batche_id_local_fn_dic[batch_id] = ",".join(local_fn_list)
    
    assert(set(batch_ids).intersection(set(local_batch_ids))==set(local_batch_ids))
    return BatchStatusResponse(
        batch_ids=local_batch_ids,
        batches=batche_id_local_fn_dic,#batches_json
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
