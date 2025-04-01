import json
import logging
import time

from typing import Union, Optional, List

import requests

from utils import read_jpeg_images

IP_ADDRESS="localhost"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)



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

        single string:
        "input_string": "boy riding a bike"
    """
    url = f"http://{IP_ADDRESS}:8000/process"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    if isinstance(input_strings, str):
        payload = { "input_string" : f"{str(input_strings)}"}
    elif isinstance(input_strings, list):
        l1 = [ f"'{s}'"for s in input_strings]
        l2 = ",".join(l1)
        payload = { "input_string" : input_strings} #  f"[{l2}]"}
    else:
        return None 

    logger.info(f" ***** paylod = {payload}")

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


def check_status(batch_id_list: List[str])->int:
    """
    curl -X 'POST' \
    'http://localhost:8000/status' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "batch_id": "c901313f-b77d-4091-93cf-0f522db3872c"
    }'

    """
    url = f"http://{IP_ADDRESS}:8000/status"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    time_start = time.time()
    # {time.time() - time_start:.2f}
    num_success = 0
    num_tasks_done = 0
    while num_success != len(batch_id_list) and time.time() - time_start < 100:
        num_success = 0
        num_tasks_done = 0
        start_time = time.time()
        for batch_id in batch_id_list:

            payload = {"batch_id" : batch_id}
            # logger.info(f" payload={payload}")

            response = requests.post(url, headers=headers, data=json.dumps(payload))

            #print(response.status_code) 
            #print(response.json())
            #print(f"payload={payload}")
            result = response.json()
            if response.status_code == 200:  
                #for key, value in result.items():
                    #logger.info(f"*** {key}: {value}")           
                if result['ready']: # results, tasks
                    num_success += 1  
                    num_tasks_done += len(result['tasks'])
            else:
                logger.info(f"Error: response.status_code={response.status_code} result={result}")
        logger.info(f"done={num_success} ({len(batch_id_list)}) {num_tasks_done}")
        time.sleep(1)
    
    return num_success
    
if __name__ == "__main__":
    
    # batch_id = post_request_to_uber(input_strings="********* XXXXXXX *****", logger=logger)
    batch_id_list = []

    #batch_id = post_request_to_uber(input_strings=[" **** a ***", "===b ===="], logger=logger)
    images_path = "/home/roman/PycharmProjects/comfyui/celery-main/romankazinnik_blog/zillow/images/"

    
    jpeg_images, fn_list = read_jpeg_images(images_path) # 40 images


    #batch_id = post_request_to_uber(input_strings=fn_list[0], logger=logger)
    #batch_id_list.append(batch_id)
    
    start_time = time.time()

    total_images = 0

    request_images = 2
    num_requests = 2
    fn_list = fn_list * int(1+num_requests * request_images/len(fn_list))
    for i in range(num_requests):
        images_group = fn_list[i*request_images: (i+1)*request_images]
        batch_id = post_request_to_uber(input_strings=images_group, logger=logger)
        batch_id_list.append(batch_id)
        total_images += len(images_group)
    
    num_success = check_status(batch_id_list=batch_id_list)
    total_time = time.time()-start_time
    logger.info(f"done={num_success} ({len(batch_id_list)}) {total_time:.2f}sec, {total_images/total_time:.2f} image/sec")
    
    quit(-1) # Replace with your directory path containing JPEG images
    
    # Check status 