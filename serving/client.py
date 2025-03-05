# client.py

from app1 import process_data
import time


def submit_requests_sequentially():
    # Submitting tasks sequentially
    results = []
    for i in range(1, 6):
        print(f"Submitting request {i}...")
        result = process_data.apply_async((f"Task-{i}",))  # Submit the task
        results.append(result)

    # Wait for the tasks to complete and retrieve the results
    # Check if the result is ready within 5 seconds
    for result in results[::-1]:
        start_time = time.time()
        while not result.ready() and (time.time() - start_time) < 5:
            print(f"Request {i} is still processing...")
            time.sleep(0.5)  # Sleep for 0.5 seconds and check again

        if result.ready():
            print(f"Request {i} completed. Result: {result.get()}")
        else:
            print(f"Request {i} did not complete in time.")

def retrive_completed_requests():
    results = []
    
    # Submitting 100 tasks sequentially
    for i in range(1, 10):
        print(f"Submitting request {i}...")
        result = process_data.apply_async((f"Task-{i}",))  # Submit the task to the queue
        results.append(result)
        time.sleep(0.1)  # Simulate a delay between requests (0.1 seconds)

    # Wait for up to 5 seconds to check if tasks have finished
    start_time = time.time()
    completed = 0

    while completed < len(results) and (time.time() - start_time) < 5:
        print(f"Checking status of {completed}/{len(results)} tasks...")
        for i, result in enumerate(results):
            if result.ready() and result.successful():
                completed += 1
                print(f"Task {i+1} completed. Result: {result.get()}")
            elif result.failed():
                print(f"Task {i+1} failed.")
        
        time.sleep(0.5)  # Sleep for 0.5 seconds before checking again

    if completed < len(results):
        print(f"Not all tasks completed within 5 seconds.")
    else:
        print(f"All tasks completed.")

if __name__ == "__main__":
    submit_requests_sequentially()
    retrive_completed_requests()

