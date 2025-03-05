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


if __name__ == "__main__":
    submit_requests_sequentially()

