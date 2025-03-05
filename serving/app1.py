# app.py

from celery import Celery
import time

# Configure Celery app
#app = Celery('my_app', broker='redis://localhost:6379/0')
app: Celery = Celery(
    "tasks",
    broker=f"redis://localhost:6379/0",
    backend=f"redis://localhost:6379/1",
) 
@app.task
def process_data(data):
    """Simple task to process data."""
    print(f"Hey! I am processing: {data}")
    time.sleep(int(data))  # Wait for 5 seconds
    return f"hey! Processed {data}"
 