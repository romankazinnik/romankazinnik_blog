# Maximizing Inference API Throughput: Lessons from the Trenches

## TL;DR
Built a high-performance inference API by benchmarking anlysis and identifying and addressing system bottlenecks and scaling the CPU processing. 

## Introduction

In the world of machine learning deployment, serving models efficiently at scale is often as challenging as training them. This post shares key insights from my building a high-throughput inference API prototype designed to maximize system resource utilization.


## The Inference Performance Puzzle

My goal was straightforward but ambitious: create an API that pushes maximum possible throughput by fully utilizing allavailable resources, whether CPU cores or GPUs—whichever hits their limit first.

### Initial Performance Analysis

My first task was benchmarking the sequential inference pipeline. I measured latencies across key components:

**Table 1: Component Latencies**
- Image disk I/O
- CPU image processing
- GPU single inference
- GPU batched inference

<img src="images/table1.png" width="500" height="400" />

An interesting but expected discovery emerged: while GPUs could process batches of 200 images nearly as fast as a single image, this theoretical advantage wasn't materializing in practice. 

### The Bottleneck Revelation

**Figure 1** revealed the culprit: CPU-bound image loading and processing was throttling GPU performance. This created an unexpected outcome—batching, which is considered essential for training efficiency, provided only minimal speedup for inference.

<img src="images/inf_fig1.png" width="800" height="500" />

**Table 2** confirmed this finding, showing actual observed latencies closely matching theoretical expectations based on:

<img src="images/inf_table2.png" width="600" height="500" />

```
total latency = API latency/number of workers + max(GPU latency/number of GPUs, CPU latency/number of workers)
```

## Experiments

Run sequential benchmark test instructions:

```
> python3 embed_sequence.py 1 200

cpu IO and processing latency=10.0ms 

throughput=77.71 image/sec latency=12.9ms

GPU inference latency (last cycle)=2.7ms

```

 Run inference API instructions:
```
> python3 stress_api.py 10 100    10000

> uv run uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload

> uv run celery -A worker.worker_embed  worker --loglevel=warning 
--concurrency=1000 --pool=gevent --autoscale=200,200
```


## The Solution: FastAPI with Task Brokering

Based on these insights, I developed an inference API built on FastAPI with Redis and Celery as the task broker. This architecture:

1. Delivers millisecond-level latency
2. Achieves over 100 requests per second (7.93 ms latency)
3. Scales linearly with additional workers (**Figure 2**) up to the maximum number of available cores

<img src="images/inf_fig2.png" width="800" height="400" />
## Future Optimization Paths

With the current bottleneck identified, two clear optimization strategies emerge:

1. **Scale the CPU processing** (simpler):
   - Add more nodes/cores
   - Requires no architectural changes
   
2. **Offload to GPU** (more complex):
   - Move CPU-bound image processing to the GPU
   - Add more GPUs to handle the increased workload
   - Requires code refactoring

## Conclusion

Sometimes the simplest solutions are the most elegant. By identifying that CPU processing was the actual bottleneck—not GPU inference—I was able to build a streamlined, high-performing inference API without overengineering. The system not only performs exceptionally well but offers a clear, data-driven path for future scaling. Remember: measure first, optimize second!

## The Power of Analytical Validation

One of the most valuable aspects of this project was the close alignment between analytical predictions and observed performance. When your system's real-world behavior matches your theoretical model, it provides strong evidence that your implementation is nearly optimal and likely bug-free. This analytical validation approach saved countless hours that might have been spent hunting phantom performance issues or implementing unnecessary optimizations, and focus optimization efforts precisely where they would have the greatest impact.

## References 

[1] https://roman-kazinnik.medium.com/market-ai-democratizing-gpu-and-model-access-41cb3dbf1052

[2] https://github.com/romankazinnik/romankazinnik_blog/tree/master/serving

[3] https://github.com/romankazinnik/romankazinnik_blog/tree/master/serving_gpu

### Inference API architecture:

<img src="images/inf_fig3.png" width="800" height="300" />


### Source: https://derlin.github.io/introduction-to-fastapi-and-celery/03-celery/

<img src="images/inf_fig4.png" width="800" height="400" />
