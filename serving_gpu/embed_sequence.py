
import os
import sys
import glob
import numpy as np
import time
from typing import List, Optional, Union, Tuple

from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
import torch.profiler
import torchvision.transforms as T
from PIL import Image
import PIL.JpegImagePlugin
import pandas as pd

def convert_file_to_image(file_path: str)->Optional[PIL.JpegImagePlugin.JpegImageFile]:
    """
    Converts a file path to a PIL.JpegImagePlugin.JpegImageFile object.
    """
                
    try:
        image = Image.open(file_path)
        # Check if the image is a JPEG
        if not isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
            image.close()
            raise ValueError(f" *** The image at {file_path} is not a JPEG image.")
        return image
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        #return None
        raise ValueError(f" *** {file_path} failed to open.")
    

def read_jpeg_images(directory_path)->Tuple[List[PIL.JpegImagePlugin.JpegImageFile], List[str]]:
    """
    Reads JPEG images from a specified directory and creates a list of 
    PIL.JpegImagePlugin.JpegImageFile objects.
    
    Args:
        directory_path (str): Path to the directory containing JPEG images
        
    Returns:
        tuple: (List of PIL.JpegImagePlugin.JpegImageFile objects, List of file paths)
    """
        
    # Get list of jpeg files in the directory
    jpeg_files = glob.glob(os.path.join(directory_path, "*.jpg"))
    jpeg_files.extend(glob.glob(os.path.join(directory_path, "*.jpeg")))
    
    # Sort files to ensure consistent behavior
    jpeg_files.sort()
    
    print(f" {len(jpeg_files)} JPEG images found in {directory_path}")
    
    # Read images and create list of JpegImageFile objects
    image_objects = []
    for file_path in jpeg_files:
            img = convert_file_to_image(file_path)
            # Verify this is actually a JPEG image
            if img is not None and img.format == "JPEG":
                image_objects.append(img)
            else:
                print(f"Skipped: {file_path} (Not a JPEG image)")
    
    print(f"Successfully loaded {len(image_objects)} JPEG images")
    return image_objects, jpeg_files


class EmbeddingsModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_ckpt1 = "nateraw/vit-base-beans"

        # Performance tracking
        self.perf_stats = {
            "disk_io_time": 0,
            "cpu_processing_time": 0,
            "gpu_transfer_time": 0,
            "gpu_inference_time": 0,
            "write_results_time": 0,
            "total_images": 0
        }

        # Initialize model and transformations
        print(f"Loading model {self.model_ckpt1} to {self.device}...")
        start_time = time.time()
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_ckpt1)
        self.model = AutoModel.from_pretrained(self.model_ckpt1)
        self.hidden_dim = self.model.config.hidden_size
        
        self.model.to(self.device)
        self.model_load_time = time.time() - start_time
        print(f"Model loaded in {self.model_load_time:.2f} seconds")

        self.gpu_total_time = 0
        
        # Data transformation chain
        self.transformation_chain = T.Compose(
            [
                T.Resize(int((256 / 224) * self.extractor.size["height"])),
                T.CenterCrop(self.extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(mean=self.extractor.image_mean, std=self.extractor.image_std),
            ]
        )

    def extract_embeddings(self, model: torch.nn.Module):
        """Utility to compute embeddings."""
        device = model.device

        def pp(batch):
            images = batch["image"]
            
            start_time = time.time()
            image_batch_transformed = torch.stack(
                [self.transformation_chain(image) for image in images]
            )
            self.perf_stats["cpu_processing_time"] += time.time() - start_time
            
            print(f"image_batch_transformed={image_batch_transformed.shape}")
            
            new_batch = {"pixel_values": image_batch_transformed.to(device)}
            
            with torch.no_grad():
                embeddings = self.model(**new_batch).last_hidden_state[:, 0]
                embeddings = embeddings.cpu()
            
            print(f"shape={len(images), {type(images[0])}} {type(embeddings)}")
            return {"embeddings": embeddings}

        return pp

    def process_batch_model(self, image_batch_transformed: torch.Tensor)->torch.Tensor:
        # Measure GPU transfer time
        transfer_start = time.time()
        new_batch = {"pixel_values": image_batch_transformed.to(self.device)}
        self.perf_stats["gpu_transfer_time"] += time.time() - transfer_start
        
        # Measure inference time
        inference_start = time.time()
        with torch.no_grad():
            embeddings = self.model(**new_batch).last_hidden_state[:, 0]
        
        inference_time = time.time() - inference_start
        self.gpu_total_time += inference_time
        self.perf_stats["gpu_inference_time"] += inference_time

        transfer_start = time.time()
        embeddings = embeddings.to('cpu') # cpu()
        self.perf_stats["gpu_transfer_time"] += time.time() - transfer_start
            
        return embeddings

    def process_write_results(self, embeddings: Union[List[torch.Tensor], torch.Tensor], embeddings_filename_list: List[str]):
        write_start = time.time()
        
        if isinstance(embeddings, torch.Tensor): 
            embeddings_list: List[torch.Tensor] = list(torch.unbind(embeddings, dim=0)) # inverse to stack
        elif isinstance(embeddings, list):
            embeddings_list = embeddings
        else:
            raise Exception("embeddings must be a torch.Tensor or a list of torch.Tensor")
        assert(isinstance(embeddings_list, list))
        
        for fn, embedding in zip(embeddings_filename_list, embeddings_list):
            # torch metadata 1MB: torch.save(embedding, fn)
            np.save(fn+".npy", embedding.numpy())

        self.perf_stats["write_results_time"] += time.time() - write_start
        return

    def process_files_batch(self, input_image_filename_list: List[str], image_batch_transformed: Optional[torch.Tensor]=None)->List[str]:
        # Increase counter
        self.perf_stats["total_images"] += len(input_image_filename_list)
        
        # Open all images and convert to tensors
        if image_batch_transformed is None:
            disk_io_start = time.time()
            if isinstance(input_image_filename_list[0], str):
                jpeg_image_list: List[PIL.JpegImagePlugin.JpegImageFile] = [convert_file_to_image(image_filename) for image_filename in input_image_filename_list]
            else:
                jpeg_image_list = input_image_filename_list
            self.perf_stats["disk_io_time"] += time.time() - disk_io_start
            
            cpu_start = time.time()
            image_batch_transformed = torch.stack(
                [self.transformation_chain(image) for image in jpeg_image_list]
            )
            self.perf_stats["cpu_processing_time"] += time.time() - cpu_start

        embeddings_filename_list: List[str] = [f"{image_filename}.embeddings.pt" for image_filename in input_image_filename_list]

        # _ = self.process_batch_model(image_batch_transformed)
        # device and IO time 
        embeddings: torch.Tensor = self.process_batch_model(image_batch_transformed)
        self.process_write_results(embeddings, embeddings_filename_list)

        return embeddings_filename_list

    def print_performance_stats(self):
        """Print performance statistics in a readable format"""
        total_images = self.perf_stats["total_images"]
        if total_images == 0:
            print("No images processed yet.")
            return
            
        total_time = (self.perf_stats["disk_io_time"] + 
                     self.perf_stats["cpu_processing_time"] + 
                     self.perf_stats["gpu_inference_time"] + 
                     self.perf_stats["write_results_time"])
        
        print("\n===== PERFORMANCE STATISTICS =====")
        print(f"Total images processed: {total_images}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Throughput: {total_images/total_time:.2f} images/second")
        
        # Per-image stats
        print("\n----- Per Image Statistics -----")
        print(f"Disk I/O time: {1000*self.perf_stats['disk_io_time']/total_images:.2f} ms/image")
        print(f"CPU processing time: {1000*self.perf_stats['cpu_processing_time']/total_images:.2f} ms/image")
        print(f"GPU transfer time: {1000*self.perf_stats['gpu_transfer_time']/total_images:.2f} ms/image")
        print(f"GPU inference time: {1000*self.perf_stats['gpu_inference_time']/total_images:.2f} ms/image")
        print(f"Result writing time: {1000*self.perf_stats['write_results_time']/total_images:.2f} ms/image")
        
        # Percentage breakdown
        print("\n----- Pipeline Breakdown -----")
        print(f"Disk I/O: {100*self.perf_stats['disk_io_time']/total_time:.2f}%")
        print(f"CPU processing: {100*self.perf_stats['cpu_processing_time']/total_time:.2f}%")
        print(f"GPU transfer: {100*self.perf_stats['gpu_transfer_time']/total_time:.2f}%")
        print(f"GPU inference: {100*self.perf_stats['gpu_inference_time']/total_time:.2f}%")
        print(f"Result writing: {100*self.perf_stats['write_results_time']/total_time:.2f}%")
        print("=================================\n")

def run_with_pytorch_profiler(inference_model, images_path, batch_size, num_batches=1):
    """Run inference with PyTorch profiler enabled"""
    # Read JPEG images
    jpeg_images, fn_list = read_jpeg_images(images_path)
    
    # Ensure we have enough images
    max_num_images = num_batches * batch_size
    if len(jpeg_images) < max_num_images:
        # Repeat images to fill the batch
        repeat_factor = (max_num_images + len(jpeg_images) - 1) // len(jpeg_images)
        jpeg_images = jpeg_images * repeat_factor
        fn_list = fn_list * repeat_factor
    
    print(f"\nRunning PyTorch profiler with {num_batches} batches of {batch_size} images each")

    # Define profiler activities to monitor
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ]
    
    # Create profiler with trace export
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=num_batches),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(num_batches + 5):  # +2 for wait and warmup
            if i >= 5:  # Skip reporting for wait and warmup steps
                print(f"Processing batch {i-5}/{num_batches}")
            
                # Process a batch of images
                batch = fn_list[(i % num_batches) * batch_size : ((i % num_batches) + 1) * batch_size]
                inference_model.process_files_batch(batch)
                
                # Step the profiler
                prof.step()
    
    # Print profiler results
    print("\n===== PYTORCH PROFILER SUMMARY =====")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Export profiler results to CSV for further analysis
    df = pd.DataFrame(columns=["Name", "CPU time (ms)", "CUDA time (ms)"])
    for event in prof.key_averages():
        df = pd.concat([df, pd.DataFrame({
            "Name": [event.key],
            "CPU time (ms)": [event.cpu_time_total / 1000],
#            "CUDA time (ms)": [event.cuda_time_total / 1000]
        })], ignore_index=True)
    
    df.to_csv("profiler_results.csv", index=False)
    print("Detailed profiler results exported to 'profiler_results.csv'")
    
    # Return the profiler object for further analysis if needed
    return prof

# Example usage
if __name__ == "__main__":
    EmbeddingsInference = EmbeddingsModel()

    num_requests = 10
    batch_size = 32  # Reduced for profiling
    use_pytorch_profiler = 0

    if len(sys.argv) > 1:
        num_requests = int(sys.argv[1])
    if len(sys.argv) > 2:
        batch_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        use_pytorch_profiler = int(sys.argv[3])

    print(f"\n *** num_requests={num_requests} batch_size={batch_size} use_pytorch_profiler={use_pytorch_profiler}\n\n")
    
    # Replace with your directory path containing JPEG images
    images_path = "/home/roman/PycharmProjects/comfyui/celery-main/romankazinnik_blog/zillow/images/"
    
    if use_pytorch_profiler == 1:
        # Run with PyTorch profiler
        prof = run_with_pytorch_profiler(EmbeddingsInference, images_path, batch_size, num_batches=10)
        # Print our custom performance stats too
        EmbeddingsInference.print_performance_stats()
        sys.exit(0)
    
    # Read JPEG images and create list of JpegImageFile objects
    jpeg_images, fn_list = read_jpeg_images(images_path)

    # Verify the objects are of the expected type
    if jpeg_images:
        img = jpeg_images[0]
        print(f"Image example: {type(img).__module__}.{type(img).__name__}  Size: {img.size} Mode: {img.mode}")

    # Ensure we have enough images for all requests
    max_num_images = num_requests * batch_size
    repeat_factor = (max_num_images + len(jpeg_images) - 1) // len(jpeg_images)
    jpeg_images = jpeg_images * repeat_factor
    fn_list = fn_list * repeat_factor
    
    start_time = time.time()
    # (2) CPU mem -> device mem
    disk_io_start = time.time()
    jpeg_image_list = [convert_file_to_image(image_filename) for image_filename in fn_list[:batch_size]]
    disk_io_time = time.time() - disk_io_start
    
    # (3) CPU processing
    cpu_start = time.time()
    image_batch_transformed_cpu = torch.stack([EmbeddingsInference.transformation_chain(image) for image in jpeg_image_list]).to('cpu')
    cpu_time = time.time() - cpu_start
    
    print(f"\n *** Disk I/O WARMUP latency={1000*disk_io_time/batch_size:.1f}ms/image")
    print(f"\n *** CPU WARMUP processing latency={1000*cpu_time/batch_size:.1f}ms/image")
    
    # GPU transfer
    gpu_transfer_start = time.time()
    image_batch_transformed_gpu = image_batch_transformed_cpu.to(EmbeddingsInference.device)
    gpu_transfer_time = time.time() - gpu_transfer_start
    print(f"\n *** GPU WARMUP batch-tensor transfer latency={1000*gpu_transfer_time/batch_size:.1f}ms/image")

    gpu_render_start = time.time()
    _ = EmbeddingsInference.process_files_batch(jpeg_image_list, image_batch_transformed_gpu)
    gpu_render_time = time.time() - gpu_render_start
    print(f"\n *** GPU WARMUP render latency={1000*gpu_render_time/batch_size:.1f}ms/image\n")


    for i in range(2+10):
        if i == 2: # warmup
            gpu_render_start = time.time()
        _ = EmbeddingsInference.process_files_batch(jpeg_image_list, image_batch_transformed_gpu)
    gpu_render_time = time.time() - gpu_render_start
    print(f"\n *** GPU (10 requests) render latency={1000*gpu_render_time/batch_size/10:.1f}ms/image\n")

    EmbeddingsInference.gpu_total_time = 0
    
    num_success = 0        
    log_rate = max(1, int(num_requests/10))
    EmbeddingsInference = EmbeddingsModel()
    start_time = time.time()
    for i in range(num_requests):
        #if i % log_rate == 0: 
        #    print(f" *** Processing batch {i+1}/{num_requests}")
        batch = fn_list[i*batch_size:(i+1)*batch_size]  
        embed_fn_list = EmbeddingsInference.process_files_batch(batch)                
        num_success += batch_size

    total_time = time.time() - start_time
    print(f"\n batch size={batch_size} \n Processed {num_success} images")
    print(f"Throughput: {num_success/total_time:.2f} images/sec")
    print(f"Average latency: {1000*total_time/num_success:.1f}ms/image")
    
    # Print detailed performance statistics
    EmbeddingsInference.print_performance_stats()

    # Run one final batch with GPU preloaded data (fastest case)
    print("\nRunning one final batch with preloaded GPU data...")
    start_time = time.time()
    image_batch_transformed_gpu = image_batch_transformed_cpu.to(EmbeddingsInference.device)
    embed_fn_list_gpu = EmbeddingsInference.process_files_batch(jpeg_image_list, image_batch_transformed_gpu)
    print(f"GPU-only inference latency: {1000*(time.time()-start_time)/batch_size:.1f}ms/image")