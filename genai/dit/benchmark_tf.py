import tensorflow as tf
import os

def create_image_dataset(image_path, batch_size=100, num_workers=16):
    """
    Create a TensorFlow dataset for loading and preprocessing images.
    
    Args:
        image_path (str): Path to directory containing images (e.g., "path/class1/")
        batch_size (int): Number of images per batch
        num_workers (int): Number of parallel workers for data loading
    
    Returns:
        tf.data.Dataset: Preprocessed dataset ready for training
    """
    
    # Get list of all jpg files
    image_pattern = os.path.join(image_path, "*.jpg")
    
    # Create dataset from file paths
    dataset = tf.data.Dataset.list_files(image_pattern, shuffle=True)
    
    def preprocess_image(image_path):
        """Load and preprocess a single image."""
        # Read image file
        image = tf.io.read_file(image_path)
        
        # Decode JPEG image
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Convert to float32 and normalize to [0,1] range
        image = tf.cast(image, tf.float32) / 255.0
        
        # Crop from 512x256 to 256x256 (center crop)
        # Since original is 512x256, we'll crop 128 pixels from each side horizontally
        image = tf.image.crop_to_bounding_box(image, 0, 128, 256, 256)
        
        # Resize from 256x256 to 1024x1024 using bilinear interpolation
        image = tf.image.resize(image, [512, 512], method='bilinear')
        image = tf.image.resize(image, [1024, 1024], method='bilinear')

        # Transpose from HWC to CHW format (256, 256, 3) -> (3, 256, 256)
        image = tf.transpose(image, [2, 0, 1])
                                     
        return image
    
    # Apply preprocessing with parallel calls
    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=num_workers
    )
    
    # Create batches
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

import time
from matplotlib import pyplot as plt
from dit_aux import plot, plot2
# Usage example
if __name__ == "__main__":
    def busy_wait_with_work(duration):
        start_time = time.time()
        counter = 0
        while time.time() - start_time < duration:
            counter += 1  # Some trivial work to keep CPU busy
            
    total_iter = 10 # 1000

    # Create the dataset
    image_path = "./pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/class_train"
    def run_dataload(batch_size=100, num_workers=16):
        dataset = create_image_dataset(
            image_path=image_path,
            batch_size=batch_size,
            num_workers=num_workers
        )    
        # Iterate through batches
        runtime_list = []
        start = time.time()
        for batch_idx, batch_images in enumerate(dataset.take(total_iter)):  # Take first 5 batches
            if False:
                print(f"Batch {batch_idx + 1}:")
                print(f"  Shape: {batch_images.shape}")
                print(f"  Data type: {batch_images.dtype}")
                print(f"  Min value: {tf.reduce_min(batch_images):.3f}")
                print(f"  Max value: {tf.reduce_max(batch_images):.3f}")
            end = time.time()
            runtime_ms = (end - start) * 1000
            runtime_list.append(runtime_ms)                
            start = time.time()            
    
        # For training with a model
        # model.fit(dataset, epochs=10)
        return runtime_list[0], sum(runtime_list[1:]) / len(runtime_list[1:])
    
    batch_size_list = [100 , 200] # [8, 16] # 100, 200
    num_workers_list = [1, 2, 4, 8, 16, 24, 32, 40, 48, 60] # 16, 24, 32, 48] # [1,2,4,8,16,24,48,60]
    m1 = {}
    m2 = {}
    for batch_size in batch_size_list:
        ms_list = []
        for num_workers in num_workers_list:
            time_first, time_after_first_ms = run_dataload(batch_size, num_workers)
            
            # normalize by batch size
            time_after_first_ms /= batch_size
            # round to 1 decimal place
            time_after_first_ms = int(0.5 + time_after_first_ms * 20) / 20.0

            ms_list.append(time_after_first_ms)
            print(f"\nnum_workers={num_workers}: time_first={time_first:.2f} ms, time_after_first_ms={time_after_first_ms:.2f} ms\n------\n")
        ms_list_invert = [1000/x for x in ms_list]
        m1[batch_size] = ms_list
        m2[batch_size] = ms_list_invert

    plot(ms_list, num_workers_list, y_title="Time per image (ms)", image_fn="tf_image_ms_per_worker")
    plot(ms_list_invert, num_workers_list, y_title="Images per second", image_fn="tf_image_batches_per_second")
    
    plot2(m1[batch_size_list[0]], m1[batch_size_list[1]], num_workers_list=num_workers_list, 
          y_title="Time per Image (ms)", x_title="Number of workers", image_fn="tf_image_ms_per_worker_2")
    plot2(m2[batch_size_list[0]], m2[batch_size_list[1]], num_workers_list=num_workers_list, 
          y_title="Images per second", x_title="Number of workers", image_fn="tf_image_batches_per_second_2")
