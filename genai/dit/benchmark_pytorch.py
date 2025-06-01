import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from PIL import Image
import os
import gc
class CenterCropAndResize(torch.nn.Module):
    """Custom transform to center crop from 512x256 to 256x256, then resize to 1024x1024."""
    
    def __init__(self, crop_size=(256, 256), resize_size=(1024, 1024), interpolation=Image.BILINEAR):
        super().__init__()
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.interpolation = interpolation
    
    def forward(self, img):
        """
        Apply center crop and resize to image.
        
        Args:
            img (PIL Image): Input image of size 512x256
            
        Returns:
            PIL Image: Processed image of size 1024x1024
        """
        # Get image dimensions
        width, height = img.size  # PIL format is (width, height)
        
        # Center crop from 512x256 to 256x256
        # Calculate crop coordinates (left, top, right, bottom)
        crop_width, crop_height = self.crop_size
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        # Crop the image
        img = img.crop((left, top, right, bottom))
        
        # Resize to 1024x1024 using bilinear interpolation
        img = img.resize((self.resize_size[0]//2, self.resize_size[1]//2), Image.NEAREST)
        img = img.resize(self.resize_size, self.interpolation)
        if img.shape[-1] == 3 or img.shape[-1] == 1:  # If it's [bs, Y, X, C]
            # Transpose from [bs, Y, X, C] to [bs, C, Y, X]
            img = img.permute(2, 0, 1)        

        return img

def create_image_dataloader(image_path, batch_size=100, num_workers=16, shuffle=True):
    """
    Create a PyTorch DataLoader for loading and preprocessing images.
    
    Args:
        image_path (str): Path to directory containing class folders (e.g., "path/")
        batch_size (int): Number of images per batch
        num_workers (int): Number of parallel workers for data loading
        shuffle (bool): Whether to shuffle the dataset
    
    Returns:
        DataLoader: PyTorch DataLoader ready for training
    """
    
    # Define transforms
    transform = transforms.Compose([
        CenterCropAndResize(crop_size=(256, 256), resize_size=(1024, 1024), interpolation=Image.BILINEAR),
        transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
        # ToTensor automatically converts from HWC to CHW format
    ])
    
    # Create dataset using ImageFolder
    dataset = datasets.ImageFolder(
        root=image_path,
        transform=transform,
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # Faster GPU transfer
        drop_last=False,   # Keep incomplete last batch
        persistent_workers=False,
        prefetch_factor=2,
    )
    
    return dataloader, dataset

import time
from dit_aux import plot, plot2

# Usage examples
if __name__ == "__main__":

    image_path = "pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/train_class"

    total_iter = 100

    # Option 1: If you have class folder structure (path/class1/, path/class2/, etc.)
    def run_dataload(batch_size=100, num_workers=16):
        try:
            dataloader, dataset = create_image_dataloader(
                image_path=image_path,  # Parent directory containing class folders
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True
            )
            print(f"Dataloader: {dataloader.__len__()} dataset: {dataset.__len__()}")
        
            print(f"Dataset size: {len(dataset)}")
            print(f"Number of classes: {len(dataset.classes)}")
            print(f"Classes: {dataset.classes}")
            
            # Test first batch
            runtime_list = []
            start = time.time()
            for batch_idx, (images, labels) in enumerate(dataloader):
                end = time.time()
                if batch_idx > total_iter:
                    break
                if False:
                    print(f"\nBatch {batch_idx + 1}:")
                    print(f"  Images shape: {images.shape}")  # Should be (100, 3, 1024, 1024)
                    print(f"  Labels shape: {labels.shape}")   # Should be (100,)
                    print(f"  Image dtype: {images.dtype}")
                    print(f"  Min value: {images.min():.3f}")
                    print(f"  Max value: {images.max():.3f}")
                runtime_ms = (end - start) * 1000
                runtime_list.append(runtime_ms)
                start = time.time()                
        except Exception as e:
            print(f"ImageFolder failed: {e}")

        dataloader, dataset = None, None
        gc.collect()  
        return runtime_list[0], sum(runtime_list[1:]) / len(runtime_list[1:])
 
    batch_size_list = [10 , 20] # [8, 16] # 100, 200
    num_workers_list =  [1, 2, 4, 8, 16, 24, 32, 40, 48, 60] # 16, 24, 32, 48] # [1,2,4,8,16,24,48,60]
    m1 = {}
    m2 = {}
    for batch_size in batch_size_list:
        ms_list = []
        for num_workers in num_workers_list:
            time_first, time_after_first_ms = run_dataload(batch_size, num_workers)
            
            # time_after_first_ms += time_first


            # normalize by batch size
            time_after_first_ms /= batch_size
            # round to 1 decimal place
            #time_after_first_ms = int(0.5 + time_after_first_ms * 20) / 20.0

            ms_list.append(time_after_first_ms)
            print(f"\nnum_workers={num_workers}: time_first={time_first:.2f} ms, time_after_first_ms={time_after_first_ms:.2f} ms\n------\n")
        ms_list_invert = [1000/x for x in ms_list]
        m1[batch_size] = ms_list
        m2[batch_size] = ms_list_invert

    plot(ms_list, num_workers_list, y_title="Time per image (ms)", image_fn="torch_image_ms_per_worker")
    plot(ms_list_invert, num_workers_list, y_title="Images per second", image_fn="torch_image_batches_per_second")
    
    plot2(m1[batch_size_list[0]], m1[batch_size_list[1]], num_workers_list=num_workers_list, 
          y_title="Time per Image (ms)", x_title="Number of workers", image_fn="torch_image_ms_per_worker_2",
          batth_sizes=batch_size_list)
    plot2(m2[batch_size_list[0]], m2[batch_size_list[1]], num_workers_list=num_workers_list, 
          y_title="Images per second", 
          x_title="Number of workers", 
          image_fn="torch_image_batches_per_second_2",
          batth_sizes=batch_size_list)
