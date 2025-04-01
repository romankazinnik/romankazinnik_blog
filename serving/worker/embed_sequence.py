
import os
import sys
import glob
import numpy as np
import time
from typing import List, Optional, Union

from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
import torch
import torchvision.transforms as T
from PIL import Image
import PIL.JpegImagePlugin

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
        num_images (int): Number of images to read (default: 10)
        
    Returns:
        list: List of PIL.JpegImagePlugin.JpegImageFile objects
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
                #print(f"Loaded: {file_path}")
            else:
                print(f"Skipped: {file_path} (Not a JPEG image)")
    
    print(f"Successfully loaded {len(image_objects)} JPEG images")
    return image_objects, jpeg_files


class EmbeddingsModel:
    def __init__(self):
        self.device = "cuda" #  if torch.cuda.is_available() else "cpu"
        self.model_ckpt1 = "nateraw/vit-base-beans"
        #model_ckpt2 = "IDEA-Research/grounding-dino-tiny"

        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_ckpt1)
        self.model = AutoModel.from_pretrained(self.model_ckpt1)
        self.hidden_dim = self.model.config.hidden_size

        self.model.to(self.device)

        self.gpu_total_time = 0
        self.debug = False
        # Data transformation chain.
        self.transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
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
            image_batch_transformed = torch.stack(
                [self.transformation_chain(image) for image in images]
            )
            print(f"image_batch_transformed={image_batch_transformed.shape}")
            
            new_batch = {"pixel_values": image_batch_transformed.to(device)}
            
            with torch.no_grad():
                embeddings = self.model(**new_batch).last_hidden_state[:, 0]
                embeddings = embeddings.cpu()
            
            #embeddings = model(image_batch_transformed.to(device)).last_hidden_state[:, 0].cpu()
            #embeddings = processor(image_batch_transformed, return_tensors="pt").to(device)
            print(f"shape={len(images), {type(images[0])}} {type(embeddings)}"); #fff()
            return {"embeddings": embeddings}

        return pp

    def process_batch_model(self, image_batch_transformed)->torch.Tensor:
        new_batch = {"pixel_values": image_batch_transformed.to(self.device)}
        
        start_time = time.time()

        with torch.no_grad():
            embeddings = self.model(**new_batch).last_hidden_state[:, 0]#.cpu()
        
        self.gpu_total_time += time.time() - start_time

        if not self.debug and self.device == "cuda":
            embeddings = embeddings.cpu() 
            # print(f"embeddings.shape={embeddings.shape}")
        
        return embeddings


    def process_write_results(self, embeddings: Union[List[torch.Tensor],torch.Tensor], embeddings_filename_list: List[str]):
        
        if isinstance(embeddings, torch.Tensor): 
            embeddings_list: List[torch.Tensor] = list(torch.unbind(embeddings,dim=0)) # invers to stack
        elif isinstance(embeddings, list):
            embeddings_list = embeddings
        else:
            raise Exception("embeddings must be a torch.Tensor or a list of torch.Tensor")
        assert(isinstance(embeddings_list, list))
        
        for fn, embedding in zip(embeddings_filename_list, embeddings_list):
            # torch metadata 1MB: torch.save(embedding, fn)
            np.save(fn+".npy",embedding.numpy())

        return 


    def process_files_batch(self, input_image_filename_list: List[str], image_batch_transformed: Optional[torch.Tensor]=None)->List[str]:
        # open all images and convert to tensors        
        if image_batch_transformed is None:
            if isinstance(input_image_filename_list[0], str):
                jpeg_image_list: List[PIL.JpegImagePlugin.JpegImageFile] = [convert_file_to_image(image_filename) for image_filename in input_image_filename_list]
            else:
                jpeg_image_list = input_image_filename_list
            image_batch_transformed = torch.stack(
                [self.transformation_chain(image) for image in jpeg_image_list]
            )    

        embeddings_filename_list: List[str] = [f"{image_filename}.embeddings.pt" for image_filename in input_image_filename_list]

        if self.debug:
            # No device time
            _  = self.process_batch_model(image_batch_transformed)
        else:
            # device and IO time 
            embeddings: torch.Tensor = self.process_batch_model(image_batch_transformed)
            self.process_write_results(embeddings, embeddings_filename_list)

        return embeddings_filename_list
# Example usage

if __name__ == "__main__":

    EmbeddingsInference = EmbeddingsModel()

    num_requests = 10
    batch_size = 600
    rate = 1000
    EmbeddingsInference.debug = False

    if len(sys.argv) != 4:
        print("Usage: python script.py num_requests batch_size rate debug")
    if len(sys.argv) > 1:
        num_requests = int(sys.argv[1])
    if len(sys.argv) > 2:
        batch_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        rate = int(sys.argv[3])
    if len(sys.argv) > 4:
        EmbeddingsInference.debug = int(sys.argv[4])
        
    print(f"\n num_requests={num_requests} batch_size={batch_size} rate={rate} EmbeddingsInference.debug={EmbeddingsInference.debug}\n")
    
    # Replace with your directory path containing JPEG images
    images_path = "/home/roman/PycharmProjects/comfyui/celery-main/romankazinnik_blog/zillow/images/"
    
    
    # Read 10 JPEG images and create list of JpegImageFile objects
    #jpeg_images: List[PIL.JpegImagePlugin.JpegImageFile], fn_list:List[str] = read_jpeg_images(image_directory) # 40 images
    jpeg_images, fn_list = read_jpeg_images(images_path) # 40 images

    # Verify the objects are of the expected type
    img = jpeg_images[0]
    print(f"Image example: {type(img).__module__}.{type(img).__name__}  Size: {img.size} Mode: {img.mode}")

    start_time = time.time()

    max_num_images = num_requests * batch_size
    jpeg_images = jpeg_images * max_num_images # 40*1000=400009 images
    fn_list = fn_list * max_num_images
    
    num_success = 0        
    sleep_time = 1/rate
    log_rate = int(num_requests/10)

    # (2) CPU mem -> device mem
    jpeg_image_list: List[PIL.JpegImagePlugin.JpegImageFile] = [convert_file_to_image(image_filename) for image_filename in fn_list[:batch_size]]
    # (3) GPUI-100% test: on device mem
    image_batch_transformed = torch.stack([EmbeddingsInference.transformation_chain(image) for image in jpeg_image_list]).to(EmbeddingsInference.device)

    for i in range(num_requests):
        if i % log_rate == 0: 
            print(f"i={i}")
        
        # (1) IO disc -> CPU mem -> device mem
        
        batch = fn_list[i*batch_size:(i+1)*batch_size]  
        embed_fn_list: List[str] = EmbeddingsInference.process_files_batch(batch)
        
        # (2) CPU mem -> device mem
        
        # embed_fn_list: List[str] = EmbeddingsInference.process_files_batch(jpeg_image_list)
        
        # GPUI-100% test: on device mem
        
        #embed_fn_list: List[str] = EmbeddingsInference.process_files_batch(jpeg_image_list,image_batch_transformed)
        
        num_success += batch_size

        time.sleep(sleep_time)
        
    
    total_time = time.time()-start_time
    print(f"done={num_success} {total_time:.2f}sec, {num_success/total_time:.2f} image/sec model only={num_success/EmbeddingsInference.gpu_total_time:.2f}")