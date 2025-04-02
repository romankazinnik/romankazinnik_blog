
import os
from PIL import Image
import glob
from typing import List
import numpy as np

import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
import torch
import torchvision.transforms as T
from PIL import Image
import PIL.JpegImagePlugin

from utils import read_jpeg_images
from utils import convert_file_to_image

device = "cuda" #  if torch.cuda.is_available() else "cpu"
model_ckpt1 = "nateraw/vit-base-beans"
#model_ckpt2 = "IDEA-Research/grounding-dino-tiny"

extractor = AutoFeatureExtractor.from_pretrained(model_ckpt1)
model = AutoModel.from_pretrained(model_ckpt1)
hidden_dim = model.config.hidden_size

model.to(device)

#processor = AutoProcessor.from_pretrained(model_ckpt1)
#processor.to(device)


# Data transformation chain.
transformation_chain = T.Compose(
    [
        # We first resize the input image to 256x256 and then we take center crop.
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)

def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["image"]
        image_batch_transformed = torch.stack(
            [transformation_chain(image) for image in images]
        )
        print(f"image_batch_transformed={image_batch_transformed.shape}")
        
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0]
            embeddings = embeddings.cpu()
        
        #embeddings = model(image_batch_transformed.to(device)).last_hidden_state[:, 0].cpu()
        #embeddings = processor(image_batch_transformed, return_tensors="pt").to(device)
        print(f"shape={len(images), {type(images[0])}} {type(embeddings)}"); #fff()
        return {"embeddings": embeddings}

    return pp

def process_batch_model(image_batch_transformed: torch.Tensor)->torch.Tensor:
    pass

def process_batch(image_list: List[PIL.JpegImagePlugin.JpegImageFile])->torch.Tensor:
    """
    Process a batch of input tensors using the model.
    
    Args:
        model: The PyTorch model to use for processing.
    """
    image_batch_transformed = torch.stack(
        [transformation_chain(image) for image in image_list]
    )    
    
    new_batch = {"pixel_values": image_batch_transformed.to(device)}
    
    with torch.no_grad():
        embeddings = model(**new_batch).last_hidden_state[:, 0]#.cpu()
    
    embeddings = embeddings.cpu() # .numpy()

    return embeddings   
    


def process_files_batch(input_image_filename_list: List[str])->List[str]:
    # open all images and convert to tensors
    jpeg_image_list: List[PIL.JpegImagePlugin.JpegImageFile] = [convert_file_to_image(image_filename) for image_filename in input_image_filename_list]
    
    # process the batch
    embeddings: torch.Tensor = process_batch(jpeg_image_list)
    # convert embeddings to list of tensors
    
    embeddings_list: List[torch.Tensor] = torch.unbind(embeddings) # invers to stack
    embeddings_filename_list: List[str] = [f"{image_filename}.embeddings.pt" for image_filename in input_image_filename_list]
    for fn, embedding in zip(embeddings_filename_list, embeddings_list):
        # torch metadata 1MB: torch.save(embedding, fn)
        np.save(fn+".npy",embedding.numpy())

    return embeddings_filename_list

# Example usage
if __name__ == "__main__":
    # Replace with your directory path containing JPEG images
    image_directory = "./images"
    
    
    
    if True:
        # Read 10 JPEG images and create list of JpegImageFile objects
        #jpeg_images: List[PIL.JpegImagePlugin.JpegImageFile], fn_list:List[str] = read_jpeg_images(image_directory) # 40 images
        jpeg_images, fn_list = read_jpeg_images(image_directory) # 40 images
    else:
        import requests

        batch_size = 400
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw)
        jpeg_images = [image] * batch_size

    # Verify the objects are of the expected type
    for i, img in enumerate(jpeg_images[:2]):
        print(f"Image {i}: {type(img).__module__}.{type(img).__name__}")
        print(f"  Size: {img.size}")
        print(f"  Mode: {img.mode}")

    jpeg_images = jpeg_images * 10000 # 40*100=4000 images
    fn_list = fn_list * 10000
    batch_size = 600
    for i in range(100):
        print(f"i={i}")
        
        if True:
            # Disc IO  
            batch = fn_list[i*batch_size:(i+1)*batch_size]
            embed_fn_list: List[str] = process_files_batch(batch)
        else:
            # No disc IO": GPU util 100%
            batch = jpeg_images[i*batch_size:(i+1)*batch_size]
            embeds = extract_embeddings(model)({"image": batch})
        #print(f"embeds={embeds['embeddings'].shape}")