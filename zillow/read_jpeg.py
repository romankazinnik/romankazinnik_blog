
import os
from PIL import Image
import glob

def read_jpeg_images(directory_path, num_images=10):
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
    
    # Limit to the requested number of images
    jpeg_files = jpeg_files[:num_images]
    
    if len(jpeg_files) < num_images:
        print(f"Warning: Only {len(jpeg_files)} JPEG images found in {directory_path}")
    
    # Read images and create list of JpegImageFile objects
    image_objects = []
    for file_path in jpeg_files:
        try:
            img = Image.open(file_path)
            # Verify this is actually a JPEG image
            if img.format == "JPEG":
                image_objects.append(img)
                print(f"Loaded: {file_path}")
            else:
                print(f"Skipped: {file_path} (Not a JPEG image)")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Successfully loaded {len(image_objects)} JPEG images")
    return image_objects



import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_ckpt1 = "nateraw/vit-base-beans"
#model_ckpt2 = "IDEA-Research/grounding-dino-tiny"

extractor = AutoFeatureExtractor.from_pretrained(model_ckpt1)
model = AutoModel.from_pretrained(model_ckpt1)
hidden_dim = model.config.hidden_size

model.to(device)

processor = AutoProcessor.from_pretrained(model_ckpt1)
#processor.to(device)
import torch
import torchvision.transforms as T



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
            embeddings = model(**new_batch).last_hidden_state[:, 0]#.cpu()
            #embeddings = model(image_batch_transformed.to(device)).last_hidden_state[:, 0].cpu()
            #embeddings = processor(image_batch_transformed, return_tensors="pt").to(device)
            print(f"shape={len(images), {type(images[0])}} {type(embeddings)}"); #fff()
        return {"embeddings": embeddings}

    return pp

# Example usage
if __name__ == "__main__":
    # Replace with your directory path containing JPEG images
    image_directory = "./images"
    
    
    
    if True:
        # Read 10 JPEG images and create list of JpegImageFile objects
        jpeg_images = read_jpeg_images(image_directory)
        jpeg_images = jpeg_images * 50
    else:
        import requests

        batch_size = 400
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw)
        jpeg_images = [image] * batch_size

    # Verify the objects are of the expected type
    for i, img in enumerate(jpeg_images[:2]):
        print(f"Image {i+1}: {type(img).__module__}.{type(img).__name__}")
        print(f"  Size: {img.size}")
        print(f"  Mode: {img.mode}")

    for i in range(100):
        print(f"i={i}")
        long_jpeg_images = jpeg_images
        embeds = extract_embeddings(model)({"image": long_jpeg_images})
        #print(f"embeds={embeds['embeddings'].shape}")