from huggingface_hub import InferenceClient
from diffusers import AutoPipelineForText2Image,StableDiffusionPipeline
from diffusers.utils import load_image
import torch
from diffusers import DiffusionPipeline

def google_storage_file_upload(up_file: str = "/tmp/surfer_wave.png", bucket_name: str = "blog_inference", uploaded_file_name: str = "test_up.png")->str:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload a file to GCP bucket    
    blob = bucket.blob(uploaded_file_name)
    blob.upload_from_filename(up_file)
    #logger.info(f"File uploaded successfully:{bucket_name}, {up_file}, {uploaded_file_name}")
    gcs_link = f"gs://{bucket_name}/{uploaded_file_name}"
    return gcs_link 

def google_storage_file_download(uploaded_file_name: str = "test_up.png", bucket_name: str = "blog_inference", down_file: str = "/tmp/test_down.png")->None:
    """
     gsutil ls gs://blog_inference
     """
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # Download a file from GCP bucket
    blob = bucket.blob(uploaded_file_name)
    blob.download_to_filename(down_file)
    #logger.info(f"File downloaded successfully: {bucket_name}, {uploaded_file_name}, {down_file}")
    return

def run_ai(prompt: str, filename: str, is_cpu: bool=False, num_inference_steps:int=50)->dict:
    if is_cpu:
        # cpu only
        pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
        
        # prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        
        image = pipe(prompt=prompt,num_inference_steps=num_inference_steps).images[0]
    else:
        # Load the pipeline
        # pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        # Load the Stable Diffusion pipeline
        pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base") # 5GB
        
        # Move the pipeline to GPU if available
        pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        # Generate the image
        image = pipeline(prompt=prompt,num_inference_steps=num_inference_steps, guidance_scale=0.0).images[0]
    # Save the image
    image.save(filename)
    return {'filename':filename}

# uv pip install transformers torch torchaudio diffusers accelerate pillow
if __name__ == "__main__":
    print("start process")
    # Define the prompt
    prompt = "A cinematic photo of a surfer riding a giant wave at sunset"
    result = run_ai(prompt, filename="surfer_wave.png")
    print(f"result: {result}")