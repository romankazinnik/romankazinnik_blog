import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
batch_size = 2
images = [image, image] * batch_size
texts = [
    "a cat. a remote control.",
    "a cat. a remote control. a sofa.",
] * batch_size

print(len(images), len(texts))
for i in range(40):
    print(f"i={i}")
    inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

w, h = image.size
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[(h, w), (h, w)],
)
print(results)