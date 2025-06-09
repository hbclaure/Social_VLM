from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests
import torch

# Load model and processor
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

# Load image
image = Image.open(requests.get("https://example.com/your_image.jpg", stream=True).raw)

# Your prompt
prompt = "What should the robot do next to be fair?"

# Preprocess
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

# Generate
outputs = model.generate(**inputs, max_new_tokens=50)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
