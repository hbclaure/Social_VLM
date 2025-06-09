from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests
import torch

# Load model and processor
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load image
image = Image.open("/home/hc864/Social_VLM/images/longvid0013.jpg")

# Your prompt
prompt = "What is the robot doing?"

# Preprocess
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

# Generate
outputs = model.generate(**inputs, max_new_tokens=50)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])

