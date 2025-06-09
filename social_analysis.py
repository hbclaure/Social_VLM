from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests
import torch
import os
from transformers import BitsAndBytesConfig

# Set CUDA memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load model and processor
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-7b",
    quantization_config=quantization_config,
    device_map="auto"
)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

# No need to manually move model to device as device_map="auto" handles it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image
image = Image.open("/home/hc864/Social_VLM/images/longvid0013.jpg")

# Your prompt
prompt = "What is the robot doing?"

# Preprocess
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

# Generate
outputs = model.generate(**inputs, max_new_tokens=50)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])

