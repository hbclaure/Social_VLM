from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests
import torch
import os

# Set GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enable memory efficient attention
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load model and processor with fast image processing
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-7b",
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
processor = InstructBlipProcessor.from_pretrained(
    "Salesforce/instructblip-vicuna-7b",
    use_fast=True  # Enable fast image processing
)

# No need to manually move model to device as device_map handles it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable model caching
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    # Load image
    image = Image.open("/home/hc864/Social_VLM/images/short1.png")

    # Your prompt
    prompt = "How many *distinct human participants* are seated around the table? Only count people who are clearly visible. Answer with a number (e.g., 1, 2, 3)."
    
    #How many people are visible around the table? Answer with a number only (e.g., 1, 2, 3)."
    
    #"Given the image, how many human participants are visible around the table? Answer with a single number. Please answer with a single number (e.g., 1, 2, 3)."
    #"In the image the robot is in charge of distributing wooden blocks to participants.  There are two human teammates. The task is for the people to create the tallest possible tower using these blocks. The person who places the most block gains an additional reward. The robot's role is to pass these blocks to people. It can only give a block one at a time. Do you pass it to the individual on your: a) right or b) left?. Respond with A or B depending on your answer."

    # Preprocess
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    # Generate with optimized parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=3,  # Reduced from 5 to 3 for faster generation
        min_length=50,
        repetition_penalty=1.5,
        length_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
        use_cache=True  # Enable KV-caching
    )

    # Print only the model's response (excluding the prompt)
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    response = response.replace(prompt, "").strip()  # Remove the prompt from the response

    print("\nModel's Response:")
    print(response)

