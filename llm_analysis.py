from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests 

import json
import pandas as pd
import re
import time
import numpy as np

# Vertex & Gemini
import vertexai
import google.generativeai as genai
from langchain_google_vertexai import VertexAI
from google.auth.exceptions import GoogleAuthError
from args import parse_args

# Set API and project
genai.configure(api_key="AIzaSyBYDGNyIbCxKU91j1UD1c0V5m2DuvcHaws") 
vertexai.init(project="llm-exploration")

# Set GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enable memory efficient attention
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load LLM model and tokenizer
model = genai.GenerativeModel("gemini-1.5-pro-001")
model_type = "gemini"



# No need to manually move model to device as device_map handles it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vlm_analysis():
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
        prompt = "Describe each person that is visible in the image. Then tell me how many people are visible in the image. Answer with a number only (e.g., 1, 2, 3)."
        #"How many *distinct human participants* are seated around the table? Only count people who are clearly visible. Answer with a number (e.g., 1, 2, 3)."
        
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

        #print("\nModel's Response:")
        #print(response)
        return response


def analyze_vlm_output(vlm_description):
    """
    Analyze the VLM's image description using an LLM
    """
    with torch.no_grad():
        # Create prompt for LLM using VLM's output
        llm_prompt = f"""Based on the following image description, analyze the social dynamics and provide insights:
        
            Image Description: {vlm_description}

            Please analyze:
            1. The social context
            2. Tell me how many people are visible in the image. Answer with a number only (e.g., 1, 2, 3)."""

        # Generate LLM response
        response = model.generate_content(llm_prompt)
        response_content = response.text




        return response_content

if __name__ == "__main__":

    print("\nVLM's Description:")
    vlm_output = vlm_analysis()
    print(vlm_output)
    # Example usage
    vlm_adjusted_output = f"There are {vlm_output} people visible in the image. They appear to be engaged in a task involving wooden blocks."
    
    print(vlm_adjusted_output)

    print("\nLLM's Analysis:")
    print(analyze_vlm_output(vlm_adjusted_output)) 