from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

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
            2. Any potential power dynamics
            3. The role of each person in the scene
            4. Any notable interactions or relationships

            Provide a detailed analysis:"""

        # Generate LLM response
        response = model.generate_content(llm_prompt)
        response_content = response.text




        return response_content

if __name__ == "__main__":
    # Example usage
    vlm_output = "There are 2 people visible in the image. One person is standing on the left side of the table, and another person is seated on the right side. They appear to be engaged in a task involving wooden blocks."
    
    print("\nVLM's Description:")
    print(vlm_output)
    
    print("\nLLM's Analysis:")
    print(analyze_vlm_output(vlm_output)) 