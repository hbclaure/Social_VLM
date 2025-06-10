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
    image = Image.open("/home/hc864/Social_VLM/images/longvid0140.jpg")

    # Your prompt
    prompt = "In the image the robot is in charge of distributing wooden blocks to participants. The robot is an orange robot arm. The wooden blocks are stacked behind it. There are two human teammates. The task is for the people to create the tallest possible tower using these blocks. The person who places the most block gains an additional reward. The robot's role is to pass these blocks to people. It can only give a block one at a time. Do you pass it to the individual on your: a) right or b) left?. Respond with A or B depending on your answer."

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
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True  # Enable KV-caching
    )

    # Get the generated sequence
    generated_sequence = outputs.sequences[0]

    # Compute log probabilities
    transition_scores = model.compute_transition_scores(
        outputs.sequences, 
        outputs.scores, 
        normalize_logits=True
    )

    # Calculate average log probability
    log_probs = transition_scores[0].cpu().numpy()
    avg_log_prob = log_probs.mean()

    # Print only the model's response (excluding the prompt)
    response = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    response = response.replace(prompt, "").strip()  # Remove the prompt from the response

    print("\nModel's Response:")
    print(response)
    print(f"\nAverage Log Probability: {avg_log_prob:.4f}")
    print(f"Per-token Log Probabilities: {log_probs}")

