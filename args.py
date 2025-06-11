import argparse
import glob
import os



def parse_args(require_experiment=True):
    parser = argparse.ArgumentParser(description="")


    parser.add_argument(
            "--model",
            type=str,
            help="name of the model to use",
            choices=[
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-001",
            "vertex/publishers/meta/models/llama-3.3-70b-instruct-maas",
            "vertex/publishers/google/models/gemini-1.5-pro-001",
            "vertex/publishers/google/models/gemini-1.5-flash-001",
            "vertex/publishers/google/models/gemma3"

            ],
            default="gemini-1.5-pro-001"
        )
    
    args = parser.parse_args()

    return args