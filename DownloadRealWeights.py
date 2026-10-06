import os
import sys
from huggingface_hub import hf_hub_download

os.makedirs('C:\\Fractal-BLT\\Models', exist_ok=True)

models_to_download = [
    ("Qwen/Qwen2.5-0.5B", "model.safetensors"),
    ("stabilityai/stablelm-zephyr-3b", "model.safetensors"), 
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "model.safetensors")
]

for repo, filename in models_to_download:
    try:
        print(f"Downloading {repo}...")
        path = hf_hub_download(repo_id=repo, filename=filename, cache_dir='C:\\Fractal-BLT\\Models')
        print(f"Success! Saved to {path}")
    except Exception as e:
        print(f"Error downloading {repo}: {e}")
