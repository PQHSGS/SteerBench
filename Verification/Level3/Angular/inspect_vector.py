
import torch
import sys

vec_path = "Vector/Angular/refusal_qwen_split_max_sim.pt"
try:
    data = torch.load(vec_path)
    print(f"Keys in vector file: {list(data.keys())}")
except Exception as e:
    print(f"Error loading vector: {e}")
