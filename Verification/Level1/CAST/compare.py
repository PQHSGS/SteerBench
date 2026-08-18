import torch
import json
import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def load_svec(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return {int(k): torch.tensor(v) for k, v in data["directions"].items()}

def run_level1():
    ref_vec_path = os.path.join(BASE_DIR, "Vector/CAST/ground_truth/steering_vector.pt.svec")
    ref_cond_path = os.path.join(BASE_DIR, "Vector/CAST/ground_truth/conditional_vector.pt.svec")
    
    my_vec_path = os.path.join(BASE_DIR, "Vector/CAST/extracted/steering_vector.pt")
    my_cond_path = os.path.join(BASE_DIR, "Vector/CAST/extracted/new_condition.pt")
    
    print("Loading vectors...")
    ref_vec = load_svec(ref_vec_path)
    ref_cond = load_svec(ref_cond_path)
    
    my_vec_data = torch.load(my_vec_path)
    if isinstance(my_vec_data, dict) and "steering_vector" in my_vec_data:
        my_vec = {int(k): v for k, v in my_vec_data["steering_vector"].items()}
    else:
        my_vec = my_vec_data.vectors if hasattr(my_vec_data, "vectors") else my_vec_data
        
    my_cond_data = torch.load(my_cond_path)
    # The condition is saved as a SteeringVector which might be a dict or a single tensor
    if hasattr(my_cond_data, "vector"):
        # If it's a SteeringVector with a single tensor
        my_cond = {7: my_cond_data.vector} # Assuming layer 7 for comparison as it's common
    elif hasattr(my_cond_data, "vectors"):
        my_cond = my_cond_data.vectors
    else:
        my_cond = my_cond_data

    print("\n" + "="*50)
    print("Level 1: Vector Alignment (Cosine Similarity)")
    print("="*50)
    
    # Steering Vector Alignment
    print("\nSteering Vector Layers:")
    common_layers = sorted(set(ref_vec.keys()) & set(my_vec.keys()))
    for l in common_layers:
        v1 = ref_vec[l].float().cpu().flatten()
        v2 = my_vec[l].float().cpu().flatten()
        cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        print(f"Layer {l:2d}: Cosine Similarity = {cos:.6f}")

    # Conditional Vector Alignment
    print("\nConditional Vector Layers:")
    if isinstance(my_cond, torch.Tensor):
        # If it's just one tensor, compare against ref_cond[7]
        v1 = ref_cond[7].float().cpu().flatten()
        v2 = my_cond.float().cpu().flatten()
        cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        print(f"Layer 7 (Assumption): Cosine Similarity = {cos:.6f}")
    else:
        common_cond_layers = sorted(set(ref_cond.keys()) & set(my_cond.keys()))
        for l in common_cond_layers:
            v1 = ref_cond[l].float().cpu().flatten()
            v2 = my_cond[l].float().cpu().flatten()
            cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            print(f"Layer {l:2d}: Cosine Similarity = {cos:.6f}")

if __name__ == "__main__":
    run_level1()
