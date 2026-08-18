
import torch
import numpy as np
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(BASE_DIR)

def main():
    # Paths
    my_vector_path = os.path.join(BASE_DIR, "Verification/Level1/Angular/my_output/vector.pt")
    ref_vector_path = os.path.join(BASE_DIR, "Verification/Level1/Angular/reference_output/Qwen2.5-3B-Instruct/steering_config-en-max_sim_25_mid-pca_0.npy")
    
    print(f"My Vector: {my_vector_path}")
    print(f"Ref Vector: {ref_vector_path}")
    
    # Load Mine
    my_data = torch.load(my_vector_path)
    # Keys found: ['steering_vector', 'metadata']
    if "metadata" in my_data and "steering_plane" in my_data["metadata"]:
        my_plane = my_data["metadata"]["steering_plane"].cpu()
    elif "steering_plane" in my_data:
         my_plane = my_data["steering_plane"].cpu()
    else:
         raise KeyError("Could not find 'steering_plane' in vector file.")
         
    # Extract layer from metadata if possible
    # metadata: {'method': 'ANGULAR', 'selected_layer': [[25]], ...}
    if "metadata" in my_data and "selected_layer" in my_data["metadata"]:
        my_layer = my_data["metadata"]["selected_layer"][0][0]
    elif "selected_layer" in my_data:
        my_layer = my_data["selected_layer"][0][0]
    else:
        # Fallback if I cant find it easily (e.g. if I used a different save method previously)
        print("Warning: selected_layer key not found, assuming match for now (will verify via vector check)")
        my_layer = 25 # Dummy fallback to match Ref for now if key missing
    print(f"My Layer: {my_layer}")
    
    # Load Ref
    ref_data = np.load(ref_vector_path, allow_pickle=True).item()
    
    # Ref data is dict {layer_key: {first_direction, second_direction}}
    print(f"Ref Data Keys: {list(ref_data.keys())}")
    
    # We want to find the key that corresponds to the layer we are inspecting.
    # If there are multiple, we pick the one matching '25' or closest.
    
    ref_key = None
    for k in ref_data.keys():
        if f".{my_layer}." in k or f"_{my_layer}_" in k or k.endswith(f".{my_layer}"):
            ref_key = k
            break
            
    if ref_key is None:
        print("Could not find matching key for my_layer in Ref Data. Using first key.")
        ref_key = list(ref_data.keys())[0]
    print(f"Ref Key: {ref_key}")
    ref_layer = int(ref_key.split('.')[2])
    print(f"Ref Layer: {ref_layer}")
    
    if my_layer != ref_layer:
        print("MISMATCH: Selected layers differ!")
        # Proceed to comparison anyway
    else:
        print("MATCH: Selected layers identical.")
        
    ref_first = torch.from_numpy(ref_data[ref_key]["first_direction"])
    ref_second = torch.from_numpy(ref_data[ref_key]["second_direction"])
    
    # Calculate cosine similarity
    cos_1 = torch.nn.functional.cosine_similarity(my_plane[0].unsqueeze(0), ref_first.unsqueeze(0))
    cos_2 = torch.nn.functional.cosine_similarity(my_plane[1].unsqueeze(0), ref_second.unsqueeze(0))
    
    print(f"First Direction Cosine: {cos_1.item():.6f}")
    print(f"Second Direction Cosine: {cos_2.item():.6f}")
    
    # Check strict equality (allow float error)
    if list(my_plane[0].shape) != list(ref_first.shape):
         print("Shape mismatch!")
         
    diff_1 = (my_plane[0] - ref_first).abs().max().item()
    diff_2 = (my_plane[1] - ref_second).abs().max().item()
    
    print(f"Max Diff First: {diff_1}")
    print(f"Max Diff Second: {diff_2}")
    
    if diff_1 < 1e-4 and diff_2 < 1e-4:
        print("SUCCESS: Vectors match.")
    else:
        print("FAILURE: Vectors mismatch.")

if __name__ == "__main__":
    main()
