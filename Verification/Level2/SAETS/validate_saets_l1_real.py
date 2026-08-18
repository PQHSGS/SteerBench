
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
from pathlib import Path

# Import the reference LinearAdapter
sae_ts_path = str(Path("/home/aiotlab/mnt/hoplt/Benchmark/Code/SAE-TS/src"))
if sae_ts_path not in sys.path:
    sys.path.append(sae_ts_path)
from sae_ts.ft_effects.utils import LinearAdapter

def main():
    print("Validating SAE-TS Level 1 on Real Dataset...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Paths
    base_dir = "/home/aiotlab/mnt/hoplt/Benchmark/Verification_Results"
    adapter_path = f"{base_dir}/saets_adapter_real.pt"
    vector_path = f"{base_dir}/saets_vector_real.pt"
    
    # 1. Load Real Adapter
    print(f"Loading adapter from {adapter_path}...")
    # Need dimensions. Gemma 2b: d_model=2304. SAE: d_sae=16384 (width_16k)
    d_model = 2304
    d_sae = 16384
    
    adapter = LinearAdapter(d_model, d_sae).to(device)
    try:
        adapter.load_state_dict(torch.load(adapter_path, map_location=device))
    except Exception as e:
        print(f"Failed to load adapter: {e}")
        return

    # 2. Load Generated Vector
    print(f"Loading vector from {vector_path}...")
    try:
        loaded_data = torch.load(vector_path, map_location=device)
    except Exception as e:
         print(f"Failed to load vector: {e}")
         return
         
    # Check keys
    if 'metadata' not in loaded_data:
        print("Error: Metadata not found in vector file.")
        return
        
    feature_idx = loaded_data['metadata']['feature_idx']
    layer = loaded_data['metadata']['layer'][0] # list [12]
    
    print(f"Target Feature Index: {feature_idx}")
    
    my_vector = loaded_data['steering_vector'][layer]
    
    # 3. Compute Reference Vector (Reference Logic)
    # s = M_j_norm - lambda * M_b_norm
    # M_j = W[:, idx]
    # M_b = W @ b
    
    lambda_reg = loaded_data['metadata']['lambda']
    print(f"Lambda: {lambda_reg}")
    
    M_j = adapter.W[:, feature_idx]
    M_j_norm = M_j / (M_j.norm() + 1e-8)
    
    b_vec = adapter.W @ adapter.b
    b_vec_norm = b_vec / (b_vec.norm() + 1e-8)
    
    s = M_j_norm - lambda_reg * b_vec_norm
    ref_vector = s / (s.norm() + 1e-8)
    
    # 4. Compare
    sim = F.cosine_similarity(my_vector.unsqueeze(0), ref_vector.unsqueeze(0)).item()
    print(f"Cosine Similarity via Reference Logic: {sim:.6f}")
    
    if sim > 0.99:
        print("Level 1 Real Validation: PASSED")
    else:
        print("Level 1 Real Validation: FAILED")

if __name__ == "__main__":
    main()
