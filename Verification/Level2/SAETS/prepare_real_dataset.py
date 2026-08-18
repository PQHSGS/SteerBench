
import os
import json
import torch
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_effects():
    print("Downloading effects_2b.pt...")
    repo_id = "schalnev/sae-ts-effects"
    filename = "effects_2b.pt"
    local_dir = "/home/aiotlab/mnt/hoplt/Benchmark/Verification_Results"
    
    # Check if exists
    local_path = Path(local_dir) / filename
    if local_path.exists():
        print(f"File already exists at {local_path}")
        return str(local_path)
        
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded to {path}")
        return path
    except Exception as e:
        print(f"Error downloading: {e}")
        return None

def load_dataset():
    path = "/home/aiotlab/mnt/hoplt/Benchmark/TrainDataset/behaviour/refusal/CAA.jsonl"
    print(f"Loading dataset from {path}...")
    
    target_data = []
    contrast_data = []
    
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Schema: binary_choice
            # target_key="correct_prompt" -> "I want to kill..." (No, typically CAA correct is refusing)
            # data_registry says: target_key="correct_prompt", contrast_key="false_prompt"
            # In refusal context: 
            # "correct" usually means "refusal" (safe model behavior).
            # "false" usually means "compliance" (unsafe behavior).
            # But for steering "refusal", we want to steer TOWARDS refusal?
            # Or usually we want to steer towards "jailbreak" (compliance)?
            # The Registry for 'refusal' says: condition_harmful.json
            # target_key="correct_prompt"
            
            # For 'refusal_caa', let's check content.
            if 'correct_prompt' in item:
                target_data.append(item['correct_prompt'])
            if 'false_prompt' in item:
                contrast_data.append(item['false_prompt'])
                
    print(f"Loaded {len(target_data)} target samples and {len(contrast_data)} contrast samples.")
    return target_data[:10], contrast_data[:10] # Return subset for quick verify

if __name__ == "__main__":
    effects_path = download_effects()
    if effects_path:
        # Validate file integrity
        try:
            data = torch.load(effects_path, map_location='cpu')
            print(f"Effects file loaded. Keys: {data.keys()}")
            print(f"Features shape: {data['features'].shape}")
            print(f"Effects shape: {data['effects'].shape}")
        except Exception as e:
            print(f"Error loading effects file: {e}")
            
    load_dataset()
