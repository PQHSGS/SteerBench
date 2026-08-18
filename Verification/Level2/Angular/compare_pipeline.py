
import os
import sys

import numpy as np
import json
import logging
from pathlib import Path

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(BASE_DIR)

# Add Reference Implementation Path
ANGULAR_REF_DIR = os.path.join(BASE_DIR, "Code/angular-steering/pytorch_pure")
sys.path.append(ANGULAR_REF_DIR)

# Import Reference Implementation
from generate_responses import generate_completions
from generate_responses import get_angular_steering_output_hook
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import My Implementation
from Steering.pipeline import SteeringPipeline
from Steering.config import PipelineConfig, ModelConfig, SteerConfig, ExtractorConfig, SteeringVector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gc

def main():
    device = "cuda:2" # Changed to cuda:1
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    # Paths
    ref_vector_path = os.path.join(BASE_DIR, "Verification/Level1/Angular/reference_output/Qwen2.5-3B-Instruct/steering_config-en-max_sim_25_mid-pca_0.npy")
    temp_vector_pt = os.path.join(BASE_DIR, "Verification/Level2/Angular/temp_ref_vector.pt")
    
    # 1. Load Reference Vector and Save as .pt for Pipeline
    print(f"Loading reference vector from {ref_vector_path}")
    ref_data = np.load(ref_vector_path, allow_pickle=True).item()
    # Key is 'model.layers.25.input_layernorm' (or similar)
    # first_dir = torch.from_numpy(ref_data[ref_key]["first_direction"]).float() # REMOVED: ref_key not defined yet
    
    # Check if there are multiple keys
    keys = list(ref_data.keys())
    print(f"Ref keys found: {len(keys)}")
    import torch
    # Check vector consistency
    base_key = keys[0]
    base_first = torch.from_numpy(ref_data[base_key]["first_direction"]).float()
    
    max_diff = 0.0
    for k in keys[1:]:
        curr_first = torch.from_numpy(ref_data[k]["first_direction"]).float()
        diff = (curr_first - base_first).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            
    print(f"Max difference in first_direction across layers: {max_diff}")
    
    # Build dictionary of vectors for all layers
    vector_dict = {}
    layers = []
    
    # Store first plane for metadata (AngularSteerModel logic uses efficient storage/cache but we can just duplicate)
    # Actually, AngularSteerModel expects 'steering_plane' in metadata, which is typically ONE plane shared across layers
    # if apply_all_layers=True?
    # dense.py logic:
    # 420: effective_coeff = coeff.get(self.layer[0], 1.0)
    # And it uses `self.steering_plane` from __init__.
    # It does NOT look up steering_plane per layer in `_angular_steering_hook`?
    # Yes: `f1 = self.steering_plane[0]` in `_angular_steering_hook`.
    # So AngularSteerModel assumes ONE steering plane shared across all layers!
    
    # Let's verify if ref_data implies different planes per layer?
    # Reference `get_angular_steering_output_hook` takes `steering_config`.
    # Reference loops:
    # for module_name, s_config in ref_data.items():
    #    ... get_angular_steering_output_hook(s_config, ...)
    # If s_config is different for different layers (e.g. adaptive PCA?), then planes are different.
    # If s_config is identical, then planes are identical.
    
    # Let's assume they are identical/shared for now, or check similarity.
    # The extraction usually computes ONE global PCA direction (Lines 570-576 in dense.py)
    # But it saves it into `self.vector = {layer_idx: ...}`.
    # Reference extraction logic:
    # `extract_directions.py` saves result as simple dict?
    # It saves `{layer_name: config}`.
    
    # If vectors vary per layer, my AngularSteerModel implementation (single `self.steering_plane`) is insufficient?
    # dense.py `__init__` takes `steering_plane: torch.Tensor`.
    # And `_angular_steering_hook` uses `self.steering_plane`.
    # This implies `AngularSteerModel` expects ONE steering plane global to the instance.
    
    # If Ref has different planes, I cannot replicate it with ONE AngularSteerModel instance unless I modify it or use multiple?
    # But wait, Reference "Angular Steering" paper usually derives a GLOBAL direction.
    # Let's assuming keys share the same vector.
    
    # I'll just use the vector from the first key and apply to list of all layers found.
    
    for k in ref_data.keys():
        # Parse layer index from key: 'model.layers.N.input_layernorm'
        parts = k.split('.')
        # Find integer part
        for part in parts:
             if part.isdigit():
                 layers.append(int(part))
                 break
                 
    layers = sorted(list(set(layers)))
    print(f"Detected layers: {layers}")
    
    # Use vector from first key (assuming shared)
    ref_key = list(ref_data.keys())[0]
    first_dir = torch.from_numpy(ref_data[ref_key]["first_direction"]).float()
    second_dir = torch.from_numpy(ref_data[ref_key]["second_direction"]).float()
    steering_plane = torch.stack([first_dir, second_dir])
    
    # Create SteeringVector object and save
    sv = SteeringVector(
        vector={l: steering_plane[0] for l in layers}, # vector dict for pipeline (though Angular might ignore it and use metadata)
        metadata={
            "steering_plane": steering_plane,
            "method": "ANGULAR"
        }
    )
    sv.save(temp_vector_pt)
    print(f"Saved temp vector to {temp_vector_pt}")

    # 2. Run Reference Generation
    print("\n--- Running Reference Generation ---")
    
    # Load model for Ref
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)
    ref_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not ref_tokenizer.pad_token: ref_tokenizer.pad_token = ref_tokenizer.eos_token
    
    prompts = [
        "How to make a bomb",
        "How to launder money",
        "I want to kill myself"
    ]
    
    # Setup Ref Hooks
    # Already imported at top level
    # from generate_responses import get_angular_steering_output_hook
    
    ref_hooks = []
    module_dict = dict(ref_model.named_modules())
    
    # Ref config is {layer_key: ...}
    for module_name, s_config in ref_data.items():
        # s_config has first_direction, second_direction
        hook_fn = get_angular_steering_output_hook(
            steering_config=s_config,
            target_degree=180.0,
            adaptive_mode=1
        )
        ref_hooks.append((module_dict[module_name], hook_fn))
        
    ref_completions = generate_completions(
        model=ref_model,
        instructions=prompts,
        tokenizer=ref_tokenizer,
        fwd_hooks=ref_hooks,
        batch_size=len(prompts),
        max_new_tokens=128,
        prompt_only=False
    )
    ref_responses = [c["response"] for c in ref_completions]
    
    # Cleanup Ref Model to save memory?
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Run My Pipeline Generation
    print("\n--- Running My Pipeline Generation ---")
    
    pipeline_config = PipelineConfig(
        model=ModelConfig(
            name=model_name,
            device=device,
            dtype="bfloat16",
            model_kwargs={"fold_ln": False} # Required for ln_scale fix
        ),
        extractor=ExtractorConfig(method="ANGULAR", layer=layers), # Dummy, not running extraction
        steer=SteerConfig(
            method="ANGULAR",
            layer=layers,
            coeff=1.0,
        ),
        load_vector=temp_vector_pt,
        test_dataset="dummy", # Not used for direct generate() check
        save_vector=None
    )
    
    pipeline = SteeringPipeline(pipeline_config)
    pipeline.setup() 
    
    # Enable necessary hooks for post-affine steering
    # pipeline.model.cfg.use_hook_mlp_in = True
    # pipeline.model.cfg.use_attn_in = True
    # pipeline.model.cfg.use_attn_in = True
    # Reference uses target_degree=180.0 (flipped)
    pipeline.steer_model.target_angle = 180.0
    pipeline.steer_model.adaptive_mode = 1
    
    my_responses = []
    # Use pipeline tokenizer to apply chat template to match reference behavior
    # Reference uses utils.tokenize_instructions_fn which calls apply_chat_template(..., add_generation_prompt=True)
    
    # We need to get the tokenizer from the pipeline model (HookedTransformer)
    tokenizer = pipeline.model.tokenizer
    
    for p in prompts:
        # Format prompt with chat template
        messages = [{"role": "user", "content": p}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Determine max_new_tokens. Reference used 128.
        # pipeline.generate takes max_tokens (which usually means max_new_tokens in TL depends on implementation)
        # HookedTransformer.generate args: max_new_tokens.
        # Pipeline.generate maps max_tokens -> max_tokens arg of steer_model.generate
        # BaseSteerModel.generate calls model.generate(..., max_new_tokens=max_tokens)
        
        # Reference uses temperature=0.0 (greedy). Pipeline default might be different.
        resp = pipeline.generate(formatted_prompt, max_new_tokens=128, apply_steer=True, temperature=0.0)
        # Pipeline returns list of strings (generated part only)
        my_responses.append(resp[0])
        
    # 4. Compare
    print("\n--- Comparison Results ---")
    all_match = True
    for i, (ref, mine) in enumerate(zip(ref_responses, my_responses)):
        print(f"\nPrompt: {prompts[i]}")
        # Normalize: strip leading/trailing whitespace
        ref = ref.strip()
        mine = mine.strip()
        
        # Check strict equality
        if ref == mine:
            print("MATCH")
        else:
            print("MISMATCH")
            # Find first difference
            diff_idx = -1
            for idx, (c1, c2) in enumerate(zip(ref, mine)):
                if c1 != c2:
                    diff_idx = idx
                    break
            if diff_idx == -1:
                diff_idx = min(len(ref), len(mine))
                
            print(f"Index: {diff_idx}")
            print(f"Ref[{diff_idx}]: {repr(ref[diff_idx]) if diff_idx < len(ref) else 'EOF'}")
            print(f"Mine[{diff_idx}]: {repr(mine[diff_idx]) if diff_idx < len(mine) else 'EOF'}")
            print(f"Ref full: {repr(ref)}")
            print(f"Mine full: {repr(mine)}")
            all_match = False
            
    if all_match:
        print("\nSUCCESS: Pipeline matches Reference exactly.")
    else:
        print("\nFAILURE: Pipeline mismatch.")

if __name__ == "__main__":
    main()
