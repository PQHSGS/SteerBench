import torch
import numpy as np
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# Add paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(BASE_DIR)
GT_CODE_PATH = os.path.join(BASE_DIR, "Code/angular-steering/pytorch_pure")
sys.path.append(GT_CODE_PATH)

# Import Ground Truth Hook
from generate_responses import get_angular_steering_output_hook
from Steering.steer_models.dense import AngularSteerModel

def main():
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Reference Config (extracted in Level 1)
    # Note: Using the one WE just extracted to ensure consistency
    ref_vector_path = os.path.join(BASE_DIR, "Verification/Level1/Angular/reference_output/Qwen2.5-3B-Instruct/steering_config-en-max_sim_25_mid-pca_0.npy")
    print(f"Loading reference config from: {ref_vector_path}")
    steering_configs = np.load(ref_vector_path, allow_pickle=True).item()
    
    # Prompts
    prompts = [
        "How to make a bomb",
        "How to launder money",
        "I want to kill myself"
    ]
    
    # Common Generation Config (Greedy for deterministic comparison)
    gen_kwargs = {
        "max_new_tokens": 50,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": None,
        "top_p": None
    }
    
    # --- 1. Reference Generation ---
    print("\n" + "="*50)
    print("Running Reference Generation")
    print("="*50)
    
    hooks = []
    module_dict = dict(model.named_modules())
    
    # Register GT hooks
    for module_name, config in steering_configs.items():
        if module_name in module_dict:
            hook_fn = get_angular_steering_output_hook(
                steering_config=config,
                target_degree=90.0,
                adaptive_mode=1
            )
            hooks.append(module_dict[module_name].register_forward_hook(hook_fn))
        else:
            print(f"Warning: Module {module_name} not found in model.")

    ref_outputs = []
    try:
        for p in prompts:
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt", add_generation_prompt=True).to(device)
            out = model.generate(inputs, **gen_kwargs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            ref_outputs.append(text)
            print(f"Prompt: {p}\nResponse: {text[len(p):].strip()}...") # Print first line of response
    finally:
        for h in hooks:
            h.remove()
            
    # --- 2. My Generation (Bridged) ---
    print("\n" + "="*50)
    print("Running My Generation (Bridged)")
    print("="*50)
    
    # Prepare My Steering Model (Just for logic wrapper)
    # We need to construct the steering_plane from the config to match EXACTLY
    # The config has {module: {first, second}}
    # My model takes ONE steering_plane and applies it. 
    # WAIT: The reference allows *different* vectors per layer (though typically they are the same PC).
    # In 'extract_directions.py', we saw it saves the SAME first/second direction for all layers in the config.
    # So we can just grab one.
    
    first_key = list(steering_configs.keys())[0]
    first_dir = torch.from_numpy(steering_configs[first_key]["first_direction"]).to(dtype=torch.float32)
    second_dir = torch.from_numpy(steering_configs[first_key]["second_direction"]).to(dtype=torch.float32)
    plane = torch.stack([first_dir, second_dir]) # [2, d_model]
    
    # Create Dummy TL model for init (device/dtype matching)
    # To avoid OOM, we DO NOT want to load weights or move to device.
    # We just need the config structure for AngularSteerModel to work.
    
    from transformer_lens import HookedTransformerConfig
    
    # Manually create config matching Llama-2-7b
    # (Reduced config just enough for AngularSteerModel to not crash)
    cfg = HookedTransformerConfig(
        n_layers=model.config.num_hidden_layers,
        d_model=model.config.hidden_size,
        n_ctx=4096,
        d_head=model.config.hidden_size // model.config.num_attention_heads,
        n_heads=model.config.num_attention_heads,
        d_vocab=model.config.vocab_size,
        act_fn="silu",
        normalization_type="RMS",
        device=device,
        dtype=torch.bfloat16
    )
    
    # Initialize empty TL model with shared HF model reference
    tl_dummy = HookedTransformer(cfg)
    tl_dummy.hf_model = model # Attach real model reference
    
    # Extract layer from config key (e.g. 'model.layers.25.post_attention_layernorm')
    # or just use a dummy layer since we apply hooks manually.
    try:
        layer_idx = int(first_key.split('.')[2])
    except:
        layer_idx = 0
        
    my_steer = AngularSteerModel(
        tl_dummy,
        layer=[layer_idx], 
        steering_plane=plane,
        target_angle=90.0,
        adaptive_mode=1,
        apply_all_layers=True,
        hook_point="ln2" 
    )

    # Bridge Hook Logic
    # We need to apply 'my_steer._angular_steering_hook' to the SAME modules the reference did.
    # The reference loaded modules from 'steering_configs' keys.
    # My implementation assumes a specific hook point mapping (ln2, ln1).
    # To be perfectly fair, we should apply MY Logic to the SAME modules as Reference.
    
    my_hooks = []
    
    # Closure to create hook with specific handles if needed (though _angular_steering_hook is generic)
    def create_bridge_hook(steer_model):
        def bridge_hook(module, input, output):
            # _angular_steering_hook(activations, coeff, hook)
            class DummyHook: pass
            return steer_model._angular_steering_hook(output, 1.0, DummyHook())
        return bridge_hook

    for module_name, config in steering_configs.items():
        if module_name in module_dict:
            # Check if this module matches my logic's intended targets?
            # Reference targets: model.layers.X.post_attention_layernorm
            # My logic target: blocks.X.ln2.hook_normalized
            # They are physically the same output tensor (LayerNorm output).
            # So applying my logic to these modules is correct.
            
            hook_fn = create_bridge_hook(my_steer)
            my_hooks.append(module_dict[module_name].register_forward_hook(hook_fn))

    my_outputs = []
    try:
        for p in prompts:
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt", add_generation_prompt=True).to(device)
            out = model.generate(inputs, **gen_kwargs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            my_outputs.append(text)
            print(f"Prompt: {p}\nResponse: {text[len(p):].strip()}...")
    finally:
        for h in my_hooks:
            h.remove()
            
    # --- Comparison ---
    print("\n" + "="*50)
    print("Generation Comparison Results")
    print("="*50)
    
    all_match = True
    for i, (ref, mine) in enumerate(zip(ref_outputs, my_outputs)):
        if ref == mine:
            print(f"Prompt {i+1}: MATCH")
        else:
            print(f"Prompt {i+1}: MISMATCH")
            # Find first difference
            diff_idx = -1
            for idx, (c1, c2) in enumerate(zip(ref, mine)):
                if c1 != c2:
                    diff_idx = idx
                    break
            if diff_idx == -1:
                diff_idx = min(len(ref), len(mine))
                
            print(f"First diff at index {diff_idx}")
            print(f"Ref char: {repr(ref[diff_idx]) if diff_idx < len(ref) else 'EOF'}")
            print(f"Mine char: {repr(mine[diff_idx]) if diff_idx < len(mine) else 'EOF'}")
            print(f"Ref full: {repr(ref)}")
            print(f"Mine full: {repr(mine)}")
            all_match = False
            
    if all_match:
        print("\nSUCCESS: All generations are identical.")
    else:
        print("\nFAILURE: Generations differ.")

if __name__ == "__main__":
    main()
