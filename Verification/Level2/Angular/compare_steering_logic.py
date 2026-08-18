import torch
import numpy as np
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# Add path to include Steering and Ground Truth code
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(BASE_DIR)
GT_CODE_PATH = os.path.join(BASE_DIR, "Code/angular-steering/pytorch_pure")
sys.path.append(GT_CODE_PATH)

# Import actual ground truth functions
from generate_responses import get_angular_steering_output_hook

from Steering.steer_models.dense import AngularSteerModel

def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen1.5-0.5B-Chat"
    
    # We'll test on Layer 16
    TARGET_LAYER = 16 
    ground_truth_path = os.path.join(BASE_DIR, "Verification/Level1/Angular/reference_output/Qwen1.5-0.5B-Chat/steering_config-en-max_sim_16_mid-pca_0.npy")
    
    print(f"Loading ground truth vector from: {ground_truth_path}")
    config_dict = np.load(ground_truth_path, allow_pickle=True).item()
    
    target_key = f"model.layers.{TARGET_LAYER}.post_attention_layernorm"
    steering_config = config_dict[target_key]
    first_dir = torch.from_numpy(steering_config["first_direction"]).to(dtype=torch.float32)
    second_dir = torch.from_numpy(steering_config["second_direction"]).to(dtype=torch.float32)
    
    print(f"Vectors loaded for {target_key}")

    # Load Model (Hugging Face)
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We use a dummy HookedTransformer JUST to initialize AngularSteerModel easily
    # but we will extract the logic and run it on HF model directly for strict comparison.
    tl_dummy = HookedTransformer.from_pretrained(
        model_name, 
        hf_model=model, 
        device=device, 
        dtype=torch.bfloat16, 
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False
    )
    
    prompt = "How to make a bomb"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # --- Reference Run (GT Code) ---
    print("\n" + "="*50)
    print("Reference Run (Actual GT Hook)")
    print("="*50)
    
    gt_hook_fn = get_angular_steering_output_hook(
        steering_config=steering_config,
        target_degree=90,
        adaptive_mode=1
    )
    
    module_name_map = {n: m for n, m in model.named_modules()}
    target_module = module_name_map[target_key]
    
    handle = target_module.register_forward_hook(gt_hook_fn)
    try:
        with torch.no_grad():
            ref_logits = model(input_ids).logits
    finally:
        handle.remove()
        
    # --- My Run (Our AngularSteerModel via Pipeline) ---
    print("\n" + "="*50)
    print("My Run (SteeringPipeline)")
    print("="*50)
    
    from Steering.pipeline import SteeringPipeline
    from Steering.config import PipelineConfig, SteerConfig, ExtractorConfig, ModelConfig

    import tempfile
    
    # We must convert the .npy reference vector to a .pt file because the pipeline expects .pt
    temp_pt_path = os.path.join(BASE_DIR, "Verification/Level2/Angular/temp_converted_vector.pt")
    
    # Provide expected structure: {"steering_vector": plane}
    plane = torch.stack([first_dir, second_dir])
    torch.save({
        "steering_vector": plane,
        "metadata": {"steering_plane": plane}
    }, temp_pt_path)

    pipeline_config = PipelineConfig(
        model=ModelConfig(
            name=model_name,
            device=device,
            dtype="bfloat16",
            use_compile=False,
            model_kwargs={"fold_ln": False}
        ),
        extractor=ExtractorConfig(method="ANGULAR", layer=[TARGET_LAYER]),
        steer=SteerConfig(
            method="ANGULAR",
            layer=[TARGET_LAYER],
            coeff=1.0
        ),
        load_vector=temp_pt_path,
        test_dataset="dummy",
        train_dataset="dummy",
        save_vector=None
    )

    pipeline = SteeringPipeline(pipeline_config)
    print(pipeline.steer_config.hook_point)
    pipeline.setup()
    # Force settings
    pipeline.steer_model.target_angle = 90
    pipeline.steer_model.adaptive_mode = 1
    pipeline.steer_model.apply_all_layers = False
    
    pipeline_tokens = pipeline.model.to_tokens(prompt)
    print(pipeline.steer_model.hook_point)
    pipeline.steer_model.setup_hooks({TARGET_LAYER: 1.0})
    try:
        with torch.no_grad():
            my_logits = pipeline.model(pipeline_tokens)
    finally:
        pipeline.model.reset_hooks()

        
    # --- Compare ---
    print("\n" + "="*50)
    print("In-Memory Direct Logic Comparison")
    print("="*50)
    
    diff = (ref_logits.cpu() - my_logits.cpu()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Logit Diff: {max_diff:.8f}")
    print(f"Mean Logit Diff: {mean_diff:.8f}")
    
    # Token check
    ref_top = torch.argmax(ref_logits, dim=-1)
    my_top = torch.argmax(my_logits, dim=-1)
    match = (ref_top == my_top).all().item()
    print(f"Top Tokens Match: {match}")
    
    if max_diff < 1e-5:
        print("\nSUCCESS: Mathematical equivalence verified!")
    else:
        print("\nWARNING: Mathematical divergence found even on same model instance.")
        # If divergence, let's see if it's just very small
        if max_diff < 1e-4:
             print("(But it is very small, likely negligible floating point noise)")

if __name__ == "__main__":
    main()
