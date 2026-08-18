
import torch
from transformer_lens import HookedTransformer

model_name = "Qwen/Qwen1.5-0.5B-Chat"
device = "cuda"

print(f"Loading {model_name}...")
model = HookedTransformer.from_pretrained(
    model_name, 
    device=device, 
    dtype="bfloat16",
    default_prepend_bos=False
)

prompt = "To be or not to be"
print(f"Running with hooks for prompt: {prompt}")

# We want to check Layer 0 resid_mid
layer = 0
hook_name = f"blocks.{layer}.hook_resid_mid"
print(f"Target hook: {hook_name}")

_, cache = model.run_with_cache(
    prompt,
    names_filter=lambda x: x == hook_name
)

if hook_name in cache:
    act = cache[hook_name]
    print(f"Captured shape: {act.shape}")
    print(f"Norm: {act.float().norm().item()}")
    print(f"Mean: {act.float().mean().item()}")
    if act.norm() == 0:
        print("ALERT: Activation is ZERO!")
else:
    print(f"ALERT: Hook {hook_name} NOT FOUND in cache!")
    print("Available hooks (first 10):")
    all_hooks = list(model.hook_dict.keys())
    print(all_hooks[:10])
    # Search for layer 0 hooks
    l0_hooks = [h for h in all_hooks if "blocks.0." in h]
    print("Layer 0 hooks:", l0_hooks)
