"""
Extract source activations with config-matching pooling.
Saves as {task}_source_acts_{pooling}.pt for each pooling mode.
"""
import sys, json
from pathlib import Path
import torch

sys.path.insert(0, str(Path("/home/caotue/SAESteeringBench")))

from Steering.data.loader import DataLoader
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-2b-it"
LAYER = 14
N_SAMPLES = 500
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("/home/caotue/SAESteeringBench/Vector/CHARS/Gemma")

# Each task's config position and extraction dataset info
TASKS = [
    {
        "name": "toxic",
        "train_dataset": "toxic_jigsaw",
        "source_key": "false_prompt",
        "position": "mask",
    },
    {
        "name": "evil",
        "train_dataset": "evil",
        "source_key": "false_prompt",
        "position": "mask",
    },
    {
        "name": "deception",
        "train_dataset": "liarbench",
        "source_key": "correct_prompt",
        "position": "last",
    },
    {
        "name": "refusal",
        "train_dataset": "refusal_caa",
        "source_key": "false_prompt",
        "position": "last",
    },
]

POOLING_MODES = ["mask", "last"]

def pool_acts(acts, attention_mask, mode):
    """Match Steering.utils._pool logic."""
    if mode == "last":
        return acts[:, -1, :]
    elif mode == "mask":
        if attention_mask is None:
            return acts.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(dtype=acts.dtype)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=acts.dtype)
        return (acts * mask).sum(dim=1) / denom
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading model {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        dtype=torch.bfloat16,
        default_padding_side="left",
    )
    model.to(DEVICE)
    tokenizer = model.tokenizer
    print("Model loaded.")

    for task_cfg in TASKS:
        name = task_cfg["name"]
        config_position = task_cfg["position"]
        
        for pool_mode in POOLING_MODES:
            out_path = OUT_DIR / f"{name}_source_acts_{pool_mode}.pt"
            if out_path.exists():
                print(f"Skipping {name} {pool_mode}: {out_path} exists")
                continue

            print(f"\n--- {name} (position={pool_mode}) ---")
            print(f"Loading {task_cfg['train_dataset']}...")
            loader = DataLoader()
            data = loader.load(
                task_cfg["train_dataset"],
                n_samples=N_SAMPLES,
                format=True,
                apply_chat_template=False,
            )
            source_texts = [d[task_cfg["source_key"]] for d in data]
            print(f"  {len(source_texts)} source samples")

            all_acts = []
            for i in range(0, len(source_texts), BATCH_SIZE):
                batch = source_texts[i:i+BATCH_SIZE]
                batch = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
                ) for t in batch]
                
                # Tokenize to get attention_mask
                tok = tokenizer(batch, padding=True, return_tensors="pt")
                input_ids = tok["input_ids"].to(DEVICE)
                attn_mask = tok["attention_mask"].to(DEVICE)
                
                with torch.no_grad():
                    _, cache = model.run_with_cache(
                        input_ids,
                        attention_mask=attn_mask,
                        names_filter=lambda n: n == f"blocks.{LAYER}.hook_resid_pre",
                        prepend_bos=True,
                    )
                acts = cache[f"blocks.{LAYER}.hook_resid_pre"]  # [B, seq_len, D]
                pooled = pool_acts(acts, attn_mask, pool_mode)
                all_acts.append(pooled.detach().cpu())
            
            acts_tensor = torch.cat(all_acts, dim=0).float()
            print(f"  shape: {acts_tensor.shape}")
            print(f"  norm range: {acts_tensor.norm(dim=1).min().item():.1f} - {acts_tensor.norm(dim=1).max().item():.1f}")
            
            torch.save(acts_tensor, str(out_path))
            print(f"  Saved to {out_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
