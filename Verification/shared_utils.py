"""
Shared utilities for L1/L2 validation scripts.

Provides:
- load_sycophancy_data: Load real data from TrainDataset/behaviour/sycophancy/sycophancy.jsonl
- load_model_and_sae: Load gemma-2-2b + SAE 16k (shared across all tests)
- get_sae_activations: Get SAE latent activations from shared model
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

ROOT = Path(__file__).parent.parent


def load_sycophancy_data(n_per_class: int = 100) -> Tuple[List[str], List[str]]:
    """Load real sycophancy data as (target_texts, contrast_texts).
    
    Target = answer_matching_behavior direction
    Contrast = answer_not_matching_behavior direction
    
    Returns exactly n_per_class target and n_per_class contrast texts.
    """
    data_path = ROOT / "TrainDataset" / "behaviour" / "sycophancy" / "sycophancy.jsonl"
    
    all_items = []
    with open(data_path, "r") as f:
        for line in f:
            all_items.append(json.loads(line.strip()))
    
    # Use first 2*n_per_class items deterministically
    items = all_items[:2 * n_per_class]
    
    target_texts = []
    contrast_texts = []
    for item in items:
        q = item["question"]
        match = item["answer_matching_behavior"].strip()
        not_match = item["answer_not_matching_behavior"].strip()
        
        target_texts.append(q + " " + match)
        contrast_texts.append(q + " " + not_match)
    
    return target_texts[:n_per_class], contrast_texts[:n_per_class]


def load_model_and_sae(
    layer: int = 20,
    device: str = "cuda:0",
    dtype=torch.bfloat16,
):
    """Load gemma-2-2b (TransformerLens) + SAE 16k for the given layer.
    
    Returns (model, sae, layer).
    """
    from sae_lens import SAE, HookedSAETransformer
    
    print(f"Loading gemma-2-2b on {device}...")
    model = HookedSAETransformer.from_pretrained(
        "gemma-2-2b", device=device, dtype=dtype
    )
    
    print(f"Loading SAE layer_{layer}/width_16k/canonical...")
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{layer}/width_16k/canonical",
    )
    sae = sae.to(device)
    
    return model, sae, layer


def get_sae_activations(
    model,
    sae,
    texts: List[str],
    layer: int,
    batch_size: int = 8,
    position: str = "last",
) -> List[torch.Tensor]:
    """Get SAE latent activations for each text at the specified token position.
    
    Returns a list of 1-D tensors (one per text), shape [d_sae].
    """
    hook_name = f"hook_sae_acts_post"
    all_latents = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_latents = []
        
        def hook_fn(sae_acts, hook):
            if position == "last":
                # Last non-padding token for each sample
                for b in range(sae_acts.shape[0]):
                    batch_latents.append(sae_acts[b, -1, :].cpu())
            elif position == "mean":
                for b in range(sae_acts.shape[0]):
                    batch_latents.append(sae_acts[b].mean(dim=0).cpu())
        sae.add_hook(hook_name, hook_fn)
        with torch.no_grad():
            model.run_with_saes(
                batch,
                saes=sae
            )
        sae.reset_hooks()
        all_latents.extend(batch_latents)
        torch.cuda.empty_cache()
    
    return all_latents


def get_resid_activations(
    model,
    texts: List[str],
    layer: int,
    batch_size: int = 8,
    position: str = "last",
) -> List[torch.Tensor]:
    """Get residual stream activations for each text at the specified position.
    
    Returns list of 1-D tensors shape [d_model].
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_acts = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_acts = []
        
        def hook_fn(act, hook):
            if position == "last":
                for b in range(act.shape[0]):
                    batch_acts.append(act[b, -1, :].cpu().float())
            elif position == "mean":
                for b in range(act.shape[0]):
                    batch_acts.append(act[b].mean(dim=0).cpu().float())
        
        with torch.no_grad():
            model.run_with_hooks(
                batch,
                fwd_hooks=[(hook_name, hook_fn)],
            )
        
        all_acts.extend(batch_acts)
        torch.cuda.empty_cache()
    
    return all_acts
