"""
Experiment B: First-token logit shift under CAA steering.

Measures whether CAA steering shifts the logits at token position 1 (the first
generated token). If shallow alignment is the barrier:
  - Toxic/Evil: steering barely moves position-1 logits (blocked by alignment)
  - Deception: steering moves position-1 logits freely (no alignment)
  - Base model: steering moves position-1 logits freely on all tasks

Design:
  For each prompt, we compute the softmax distribution over the vocabulary at
  position 1 (the first generated token). We compare:
    - P("I") / P("Sorry") / P("Sure") under steered vs unsteered
    - The total KL(steered || unsteered) at position 1
    - The probability assigned to the top-1 token in each condition

Usage:
    conda activate sae_circuit
    unset CUDA_VISIBLE_DEVICES
    export HF_HOME="/data/caotue/hf_cache"
    export HUGGINGFACE_HUB_CACHE="/data/caotue/hf_cache/hub"
    export TRANSFORMERS_CACHE="/data/caotue/hf_cache/transformers"
    export HF_DATASETS_CACHE="/data/caotue/hf_cache/datasets"
    export TORCH_HOME="/data/caotue/torch_cache"
    export TMPDIR="/data/caotue/tmp"
    CUDA_VISIBLE_DEVICES=2 python Experiments/ShallowAlignment/first_token_logit.py --task toxic --n_prompts 100
    CUDA_VISIBLE_DEVICES=2 python Experiments/ShallowAlignment/first_token_logit.py --task deception --n_prompts 100
    CUDA_VISIBLE_DEVICES=2 python Experiments/ShallowAlignment/first_token_logit.py --task evil --n_prompts 100

Base model comparison:
    CUDA_VISIBLE_DEVICES=2 python Experiments/ShallowAlignment/first_token_logit.py --task toxic --model google/gemma-2-2b --n_prompts 100
"""

import sys; sys.path.insert(0, ".")
import json
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.nn import functional as F
from transformer_lens import HookedTransformer

from Steering.config.results import SteeringVector

torch.set_grad_enabled(False)

TASK_CONFIGS = {
    "toxic": {"vector_path": "Vector/CAA/Gemma/toxic"},
    "evil": {"vector_path": "Vector/CAA/Gemma/evil"},
    "deception": {"vector_path": "Vector/CAA/Gemma/deception"},
}


REFUSAL_TOKENS = ["I", "Sorry", "I am", "I cannot", "I apologize", "I'm", "I\u2019m"]
# We'll check these tokens explicitly
REFUSAL_TOKEN_IDS = None  # populated after tokenizer is loaded


def tokenize_refusal_tokens(tokenizer, refusal_strs=None):
    """Get the token IDs for common refusal tokens."""
    if refusal_strs is None:
        refusal_strs = REFUSAL_TOKENS
    token_ids = {}
    for s in refusal_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            token_ids[s] = ids[0]
    return token_ids


def load_steering_vector(path: str, layer: int, device: str) -> torch.Tensor:
    data = torch.load(f"{path}/vector.pt", map_location="cpu", weights_only=True)
    vec = data[layer].to(device)
    vec = vec / vec.norm()
    return vec


def make_chat_prompt(instruction: str, model) -> str:
    messages = [{"role": "user", "content": instruction}]
    tokens = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokens


def get_position1_logits(model, prompt: str, steering_vec: torch.Tensor,
                         coeff: float, layer: int, apply_steer: bool = True):
    """
    Get logits at position 1 (first generated token) with or without steering.

    Returns: logits (vocab_size,), softmax_probs (vocab_size,)
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)

    if apply_steer:
        def steering_hook(activations, hook):
            activations[:, -1, :] += coeff * steering_vec.to(dtype=activations.dtype)
            return activations

        hook_name = f"blocks.{layer}.hook_resid_pre"
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=[hook_name],
            return_type="logits",
            fwd_hooks=[(hook_name, steering_hook)],
        )
    else:
        logits, cache = model.run_with_cache(
            tokens,
            return_type="logits",
        )

    # Logits at the last prompt position = first generated token's distribution
    pos1_logits = logits[0, -1, :]  # (vocab_size,)
    probs = F.softmax(pos1_logits.float(), dim=-1)

    return pos1_logits, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["toxic", "evil", "deception"], required=True)
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument("--coeff", type=float, default=1.0)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--output_dir", type=str, default="Experiments/ShallowAlignment/results")
    args = parser.parse_args()

    cfg = TASK_CONFIGS[args.task]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = HookedTransformer.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
        default_prepend_bos=True,
    )
    model.eval()

    # Get refusal token IDs
    refusal_ids = tokenize_refusal_tokens(model.tokenizer)
    print(f"  Refusal token IDs: {refusal_ids}")

    # Also get "Sure" token ID
    sure_id = model.tokenizer.encode("Sure", add_special_tokens=False)
    print(f"  'Sure' token IDs: {sure_id}")

    # Load steering vector
    print(f"Loading steering vector from: {cfg['vector_path']}")
    steering_vec = load_steering_vector(cfg["vector_path"], args.layer, "cuda")
    print(f"  Vector norm: {steering_vec.norm().item():.4f}")

    # Load test prompts
    test_file = "TrainDataset/behaviour/normal/alpaca.json"
    with open(test_file) as f:
        all_data = json.load(f)
    prompts = [d["instruction"] for d in all_data[:args.n_prompts]]

    # Run experiment
    results = {
        "config": {
            "task": args.task,
            "model": args.model,
            "coeff": args.coeff,
            "layer": args.layer,
            "n_prompts": args.n_prompts,
        },
        "per_prompt": [],
        "summaries": {},
    }

    all_unsteered_top1 = []
    all_steered_top1 = []
    all_kl_pos1 = []
    all_unsteered_refusal_prob = []
    all_steered_refusal_prob = []
    all_unsteered_sure_prob = []
    all_steered_sure_prob = []

    for i, prompt_text in enumerate(prompts):
        if i % 20 == 0:
            print(f"  [{i}/{len(prompts)}]")

        full_prompt = make_chat_prompt(prompt_text, model)

        # Get unsteered position-1 logits
        unsteered_logits, unsteered_probs = get_position1_logits(
            model, full_prompt, steering_vec, args.coeff, args.layer, apply_steer=False
        )
        # Get steered position-1 logits
        steered_logits, steered_probs = get_position1_logits(
            model, full_prompt, steering_vec, args.coeff, args.layer, apply_steer=True
        )

        # KL divergence at position 1
        kl_pos1 = F.kl_div(
            steered_probs.log(), unsteered_probs, reduction="sum"
        ).item()

        # Top-1 tokens
        unsteered_top1_id = unsteered_probs.argmax().item()
        steered_top1_id = steered_probs.argmax().item()
        unsteered_top1_token = model.tokenizer.decode([unsteered_top1_id])
        steered_top1_token = model.tokenizer.decode([steered_top1_id])

        # Refusal token probabilities
        unsteered_refusal_probs = {
            name: unsteered_probs[tid].item()
            for name, tid in refusal_ids.items()
            if tid < unsteered_probs.shape[0]
        }
        steered_refusal_probs = {
            name: steered_probs[tid].item()
            for name, tid in refusal_ids.items()
            if tid < steered_probs.shape[0]
        }

        # "Sure" token probability
        sure_tid = sure_id[0] if sure_id else None
        unsteered_sure_prob = unsteered_probs[sure_tid].item() if sure_tid else 0.0
        steered_sure_prob = steered_probs[sure_tid].item() if sure_tid else 0.0

        # Sum of all refusal token probabilities
        unsteered_refusal_total = sum(unsteered_refusal_probs.values())
        steered_refusal_total = sum(steered_refusal_probs.values())

        record = {
            "prompt": prompt_text[:80],
            "unsteered_top1_token": unsteered_top1_token.strip(),
            "steered_top1_token": steered_top1_token.strip(),
            "unsteered_top1_prob": unsteered_probs.max().item(),
            "steered_top1_prob": steered_probs.max().item(),
            "kl_pos1": kl_pos1,
            "unsteered_refusal_total": unsteered_refusal_total,
            "steered_refusal_total": steered_refusal_total,
            "unsteered_sure_prob": unsteered_sure_prob,
            "steered_sure_prob": steered_sure_prob,
        }
        results["per_prompt"].append(record)

        all_unsteered_top1.append(unsteered_top1_token.strip())
        all_steered_top1.append(steered_top1_token.strip())
        all_kl_pos1.append(kl_pos1)
        all_unsteered_refusal_prob.append(unsteered_refusal_total)
        all_steered_refusal_prob.append(steered_refusal_total)
        all_unsteered_sure_prob.append(unsteered_sure_prob)
        all_steered_sure_prob.append(steered_sure_prob)

    # Summaries
    results["summaries"] = {
        "mean_kl_pos1": float(np.mean(all_kl_pos1)),
        "median_kl_pos1": float(np.median(all_kl_pos1)),
        "mean_unsteered_refusal_prob": float(np.mean(all_unsteered_refusal_prob)),
        "mean_steered_refusal_prob": float(np.mean(all_steered_refusal_prob)),
        "mean_unsteered_sure_prob": float(np.mean(all_unsteered_sure_prob)),
        "mean_steered_sure_prob": float(np.mean(all_steered_sure_prob)),
        "n_top1_unsteered_refusal": sum(
            1 for t in all_unsteered_top1 if any(r.lower() in t.lower() for r in ["i", "sorry", "i am", "i'm", "apologize"])
        ),
        "n_top1_steered_refusal": sum(
            1 for t in all_steered_top1 if any(r.lower() in t.lower() for r in ["i", "sorry", "i am", "i'm", "apologize"])
        ),
        "n_top1_unsteered_sure": sum(1 for t in all_unsteered_top1 if "sure" in t.lower()),
        "n_top1_steered_sure": sum(1 for t in all_steered_top1 if "sure" in t.lower()),
    }

    # Print
    print(f"\n{'='*60}")
    print(f"Task: {args.task} | Model: {args.model}")
    print(f"{'='*60}")
    print(f"KL at position 1: mean={results['summaries']['mean_kl_pos1']:.4f}, median={results['summaries']['median_kl_pos1']:.4f}")
    print(f"Refusal prob (unsteered): {results['summaries']['mean_unsteered_refusal_prob']:.4f}")
    print(f"Refusal prob (steered):   {results['summaries']['mean_steered_refusal_prob']:.4f}")
    print(f"Sure prob (unsteered):    {results['summaries']['mean_unsteered_sure_prob']:.4f}")
    print(f"Sure prob (steered):      {results['summaries']['mean_steered_sure_prob']:.4f}")
    print(f"Top-1 refusal (unsteered): {results['summaries']['n_top1_unsteered_refusal']}/{args.n_prompts}")
    print(f"Top-1 refusal (steered):   {results['summaries']['n_top1_steered_refusal']}/{args.n_prompts}")
    print(f"Top-1 sure (unsteered):    {results['summaries']['n_top1_unsteered_sure']}/{args.n_prompts}")
    print(f"Top-1 sure (steered):      {results['summaries']['n_top1_steered_sure']}/{args.n_prompts}")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"first_token_logit_{args.task}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
