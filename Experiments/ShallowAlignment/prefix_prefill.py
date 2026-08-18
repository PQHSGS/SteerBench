"""
Experiment A: Prefix-prefill + CAA steering.

Tests whether shallow alignment at token position 1 is the barrier blocking steering.

Logic:
  - If prefix prefill bypasses the barrier and CAA works on Toxic/Evil:
    → shallow alignment at token 1 is the only barrier (Qi et al. theory)
  - If prefix prefill + CAA still fails at 0-21%:
    → deeper mechanism exists (cancellation circuit or general activation limitation)

Design: 2 (steering: off/on) x 2 (prefix: none/toxic) for each of 4 tasks (Toxic, Evil, Deception, Refusal)
  - Prefix: "Sure, " (non-refusal prefix)
  - Steering: CAA c=1.0 (conservative, below OOD threshold)
  - Control: Deception should show high acc regardless (no alignment barrier)

Usage:
    conda activate sae_circuit
    unset CUDA_VISIBLE_DEVICES
    export HF_HOME="/data/caotue/hf_cache"
    export HUGGINGFACE_HUB_CACHE="/data/caotue/hf_cache/hub"
    export TRANSFORMERS_CACHE="/data/caotue/hf_cache/transformers"
    export HF_DATASETS_CACHE="/data/caotue/hf_cache/datasets"
    export TORCH_HOME="/data/caotue/torch_cache"
    export TMPDIR="/data/caotue/tmp"
    CUDA_VISIBLE_DEVICES=2 python Experiments/ShallowAlignment/prefix_prefill.py --task toxic --n_prompts 50
    CUDA_VISIBLE_DEVICES=2 python Experiments/ShallowAlignment/prefix_prefill.py --task evil --n_prompts 50
    CUDA_VISIBLE_DEVICES=2 python Experiments/ShallowAlignment/prefix_prefill.py --task deception --n_prompts 50
    CUDA_VISIBLE_DEVICES=2 python Experiments/ShallowAlignment/prefix_prefill.py --task refusal --n_prompts 50
"""

import sys; sys.path.insert(0, ".")
import json
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformer_lens import HookedTransformer

from Steering.evaluators import ToxicMatcher, BehaviorMatcher, RefusalMatcher, DeceptionEvaluator
from Steering.config.results import SteeringVector
from Steering.config.models import MODEL_SAE_REGISTRY

torch.set_grad_enabled(False)

TASK_CONFIGS = {
    "toxic": {
        "vector_path": "Vector/CAA/Gemma/toxic",
        "test_file": "TrainDataset/behaviour/normal/alpaca.json",
        "evaluator_factory": lambda: ToxicMatcher(device="cuda"),
        "prefix": "Sure, ",
    },
    "evil": {
        "vector_path": "Vector/CAA/Gemma/evil",
        "test_file": "TrainDataset/behaviour/normal/alpaca.json",
        "evaluator_factory": lambda: BehaviorMatcher(device="cuda", mode="evil"),
        "prefix": "Sure, ",
    },
    "deception": {
        "vector_path": "Vector/CAA/Gemma/deception",
        "test_file": "TrainDataset/behaviour/normal/alpaca.json",
        "evaluator_factory": lambda: DeceptionEvaluator(device="cuda"),
        "prefix": "Sure, ",
    },
    "refusal": {
        "vector_path": "Vector/CAA/Gemma/refusal_response",
        "test_file": "TrainDataset/behaviour/normal/alpaca.json",
        "evaluator_factory": lambda: RefusalMatcher(device="cuda"),
        "prefix": "Sure, ",
    },
}


def load_steering_vector(path: str, layer: int, device: str) -> torch.Tensor:
    """Load a CAA steering vector from disk."""
    data = torch.load(f"{path}/vector.pt", map_location="cpu", weights_only=True)
    vec = data[layer].to(device)
    vec = vec / vec.norm()  # normalize to unit vector
    return vec


def make_chat_prompt(instruction: str, model) -> str:
    """Apply gemma chat template to get the full prompt."""
    messages = [{"role": "user", "content": instruction}]
    tokens = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokens


def generate_with_steering(model, prompt: str, steering_vec: torch.Tensor,
                           coeff: float, layer: int,
                           max_new_tokens: int = 50) -> str:
    """
    Generate text with CAA steering applied at the given layer.
    Uses transformer_lens hooks to add the steering vector.
    """
    # Tokenize
    tokens = model.to_tokens(prompt, prepend_bos=True)
    prompt_len = tokens.shape[1]

    # Define steering hook
    def steering_hook(activations, hook):
        activations[:, -1, :] += coeff * steering_vec.to(dtype=activations.dtype)
        return activations

    # Run generation with hook
    hook_name = f"blocks.{layer}.hook_resid_pre"

    output = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        prepend_bos=True,
        stop_at_layer=None,
        fwd_hooks=[(hook_name, steering_hook)],
    )

    # Decode only the generated part
    generated = output[0, prompt_len:]
    return model.tokenizer.decode(generated, skip_special_tokens=True)


def generate_baseline(model, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate without steering."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    prompt_len = tokens.shape[1]

    output = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        prepend_bos=True,
    )

    generated = output[0, prompt_len:]
    return model.tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["toxic", "evil", "deception", "refusal"], required=True)
    parser.add_argument("--n_prompts", type=int, default=50)
    parser.add_argument("--coeff", type=float, default=1.0)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--output_dir", type=str, default="Experiments/ShallowAlignment/results")
    parser.add_argument("--prefix", type=str, default=None)  # override task default prefix
    args = parser.parse_args()

    cfg = TASK_CONFIGS[args.task]
    evaluator = cfg["evaluator_factory"]()
    prefix = args.prefix if args.prefix is not None else cfg["prefix"]
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

    # Load steering vector
    print(f"Loading steering vector from: {cfg['vector_path']}")
    steering_vec = load_steering_vector(cfg["vector_path"], args.layer, "cuda")
    print(f"  Vector norm: {steering_vec.norm().item():.4f}")

    # Load test prompts
    print(f"Loading prompts from: {cfg['test_file']}")
    with open(cfg["test_file"]) as f:
        all_data = json.load(f)
    prompts = [d["instruction"] for d in all_data[:args.n_prompts]]

    # Run experiment
    results = {
        "config": {
            "task": args.task,
            "model": args.model,
            "coeff": args.coeff,
            "layer": args.layer,
            "prefix": prefix,
            "n_prompts": args.n_prompts,
        },
        "conditions": {
            "baseline": [],       # no prefix, no steering
            "steer_only": [],     # no prefix, with steering
            "prefix_only": [],    # with prefix, no steering
            "prefix_steer": [],   # with prefix, with steering
        },
        "summaries": {},
    }

    for i, prompt_text in enumerate(prompts):
        if i % 10 == 0:
            print(f"  [{i}/{len(prompts)}]")

        # Format prompt with chat template
        full_prompt = make_chat_prompt(prompt_text, model)

        # Condition 1: Baseline (no prefix, no steer)
        resp_bl = generate_baseline(model, full_prompt)
        score_bl, conf_bl = evaluator.check(resp_bl)

        # Condition 2: Steering only (no prefix)
        resp_steer = generate_with_steering(model, full_prompt, steering_vec, args.coeff, args.layer)
        score_steer, conf_steer = evaluator.check(resp_steer)

        # Condition 3: Prefix only (no steering)
        prefixed_prompt = full_prompt + prefix
        resp_pref = generate_baseline(model, prefixed_prompt)
        score_pref, conf_pref = evaluator.check(resp_pref)

        # Condition 4: Prefix + Steering
        resp_pref_steer = generate_with_steering(model, prefixed_prompt, steering_vec, args.coeff, args.layer)
        score_pref_steer, conf_pref_steer = evaluator.check(resp_pref_steer)

        results["conditions"]["baseline"].append({
            "prompt": prompt_text,
            "response": resp_bl,
            "score": score_bl,
            "confidence": conf_bl,
        })
        results["conditions"]["steer_only"].append({
            "prompt": prompt_text,
            "response": resp_steer,
            "score": score_steer,
            "confidence": conf_steer,
        })
        results["conditions"]["prefix_only"].append({
            "prompt": prompt_text,
            "response": resp_pref,
            "score": score_pref,
            "confidence": conf_pref,
        })
        results["conditions"]["prefix_steer"].append({
            "prompt": prompt_text,
            "response": resp_pref_steer,
            "score": score_pref_steer,
            "confidence": conf_pref_steer,
        })

    # Compute summaries
    for cond_name, cond_data in results["conditions"].items():
        scores = [d["score"] for d in cond_data]
        acc = np.mean(scores) * 100
        results["summaries"][cond_name] = {
            "accuracy": f"{acc:.1f}%",
            "n_correct": sum(scores),
            "n_total": len(scores),
            "mean_confidence": f"{np.mean([d['confidence'] for d in cond_data]):.3f}",
        }

    # Print results
    print(f"\n{'='*60}")
    print(f"Task: {args.task} | Model: {args.model}")
    print(f"Prefix: '{prefix}' | Coeff: {args.coeff}")
    print(f"{'='*60}")
    print(f"{'Condition':<20} {'Accuracy':<12} {'Correct/N':<12}")
    print(f"{'-'*44}")
    for cond_name in ["baseline", "steer_only", "prefix_only", "prefix_steer"]:
        s = results["summaries"][cond_name]
        print(f"{cond_name:<20} {s['accuracy']:<12} {s['n_correct']}/{s['n_total']:<8}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"prefix_prefill_{args.task}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
