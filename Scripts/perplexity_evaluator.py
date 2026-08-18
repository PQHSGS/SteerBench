#!/usr/bin/env python3
"""
Perplexity Evaluator for Model Outputs

This script computes conditional perplexity PPL(response | prompt)
from a JSON result file produced by the steering pipeline.

Expected input structure (same as existing evaluators):
- data["result"]["samples"]: List[dict] with "prompt" and "response"
- data["result"]["baseline_samples"]: optional List[dict] with same fields

Usage:
    python perplexity_evaluator.py <file> [--output <output_file>]

Example:
    python perplexity_evaluator.py Results/cast/eval_cast_llama_sorrybench_20260223_005605.json --compare
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json_results(json_path: str) -> dict:
    """Load results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def extract_samples(data: dict) -> Tuple[List[dict], List[dict]]:
    """Extract samples and baseline_samples from result JSON."""
    result = data.get("result", {})
    samples = result.get("samples", [])
    baseline_samples = result.get("baseline_samples", [])
    return samples, baseline_samples


def infer_model_name(data: dict) -> Optional[str]:
    """Infer model name from config fields in the input JSON."""
    config = data.get("config", {})
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}

    # Primary expected location from PipelineConfig serialization.
    if isinstance(model_cfg, dict) and model_cfg.get("name"):
        return model_cfg["name"]

    # Backward-compatible fallbacks for older result schemas.
    if isinstance(config, dict) and config.get("model_name"):
        return config["model_name"]
    if data.get("model_name"):
        return data["model_name"]
    return None


def _resolve_dtype(dtype_name: str) -> Optional[torch.dtype]:
    """Map CLI dtype argument to torch dtype."""
    name = dtype_name.lower()
    if name == "auto":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


class PerplexityEvaluator:
    """Compute conditional perplexity for prompt/response pairs."""

    def __init__(self, model_name: str, device: str, dtype_name: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.dtype_name = dtype_name

        torch_dtype = _resolve_dtype(dtype_name)
        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.to(device)
        self.model.eval()

    def compute_conditional_perplexity(self, prompt: str, response: str, max_length: int = 2048) -> float:
        """Compute PPL(response | prompt) using LM cross-entropy."""
        prompt = prompt or ""
        response = response or ""
        if not response.strip():
            return float("inf")

        with torch.no_grad():
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
            full_text = prompt + response
            full_ids = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids

            prompt_len = prompt_ids.shape[1]
            total_len = full_ids.shape[1]

            if total_len == 0:
                return float("inf")

            # Keep the last max_length tokens to satisfy model context limits.
            if total_len > max_length:
                overflow = total_len - max_length
                full_ids = full_ids[:, overflow:]
                prompt_len = max(0, prompt_len - overflow)

            if prompt_len >= full_ids.shape[1]:
                return float("inf")

            input_ids = full_ids.to(self.device)
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100

            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            return math.exp(loss.item())


def evaluate_samples(
    samples: List[dict],
    evaluator: PerplexityEvaluator,
    max_length: int,
    crop: int = None,
    verbose: bool = False,
) -> List[dict]:
    """Compute perplexity for a sample list."""
    results = []
    for i, sample in enumerate(samples):
        if verbose:
            print(f"Processing sample {i + 1}/{len(samples)}...")

        prompt = sample.get("prompt", "")
        response = sample.get("response", "")
        if crop is not None:
            response = response[:crop]
        ppl = evaluator.compute_conditional_perplexity(prompt=prompt, response=response, max_length=max_length)

        results.append(
            {
                "index": i,
                "prompt": prompt,
                "response": response,
                "perplexity": ppl,
                "is_finite": math.isfinite(ppl),
            }
        )
    return results


def summarize_perplexities(results: List[dict]) -> Dict[str, float]:
    """Compute aggregate statistics for perplexity results."""
    values = [r["perplexity"] for r in results if math.isfinite(r["perplexity"])]
    if not values:
        return {
            "count": len(results),
            "finite_count": 0,
            "mean_perplexity": float("inf"),
            "median_perplexity": float("inf"),
            "min_perplexity": float("inf"),
            "max_perplexity": float("inf"),
        }

    return {
        "count": len(results),
        "finite_count": len(values),
        "mean_perplexity": sum(values) / len(values),
        "median_perplexity": median(values),
        "min_perplexity": min(values),
        "max_perplexity": max(values),
    }


def evaluate_comparison(
    samples: List[dict],
    baseline_samples: List[dict],
    evaluator: PerplexityEvaluator,
    max_length: int,
    verbose: bool = False,
) -> Dict:
    """Compare steered vs baseline perplexities sample-wise."""
    if len(samples) != len(baseline_samples):
        print(
            f"Warning: Number of samples ({len(samples)}) != baseline_samples ({len(baseline_samples)})"
        )

    n_samples = min(len(samples), len(baseline_samples))
    steered_results: List[dict] = []
    baseline_results: List[dict] = []
    comparisons: List[dict] = []

    for i in range(n_samples):
        if verbose:
            print(f"Processing sample {i + 1}/{n_samples}...")

        steered_prompt = samples[i].get("prompt", "")
        steered_response = samples[i].get("response", "")
        baseline_prompt = baseline_samples[i].get("prompt", "")
        baseline_response = baseline_samples[i].get("response", "")

        steered_ppl = evaluator.compute_conditional_perplexity(
            prompt=steered_prompt,
            response=steered_response,
            max_length=max_length,
        )
        baseline_ppl = evaluator.compute_conditional_perplexity(
            prompt=baseline_prompt,
            response=baseline_response,
            max_length=max_length,
        )

        steered_results.append(
            {
                "index": i,
                "prompt": steered_prompt,
                "response": steered_response,
                "perplexity": steered_ppl,
                "is_finite": math.isfinite(steered_ppl),
            }
        )
        baseline_results.append(
            {
                "index": i,
                "prompt": baseline_prompt,
                "response": baseline_response,
                "perplexity": baseline_ppl,
                "is_finite": math.isfinite(baseline_ppl),
            }
        )

        delta = (
            steered_ppl - baseline_ppl
            if math.isfinite(steered_ppl) and math.isfinite(baseline_ppl)
            else float("nan")
        )
        comparisons.append(
            {
                "index": i,
                "steered_perplexity": steered_ppl,
                "baseline_perplexity": baseline_ppl,
                "delta": delta,
            }
        )

    steered_summary = summarize_perplexities(steered_results)
    baseline_summary = summarize_perplexities(baseline_results)

    return {
        "steered": steered_results,
        "baseline": baseline_results,
        "comparison": comparisons,
        "summary": {
            "total_samples": n_samples,
            "steered_mean_perplexity": steered_summary["mean_perplexity"],
            "baseline_mean_perplexity": baseline_summary["mean_perplexity"],
            "steered_median_perplexity": steered_summary["median_perplexity"],
            "baseline_median_perplexity": baseline_summary["median_perplexity"],
            "delta_mean_perplexity": (
                steered_summary["mean_perplexity"] - baseline_summary["mean_perplexity"]
                if math.isfinite(steered_summary["mean_perplexity"])
                and math.isfinite(baseline_summary["mean_perplexity"])
                else float("nan")
            ),
        },
    }


def _default_output_path(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}_perplexity{input_path.suffix}"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity of model responses from a steering result JSON file"
    )
    parser.add_argument("file", help="Path to the JSON result file")
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save the output JSON file",
        default=None,
    )
    parser.add_argument(
        "--model-name",
        help=(
            "Optional override for LM used in perplexity calculation. "
            "Normally this is auto-loaded from JSON config.model.name."
        ),
        default=None,
    )
    parser.add_argument(
        "--device",
        help="Device to run the LM on",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype for perplexity LM",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length used for PPL computation",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=None,
        help="Optional max token length to crop responses before evaluation (after concatenation with prompt)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Force steered vs baseline comparison (default: auto if baseline exists)",
    )

    args = parser.parse_args()

    print(f"Loading data from {args.file}...")
    data = load_json_results(args.file)
    samples, baseline_samples = extract_samples(data)

    print(f"Found {len(samples)} steered samples")
    print(f"Found {len(baseline_samples)} baseline samples")

    inferred_model = infer_model_name(data)
    model_name = args.model_name or inferred_model
    if not model_name:
        raise ValueError(
            "Could not infer model name from JSON. "
            "Expected config.model.name in the result file. "
            "You can still pass --model-name as an override."
        )

    auto_compare = bool(baseline_samples)
    do_compare = args.compare or auto_compare

    print(f"Initializing perplexity model: {model_name} on {args.device} (dtype={args.dtype})...")
    evaluator = PerplexityEvaluator(model_name=model_name, device=args.device, dtype_name=args.dtype)

    if do_compare and baseline_samples:
        print("Computing perplexity for steered and baseline responses...")
        output = evaluate_comparison(
            samples=samples,
            baseline_samples=baseline_samples,
            evaluator=evaluator,
            max_length=args.max_length,
            verbose=args.verbose,
        )

        summary = output["summary"]
        print("\n=== Results Summary ===")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Steered mean perplexity: {summary['steered_mean_perplexity']:.3f}")
        print(f"Baseline mean perplexity: {summary['baseline_mean_perplexity']:.3f}")
        print(f"Delta mean perplexity (steered - baseline): {summary['delta_mean_perplexity']:.3f}")
    else:
        if args.compare and not baseline_samples:
            print("--compare requested but no baseline_samples found; evaluating steered samples only.")
        print("Computing perplexity for steered responses...")
        sample_results = evaluate_samples(
            samples=samples,
            evaluator=evaluator,
            max_length=args.max_length,
            crop = args.crop,
            verbose=args.verbose,
        )
        summary = summarize_perplexities(sample_results)

        output = {
            "samples": sample_results,
            "summary": summary,
        }

        print("\n=== Results Summary ===")
        print(f"Total samples: {summary['count']}")
        print(f"Finite perplexities: {summary['finite_count']}")
        print(f"Mean perplexity: {summary['mean_perplexity']:.3f}")
        print(f"Median perplexity: {summary['median_perplexity']:.3f}")

    output_path = Path(args.output) if args.output else _default_output_path(Path(args.file))
    print(f"\nSaving results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
