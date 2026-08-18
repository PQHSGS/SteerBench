"""
KL Decay 02 — Multi-Token Per-Generation-Step Analysis

Uses SteeringPipeline for ALL methods (CAA, CHARS, LinearAcT, REPS, etc.).
No manual hooks — the library manages hook functions correctly.

Metrics:
  1. Full KL divergence via logit lens
  2. Top-K KL: KL contribution from top-K tokens of steered distribution
  3. Top-K overlap: fraction of steered top-K also in base top-K

Usage:
    conda activate sae_circuit
    unset CUDA_VISIBLE_DEVICES

    python Experiments/KL_decay/02_multi_token.py --task toxic --method CAA
    python Experiments/KL_decay/02_multi_token.py --task toxic --all_methods
    python Experiments/KL_decay/02_multi_token.py --all_tasks --all_methods
    python Experiments/KL_decay/02_multi_token.py --task toxic --method CAA --smoke
"""

import argparse
import gc
import json
import time
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, ".")
from Steering.pipeline import SteeringPipeline
from Steering.config.pipeline import PipelineConfig
from Steering.steer_models.weight import WeightSteerModel

torch.set_grad_enabled(False)

CONFIG_FILE = Path(__file__).parent / "config.json"
OUTPUT_DIR = Path(__file__).parent


def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="KL Decay Multi-Token")
    parser.add_argument("--task", type=str, default=None,
                        choices=["toxic", "evil", "deception", "refusal"])
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--all_methods", action="store_true")
    parser.add_argument("--coeff", type=float, default=None)
    parser.add_argument("--n_prompts", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


# ──────────────────────────────────────────────
# KL helper — full + top-K metrics
# ──────────────────────────────────────────────

def logit_lens_metrics(r_steer, r_base, model, topk_values=(10, 100, 500)):
    """
    Compute full KL and top-K filtered metrics between steered and base logit distributions.
    
    Full KL: KL(base || steered) over all tokens (backward compatible).
    Top-K KL: Σ_{x ∈ topK(P_steer)} P_steer(x) * log(P_steer(x)/P_base(x))
      — KL contribution from the top-K tokens of the steered distribution.
      — If this stays high while full KL decays, the harmful signal persists
        but is diluted by the tail.
    """
    dev = model.cfg.device
    logits_s = model.ln_final(r_steer.to(dev)) @ model.W_U.detach()
    logits_b = model.ln_final(r_base.to(dev)) @ model.W_U.detach()
    
    ls = F.log_softmax(logits_s.float(), dim=-1)
    lb = F.log_softmax(logits_b.float(), dim=-1)
    
    ps = ls.exp()  # steered probs
    pb = lb.exp()  # base probs
    
    # Full KL: KL(base || steered) — backward compatible
    full_kl = F.kl_div(ls, lb, log_target=True, reduction='sum').item()
    
    result = {"kl_full": full_kl}
    
    vocab_size = ps.shape[-1]
    for k in topk_values:
        actual_k = min(k, vocab_size)
        # Top-K of steered distribution
        topk_s_vals, topk_s_idx = ps.topk(actual_k)
        
        # KL contribution from top-K of steered: P_s(x) * log(P_s(x)/P_b(x))
        log_ps_topk = ls.gather(-1, topk_s_idx).squeeze(0)
        log_pb_topk = lb.gather(-1, topk_s_idx).squeeze(0)
        ps_topk = ps.gather(-1, topk_s_idx).squeeze(0)
        
        kl_topk = (ps_topk * (log_ps_topk - log_pb_topk)).sum().item()
        result[f"kl_top{k}"] = kl_topk
        result[f"frac_top{k}"] = kl_topk / full_kl if abs(full_kl) > 1e-10 else 0.0
        
        # Top-K overlap: what fraction of steered top-K are also in base top-K?
        topk_b_idx = pb.topk(actual_k).indices
        overlap = len(set(topk_s_idx.tolist()) & set(topk_b_idx.tolist()))
        result[f"overlap_top{k}"] = overlap / actual_k
    
    return result


# ──────────────────────────────────────────────
# Pipeline setup
# ──────────────────────────────────────────────

def setup_pipeline(config_path, device, model=None, coeff_override=None, n_prompts_override=None):
    """
    Create SteeringPipeline from a method/task config JSON.
    Returns (pipeline, steer, model, coeff_dict).
    """
    config = PipelineConfig.load(config_path)
    config.model.device = device
    if coeff_override is not None:
        config.steer.coeff = coeff_override
    if n_prompts_override is not None:
        config.n_test = n_prompts_override

    # Use pre-extracted vector (skip extraction)
    if not getattr(config, "load_vector", None):
        config.load_vector = config.save_vector

    pipeline = SteeringPipeline(config)
    if model is not None:
        pipeline.model = model
        pipeline._current_model_type = "tl"
    pipeline.setup()

    # After config.resolve(), coeff is always Dict[int, float]
    coeff_dict = pipeline.steer_config.coeff

    return pipeline, pipeline.steer_model, pipeline.model, coeff_dict


# ──────────────────────────────────────────────
# Multi-token KL tracking
# ──────────────────────────────────────────────

def track_multi_token_kl(model, steer, coeff_dict, prompts, track_layers, max_tokens, topk_values=(10, 100, 500)):
    """
    For each prompt, run step-by-step generation:
      - BASE: no hooks, capture residuals
      - STEERED: steer hooks active, capture residuals
      - Compute full KL + top-K KL at each (step, layer)
    """
    is_weight = isinstance(steer, WeightSteerModel)

    all_metrics = []  # list of dict[step][layer] = metrics dict

    for prompt in tqdm(prompts, desc="Prompts"):
        tokens = model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device)
        generated = tokens.clone()
        metrics_by_step = defaultdict(dict)

        for step in range(max_tokens):
            # BASE forward (no hooks)
            base_residuals = {}
            def make_base_hook(layer):
                def hook_fn(value, hook):
                    base_residuals[layer] = value[0, -1, :].detach().cpu()
                    return value
                return hook_fn

            base_fwd_hooks = [(f"blocks.{l}.hook_resid_post", make_base_hook(l)) for l in track_layers]
            model.reset_hooks()
            base_logits = model.run_with_hooks(generated, fwd_hooks=base_fwd_hooks, return_type="logits")

            # STEERED forward
            steered_residuals = {}
            def make_steered_hook(layer):
                def hook_fn(value, hook):
                    steered_residuals[layer] = value[0, -1, :].detach().cpu()
                    return value
                return hook_fn

            steered_fwd_hooks = [(f"blocks.{l}.hook_resid_post", make_steered_hook(l)) for l in track_layers]
            model.reset_hooks()
            if is_weight:
                steer._apply_weight_modifications(coeff_dict)
                try:
                    steered_logits = model.run_with_hooks(generated, fwd_hooks=steered_fwd_hooks, return_type="logits")
                finally:
                    steer._restore_weights(coeff_dict)
            else:
                steer.setup_hooks(coeff_dict)
                try:
                    steered_logits = model.run_with_hooks(generated, fwd_hooks=steered_fwd_hooks, return_type="logits")
                finally:
                    model.reset_hooks()

            # Full + top-K metrics at each layer
            for l in track_layers:
                if l in base_residuals and l in steered_residuals:
                    m = logit_lens_metrics(steered_residuals[l], base_residuals[l], model, topk_values)
                    metrics_by_step[step][l] = m

            # Next token from BASE
            next_tok = base_logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_tok], dim=1)

        all_metrics.append(dict(metrics_by_step))

    return all_metrics


# ──────────────────────────────────────────────
# Aggregation & printing
# ──────────────────────────────────────────────

def aggregate_multi_token(all_metrics, track_layers, max_tokens):
    """Aggregate per-prompt metrics dicts into mean/std per (step, layer, metric)."""
    result = {}
    for step in range(max_tokens):
        result[step] = {}
        for l in track_layers:
            # Collect all metric keys from first prompt that has this step/layer
            metric_keys = None
            for pm in all_metrics:
                if step in pm and l in pm[step]:
                    metric_keys = list(pm[step][l].keys())
                    break
            if not metric_keys:
                continue

            result[step][l] = {}
            for key in metric_keys:
                vals = [pm[step][l][key] for pm in all_metrics if step in pm and l in pm[step] and key in pm[step][l]]
                if vals:
                    result[step][l][f"{key}_mean"] = float(np.mean(vals))
                    result[step][l][f"{key}_std"] = float(np.std(vals))
    return result


def print_multi_token_table(label, agg, track_layers, max_tokens):
    print(f"\n  {'='*100}")
    print(f"  {label}")
    print(f"  {'='*100}")

    # Print full KL table
    print(f"\n  Full KL by generation step:")
    header = f"  {'Step':<6}"
    for l in track_layers:
        header += f" {'L'+str(l):>10}"
    print(header)
    print(f"  {'-'*80}")
    for step in range(max_tokens):
        row = f"  {step+1:<6}"
        for l in track_layers:
            d = agg.get(step, {}).get(l, {})
            mean = d.get("kl_full_mean", 0)
            row += f" {mean:>10.4f}"
        print(row)

    # Print top-100 KL table
    print(f"\n  Top-100 KL by generation step:")
    header = f"  {'Step':<6}"
    for l in track_layers:
        header += f" {'L'+str(l):>10}"
    print(header)
    print(f"  {'-'*80}")
    for step in range(max_tokens):
        row = f"  {step+1:<6}"
        for l in track_layers:
            d = agg.get(step, {}).get(l, {})
            mean = d.get("kl_top100_mean", 0)
            row += f" {mean:>10.4f}"
        print(row)

    # Print overlap at step 0 and last step
    print(f"\n  Top-K overlap (step1 vs last step):")
    for l in track_layers:
        d0 = agg.get(0, {}).get(l, {})
        dN = agg.get(max_tokens - 1, {}).get(l, {})
        o10_0 = d0.get("overlap_top10_mean", 0)
        o10_N = dN.get("overlap_top10_mean", 0)
        o100_0 = d0.get("overlap_top100_mean", 0)
        o100_N = dN.get("overlap_top100_mean", 0)
        print(f"    L{l}: top10: {o10_0:.2f} -> {o10_N:.2f}  |  top100: {o100_0:.2f} -> {o100_N:.2f}")

    # Ratios
    print(f"\n  Step1/Step{max_tokens} ratio per layer:")
    print(f"  {'Layer':<6} {'KL_full':>10} {'KL_top10':>10} {'KL_top100':>10} {'KL_top500':>10}")
    for l in track_layers:
        d1 = agg.get(0, {}).get(l, {})
        dN = agg.get(max_tokens - 1, {}).get(l, {})
        kf1 = d1.get("kl_full_mean", 0)
        kfN = dN.get("kl_full_mean", 0)
        k101 = d1.get("kl_top10_mean", 0)
        k10N = dN.get("kl_top10_mean", 0)
        k1001 = d1.get("kl_top100_mean", 0)
        k100N = dN.get("kl_top100_mean", 0)
        k5001 = d1.get("kl_top500_mean", 0)
        k500N = dN.get("kl_top500_mean", 0)
        r_full = kfN / kf1 if kf1 > 0 else 0
        r10 = k10N / k101 if k101 > 0 else 0
        r100 = k100N / k1001 if k1001 > 0 else 0
        r500 = k500N / k5001 if k5001 > 0 else 0
        print(f"  L{l:<5} {r_full:>10.4f} {r10:>10.4f} {r100:>10.4f} {r500:>10.4f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    exp_cfg = load_config()

    if args.coeff is not None:
        exp_cfg["coeff"] = args.coeff
    if args.n_prompts is not None:
        exp_cfg["n_prompts"] = args.n_prompts
    if args.smoke:
        exp_cfg["n_prompts"] = 3
        exp_cfg["max_tokens"] = 5

    tasks = [args.task] if args.task else (exp_cfg["tasks"] if args.all_tasks else ["toxic"])
    methods = [args.method.upper()] if args.method else []
    if args.all_methods:
        methods = exp_cfg["methods"]
    if not methods:
        methods = ["CAA"]

    track_layers = list(range(exp_cfg["track_layers_start"], exp_cfg["track_layers_end"]))
    max_tokens = exp_cfg["max_tokens"]
    coeff = exp_cfg["coeff"]
    topk_values = tuple(exp_cfg.get("topk_values", [10, 100, 500]))

    print(f"{'='*80}")
    print(f"  KL DECAY MULTI-TOKEN (SteeringPipeline)")
    print(f"{'='*80}")
    print(f"  Tasks: {tasks}")
    print(f"  Methods: {methods}")
    print(f"  Track layers: L{track_layers[0]}-L{track_layers[-1]}")
    print(f"  Top-K values: {topk_values}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Coefficient: {coeff}")
    print(f"  Prompts: {exp_cfg['n_prompts']}")
    print(f"{'='*80}\n")

    t0 = time.time()
    all_results = {}
    model = None

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*60}")
        task_results = {}

        for method in methods:
            print(f"\n  --- {method} ---")

            # Find config path (with task_config_suffix support for refusal)
            method_info = exp_cfg.get("method_config_map", {}).get(method, {})
            cfg_dir = method_info.get("cfg_dir", "")
            suffix = exp_cfg.get("task_config_suffix", {}).get(task, task)
            config_path = f"{cfg_dir}/gemma_{suffix}.json"

            if not Path(config_path).exists():
                print(f"    No config for {method}/{task}, skipping")
                continue

            # Setup pipeline (one per method — model loaded once, steer_model varies)
            try:
                pipeline, steer, model, coeff_dict = setup_pipeline(
                    config_path, args.device, model=model,
                    coeff_override=coeff,
                    n_prompts_override=exp_cfg["n_prompts"],
                )
                print(f"    Pipeline ready (method={steer.__class__.__name__}, "
                      f"layers={steer.layer}, coeff={coeff_dict})")
            except Exception as e:
                print(f"    Error setting up pipeline: {e}")
                continue

            # Load prompts
            test_data = pipeline.load_test_data(
                dataset_name=pipeline.config.test_dataset,
                n_samples=exp_cfg["n_prompts"],
                apply_chat_template=pipeline.steer_config.apply_chat_template,
            )
            prompts = [str(s.get("question", s)) for s in test_data]
            print(f"    Loaded {len(prompts)} prompts")

            # Run multi-token KL (full + top-K)
            all_kl = track_multi_token_kl(
                model, steer, coeff_dict, prompts, track_layers, max_tokens,
                topk_values=topk_values
            )

            # Aggregate
            agg = aggregate_multi_token(all_kl, track_layers, max_tokens)
            label = f"{method} — {task} — KL by generation step"
            print_multi_token_table(label, agg, track_layers, max_tokens)
            task_results[method] = agg

            # Cleanup
            model.reset_hooks()
            del pipeline, steer, coeff_dict, all_kl, agg, prompts
            gc.collect(); torch.cuda.empty_cache(); time.sleep(1)

        all_results[task] = task_results

    if model is not None:
        model.reset_hooks()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"multi_token_kl_{'_'.join(tasks)}_{'_'.join(methods)}_c{coeff}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Cross-task comparison
    if len(tasks) > 1 and len(methods) > 0:
        print(f"\n{'='*80}")
        print(f"  CROSS-TASK COMPARISON (step1 vs step{max_tokens} KL ratio)")
        print(f"{'='*80}")
        for method in methods:
            print(f"\n  Method: {method}")
            for task in tasks:
                agg = all_results.get(task, {}).get(method, {})
                if not agg:
                    continue
                ratios_full, ratios_top100 = [], []
                for l in track_layers:
                    s1 = agg.get(0, {}).get(l, {})
                    sN = agg.get(max_tokens - 1, {}).get(l, {})
                    kf1 = s1.get("kl_full_mean", 0)
                    kfN = sN.get("kl_full_mean", 0)
                    kt100_1 = s1.get("kl_top100_mean", 0)
                    kt100_N = sN.get("kl_top100_mean", 0)
                    ratios_full.append(kfN / kf1 if kf1 > 0 else 0)
                    ratios_top100.append(kt100_N / kt100_1 if kt100_1 > 0 else 0)
                print(f"    {task:<12}: full={np.mean(ratios_full):.4f}  top100={np.mean(ratios_top100):.4f}")

    print(f"\n{'='*80}")
    print(f"  Saved to {out}")
    print(f"  Total: {time.time()-t0:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
