"""
KL Decay Experiment — Consolidated (SteeringPipeline)

Measures perturbation survival across layers after injection.
Compares multiple methods (CAA, FLOW, REPS, etc.) across tasks (Toxic, Evil, Deception).
Includes random vector control for natural decay baseline.

Metrics:
  1. KL divergence via logit lens (full vocab + top-K filtered)
  2. Top-K overlap: fraction of steered top-K also in base top-K
  3. Norm survival: ||steered - unsteered|| at each layer
  4. Direction persistence: cos(steered - unsteered, v_original) at each layer

Usage:
    conda activate sae_circuit
    unset CUDA_VISIBLE_DEVICES
    
    python Experiments/KL_decay/01_kl_decay.py --task toxic --method CAA
    python Experiments/KL_decay/01_kl_decay.py --task toxic --all_methods
    python Experiments/KL_decay/01_kl_decay.py --all_tasks --all_methods
    python Experiments/KL_decay/01_kl_decay.py --task toxic --method CAA --smoke
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
from functools import partial
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
    parser = argparse.ArgumentParser(description="KL Decay")
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
# Pipeline setup
# ──────────────────────────────────────────────

def setup_pipeline(config_path, device, model=None, coeff_override=None, n_prompts_override=None):
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

    coeff_dict = pipeline.steer_config.coeff
    return pipeline, pipeline.steer_model, pipeline.model, coeff_dict


# ──────────────────────────────────────────────
# Residual capture
# ──────────────────────────────────────────────

def logit_lens_metrics(r_steer, r_base, model, topk_values=(10, 100, 500)):
    """
    Compute full KL and top-K filtered metrics between steered and base logit distributions.
    
    Full KL: KL(base || steered) over all tokens (backward compatible).
    Top-K KL: Σ_{x ∈ topK(P_steer)} P_steer(x) * log(P_steer(x)/P_base(x))
      — KL contribution from the top-K tokens of the steered distribution.
      — If this stays high while full KL decays, the harmful signal persists
        but is diluted by the tail.
    
    Returns dict with:
      kl_full, kl_top{k}, frac_top{k}, overlap_top{k}
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
# Steered vs base residual comparison
# ──────────────────────────────────────────────

def run_steered_vs_base(model, steer, coeff_dict, tokens, track_layers):
    """
    Run BASE (no steer) and STEERED (with steer hooks) forward passes.
    Return (base_residuals, steered_residuals) as dicts[layer] = tensor.
    """
    is_weight = isinstance(steer, WeightSteerModel)

    # BASE
    base_residuals = {}
    def make_base_hook(layer):
        def hook_fn(value, hook):
            base_residuals[layer] = value[0, -1, :].detach().cpu()
            return value
        return hook_fn

    base_fwd_hooks = [(f"blocks.{l}.hook_resid_post", make_base_hook(l)) for l in track_layers]

    model.reset_hooks()
    model.run_with_hooks(tokens, fwd_hooks=base_fwd_hooks, return_type="logits")

    # STEERED
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
            model.run_with_hooks(tokens, fwd_hooks=steered_fwd_hooks, return_type="logits")
        finally:
            steer._restore_weights(coeff_dict)
    else:
        steer.setup_hooks(coeff_dict)
        try:
            model.run_with_hooks(tokens, fwd_hooks=steered_fwd_hooks, return_type="logits")
        finally:
            model.reset_hooks()

    return base_residuals, steered_residuals


def run_experiment(model, steer, coeff_dict, prompts, track_layers, topk_values=(10, 100, 500)):
    """Run KL decay for one method across prompts. Returns full + top-K metrics."""
    inject_layer = steer.layer[0] if isinstance(steer.layer, list) else steer.layer
    sv = list(steer.steering_vector.values())[0]

    # Collect all metric values per layer
    metrics_by_layer = defaultdict(lambda: defaultdict(list))

    for idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        tokens = model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device)

        base, steered = run_steered_vs_base(model, steer, coeff_dict, tokens, track_layers)

        for l in track_layers:
            if l not in base or l not in steered:
                continue
            diff = steered[l] - base[l]

            # Full + top-K metrics
            m = logit_lens_metrics(steered[l], base[l], model, topk_values=topk_values)
            for key, val in m.items():
                metrics_by_layer[l][key].append(val)

            # Norm and cosine
            norm = diff.norm().item()
            cos = F.cosine_similarity(
                diff.unsqueeze(0).to(sv.device),
                sv.unsqueeze(0).float()
            ).item()

            metrics_by_layer[l]["norm"].append(norm)
            metrics_by_layer[l]["cos"].append(cos)

        if (idx + 1) % 10 == 0:
            print(f"    {idx+1}/{len(prompts)} prompts processed")

    # Aggregate: mean and std for each metric
    results = {}
    for l in track_layers:
        results[l] = {}
        for key, vals in metrics_by_layer[l].items():
            results[l][f"{key}_mean"] = float(np.mean(vals))
            results[l][f"{key}_std"] = float(np.std(vals))
    return results


# ──────────────────────────────────────────────
# Random control (same vector shape, no semantic meaning)
# ──────────────────────────────────────────────

def make_random_steer(pipeline, model, coeff_dict, task, n_prompts):
    """
    Create a SteeringPipeline with a random vector for control.
    Reuses the same config but replaces the vector.
    """
    import torch
    from Steering.config import SteeringVector
    steer = pipeline.steer_model
    real_sv = steer.steering_vector
    random_sv = {k: F.normalize(torch.randn_like(v.float()), dim=-1) for k, v in real_sv.items()}
    steer.steering_vector = random_sv
    return steer


# ──────────────────────────────────────────────
# Printing
# ──────────────────────────────────────────────

def compute_decay_ratios(results, first_layer, last_layer):
    """Compute decay ratios for full KL and top-K KL."""
    ratios = {}
    for metric in ["kl_full", "kl_top10", "kl_top100", "kl_top500", "norm"]:
        first = results.get(first_layer, {}).get(f"{metric}_mean", 0)
        last = results.get(last_layer, {}).get(f"{metric}_mean", 0)
        ratios[metric] = last / first if first > 0 else 0
    ratios["cos_first"] = results.get(first_layer, {}).get("cos_mean", 0)
    ratios["cos_last"] = results.get(last_layer, {}).get("cos_mean", 0)
    return ratios


def print_results(label, results, track_layers):
    print(f"\n  {'='*100}")
    print(f"  {label}")
    print(f"  {'='*100}")
    print(f"  {'Layer':<6} {'KL_full':>10} {'KL_top10':>10} {'KL_top100':>10} {'KL_top500':>10} {'Norm':>8} {'Cos':>8}")
    print(f"  {'-'*70}")
    for l in track_layers:
        d = results.get(l, {})
        kl_full = d.get("kl_full_mean", 0)
        kl10 = d.get("kl_top10_mean", 0)
        kl100 = d.get("kl_top100_mean", 0)
        kl500 = d.get("kl_top500_mean", 0)
        nm = d.get("norm_mean", 0)
        cos = d.get("cos_mean", 0)
        print(f"  L{l:<5} {kl_full:>10.4f} {kl10:>10.4f} {kl100:>10.4f} {kl500:>10.4f} {nm:>8.2f} {cos:>8.4f}")

    # Overlap at L15 and L25
    print(f"\n  Top-K overlap (steered vs base):")
    for l in [track_layers[0], track_layers[-1]]:
        d = results.get(l, {})
        o10 = d.get("overlap_top10_mean", 0)
        o100 = d.get("overlap_top100_mean", 0)
        o500 = d.get("overlap_top500_mean", 0)
        print(f"    L{l}: top10={o10:.2f}  top100={o100:.2f}  top500={o500:.2f}")

    # Fraction of KL in top-K
    print(f"\n  Fraction of full KL in top-K:")
    for l in [track_layers[0], track_layers[-1]]:
        d = results.get(l, {})
        f10 = d.get("frac_top10_mean", 0)
        f100 = d.get("frac_top100_mean", 0)
        f500 = d.get("frac_top500_mean", 0)
        print(f"    L{l}: top10={f10:.2%}  top100={f100:.2%}  top500={f500:.2%}")

    ratios = compute_decay_ratios(results, track_layers[0], track_layers[-1])
    print(f"\n  Decay ratios (L{track_layers[0]}\u2192L{track_layers[-1]}):")
    print(f"    KL_full:   {ratios['kl_full']:.4f}")
    print(f"    KL_top10:  {ratios['kl_top10']:.4f}")
    print(f"    KL_top100: {ratios['kl_top100']:.4f}")
    print(f"    KL_top500: {ratios['kl_top500']:.4f}")
    print(f"    Norm:      {ratios['norm']:.4f}")
    print(f"    Cos:       {ratios['cos_first']:.4f} \u2192 {ratios['cos_last']:.4f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    exp_config = load_config()

    if args.coeff is not None:
        exp_config["coeff"] = args.coeff
    if args.n_prompts is not None:
        exp_config["n_prompts"] = args.n_prompts
    if args.smoke:
        exp_config["n_prompts"] = 3

    tasks = [args.task] if args.task else (exp_config["tasks"] if args.all_tasks else ["toxic"])

    if args.all_methods:
        methods = exp_config["methods"]
    elif args.method:
        methods = [args.method.upper()]
    else:
        methods = ["CAA"]

    track_layers = list(range(exp_config["track_layers_start"], exp_config["track_layers_end"]))
    topk_values = tuple(exp_config.get("topk_values", [10, 100, 500]))

    print(f"{'='*80}")
    print(f"  KL DECAY EXPERIMENT (SteeringPipeline)")
    print(f"{'='*80}")
    print(f"  Tasks: {tasks}")
    print(f"  Methods: {methods}")
    print(f"  Track layers: L{track_layers[0]}-L{track_layers[-1]}")
    print(f"  Top-K values: {topk_values}")
    print(f"  Coefficient: {exp_config['coeff']}")
    print(f"  Prompts: {exp_config['n_prompts']}")
    print(f"{'='*80}\n")

    t_start = time.time()
    all_results = {}

    model = None
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*60}")

        task_results = {}

        for method in methods:
            try:
                print(f"\n  --- {method} ---")

                method_info = exp_config.get("method_config_map", {}).get(method, {})
                cfg_dir = method_info.get("cfg_dir", "")
                config_suffix = exp_config.get("task_config_suffix", {}).get(task, task)
                config_path = f"{cfg_dir}/gemma_{config_suffix}.json"

                if not Path(config_path).exists():
                    print(f"    No config for {method}/{task}, skipping")
                    continue

                # Setup pipeline (model + vector + steer_model all handled by library)
                pipeline, steer, model, coeff_dict = setup_pipeline(
                    config_path, args.device, model=model,
                    coeff_override=exp_config["coeff"],
                    n_prompts_override=exp_config["n_prompts"],
                )
                print(f"    Pipeline ready ({steer.__class__.__name__}, "
                      f"layers={steer.layer}, coeff={coeff_dict})")

                # Load prompts from config's test_dataset
                test_data = pipeline.load_test_data(
                    dataset_name=pipeline.config.test_dataset,
                    n_samples=exp_config["n_prompts"],
                    apply_chat_template=pipeline.steer_config.apply_chat_template,
                )
                prompts = [str(s.get("question", s)) for s in test_data]
                print(f"    Loaded {len(prompts)} prompts from {pipeline.config.test_dataset}")

                # Run with real vector
                print(f"    Running {method} steering vector...")
                results_task = run_experiment(model, steer, coeff_dict, prompts, track_layers, topk_values)

                # Run with random vector (control) — a simple CAA injection of a random vector with the same norm
                print(f"    Running random vector (control)...")
                random_sv = {}
                for k, v in steer.steering_vector.items():
                    v_norm = v.norm().item()
                    rand_v = torch.randn_like(v.float())
                    rand_v = F.normalize(rand_v, dim=-1) * v_norm
                    random_sv[k] = rand_v.to(device=v.device, dtype=v.dtype)
                
                from Steering.steer_models.dense import DenseSteerModel
                control_steer = DenseSteerModel(
                    model=model,
                    layer=steer.layer,
                    steering_vector=random_sv,
                    hook_point=steer.hook_point,
                    position=steer.position,
                )
                results_random = run_experiment(model, control_steer, coeff_dict, prompts, track_layers, topk_values)

                # Print
                print_results(f"{method} \u2014 {task}", results_task, track_layers)
                print_results(f"{method} \u2014 random control", results_random, track_layers)

                # Suppression ratio — full KL and top-K KL
                print(f"\n    Suppression ratio (task / random) \u2014 KL:")
                print(f"    {'Layer':<6} {'KL_full':>10} {'KL_top10':>10} {'KL_top100':>10} {'KL_top500':>10}")
                for l in track_layers:
                    td = results_task.get(l, {})
                    rd = results_random.get(l, {})
                    kf = td.get("kl_full_mean", 0)
                    kr = rd.get("kl_full_mean", 0)
                    k10f = td.get("kl_top10_mean", 0)
                    k10r = rd.get("kl_top10_mean", 0)
                    k100f = td.get("kl_top100_mean", 0)
                    k100r = rd.get("kl_top100_mean", 0)
                    k500f = td.get("kl_top500_mean", 0)
                    k500r = rd.get("kl_top500_mean", 0)
                    r_full = kf / kr if kr > 0 else 0
                    r10 = k10f / k10r if k10r > 0 else 0
                    r100 = k100f / k100r if k100r > 0 else 0
                    r500 = k500f / k500r if k500r > 0 else 0
                    print(f"    L{l:<5} {r_full:>10.4f} {r10:>10.4f} {r100:>10.4f} {r500:>10.4f}")

                task_results[method] = {
                    "task_vector": results_task,
                    "random_control": results_random,
                }

            except Exception as e:
                print(f"    ERROR in {method}/{task}: {e}")
                import traceback; traceback.print_exc()
            finally:
                try:
                    model.reset_hooks()
                except Exception:
                    pass
                try:
                    del pipeline, steer, coeff_dict, results_task, results_random, prompts, control_steer
                except Exception:
                    pass
                gc.collect(); torch.cuda.empty_cache(); time.sleep(1)

        all_results[task] = task_results

    if model is not None:
        model.reset_hooks()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    method_str = "_".join(methods[:3])
    task_str = "_".join(tasks)
    output_file = OUTPUT_DIR / f"kl_decay_{task_str}_{method_str}_c{exp_config['coeff']}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    if len(tasks) > 1:
        print(f"\n{'='*80}")
        print(f"  CROSS-TASK COMPARISON (decay ratios)")
        print(f"{'='*80}")
        for method in methods:
            print(f"\n  Method: {method}")
            for task in tasks:
                if method in all_results.get(task, {}):
                    ratios = compute_decay_ratios(
                        all_results[task][method]["task_vector"],
                        track_layers[0], track_layers[-1]
                    )
                    print(f"    {task:<12}: full={ratios['kl_full']:.4f}  top10={ratios['kl_top10']:.4f}  top100={ratios['kl_top100']:.4f}  top500={ratios['kl_top500']:.4f}")

    if len(methods) > 1:
        print(f"\n{'='*80}")
        print(f"  CROSS-METHOD COMPARISON (decay ratios)")
        print(f"{'='*80}")
        for task in tasks:
            print(f"\n  Task: {task}")
            for method in methods:
                if method in all_results.get(task, {}):
                    ratios = compute_decay_ratios(
                        all_results[task][method]["task_vector"],
                        track_layers[0], track_layers[-1]
                    )
                    print(f"    {method:<12}: full={ratios['kl_full']:.4f}  top10={ratios['kl_top10']:.4f}  top100={ratios['kl_top100']:.4f}  top500={ratios['kl_top500']:.4f}")

    print(f"\n{'='*80}")
    print(f"  Saved to {output_file}")
    print(f"  Total: {time.time()-t_start:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
