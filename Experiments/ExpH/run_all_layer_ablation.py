"""
All-layer r̂ ablation via SteeringPipeline.

Wraps any method's setup_hooks to add r̂ subtraction/projection at specified layers.
Supports all methods (CAA, CHARS, LinearAcT, REPS, etc.) via existing eval configs.

Usage:
  conda activate sae_circuit; unset CUDA_VISIBLE_DEVICES
  python Experiments/ExpH/run_all_layer_ablation.py --task toxic --method CAA
  python Experiments/ExpH/run_all_layer_ablation.py --task toxic --method CAA --ablation_layers 16,18
  python Experiments/ExpH/run_all_layer_ablation.py --all_tasks --all_methods
  python Experiments/ExpH/run_all_layer_ablation.py --task toxic --method CAA --skip_coskl --smoke
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, ".")
from Steering.pipeline import SteeringPipeline
from Steering.config.pipeline import PipelineConfig
from Steering.steer_models.dense import DenseSteerModel
from Steering.steer_models.weight import WeightSteerModel

torch.set_grad_enabled(False)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

MODEL_NAME = "google/gemma-2-2b-it"
DTYPE = torch.bfloat16

CONFIG_FILE = Path(__file__).parent.parent.parent / "Experiments/KL_decay/config.json"
OUTPUT_DIR = Path(__file__).parent / "results"

TASKS = ["toxic", "evil", "deception"]
METHODS = ["CAA", "CHARS", "LinearAcT", "PID", "CURVE", "REPS", "MANIFOLD", "SRPS", "WEIGHTSTEER"]
TRACK_LAYERS = list(range(15, 26))  # L15–L25


def parse_args():
    parser = argparse.ArgumentParser(description="All-layer r̂ ablation")
    parser.add_argument("--task", type=str, default=None, choices=TASKS)
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--all_methods", action="store_true")
    parser.add_argument("--coeff", type=float, default=5.0)
    parser.add_argument("--n_prompts", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ablation_layers", type=str, default=None,
                        help="Comma-separated layer numbers, e.g. 16,18,20. Default: all layers >= steer")
    parser.add_argument("--variant", type=str, default="subtract", choices=["subtract", "project"])
    parser.add_argument("--skip_coskl", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def get_ablation_layers(steer_layer, user_specified):
    """Resolve which layers to ablate."""
    if user_specified is None:
        # Default: all layers from steer_layer+1 to L25
        return list(range(steer_layer + 1, 26))
    return [int(x) for x in user_specified.split(",")]


def load_rhat_vectors(device):
    """Load per-layer r̂ (refusal) vectors."""
    r_path = Path("Experiments/ExpH/results/refusal_vecs_all_layers.pt")
    if not r_path.exists():
        print(f"ERROR: r̂ vectors not found at {r_path}")
        print("Run extraction first: python Experiments/ExpH/analyze_kl_all.py")
        sys.exit(1)
    r_raw = torch.load(str(r_path), map_location=device)
    return {int(k): v.to(dtype=DTYPE) for k, v in r_raw.items()}


# ──────────────────────────────────────────────
# Pipeline setup (same pattern as KL_decay/01)
# ──────────────────────────────────────────────

def setup_pipeline(config_path, device, coeff_override=None, n_prompts_override=None):
    config = PipelineConfig.load(config_path)
    config.model.device = device
    if coeff_override is not None:
        config.steer.coeff = coeff_override
    if n_prompts_override is not None:
        config.n_test = n_prompts_override

    if not getattr(config, "load_vector", None):
        config.load_vector = config.save_vector

    pipeline = SteeringPipeline(config)
    pipeline.setup()

    coeff_dict = pipeline.steer_config.coeff
    return pipeline, pipeline.steer_model, pipeline.model, coeff_dict


# ──────────────────────────────────────────────
# Wrapped setup_hooks: original method + r̂ ablation
# ──────────────────────────────────────────────

def wrap_setup_hooks(steer, rhat_vecs, ablation_layers, variant):
    """
    Wrap steer.setup_hooks to add r̂ ablation after the method's own hooks.
    This is called per prompt by the pipeline (hooks are wiped and reinstalled each time).
    """
    original_setup = steer.setup_hooks

    def wrapped_setup(coeff_dict):
        # 1. Install method's own steering hooks
        handles = original_setup(coeff_dict)

        # 2. Add r̂ ablation hooks on top
        model = steer.model
        for L in ablation_layers:
            if L not in rhat_vecs:
                continue
            r = rhat_vecs[L].to(dtype=DTYPE, device=model.cfg.device)

            if variant == "subtract":
                def _make_sub(r_vec):
                    def hook(value, hook):
                        value[:, -1, :] -= r_vec
                        return value
                    return hook
                h = model.add_hook(f"blocks.{L}.hook_resid_pre", _make_sub(r), "fwd")
                handles.append(h)

            elif variant == "project":
                rns = (r.norm() ** 2).item()
                def _make_proj(r_vec, rns_val):
                    def hook(value, hook):
                        h = value[:, -1, :]
                        dots = torch.matmul(h, r_vec) / (rns_val + 1e-8)
                        proj = dots.unsqueeze(-1) * r_vec
                        value[:, -1, :] -= proj.to(dtype=value.dtype)
                        return value
                    return hook
                h = model.add_hook(f"blocks.{L}.hook_resid_pre", _make_proj(r, rns), "fwd")
                handles.append(h)

        # 3. Post-processing hooks (if any)
        if steer.post_processor is not None:
            handles.extend(steer._setup_post_process_hooks())

        return handles

    steer.setup_hooks = wrapped_setup


# ──────────────────────────────────────────────
# Residual capture (for Phase 1+2 cos/KL)
# ──────────────────────────────────────────────

def logit_lens_kl(r_steer, r_base, model):
    dev = model.cfg.device
    ls = F.log_softmax(
        (model.ln_final(r_steer.to(dev)) @ model.W_U.detach()).float(), dim=-1
    )
    lb = F.log_softmax(
        (model.ln_final(r_base.to(dev)) @ model.W_U.detach()).float(), dim=-1
    )
    return F.kl_div(ls, lb, log_target=True, reduction='sum').item()


def run_capture(model, steer, coeff_dict, tokens, track_layers, rhat_vecs):
    """Run base + steered forward, capture residuals + compute cos(h,r̂) and KL."""
    is_weight = isinstance(steer, WeightSteerModel)

    # BASE
    base_residuals = {}
    def make_base_hook(layer):
        def hook_fn(value, hook):
            base_residuals[layer] = value[0, -1, :].detach().cpu()
            return value
        return hook_fn
    base_hooks = [(f"blocks.{l}.hook_resid_post", make_base_hook(l)) for l in track_layers]
    model.reset_hooks()
    model.run_with_hooks(tokens, fwd_hooks=base_hooks, return_type="logits")

    # STEERED (with r̂ ablation)
    steered_residuals = {}
    def make_steered_hook(layer):
        def hook_fn(value, hook):
            steered_residuals[layer] = value[0, -1, :].detach().cpu()
            return value
        return hook_fn
    steered_hooks = [(f"blocks.{l}.hook_resid_post", make_steered_hook(l)) for l in track_layers]

    model.reset_hooks()
    if is_weight:
        steer._apply_weight_modifications(coeff_dict)
        try:
            model.run_with_hooks(tokens, fwd_hooks=steered_hooks, return_type="logits")
        finally:
            steer._restore_weights(coeff_dict)
    else:
        steer.setup_hooks(coeff_dict)
        try:
            model.run_with_hooks(tokens, fwd_hooks=steered_hooks, return_type="logits")
        finally:
            model.reset_hooks()

    # Compute metrics per layer
    sv = list(steer.steering_vector.values())[0]
    layer_metrics = {}
    for l in track_layers:
        if l not in base_residuals or l not in steered_residuals:
            continue
        diff = steered_residuals[l] - base_residuals[l]
        kl = logit_lens_kl(steered_residuals[l], base_residuals[l], model)
        norm = diff.norm().item()
        cos = F.cosine_similarity(
            diff.unsqueeze(0).to(sv.device), sv.unsqueeze(0).float()
        ).item()
        # cos with r̂
        r_hat = rhat_vecs.get(l)
        if r_hat is not None:
            cos_rhat = F.cosine_similarity(
                diff.unsqueeze(0).to(r_hat.device), r_hat.unsqueeze(0).float()
            ).item()
        else:
            cos_rhat = 0.0
        layer_metrics[l] = {"kl": kl, "norm": norm, "cos_v": cos, "cos_rhat": cos_rhat}

    return layer_metrics


# ──────────────────────────────────────────────
# Phase 1+2: Cos + KL analysis
# ──────────────────────────────────────────────

def run_coskl(model, steer, coeff_dict, prompts, track_layers, rhat_vecs, ablation_layers):
    """Measure cos(h,r̂), cos(h,v), KL at each layer with r̂ ablation active."""
    agg = defaultdict(lambda: defaultdict(list))
    for prompt in tqdm(prompts, desc="CosKL"):
        tokens = model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device)
        metrics = run_capture(model, steer, coeff_dict, tokens, track_layers, rhat_vecs)
        for l, m in metrics.items():
            for k, v in m.items():
                agg[l][k].append(v)

    results = {}
    for l in track_layers:
        results[str(l)] = {
            k: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for k, vals in agg[l].items()
        }
    return results


# ──────────────────────────────────────────────
# Phase 3: Accuracy eval via pipeline
# ──────────────────────────────────────────────

def run_accuracy(pipeline, n_eval, batch_size=4):
    """Run pipeline eval loop, return accuracy, ppl, repetition."""
    from Steering.data.loader import EvalDataLoader
    from Steering.data.data_registry import TEST_DATASET_REGISTRY

    test_data = pipeline.load_test_data(
        dataset_name=pipeline.config.test_dataset,
        n_samples=n_eval,
        apply_chat_template=pipeline.steer_config.apply_chat_template,
    )
    evaluator_type = TEST_DATASET_REGISTRY[pipeline.config.test_dataset].evaluator
    evaluator = pipeline.get_evaluator(evaluator_type)

    coeff_dict = pipeline.steer_config.coeff
    correct, total, samples = pipeline._run_eval_loop(
        evaluator, test_data, coeff_dict,
        batch_size, verbose=False, apply_steer=True,
        max_new_tokens=pipeline.model_config.max_new_tokens,
    )
    metrics = pipeline._compute_response_score(samples, compute_ppl=True, compute_lsp=False)

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "perplexity": metrics["perplexity"],
        "repetition_rate": metrics["repetition_rate"],
        "n_correct": correct,
        "n_total": total,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    with open(CONFIG_FILE) as f:
        exp_config = json.load(f)

    if args.smoke:
        args.n_prompts = 3

    tasks = [args.task] if args.task else (TASKS if args.all_tasks else ["toxic"])
    methods = [args.method.upper()] if args.method else (METHODS if args.all_methods else ["CAA"])

    rhat_vecs = load_rhat_vectors("cuda")

    print(f"{'='*80}")
    print(f"  ALL-LAYER r̂ ABLATION (SteeringPipeline)")
    print(f"{'='*80}")
    print(f"  Tasks: {tasks}")
    print(f"  Methods: {methods}")
    print(f"  Ablation variant: {args.variant}")
    print(f"  Ablation layers: {args.ablation_layers or 'all >= steer+1'}")
    print(f"  Coefficient: {args.coeff}")
    print(f"  Prompts: {args.n_prompts}")
    print(f"{'='*80}\n")

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

                # Setup pipeline
                pipeline, steer, model, coeff_dict = setup_pipeline(
                    config_path, args.device, model=model,
                    coeff_override=args.coeff,
                    n_prompts_override=args.n_prompts,
                )
                steer_layer = steer.layer[0] if isinstance(steer.layer, list) else steer.layer
                ablation_layers = get_ablation_layers(steer_layer, args.ablation_layers)

                print(f"    Pipeline ready ({steer.__class__.__name__}, "
                      f"layer={steer_layer}, coeff={coeff_dict})")
                print(f"    Ablating r̂ at layers: {ablation_layers} ({args.variant})")

                # Wrap setup_hooks to include r̂ ablation
                wrap_setup_hooks(steer, rhat_vecs, ablation_layers, args.variant)

                # Load prompts
                test_data = pipeline.load_test_data(
                    dataset_name=pipeline.config.test_dataset,
                    n_samples=args.n_prompts,
                    apply_chat_template=pipeline.steer_config.apply_chat_template,
                )
                prompts = [str(s.get("question", s)) for s in test_data]
                print(f"    Loaded {len(prompts)} prompts")

                method_results = {"method": method, "task": task, "steer_layer": steer_layer,
                                  "ablation_layers": ablation_layers, "variant": args.variant}

                # Phase 1+2: Cos + KL
                if not args.skip_coskl:
                    print(f"    Phase 1+2: Cos + KL analysis...")
                    coskl = run_coskl(model, steer, coeff_dict, prompts, TRACK_LAYERS, rhat_vecs, ablation_layers)
                    method_results["coskl"] = coskl

                    # Print summary
                    first_l, last_l = TRACK_LAYERS[0], TRACK_LAYERS[-1]
                    kl_first = coskl[str(first_l)]["kl"]["mean"]
                    kl_last = coskl[str(last_l)]["kl"]["mean"]
                    ratio = kl_last / kl_first if kl_first > 0 else 0
                    print(f"    KL: L{first_l}={kl_first:.4f} -> L{last_l}={kl_last:.4f} (ratio={ratio:.4f})")

                # Phase 3: Accuracy
                print(f"    Phase 3: Accuracy eval...")
                acc_results = run_accuracy(pipeline, args.n_prompts)
                method_results["accuracy"] = acc_results
                print(f"    Accuracy: {acc_results['accuracy']:.4f} "
                      f"({acc_results['n_correct']}/{acc_results['n_total']}) "
                      f"PPL={acc_results['perplexity']:.2f} "
                      f"Rep={acc_results['repetition_rate']:.4f}")

                # Baseline (no ablation, no steering)
                print(f"    Baseline (no steering)...")
                pipeline_clean, steer_clean, _, _ = setup_pipeline(
                    config_path, args.device, model=model,
                    coeff_override=0.0,
                    n_prompts_override=args.n_prompts,
                )
                baseline = run_accuracy(pipeline_clean, args.n_prompts)
                method_results["baseline"] = baseline
                print(f"    Baseline accuracy: {baseline['accuracy']:.4f}")

                task_results[method] = method_results

                # Save incrementally
                out_file = OUTPUT_DIR / f"ablation_{task}_{method}_c{int(args.coeff)}.json"
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with open(out_file, "w") as f:
                    json.dump(method_results, f, indent=2, default=str)
                print(f"    Saved to {out_file}")

            except Exception as e:
                print(f"    ERROR in {method}/{task}: {e}")
                import traceback; traceback.print_exc()
            finally:
                try:
                    model.reset_hooks()
                except Exception:
                    pass
                try:
                    del pipeline, steer, coeff_dict
                except Exception:
                    pass
                try:
                    del pipeline_clean, steer_clean, baseline
                except Exception:
                    pass
                gc.collect(); torch.cuda.empty_cache(); time.sleep(1)

        all_results[task] = task_results

    # Final save
    out_file = OUTPUT_DIR / f"ablation_all.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*80}")
    print(f"  Saved to {out_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
