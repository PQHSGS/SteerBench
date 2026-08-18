"""
DLA 02 — Full Experiment with Controls (SteeringPipeline)

Wang's DLA: target tokens from steered forward top-K.
Three ablation conditions per (method, task):
  1. Targeted: Ablate top-K suppressors (negative net_DLA)
  2. Random: Ablate K random heads (control)
  3. Non-suppressor: Ablate K promoter heads (positive net_DLA)

Uses SteeringPipeline for ALL methods. Weight methods use
_apply_weight_modifications / _restore_weights.

Usage:
    conda activate sae_circuit; unset CUDA_VISIBLE_DEVICES
    python Experiments/DLA/02_full_experiment.py --task toxic --method CAA
    python Experiments/DLA/02_full_experiment.py --task toxic --all_methods
    python Experiments/DLA/02_full_experiment.py --all_tasks --all_methods
    python Experiments/DLA/02_full_experiment.py --task toxic --method CAA --smoke
"""

import argparse, gc, json, time, sys
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
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=None, choices=["toxic", "evil", "deception"])
    p.add_argument("--all_tasks", action="store_true")
    p.add_argument("--method", default=None)
    p.add_argument("--all_methods", action="store_true")
    p.add_argument("--coeff", type=float, default=None)
    p.add_argument("--n_prompts", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# ── Pipeline ──

def setup_pipeline(config_path, device, model=None, coeff_override=None, n_prompts_override=None):
    config = PipelineConfig.load(config_path)
    config.model.device = device
    if coeff_override is not None:
        config.steer.coeff = coeff_override
    if n_prompts_override is not None:
        config.n_test = n_prompts_override
    if not getattr(config, "load_vector", None):
        config.load_vector = config.save_vector

    pipeline = SteeringPipeline(config)
    if model is not None:
        pipeline.model = model
        pipeline._current_model_type = "tl"
    pipeline.setup()
    pipeline.model.cfg.use_attn_result = True

    coeff_dict = pipeline.steer_config.coeff
    return pipeline, pipeline.steer_model, pipeline.model, coeff_dict


# ── Wang target tokens ──

def get_target_tokens_wang(model, steer, coeff_dict, prompt, n_top=10):
    """Steered forward → top-K logits as targets (Wang's method)."""
    tokens = model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device)
    is_weight = isinstance(steer, WeightSteerModel)

    model.reset_hooks()
    if is_weight:
        steer._apply_weight_modifications(coeff_dict)
        try:
            logits = model.run_with_hooks(tokens, return_type="logits")
        finally:
            steer._restore_weights(coeff_dict)
    else:
        steer.setup_hooks(coeff_dict)
        try:
            logits = model.run_with_hooks(tokens, return_type="logits")
        finally:
            model.reset_hooks()

    last = logits[0, -1, :].float()
    target_ids = last.topk(n_top).indices.tolist()
    normal_ids = last.topk(n_top, largest=False).indices.tolist()
    return target_ids, normal_ids


# ── DLA core ──

def capture_heads(model, steer, coeff_dict, tokens, search_layers):
    """Capture attn.hook_result at search_layers under method's steering."""
    is_weight = isinstance(steer, WeightSteerModel)
    captured = {}

    def make_hook(layer):
        def fn(value, hook):
            captured[layer] = value[0, -1].detach().cpu()  # [n_heads, d_model]
            return value
        return fn

    hooks = [(f"blocks.{l}.attn.hook_result", make_hook(l)) for l in search_layers]

    model.reset_hooks()
    if is_weight:
        steer._apply_weight_modifications(coeff_dict)
        try:
            model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")
        finally:
            steer._restore_weights(coeff_dict)
    else:
        steer.setup_hooks(coeff_dict)
        try:
            model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")
        finally:
            model.reset_hooks()

    return captured


def compute_dla(model, steer, coeff_dict, tokens, search_layers, target_ids, normal_ids):
    """Net DLA per head: mean(attribution[target]) - mean(attribution[safe])."""
    W_U = model.W_U
    head_outs = capture_heads(model, steer, coeff_dict, tokens, search_layers)

    dla_per_head = {}
    for l in search_layers:
        if l not in head_outs:
            continue
        ho = head_outs[l].to(W_U.device)
        attr = (ho @ W_U).float()
        target_dla = attr[:, target_ids].mean(-1)
        safe_dla = attr[:, normal_ids].mean(-1)
        net = target_dla - safe_dla
        dla_per_head[l] = net.tolist()
    return dla_per_head


# ── Ablation ──

def ablation_hook(head_idx):
    def fn(result, hook):
        result[0, -1, head_idx, :] = 0.0
        return result
    return fn


def forward_with_ablation(model, steer, coeff_dict, tokens, ablation_list):
    """Forward with steering + head ablation."""
    is_weight = isinstance(steer, WeightSteerModel)

    model.reset_hooks()
    if is_weight:
        steer._apply_weight_modifications(coeff_dict)
    else:
        steer.setup_hooks(coeff_dict)

    for li, hi in ablation_list:
        model.add_hook(f"blocks.{li}.attn.hook_z", ablation_hook(hi))

    try:
        logits = model.run_with_hooks(tokens, return_type="logits")
    finally:
        model.reset_hooks()
        if is_weight:
            steer._restore_weights(coeff_dict)

    return logits


def generate_and_eval(model, steer, coeff_dict, prompt, ablation_list,
                      max_new_tokens, evaluator=None):
    """Autoregressive generation with steering + ablation."""
    tokens = model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device)
    generated = tokens.clone()

    for _ in range(max_new_tokens):
        logits = forward_with_ablation(model, steer, coeff_dict, generated, ablation_list)
        next_tok = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        generated = torch.cat([generated, next_tok], dim=1)

    gen_text = model.tokenizer.decode(generated[0, tokens.shape[1]:], skip_special_tokens=True)
    acc = evaluator.check(gen_text)[0] if evaluator else None
    return gen_text, acc


# ── Main ──

def main():
    args = parse_args()
    exp_cfg = load_config()

    if args.coeff is not None:
        exp_cfg["coeff"] = args.coeff
    if args.n_prompts is not None:
        exp_cfg["n_prompts"] = args.n_prompts
    if args.smoke:
        exp_cfg["n_prompts"] = 3
        exp_cfg["ablating_k_values"] = [3]

    tasks = [args.task] if args.task else (exp_cfg["tasks"] if args.all_tasks else ["toxic"])
    methods = [args.method.upper()] if args.method else []
    if args.all_methods:
        methods = exp_cfg["methods"]
    if not methods:
        methods = ["CAA"]

    search = exp_cfg["search_layers"]
    inject = exp_cfg["inject_layer"]
    coeff = exp_cfg["coeff"]
    n_top = exp_cfg["n_tokens_dla"]
    max_tok = exp_cfg["max_tokens"]
    k_values = exp_cfg["ablating_k_values"]

    print(f"{'='*80}")
    print(f"  DLA FULL EXPERIMENT — Wang method (SteeringPipeline)")
    print(f"  Tasks: {tasks}  Methods: {methods}")
    print(f"  Search=L{search[0]}-L{search[-1]}  coeff={coeff}  prompts={exp_cfg['n_prompts']}")
    print(f"  Ablate-K={k_values}  Conditions: targeted, random, non-suppressor")
    print(f"{'='*80}\n")

    t0 = time.time()
    rng = np.random.RandomState(42)
    all_results = {}
    model = None

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*60}")

        from Steering.evaluators import EVALUATOR_MAP
        evaluator = EVALUATOR_MAP[task](device=args.device)
        task_results = {}

        for method in methods:
            try:
                print(f"\n  --- {method} ---")

                method_info = exp_cfg.get("method_config_map", {}).get(method, {})
                cfg_dir = method_info.get("cfg_dir", "")
                config_path = f"{cfg_dir}/gemma_{task}.json"

                if not Path(config_path).exists():
                    print(f"    No config for {method}/{task}, skipping")
                    continue

                # Setup pipeline
                pipeline, steer, model, coeff_dict = setup_pipeline(
                    config_path, args.device, model=model,
                    coeff_override=coeff,
                    n_prompts_override=exp_cfg["n_prompts"],
                )
                print(f"    Pipeline ready ({steer.__class__.__name__}, "
                      f"layers={steer.layer}, coeff={coeff_dict})")

                # Load prompts
                test_data = pipeline.load_test_data(
                    dataset_name=pipeline.config.test_dataset,
                    n_samples=exp_cfg["n_prompts"],
                    apply_chat_template=pipeline.steer_config.apply_chat_template,
                )
                prompts = [str(s.get("question", s)) for s in test_data]

                # ── Phase 1: Wang target tokens + DLA ──
                print(f"    Phase 1: Target tokens (Wang) + DLA across {len(prompts)} prompts")

                # Collect per-prompt target/normal tokens
                all_target_ids = []
                all_normal_ids = []
                for prompt in tqdm(prompts, desc="    Targets", leave=False):
                    t_ids, n_ids = get_target_tokens_wang(model, steer, coeff_dict, prompt, n_top)
                    all_target_ids.append(t_ids)
                    all_normal_ids.append(n_ids)

                # Show first prompt's targets for validation
                t0_ids, n0_ids = all_target_ids[0], all_normal_ids[0]
                print(f"    Validation — first prompt targets: {model.tokenizer.decode(t0_ids)}")
                print(f"    Validation — first prompt normals: {model.tokenizer.decode(n0_ids)}")

                # DLA across prompts (per-prompt targets)
                dla_accum = defaultdict(lambda: defaultdict(list))
                for idx, prompt in enumerate(tqdm(prompts, desc="    DLA", leave=False)):
                    tokens = model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device)
                    dla = compute_dla(model, steer, coeff_dict, tokens, search,
                                      all_target_ids[idx], all_normal_ids[idx])
                    for l in dla:
                        for h, v in enumerate(dla[l]):
                            dla_accum[l][h].append(v)

                dla_avg = {}
                for l in search:
                    dla_avg[l] = {h: float(np.mean(dla_accum[l][h]))
                                  for h in dla_accum[l]}

                # Rank heads
                all_heads = []
                for l in search:
                    for h, v in dla_avg[l].items():
                        all_heads.append((l, h, v))
                all_heads.sort(key=lambda x: x[2])

                print(f"    Top 5 suppressors: {[(f'L{l}H{h}: {v:.4f}') for l,h,v in all_heads[:5]]}")
                print(f"    Top 5 promoters:   {[(f'L{l}H{h}: {v:.4f}') for l,h,v in all_heads[-5:]]}")

                # ── Phase 2: Ablation sweep ──
                print(f"    Phase 2: Ablation sweep")

                # Baseline
                base_accs = []
                for prompt in tqdm(prompts, desc="    Baseline", leave=False):
                    _, a = generate_and_eval(model, steer, coeff_dict, prompt, [], max_tok, evaluator)
                    base_accs.append(int(a) if a else 0)
                base_acc = float(np.mean(base_accs))
                print(f"    Baseline accuracy: {base_acc:.1%}")

                results = {
                    "dla_top15_suppressors": [(l, h, dla_avg[l][h]) for l, h, _ in all_heads[:15]],
                    "dla_top5_promoters": [(l, h, dla_avg[l][h]) for l, h, _ in all_heads[-5:]],
                    "baseline_accuracy": base_acc,
                    "target_token_ids": all_target_ids[0],
                    "normal_token_ids": all_normal_ids[0],
                    "sweep": {},
                }

                for k in k_values:
                    targeted = all_heads[:k]  # most negative first
                    all_indices = list(range(len(all_heads)))
                    rand_idx = rng.choice(all_indices, size=min(k, len(all_indices)), replace=False)
                    random_heads = [all_heads[i] for i in rand_idx]
                    non_sup = all_heads[-k:]  # most positive first

                    conditions = {
                        "targeted": targeted,
                        "random": random_heads,
                        "non_suppressor": non_sup,
                    }

                    sweep_k = {}
                    for cond_name, heads in conditions.items():
                        abl_list = [(l, h) for l, h, _ in heads]
                        accs = []
                        for prompt in tqdm(prompts, desc=f"    K={k} {cond_name}", leave=False):
                            _, a = generate_and_eval(model, steer, coeff_dict, prompt,
                                                     abl_list, max_tok, evaluator)
                            accs.append(int(a) if a else 0)
                        acc = float(np.mean(accs))
                        sweep_k[cond_name] = {
                            "accuracy": acc, "delta": acc - base_acc,
                            "heads": [(l, h) for l, h, _ in heads],
                        }

                    results["sweep"][str(k)] = sweep_k

                    t_d = sweep_k["targeted"]
                    r_d = sweep_k["random"]
                    n_d = sweep_k["non_suppressor"]
                    print(f"    K={k}: targeted={t_d['accuracy']:.1%}({t_d['delta']:+.1%}) "
                          f"random={r_d['accuracy']:.1%}({r_d['delta']:+.1%}) "
                          f"non-sup={n_d['accuracy']:.1%}({n_d['delta']:+.1%})")

                    gc.collect(); torch.cuda.empty_cache()

                task_results[method] = results

            except Exception as e:
                print(f"    ERROR in {method}/{task}: {e}")
                import traceback; traceback.print_exc()
            finally:
                try: model.reset_hooks()
                except: pass
                try: del pipeline, steer, coeff_dict, results, prompts
                except: pass
                gc.collect(); torch.cuda.empty_cache(); time.sleep(1)

        all_results[task] = task_results

    if model is not None:
        model.reset_hooks()
        del model
        gc.collect(); torch.cuda.empty_cache()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"full_experiment_{'_'.join(tasks)}_{'_'.join(methods)}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Cross-task table
    if len(tasks) > 1 or len(methods) > 1:
        print(f"\n{'='*80}")
        print(f"  ACCURACY TABLE: tasks x methods (baseline -> best ablated)")
        print(f"{'='*80}")
        header = f"  {'':20}"
        for method in methods:
            header += f" {method:>16}"
        print(header)
        print(f"  {'-'*80}")
        for task in tasks:
            row = f"  {task:>20}"
            for method in methods:
                r = all_results.get(task, {}).get(method)
                if r is None:
                    row += f" {'N/A':>16}"
                    continue
                base = r["baseline_accuracy"]
                best = base
                for k, d in r["sweep"].items():
                    if d["targeted"]["accuracy"] > best:
                        best = d["targeted"]["accuracy"]
                row += f" {base:.0%}->{best:.0%}"
            print(row)

    print(f"\n  Saved: {out}")
    print(f"  Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
