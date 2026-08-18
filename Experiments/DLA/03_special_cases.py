"""
DLA 03 — Special Cases: REPS + WEIGHTSTEER

Compare DLA profiles: does REPS avoid triggering suppressors?
Does WEIGHTSTEER bypass them by modifying weights?

Both use SteeringPipeline. WEIGHTSTEER uses _apply_weight_modifications.
REPS uses setup_hooks (learned intervention, not weight mod).

Usage:
    conda activate sae_circuit; unset CUDA_VISIBLE_DEVICES
    python Experiments/DLA/03_special_cases.py --task toxic
    python Experiments/DLA/03_special_cases.py --all_tasks
    python Experiments/DLA/03_special_cases.py --task toxic --smoke
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
    p.add_argument("--coeff", type=float, default=None)
    p.add_argument("--n_prompts", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# ── Pipeline (same as 02) ──

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
    is_weight = isinstance(steer, WeightSteerModel)
    captured = {}

    def make_hook(layer):
        def fn(value, hook):
            captured[layer] = value[0, -1].detach().cpu()
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


def aggregate_dla(model, steer, coeff_dict, prompts, search_layers, target_ids, normal_ids):
    """Average DLA across prompts."""
    dla_accum = defaultdict(lambda: defaultdict(list))
    for prompt in tqdm(prompts, desc="DLA", leave=False):
        dla = compute_dla(model, steer, coeff_dict, model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device),
                          search_layers, target_ids, normal_ids)
        for l in dla:
            for h, v in enumerate(dla[l]):
                dla_accum[l][h].append(v)

    dla_avg = {}
    for l in search_layers:
        dla_avg[l] = {h: float(np.mean(dla_accum[l][h])) for h in dla_accum[l]}
    return dla_avg


def sort_heads(dla_avg):
    all_heads = []
    for l in dla_avg:
        for h, v in dla_avg[l].items():
            all_heads.append((l, h, v))
    all_heads.sort(key=lambda x: x[2])
    return all_heads


# ── Compare two methods ──

def compare_methods(model, prompts, config, device, method_a, method_b, task):
    """
    Compare DLA profiles of method_a vs method_b.
    Uses method_a's target tokens for both (fair comparison).
    """
    search = config["search_layers"]
    n_top = config["n_tokens_dla"]
    coeff = config["coeff"]

    # Setup method_a
    a_info = config["method_config_map"][method_a]
    a_path = f"{a_info['cfg_dir']}/gemma_{task}.json"
    _, steer_a, _, coeff_a = setup_pipeline(a_path, device, coeff_override=coeff,
                                             n_prompts_override=config["n_prompts"])

    # Setup method_b
    b_info = config["method_config_map"][method_b]
    b_path = f"{b_info['cfg_dir']}/gemma_{task}.json"
    _, steer_b, _, coeff_b = setup_pipeline(b_path, device, coeff_override=coeff,
                                             n_prompts_override=config["n_prompts"])

    # Target tokens from method_a's steered forward (first prompt)
    target_ids, normal_ids = get_target_tokens_wang(model, steer_a, coeff_a, prompts[0], n_top)
    print(f"      Target tokens ({method_a}): {model.tokenizer.decode(target_ids)}")
    print(f"      Normal tokens ({method_a}): {model.tokenizer.decode(normal_ids)}")

    # DLA for both methods
    print(f"      DLA for {method_a}...")
    dla_a = aggregate_dla(model, steer_a, coeff_a, prompts, search, target_ids, normal_ids)
    heads_a = sort_heads(dla_a)

    print(f"      DLA for {method_b}...")
    dla_b = aggregate_dla(model, steer_b, coeff_b, prompts, search, target_ids, normal_ids)
    heads_b = sort_heads(dla_b)

    # Overlap analysis
    sup_a = set((l, h) for l, h, _ in heads_a[:15])
    sup_b = set((l, h) for l, h, _ in heads_b[:15])
    overlap = sup_a & sup_b

    # DLA comparison for overlapping heads
    dla_diff = {}
    for l, h in overlap:
        dla_diff[f"L{l}H{h}"] = {
            f"{method_a}_dla": dla_a[l][h],
            f"{method_b}_dla": dla_b[l][h],
            "diff": dla_b[l][h] - dla_a[l][h],
        }

    # Interpretation
    if len(overlap) < 5:
        interp = f"{method_b} avoids {method_a}'s suppressors"
    elif len(overlap) > 10:
        interp = f"{method_b} triggers same suppressors as {method_a}"
    else:
        interp = f"Partial overlap between {method_a} and {method_b}"

    # Cleanup
    del steer_a, coeff_a, steer_b, coeff_b
    gc.collect(); torch.cuda.empty_cache()

    return {
        f"{method_a}_top15": [(l, h, dla_a[l][h]) for l, h, _ in heads_a[:15]],
        f"{method_b}_top15": [(l, h, dla_b[l][h]) for l, h, _ in heads_b[:15]],
        "overlap_count": len(overlap),
        "overlap_heads": [(l, h) for l, h in sorted(overlap)],
        "dla_comparison": dla_diff,
        "interpretation": interp,
    }


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

    tasks = [args.task] if args.task else (exp_cfg["tasks"] if args.all_tasks else ["toxic"])

    print(f"{'='*80}")
    print(f"  DLA SPECIAL CASES: REPS + WEIGHTSTEER (Wang method)")
    print(f"  Tasks: {tasks}")
    print(f"  Search=L{exp_cfg['search_layers'][0]}-L{exp_cfg['search_layers'][-1]}  "
          f"coeff={exp_cfg['coeff']}  prompts={exp_cfg['n_prompts']}")
    print(f"{'='*80}\n")

    t0 = time.time()
    all_results = {}
    model = None

    for task in tasks:
        try:
            print(f"\n{'='*60}")
            print(f"  TASK: {task.upper()}")
            print(f"{'='*60}")

            # Load prompts (via CAA config — any config works)
            caa_info = exp_cfg["method_config_map"]["CAA"]
            caa_path = f"{caa_info['cfg_dir']}/gemma_{task}.json"
            cfg_tmp = PipelineConfig.load(caa_path)
            cfg_tmp.n_test = exp_cfg["n_prompts"]
            from Steering.data.loader import EvalDataLoader
            test_data = EvalDataLoader().load(cfg_tmp.test_dataset, n_samples=exp_cfg["n_prompts"])
            prompts = [str(s.get("question", s)) for s in test_data]
            print(f"  Loaded {len(prompts)} prompts")

            # CAA baseline DLA
            print(f"\n  --- CAA Baseline DLA ---")
            caa_cfg_path = f"{caa_info['cfg_dir']}/gemma_{task}.json"
            _, steer_caa, model, coeff_caa = setup_pipeline(
                caa_cfg_path, args.device, coeff_override=exp_cfg["coeff"],
                n_prompts_override=exp_cfg["n_prompts"],
            )

            target_ids, normal_ids = get_target_tokens_wang(
                model, steer_caa, coeff_caa, prompts[0], exp_cfg["n_tokens_dla"]
            )
            print(f"  CAA target tokens: {model.tokenizer.decode(target_ids)}")

            dla_caa = aggregate_dla(model, steer_caa, coeff_caa, prompts,
                                    exp_cfg["search_layers"], target_ids, normal_ids)
            heads_caa = sort_heads(dla_caa)
            print(f"  CAA top 5 suppressors:")
            for r, (l, h, v) in enumerate(heads_caa[:5]):
                print(f"    {r+1}. L{l}H{h}: {v:.6f}")

            del steer_caa, coeff_caa
            gc.collect(); torch.cuda.empty_cache(); time.sleep(1)

            # REPS comparison
            print(f"\n  --- REPS vs CAA ---")
            reps_result = compare_methods(
                model, prompts, exp_cfg, args.device, "CAA", "REPS", task
            )
            print(f"  Overlap: {reps_result['overlap_count']}/15 heads")
            print(f"  {reps_result['interpretation']}")
            if reps_result["dla_comparison"]:
                print(f"  DLA comparison (overlapping heads):")
                for k, d in list(reps_result["dla_comparison"].items())[:5]:
                    print(f"    {k}: CAA={d['CAA_dla']:.4f}  REPS={d['REPS_dla']:.4f}  "
                          f"diff={d['diff']:+.4f}")

            # WEIGHTSTEER comparison
            print(f"\n  --- WEIGHTSTEER vs CAA ---")
            ws_result = compare_methods(
                model, prompts, exp_cfg, args.device, "CAA", "WEIGHTSTEER", task
            )
            print(f"  Overlap: {ws_result['overlap_count']}/15 heads")
            print(f"  {ws_result['interpretation']}")
            if ws_result["dla_comparison"]:
                print(f"  DLA comparison (overlapping heads):")
                for k, d in list(ws_result["dla_comparison"].items())[:5]:
                    print(f"    {k}: CAA={d['CAA_dla']:.4f}  WS={d['WEIGHTSTEER_dla']:.4f}  "
                          f"diff={d['diff']:+.4f}")

            all_results[task] = {
                "caa_top15": [(l, h, dla_caa[l][h]) for l, h, _ in heads_caa[:15]],
                "reps": reps_result,
                "weightsteer": ws_result,
            }

        except Exception as e:
            print(f"  ERROR in {task}: {e}")
            import traceback; traceback.print_exc()
        finally:
            try: del prompts, reps_result, ws_result, heads_caa, dla_caa
            except: pass
            try: model.reset_hooks()
            except: pass
            gc.collect(); torch.cuda.empty_cache()

    if model is not None:
        model.reset_hooks()
        del model
        gc.collect(); torch.cuda.empty_cache()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"special_cases_{'_'.join(tasks)}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Saved: {out}")
    print(f"  Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
