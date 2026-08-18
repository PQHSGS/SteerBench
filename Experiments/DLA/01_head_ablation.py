"""
DLA Head Ablation — Wang's Approach (SteeringPipeline)

Step 1: Target tokens = steered forward pass top-K (Wang's method)
Step 2: Per-head attribution via hook_result @ W_U
Step 3: Net DLA = mean(attribution[target]) - mean(attribution[safe])
Step 4: Ablate top-K suppressors → measure accuracy + KL change

CAA-only. Uses SteeringPipeline. Prints all tokens for validation.

Usage:
    conda activate sae_circuit; unset CUDA_VISIBLE_DEVICES
    python Experiments/DLA/01_head_ablation.py --task toxic
    python Experiments/DLA/01_head_ablation.py --task toxic --smoke
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

torch.set_grad_enabled(False)

CONFIG_FILE = Path(__file__).parent / "config.json"
OUTPUT_DIR = Path(__file__).parent

# ── ICL prompts for DLA target token generation (Wang's format) ──
# Used for Steps 1-3 (target identification + head ranking).
# Steered generation on these simple prompts reveals what the steering
# vector activates in the model's vocabulary, regardless of task.
ICL_PROMPTS_RAW = []  # not used — DLA uses task-specific prompts like Wang


def format_icl_prompts(prompts_raw, tokenizer):
    """Not used — DLA uses task-specific prompts."""
    return prompts_raw


ICL_PROMPTS = None


def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["toxic", "evil", "deception"])
    p.add_argument("--method", default="CAA", help="Method name (CAA, LinearAcT, CHARS, etc.)")
    p.add_argument("--coeff", type=float, default=None)
    p.add_argument("--n_prompts", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# ── Pipeline ──

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
    pipeline.model.cfg.use_attn_result = True

    coeff_dict = pipeline.steer_config.coeff
    return pipeline, pipeline.steer_model, pipeline.model, coeff_dict


# ── Step 1: Target tokens from steered forward ──

def get_target_tokens(model, steer, coeff_dict, prompt, n_top=10, n_gen=20):
    """
    Wang's approach adapted for chat models:
    1. Generate n_gen tokens under STEERED (coeff=5, hardcoded) → target set.
    2. Unsteered top-K from last-position logits → normal set.
    DLA = attribution(target) - attribution(normal).
    """
    tokens = model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device)

    # Steered generation with HIGHER coeff to get clearer directional tokens
    high_coeff = {k: 5.0 for k in coeff_dict}
    model.reset_hooks()
    steer.setup_hooks(high_coeff)
    try:
        out = model.generate(
            tokens, max_new_tokens=n_gen, temperature=0.0, do_sample=False, verbose=False,
        )
    finally:
        model.reset_hooks()

    gen_ids = out[0, tokens.shape[1]:].tolist()
    gen_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Unsteered forward → top-K = normal tokens (what model WOULD say)
    model.reset_hooks()
    try:
        logits_un = model.run_with_hooks(tokens, return_type="logits")
    finally:
        model.reset_hooks()

    last_logits = logits_un[0, -1, :].float()
    topk_vals, topk_ids = last_logits.topk(n_top)
    normal_ids = topk_ids.tolist()  # what model naturally says = normal set

    return gen_ids, gen_text, normal_ids


# ── Step 2-3: Per-head DLA ──

def capture_heads(model, steer, coeff_dict, tokens, search_layers):
    """Capture attn.hook_result at search_layers under CAA steering."""
    captured = {}

    def make_hook(layer):
        def hook_fn(value, hook):
            captured[layer] = value[0, -1].detach().cpu()  # [n_heads, d_model]
            return value
        return hook_fn

    hooks = [(f"blocks.{l}.attn.hook_result", make_hook(l)) for l in search_layers]
    model.reset_hooks()
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
        ho = head_outs[l].to(W_U.device)          # [n_heads, d_model]
        attr = (ho @ W_U).float()                  # [n_heads, vocab_size]
        target_dla = attr[:, target_ids].mean(-1)   # [n_heads]
        safe_dla = attr[:, normal_ids].mean(-1)     # [n_heads]
        net = target_dla - safe_dla
        dla_per_head[l] = net.tolist()

    return dla_per_head


# ── Step 4: Ablation ──

def ablation_hook(head_idx):
    def hook(result, hook):
        result[0, -1, head_idx, :] = 0.0
        return result
    return hook


def forward_with_ablation(model, steer, coeff_dict, tokens, ablation_list):
    model.reset_hooks()
    steer.setup_hooks(coeff_dict)
    for li, hi in ablation_list:
        model.add_hook(f"blocks.{li}.attn.hook_z", ablation_hook(hi))
    try:
        logits = model.run_with_hooks(tokens, return_type="logits")
    finally:
        model.reset_hooks()
    return logits


def generate_and_eval(model, steer, coeff_dict, prompt, ablation_list,
                      max_new_tokens, evaluator=None):
    tokens = model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device)
    generated = tokens.clone()
    for _ in range(max_new_tokens):
        logits = forward_with_ablation(model, steer, coeff_dict, generated, ablation_list)
        next_tok = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        generated = torch.cat([generated, next_tok], dim=1)
    gen_ids = generated[0, tokens.shape[1]:].tolist()
    gen_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True)
    acc = evaluator.check(gen_text)[0] if evaluator else None
    return gen_text, acc, gen_ids, tokens.shape[1]


# ── PPL + LSP ──

def compute_ppl_lsp(model, prompt_ids, response_ids, window_size=30):
    """Compute perplexity and LSP (Localized Suffix-Penalized PPL) for a response.

    prompt_ids: [1, prompt_len] tensor
    response_ids: list of int (generated token IDs)
    Returns: {"ppl": float, "lsp": float or None}
    """
    import math
    prompt_len = prompt_ids.shape[1]
    resp_tensor = torch.tensor([response_ids], device=prompt_ids.device)
    full_ids = torch.cat([prompt_ids, resp_tensor], dim=1)

    with torch.no_grad():
        loss_per_token = model(full_ids, return_type="loss", loss_per_token=True)
        response_loss = loss_per_token[:, prompt_len - 1:]
        ppl = float(torch.exp(response_loss.mean()).item())

        # LSP: per-token loss with repetition penalty
        resp_losses = response_loss[0].tolist()
        resp_token_ids = full_ids[0, prompt_len:].tolist()

        all_losses = []
        for t in range(len(resp_token_ids)):
            raw_loss = resp_losses[t]
            start_idx = max(0, t - window_size)
            history = resp_token_ids[start_idx:t]

            lcs = 0
            for idx in range(len(history) - 1):
                length = 0
                while idx - length >= 0 and history[idx - length] == resp_token_ids[t]:
                    length += 1
                if length > lcs:
                    lcs = length

            penalty = 2.0 * math.log1p(max(0, lcs - 2))
            all_losses.append(raw_loss + penalty)

        mean_loss = sum(all_losses) / len(all_losses) if all_losses else float("inf")
        try:
            lsp = math.exp(mean_loss)
        except OverflowError:
            lsp = float("inf")

    return {"ppl": ppl, "lsp": lsp}


def compute_kl_ablation(model, steer, coeff_dict, prompt, ablation_list, track_layers):
    tokens = model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device)

    # Unsteered
    resid_un = {}
    def un_hook(layer):
        def fn(value, hook):
            resid_un[layer] = value[0, -1].detach().cpu()
            return value
        return fn
    hooks_un = [(f"blocks.{l}.hook_resid_post", un_hook(l)) for l in track_layers]
    model.reset_hooks()
    model.run_with_hooks(tokens, fwd_hooks=hooks_un, return_type="logits")

    # Steered + ablated
    resid_st = {}
    def st_hook(layer):
        def fn(value, hook):
            resid_st[layer] = value[0, -1].detach().cpu()
            return value
        return fn
    hooks_st = [(f"blocks.{l}.hook_resid_post", st_hook(l)) for l in track_layers]
    model.reset_hooks()
    steer.setup_hooks(coeff_dict)
    for li, hi in ablation_list:
        model.add_hook(f"blocks.{li}.attn.hook_z", ablation_hook(hi))
    try:
        model.run_with_hooks(tokens, fwd_hooks=hooks_st, return_type="logits")
    finally:
        model.reset_hooks()

    W_U, ln, dtype, dev = model.W_U, model.ln_final, next(model.parameters()).dtype, model.cfg.device
    kl = {}
    for l in track_layers:
        if l not in resid_un or l not in resid_st:
            continue
        lu = ln(resid_un[l].to(dev).to(dtype)) @ W_U
        ls = ln(resid_st[l].to(dev).to(dtype)) @ W_U
        kl[l] = F.kl_div(
            F.log_softmax(ls.float(), -1), F.log_softmax(lu.float(), -1),
            log_target=True, reduction='sum'
        ).item()
    return kl


# ── Main ──

def main():
    args = parse_args()
    cfg = load_config()

    if args.coeff is not None:
        cfg["coeff"] = args.coeff
    if args.n_prompts is not None:
        cfg["n_prompts"] = args.n_prompts
    if args.smoke:
        cfg["n_prompts"] = 3
        cfg["ablating_k_values"] = [3]

    inject = cfg["inject_layer"]
    search = cfg["search_layers"]
    track = list(range(inject + 1, 26))
    coeff = cfg["coeff"]
    n_top = cfg["n_tokens_dla"]
    max_tok = cfg["max_tokens"]

    print(f"{'='*70}")
    print(f"  DLA HEAD ABLATION — {args.task.upper()} / {args.method}")
    print(f"  Inject=L{inject}  Search=L{search[0]}-L{search[-1]}  coeff={coeff}")
    print(f"  Prompts={cfg['n_prompts']}  Ablate-K={cfg['ablating_k_values']}")
    print(f"{'='*70}\n")

    t0 = time.time()

    try:
        # Setup
        method_info = cfg["method_config_map"][args.method]
        config_path = f"{method_info['cfg_dir']}/gemma_{args.task}.json"
        pipeline, steer, model, coeff_dict = setup_pipeline(
            config_path, args.device, coeff_override=coeff,
            n_prompts_override=cfg["n_prompts"],
        )
        print(f"  Pipeline ready ({steer.__class__.__name__}, layers={steer.layer})")
        print(f"  Loaded in {time.time()-t0:.1f}s")

        test_data = pipeline.load_test_data(
            dataset_name=pipeline.config.test_dataset,
            n_samples=cfg["n_prompts"],
            apply_chat_template=pipeline.steer_config.apply_chat_template,
        )
        prompts = [str(s.get("question", s)) for s in test_data]

        # ── Step 1: Target tokens (Wang) ──
        print(f"\n{'─'*50}")
        print(f"  STEP 1: Target tokens from steered generation (Wang)")
        print(f"{'─'*50}")
        print(f"\n  Validating target tokens on first 3 prompts:\n")

        all_gen_ids = []
        all_normal_ids = []
        for i, prompt in enumerate(prompts[:3]):
            gen_ids, gen_text, n_ids = get_target_tokens(
                model, steer, coeff_dict, prompt, n_top
            )
            all_gen_ids.append(gen_ids)
            all_normal_ids.append(n_ids)
            print(f"  Prompt [{i}]: {prompt[:80]}...")
            print(f"  Steered output ({len(gen_ids)} tok): {repr(gen_text[:100])}")
            print(f"  Steered token IDs: {gen_ids}")
            print(f"  Normal top-{n_top}: {model.tokenizer.decode(n_ids)}")
            print()

        # Collect targets from ALL prompts
        print(f"  Now generating steered targets from all {len(prompts)} prompts...")
        all_gen_ids_full = []
        all_normal_ids_full = []
        for prompt in tqdm(prompts, desc="Collecting targets", leave=False):
            gen_ids, _, n_ids = get_target_tokens(model, steer, coeff_dict, prompt, n_top)
            all_gen_ids_full.append(gen_ids)
            all_normal_ids_full.append(n_ids)

        print(f"  Collected targets from {len(prompts)} prompts")

        # ── Step 2-3: DLA analysis ──
        print(f"\n{'─'*50}")
        print(f"  STEP 2-3: Per-head DLA across {len(prompts)} prompts")
        print(f"{'─'*50}")

        # Per-prompt DLA (using each prompt's own steered tokens)
        dla_accum = defaultdict(lambda: defaultdict(list))
        for idx, prompt in enumerate(tqdm(prompts, desc="DLA")):
            tokens = model.to_tokens(prompt, truncate=True)[:, :32].to(model.cfg.device)
            t_ids = all_gen_ids_full[idx]
            n_ids = all_normal_ids_full[idx]
            if len(t_ids) == 0:
                continue  # skip if generation produced nothing
            dla = compute_dla(model, steer, coeff_dict, tokens, search, t_ids, n_ids)
            for l in dla:
                for h, v in enumerate(dla[l]):
                    dla_accum[l][h].append(v)

        # Average per-prompt DLA across prompts
        dla_avg = {}
        for l in search:
            dla_avg[l] = {h: float(np.mean(dla_accum[l][h])) for h in dla_accum[l]}

        def print_ranking(dla_dict, top_n=15):
            all_h = []
            for l in dla_dict:
                for h, v in dla_dict[l].items():
                    all_h.append((l, h, v))
            all_h.sort(key=lambda x: x[2])

            print(f"\n  Top {top_n} SUPPRESSING heads:")
            print(f"  {'Rank':<6} {'Layer':<8} {'Head':<8} {'Net DLA':>12}")
            print(f"  {'─'*40}")
            for r, (l, h, v) in enumerate(all_h[:top_n]):
                print(f"  {r+1:<6} L{l:<7} H{h:<7} {v:>12.6f}")

            print(f"\n  Top 5 PROMOTING heads:")
            print(f"  {'Rank':<6} {'Layer':<8} {'Head':<8} {'Net DLA':>12}")
            print(f"  {'─'*40}")
            for r, (l, h, v) in enumerate(reversed(all_h[-5:])):
                print(f"  {r+1:<6} L{l:<7} H{h:<7} {v:>12.6f}")

        print_ranking(dla_avg)

        # Rank heads by per-prompt DLA (most negative = suppressors)
        all_heads = []
        for l in search:
            for h, v in dla_avg[l].items():
                all_heads.append((l, h, v))
        all_heads.sort(key=lambda x: x[2])

        # ── Step 4: Ablation sweep ──
        print(f"\n{'─'*50}")
        print(f"  STEP 4: Ablation sweep")
        print(f"{'─'*50}")

        from Steering.evaluators import EVALUATOR_MAP
        evaluator = EVALUATOR_MAP[args.task](device=args.device)

        # Baseline
        print(f"\n  Baseline (no ablation)...")
        base_accs = []
        base_ppls = []
        base_lsps = []
        base_gen_texts = []
        for p in tqdm(prompts, desc="Baseline", leave=False):
            gen_text, a, gen_ids, prompt_len = generate_and_eval(
                model, steer, coeff_dict, p, [], max_tok, evaluator
            )
            base_accs.append(int(a) if a else 0)
            base_gen_texts.append(gen_text)
            prompt_toks = model.to_tokens(p, truncate=True)[:, :32].to(model.cfg.device)
            metrics = compute_ppl_lsp(model, prompt_toks, gen_ids)
            base_ppls.append(metrics["ppl"])
            base_lsps.append(metrics["lsp"])
        base_acc = float(np.mean(base_accs))
        base_ppl = float(np.mean([p for p in base_ppls if p != float("inf")])) if base_ppls else float("inf")
        base_lsp = float(np.mean([l for l in base_lsps if l is not None and l != float("inf")])) if base_lsps else None
        # OOD thresholds
        ood_thresholds = {"toxic": 20, "evil": 10, "deception": 25}
        ood_thresh = ood_thresholds.get(args.task, 20)
        base_ood = (base_lsp is not None and base_lsp > ood_thresh) if base_lsp is not None else False
        print(f"  Baseline accuracy: {base_acc:.1%}  PPL: {base_ppl:.2f}  LSP: {base_lsp:.2f}  OOD: {base_ood}")

        # Baseline KL
        base_kl = defaultdict(list)
        for p in tqdm(prompts[:10], desc="Baseline KL", leave=False):
            kl = compute_kl_ablation(model, steer, coeff_dict, p, [], track)
            for l in kl:
                base_kl[l].append(kl[l])
        base_kl_mean = {l: float(np.mean(base_kl[l])) for l in base_kl}
        print(f"  Baseline KL@L{inject+1}={base_kl_mean.get(inject+1,0):.4f}  KL@L25={base_kl_mean.get(25,0):.4f}")

        # Sweep
        sweep = {}
        for k in cfg["ablating_k_values"]:
            supps = [(l, h) for l, h, _ in all_heads[:k]]
            print(f"\n  --- Ablate top-{k} suppressors: {[(f'L{l}H{h}') for l,h in supps]} ---")

            abl_accs = []
            abl_ppls = []
            abl_lsps = []
            abl_gen_texts = []
            for p in tqdm(prompts, desc=f"K={k}", leave=False):
                gen_text, a, gen_ids, prompt_len = generate_and_eval(
                    model, steer, coeff_dict, p, supps, max_tok, evaluator
                )
                abl_accs.append(int(a) if a else 0)
                abl_gen_texts.append(gen_text)
                prompt_toks = model.to_tokens(p, truncate=True)[:, :32].to(model.cfg.device)
                metrics = compute_ppl_lsp(model, prompt_toks, gen_ids)
                abl_ppls.append(metrics["ppl"])
                abl_lsps.append(metrics["lsp"])
            abl_acc = float(np.mean(abl_accs))
            abl_ppl = float(np.mean([p for p in abl_ppls if p != float("inf")])) if abl_ppls else float("inf")
            abl_lsp = float(np.mean([l for l in abl_lsps if l is not None and l != float("inf")])) if abl_lsps else None
            abl_ood = (abl_lsp is not None and abl_lsp > ood_thresh) if abl_lsp is not None else False
            delta = abl_acc - base_acc
            print(f"    Accuracy: {abl_acc:.1%} (delta={delta:+.1%})  PPL: {abl_ppl:.2f}  LSP: {abl_lsp:.2f}  OOD: {abl_ood}")

            abl_kl = defaultdict(list)
            for p in tqdm(prompts[:10], desc=f"K={k} KL", leave=False):
                kl = compute_kl_ablation(model, steer, coeff_dict, p, supps, track)
                for l in kl:
                    abl_kl[l].append(kl[l])
            abl_kl_mean = {l: float(np.mean(abl_kl[l])) for l in abl_kl}
            kl15 = abl_kl_mean.get(inject+1, 0)
            kl25 = abl_kl_mean.get(25, 0)
            bkl15 = base_kl_mean.get(inject+1, 0)
            bkl25 = base_kl_mean.get(25, 0)
            ratio15 = kl15 / bkl15 if bkl15 > 0 else 0
            ratio25 = kl25 / bkl25 if bkl25 > 0 else 0
            print(f"    KL@L{inject+1}: {kl15:.4f} [ratio: {ratio15:.2f}x]  KL@L25: {kl25:.4f} [ratio: {ratio25:.2f}x]")

            sweep[str(k)] = {
                "accuracy": abl_acc, "delta": delta,
                "ppl": abl_ppl, "lsp": abl_lsp, "ood": abl_ood,
                "kl_mean": {str(l): v for l, v in abl_kl_mean.items()},
                "suppressors": [(l, h) for l, h in supps],
                "samples": [
                    {"prompt": prompts[i][:120], "response": abl_gen_texts[i][:200],
                     "correct": bool(abl_accs[i]), "ppl": abl_ppls[i], "lsp": abl_lsps[i]}
                    for i in range(len(prompts))
                ],
            }

            # ── Incremental save after each K ──
            results = {
                "config": {
                    "task": args.task, "method": args.method,
                    "inject_layer": inject, "search_layers": search,
                    "coeff": coeff, "n_prompts": len(prompts),
                    "n_top": n_top, "target_method": "wang_generation",
                    "ood_threshold": ood_thresh,
                },
                "dla_top15_suppressors": [(l, h, dla_avg[l][h]) for l, h, _ in all_heads[:15]],
                "dla_top5_promoters": [(l, h, dla_avg[l][h]) for l, h, _ in all_heads[-5:]],
                "baseline": {
                    "accuracy": base_acc, "ppl": base_ppl, "lsp": base_lsp, "ood": base_ood,
                    "kl_mean": {str(l): v for l, v in base_kl_mean.items()},
                },
                "ablation_sweep": sweep,
                "baseline_samples": [
                    {"prompt": prompts[i][:120], "response": base_gen_texts[i][:200],
                     "correct": bool(base_accs[i]), "ppl": base_ppls[i], "lsp": base_lsps[i]}
                    for i in range(min(5, len(prompts)))
                ],
            }
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OUTPUT_DIR / f"dla_ablation_{args.task}_{args.method}.json"
            with open(out, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"    Saved → {out.name}")

            gc.collect(); torch.cuda.empty_cache()

        print(f"\n{'='*70}")
        print(f"  SUMMARY — {args.task.upper()}")
        print(f"{'='*70}")
        print(f"  Baseline: {base_acc:.1%}  PPL={base_ppl:.2f}  LSP={base_lsp:.2f}  OOD={base_ood}")
        for k, d in sweep.items():
            print(f"  Ablate-{k}: {d['accuracy']:.1%} (delta={d['delta']:+.1%})  PPL={d['ppl']:.2f}  LSP={d['lsp']:.2f}  OOD={d['ood']}")
        print(f"  Saved: {out}")
        print(f"  Time: {time.time()-t0:.1f}s")

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback; traceback.print_exc()
    finally:
        try: model.reset_hooks()
        except: pass
        try: del pipeline, steer, model, coeff_dict
        except: pass
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
