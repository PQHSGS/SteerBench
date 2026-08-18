"""
Experiment D (Logit Lens KL Divergence on Generated Tokens):
Measures KL(steered || baseline) across layers (L8-L25) for 1st, 3rd, and 10th generated tokens of the sequence.
For each prompt, we generate 10 tokens under steering, and then evaluate the KL divergence
at generation steps 0 (1st generated), 2 (3rd generated), and 9 (10th generated) by feeding the
corresponding context prefix to both the steered and baseline models.

Usage:
    export HF_HOME="/data/caotue/hf_cache"
    export HUGGINGFACE_HUB_CACHE="/data/caotue/hf_cache/hub"
    export TRANSFORMERS_CACHE="/data/caotue/hf_cache/transformers"
    export HF_DATASETS_CACHE="/data/caotue/hf_cache/datasets"
    export TORCH_HOME="/data/caotue/torch_cache"
    export TMPDIR="/data/caotue/tmp"
    conda activate sae_circuit
    CUDA_VISIBLE_DEVICES=2 python Experiments/ExpD/expD_logit_lens.py --task evil --n_prompts 100
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformer_lens import HookedTransformer

sys.path.insert(0, ".")
from Steering.config.pipeline import PipelineConfig
from Steering.pipeline import SteeringPipeline
from Steering.steer_models.weight import WeightSteerModel

# Disable gradients globally
torch.set_grad_enabled(False)

CONFIG_PATHS = {
    "evil": {
        "CAA": "Configs/Eval/CAA/Gemma/gemma_evil.json",
        "CHARS": "Configs/Eval/CHARS/Gemma/gemma_evil.json",
        "ACT": "Configs/Eval/LinearAcT/Gemma/gemma_evil.json",
        "REPS": "Configs/Eval/REFT/gemma_evil.json",
        "FLAS": "Configs/Eval/FLAS/gemma_evil.json",
        "PID": "Configs/Eval/PID/Gemma/gemma_evil.json",
        "WeightSteer": "Configs/Eval/WEIGHTSTEER/gemma_evil.json",
    },
    "deception": {
        "CAA": "Configs/Eval/CAA/Gemma/gemma_deception.json",
        "CHARS": "Configs/Eval/CHARS/Gemma/gemma_deception.json",
        "ACT": "Configs/Eval/LinearAcT/Gemma/gemma_deception.json",
        "REPS": "Configs/Eval/REFT/gemma_deception.json",
        "FLAS": "Configs/Eval/FLAS/gemma_deception.json",
        "PID": "Configs/Eval/PID/Gemma/gemma_deception.json",
        "WeightSteer": "Configs/Eval/WEIGHTSTEER/gemma_deception.json",
    },
    "toxic": {
        "CAA": "Configs/Eval/CAA/Gemma/gemma_toxic.json",
        "CHARS": "Configs/Eval/CHARS/Gemma/gemma_toxic.json",
        "ACT": "Configs/Eval/LinearAcT/Gemma/gemma_toxic.json",
        "REPS": "Configs/Eval/REFT/gemma_toxic.json",
        "FLAS": "Configs/Eval/FLAS/gemma_toxic.json",
        "PID": "Configs/Eval/PID/Gemma/gemma_toxic.json",
        "WeightSteer": "Configs/Eval/WEIGHTSTEER/gemma_toxic.json",
    },
    "refusal": {
        "CAA": "Configs/Eval/CAA/Gemma/gemma_refusal_response.json",
        "CHARS": "Configs/Eval/CHARS/Gemma/gemma_refusal_response.json",
        "ACT": "Configs/Eval/LinearAcT/Gemma/gemma_refusal_response.json",
        "REPS": "Configs/Eval/REFT/reps_refusal_response.json",
        "FLAS": "Configs/Eval/FLAS/flas_refusal_response.json",
        "PID": "Configs/Eval/PID/Gemma/gemma_refusal_response.json",
        "WeightSteer": "Configs/Eval/WEIGHTSTEER/gemma_refusal_response.json",
    }
}

ALPACA_DATA_PATH = "TestDataset/alpaca/alpaca.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Logit Lens KL Divergence on Generated Tokens")
    parser.add_argument("--task", type=str, required=True, choices=["refusal", "deception", "evil", "toxic"], help="Evaluation task")
    parser.add_argument("--coeff", type=float, default=1.0, help="Steering coefficient")
    parser.add_argument("--n_prompts", type=int, default=100, help="Number of prompts to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    return parser.parse_args()

def load_alpaca_prompts(json_path, n):
    """Load n prompts from Alpaca json dataset."""
    if not os.path.exists(json_path):
        print(f"Error: dataset not found at {json_path}")
        sys.exit(1)
    with open(json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    
    prompts = []
    for item in data:
        q = item.get("question")
        if q:
            prompts.append(q)
        if len(prompts) >= n:
            break
    return prompts

def make_capture_hook_tl(layer_idx, captured_dict):
    def hook_fn(value, hook):
        # value is [batch, seq_len, d_model] -> detach last position
        captured_dict[layer_idx] = value[0, -1].detach().to(torch.float32).cpu()
        return value
    return hook_fn

def make_capture_hook_hf(layer_idx, captured_dict):
    def hook_fn(module, input, output):
        val = output[0] if isinstance(output, tuple) else output
        captured_dict[layer_idx] = val[0, -1].detach().to(torch.float32).cpu()
        return output
    return hook_fn

def capture_resid_post(model, tokens, track_layers):
    """Capture post-layer residual streams using forward hooks."""
    captured = {}
    handles = []
    is_tl = isinstance(model, HookedTransformer)

    if is_tl:
        for l in track_layers:
            h = model.add_hook(f"blocks.{l}.hook_resid_post", make_capture_hook_tl(l, captured))
            handles.append(h)
    else:
        for l in track_layers:
            h = model.model.layers[l].register_forward_hook(make_capture_hook_hf(l, captured))
            handles.append(h)

    try:
        model(tokens)
    finally:
        if is_tl:
            model.reset_hooks()
        else:
            for h in handles:
                h.remove()

    return captured

def get_resid_to_logits_fn(model, device):
    """Factory to create a function that maps residual stream state to logits."""
    is_tl = isinstance(model, HookedTransformer)
    model_dtype = next(model.parameters()).dtype if list(model.parameters()) else torch.bfloat16
    
    if is_tl:
        W_U = model.W_U.detach()
        ln_final = model.ln_final
        def resid_to_logits_tl(resid):
            res_dev = resid.to(device).to(model_dtype)
            ln_out = ln_final(res_dev)
            return (ln_out @ W_U)
        return resid_to_logits_tl
    else:
        norm = model.model.norm
        lm_head = model.lm_head
        def resid_to_logits_hf(resid):
            res_dev = resid.to(device).to(model_dtype)
            ln_out = norm(res_dev)
            logits = lm_head(ln_out)
            return logits
        return resid_to_logits_hf

def main():
    args = parse_args()
    print(f"=== Generated Token Config ({args.task.upper()}) ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("==============\n")

    t_start = time.time()

    # Load Prompts
    print(f"Loading {args.n_prompts} Alpaca eval prompts...")
    eval_texts = load_alpaca_prompts(ALPACA_DATA_PATH, args.n_prompts)
    print(f"Loaded {len(eval_texts)} prompts.\n")

    # Define layers to track KL (L8 to L25)
    track_layers = list(range(8, 26))
    model_cache = {}

    def setup_pipeline(m_name, config_path, load_vector_path):
        config = PipelineConfig.load(config_path)
        config.model.device = args.device
        config.load_vector = load_vector_path
        
        # Instantiate pipeline
        pipeline = SteeringPipeline(config)
        model_type = pipeline._get_model_type_for_method(pipeline.steer_config.method)
        
        if model_type in model_cache:
            pipeline.model = model_cache[model_type]
            pipeline._current_model_type = model_type
        else:
            pipeline.load_model(model_type)
            model_cache[model_type] = pipeline.model
            
        pipeline.setup()
        return pipeline

    methods = ["CAA", "CHARS", "ACT", "REPS", "FLAS", "PID", "WeightSteer", "CAA-multi"]
    
    token_positions = [
        ("1st_gen_token", 0),
        ("3rd_gen_token", 2),
        ("10th_gen_token", 9)
    ]
    
    results = {pos_name: {} for pos_name, _ in token_positions}

    for m_name in methods:
        print(f"\nSetting up method {m_name}...")
        
        cfg_name = "CAA" if m_name == "CAA-multi" else m_name
        config_path = CONFIG_PATHS[args.task][cfg_name]
        
        with open(config_path) as f:
            cfg_data = json.load(f)
        vector_path = cfg_data.get("load_vector") or cfg_data.get("save_vector")
        
        pipeline = setup_pipeline(m_name, config_path, vector_path)
        model = pipeline.model
        steer_model = pipeline.steer_model
        model_type = pipeline._current_model_type

        # Setup custom coefficients
        coeff = {layer: args.coeff for layer in steer_model.layer}

        # Apply specific modifications for CAA-multi
        if m_name == "CAA-multi":
            steer_model.layer = list(range(8, 26))
            v_14 = steer_model.steering_vector[14]
            steer_model.steering_vector = {l: v_14 for l in range(8, 26)}
            coeff = {l: args.coeff for l in range(8, 26)}

        # Setup resid to logits function
        resid_to_logits = get_resid_to_logits_fn(model, args.device)

        print(f"Running evaluation for {m_name}...")
        all_kl = {pos_name: defaultdict(list) for pos_name, _ in token_positions}
        is_weight_steer = isinstance(steer_model, WeightSteerModel)

        for idx, prompt_text in enumerate(eval_texts):
            # Tokenize prompt
            if isinstance(model, HookedTransformer):
                prompt_tokens = model.to_tokens(prompt_text, truncate=True)[:, :32].to(args.device)
                tokenizer = model.tokenizer
            else:
                tokenizer = steer_model.tokenizer
                prompt_tokens = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=32).input_ids.to(args.device)

            prompt_len = prompt_tokens.shape[1]

            # Generate 10 tokens under steering
            steer_handles = []
            if is_weight_steer:
                steer_model._apply_weight_modifications(coeff)
            elif hasattr(steer_model, "_install_hook"):
                concept_text = steer_model.flas_concept_text or args.task
                flowtime = steer_model._resolve_flowtime(coeff)
                attention_mask = torch.ones_like(prompt_tokens)
                steer_model._prepare_flow_context(concept_text, flowtime, attention_mask, prompt_len)
                steer_model._install_hook()
                steer_model._active = True
                
                class FLASCleanup:
                    def remove(self):
                        steer_model._active = False
                        steer_model._remove_hook()
                        steer_model._ctx = None
                steer_handles = [FLASCleanup()]
            else:
                steer_handles = steer_model.setup_hooks(coeff)

            try:
                # Generate 10 tokens greedily using the wrapper's native generate method
                gen_text_list = steer_model.generate(
                    prompt_text,
                    coeff=coeff,
                    max_new_tokens=10,
                    do_sample=False
                )
                gen_text = gen_text_list[0] if gen_text_list else ""
                
                gen_tokens = tokenizer(gen_text, add_special_tokens=False, return_tensors="pt").input_ids.to(args.device)
                full_tokens = torch.cat([prompt_tokens, gen_tokens], dim=-1)
            finally:
                if is_weight_steer:
                    steer_model._restore_weights(coeff)
                else:
                    if isinstance(model, HookedTransformer):
                        model.reset_hooks()
                    else:
                        for h in steer_handles:
                            if hasattr(h, "remove"):
                                h.remove()

            gen_len = full_tokens.shape[1] - prompt_len
            if gen_len <= 0:
                continue

            # Evaluate at the 1st, 3rd, and 10th generated tokens
            for pos_name, offset in token_positions:
                step_idx = prompt_len + offset
                if step_idx > full_tokens.shape[1]:
                    continue
                
                ctx = full_tokens[:, :step_idx]

                # Run baseline model on ctx
                resid_un = capture_resid_post(model, ctx, track_layers)

                # Run steered model on ctx
                if is_weight_steer:
                    steer_model._apply_weight_modifications(coeff)
                elif hasattr(steer_model, "_install_hook"):
                    concept_text = steer_model.flas_concept_text or args.task
                    flowtime = steer_model._resolve_flowtime(coeff)
                    attention_mask = torch.ones_like(ctx)
                    steer_model._prepare_flow_context(concept_text, flowtime, attention_mask, ctx.shape[1])
                    steer_model._install_hook()
                    steer_model._active = True
                else:
                    steer_handles = steer_model.setup_hooks(coeff)

                try:
                    resid_st = capture_resid_post(model, ctx, track_layers)
                finally:
                    if is_weight_steer:
                        steer_model._restore_weights(coeff)
                    else:
                        if isinstance(model, HookedTransformer):
                            model.reset_hooks()
                        else:
                            for h in steer_handles:
                                if hasattr(h, "remove"):
                                    h.remove()

                # Calculate KL divergence at the last position of prefix context
                for l in track_layers:
                    act_un = resid_un[l].unsqueeze(0).to(args.device)
                    act_st = resid_st[l].unsqueeze(0).to(args.device)

                    logits_un = resid_to_logits(act_un)
                    logits_st = resid_to_logits(act_st)

                    kl = F.kl_div(
                        F.log_softmax(logits_st[0].float(), dim=-1),
                        F.log_softmax(logits_un[0].float(), dim=-1),
                        log_target=True, reduction='sum'
                    ).item()

                    all_kl[pos_name][l].append(kl)

            if (idx + 1) % 25 == 0:
                print(f"  {idx + 1}/{len(eval_texts)} prompts processed")

        # Save aggregated averages for each position
        for pos_name, _ in token_positions:
            results[pos_name][m_name] = {
                "kl_mean": {l: float(np.mean(all_kl[pos_name][l])) if all_kl[pos_name][l] else 0.0 for l in track_layers}
            }

    # Save to task-specific JSON
    output_dir = Path("Experiments/ExpD")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"expD_results_{args.task}.json"
    
    json_results = {}
    for pos_name in results:
        json_results[pos_name] = {}
        for m_name in results[pos_name]:
            json_results[pos_name][m_name] = {
                "kl_mean": {str(k): v for k, v in results[pos_name][m_name]["kl_mean"].items()}
            }
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved generated token results to {output_file}")

    # Print Markdown Tables for each position
    for pos_name, _ in token_positions:
        print("\n" + "="*80)
        print(f"LOGIT LENS KL DIVERGENCE COMPARISON - {pos_name.upper()}")
        print("="*80)
        
        method_names = sorted(results[pos_name].keys())
        header_str = f"| {'Layer':<5} | " + " | ".join(f"{name:<11}" for name in method_names) + " |"
        sep_str = f"|:{'-'*5}:|:" + ":|:".join(f"{'-'*11}" for _ in method_names) + ":|"
        print(header_str)
        print(sep_str)

        for l in track_layers:
            row_cells = []
            for name in method_names:
                kl_val = results[pos_name][name]["kl_mean"].get(l, 0.0)
                row_cells.append(f"{kl_val:<11.6f}")
            print(f"| L{l:<4} | " + " | ".join(row_cells) + " |")
        print("="*80)

    print(f"\nTotal execution time: {time.time() - t_start:.2f}s")

if __name__ == "__main__":
    main()
