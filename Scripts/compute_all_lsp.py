#!/usr/bin/env python3
"""Compute lsp_score for all JSON results missing it, then remove
token_nlls/token_ids from metadata to keep files lean.

Reads each sample's response text, runs a forward pass to get per-token NLLs,
computes Localized Suffix-Penalized Perplexity (PPL_lsp), saves only the scalar
lsp_score back into the result dict, and discards the per-token arrays."""

import os, glob, re, json, math
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b-it"
GPU_ID = 5
RESULTS_DIR = "/home/caotue/SAESteeringBench/Results"

TARGET_DIRS = [
    "caa", "linearact", "curve", "chars", "flas", "reps",
    "weightsteer", "linneas", "pcaot",
    "linearact_ablate", "chars_ablate",
]


def compute_nlls_and_tokens(
    model, tokenizer, texts: List[str], batch_size: int = 64
) -> Tuple[List[List[float]], List[List[int]]]:
    all_nlls, all_tokens = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = tokens["input_ids"].cuda()
        attention_mask = tokens["attention_mask"].cuda()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            ).reshape(shift_logits.shape[0], -1)
            for b in range(per_token_loss.size(0)):
                mask = attention_mask[b, 1: 1 + per_token_loss.size(1)]
                valid_losses = per_token_loss[b][mask.bool()]
                valid_losses = valid_losses[valid_losses.isfinite()]
                valid_tokens = input_ids[b, 1: 1 + per_token_loss.size(1)][mask.bool()]
                finite_mask = torch.isfinite(per_token_loss[b][mask.bool()])
                valid_tokens = valid_tokens[finite_mask]
                if len(valid_losses) > 0:
                    all_nlls.append(valid_losses.cpu().tolist())
                    all_tokens.append(valid_tokens.cpu().tolist())
                else:
                    all_nlls.append([])
                    all_tokens.append([])
    return all_nlls, all_tokens


def compute_lsp_scores(
    samples_nlls: List[List[float]],
    samples_tokens: List[List[int]],
    window_size: int = 30,
    penalty_weight: float = 2.0,
) -> Dict[str, Any]:
    all_losses = []
    n_valid = 0
    for nlls, tokens in zip(samples_nlls, samples_tokens):
        if not nlls or not tokens or len(nlls) != len(tokens) or len(tokens) <= 1:
            continue
        n_valid += 1
        for t in range(len(tokens)):
            raw_loss = nlls[t]
            start_idx = max(0, t - window_size)
            history = tokens[start_idx:t]
            lcs = 0
            for idx in range(len(history) - 1):
                length = 0
                while idx - length >= 0 and history[idx - length] == history[-1 - length]:
                    length += 1
                if length > lcs:
                    lcs = length
            penalty = penalty_weight * math.log1p(max(0, lcs - 2))
            all_losses.append(raw_loss + penalty)
    if not all_losses:
        return {"lsp_score": None, "n_valid": 0}
    mean_loss = sum(all_losses) / len(all_losses)
    try:
        lsp_score = math.exp(mean_loss)
    except OverflowError:
        lsp_score = float("inf")
    return {"lsp_score": lsp_score, "n_valid": n_valid}


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    files_to_process = []
    for mdir in TARGET_DIRS:
        dirpath = os.path.join(RESULTS_DIR, mdir)
        if not os.path.isdir(dirpath):
            continue
        for fpath in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
            try:
                with open(fpath) as fh:
                    data = json.load(fh)
            except:
                continue
            result_dict = data.get("result", data)
            if not isinstance(result_dict, dict):
                continue
            ls = result_dict.get("lsp_score")
            if ls is not None:
                continue
            files_to_process.append((fpath, data))

    print(f"Found {len(files_to_process)} files missing lsp_score. Loading model...")
    if not files_to_process:
        return

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("Model loaded.")

    for fpath, data in tqdm(files_to_process, desc="Computing LSP"):
        try:
            result_dict = data.get("result", data)
            samples = result_dict.get("samples", [])
            responses = [s["response"] for s in samples]
            if not responses:
                continue

            nlls, tokens = compute_nlls_and_tokens(model, tokenizer, responses, batch_size=64)
            res = compute_lsp_scores(nlls, tokens)
            ls = res["lsp_score"]

            if "result" in data:
                data["result"]["lsp_score"] = ls
            else:
                data["lsp_score"] = ls

            # Remove per-token arrays from metadata (keep JSON lean)
            for s in samples:
                meta = s.get("metadata")
                if meta:
                    meta.pop("token_nlls", None)
                    meta.pop("token_ids", None)

            with open(fpath, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error: {fpath}: {e}")


if __name__ == "__main__":
    main()
