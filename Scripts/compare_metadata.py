#!/usr/bin/env python3
"""
Activation Metadata Score Comparison.
Loads metadata.pt for swept trials, computes the discriminative score:
  Score = sum(|target_freq - contrast_freq| + |target_mean - contrast_mean|)
and reports the optimal hyperparameters.
"""

import os
import torch

# =============================================================================
# SWEEP VALUES — edit here
# =============================================================================
LAYER = 14

SAS_ACT_FRACS      = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

SRPS_THRESHOLDS    = [3.0, 10.0, 20.0, 50.0, 100.0]
SRPS_BETAS         = [1.0, 2.0, 5.0, 10.0]

SPARE_PROPS        = [0.01, 0.05, 0.1, 0.2, 0.5]
SPARE_LOSS_WEIGHTS = [True, False]
SPARE_N_NEIGHBORS  = [1, 3, 5, 10]
SPARE_DEFAULT_PROP = 0.05
SPARE_DEFAULT_LW   = True
SPARE_DEFAULT_NN   = 3

SAECOT_MODES       = ["target_mean", "selected"]
SAECOT_ACTS        = [0.5, 1.0, 2.0]
# =============================================================================


def compute_score(metadata_path, layer=LAYER):
    if not os.path.exists(metadata_path):
        return None
    try:
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        feature_tracker_block = metadata.get("feature_tracker", {})
        tracker = feature_tracker_block.get(str(layer), feature_tracker_block.get(int(layer), None))
        if tracker is None:
            return None
        score = 0.0
        for feature in tracker:
            t_freq = float(feature.get("target_activation_frequency", feature.get("activation_frequency", 0.0)))
            c_freq = float(feature.get("contrast_activation_frequency", 0.0))
            t_mean = float(feature.get("target_feature_mean", feature.get("feature_mean", 0.0)))
            c_mean = float(feature.get("contrast_feature_mean", 0.0))
            score += abs(t_mean - c_mean)
            score += abs(t_freq - c_freq)
        return score
    except Exception as e:
        print(f"Error reading {metadata_path}: {e}")
        return None


def report(results, label, winner_fmt):
    if results:
        best = max(results, key=lambda x: x[1])
        print(f"  ==> Winner: {winner_fmt(best[0])} (Score: {best[1]:.4f})")
    else:
        print(f"  No results found yet.")


def main():
    print("=" * 60)
    print("SAE Steering Benchmark Sweeps: Metadata Comparison Report")
    print("=" * 60)

    # 1. SAS
    print("\n--- SAS (act_frac sweep) ---")
    results = []
    for val in SAS_ACT_FRACS:
        tag  = str(val).replace(".", "p")
        path = f"Vector/SAS/Gemma/refusal_response_act_frac_{tag}/metadata.pt"
        score = compute_score(path)
        if score is not None:
            results.append((val, score))
            print(f"  act_frac={val:<5} -> Score: {score:.4f}")
    report(results, "SAS", lambda v: f"act_frac={v}")

    # 2. SRPS
    print("\n--- SRPS (act_threshold & beta sweep) ---")
    results = []
    for thresh in SRPS_THRESHOLDS:
        for beta in SRPS_BETAS:
            t_tag = str(thresh).replace(".", "p")
            b_tag = str(beta).replace(".", "p")
            path  = f"Vector/SRPS/Gemma/refusal_response_thresh_{t_tag}_beta_{b_tag}/metadata.pt"
            score = compute_score(path)
            if score is not None:
                results.append(((thresh, beta), score))
                print(f"  act_thresh={thresh:<5} beta={beta:<5} -> Score: {score:.4f}")
    report(results, "SRPS", lambda v: f"act_threshold={v[0]}, beta={v[1]}")

    # 3. SPARE — top_k_proportion sweep
    print("\n--- SPARE (top_k_proportion sweep) ---")
    results = []
    for prop in SPARE_PROPS:
        tag  = f"prop_{str(prop).replace('.', 'p')}"
        path = f"Vector/SPARE/Gemma/refusal_response_{tag}/metadata.pt"
        score = compute_score(path)
        if score is not None:
            results.append((prop, score))
            print(f"  top_k_proportion={prop:<5} -> Score: {score:.4f}")
    report(results, "SPARE prop", lambda v: f"top_k_proportion={v}")

    # 4. SPARE — loss_weight sweep
    print("\n--- SPARE (loss_weight sweep) ---")
    results = []
    for lw in SPARE_LOSS_WEIGHTS:
        lw_str = str(lw).lower()
        tag    = f"lw_{lw_str}_pos_last_nn{SPARE_DEFAULT_NN}"
        path   = f"Vector/SPARE/Gemma/refusal_response_{tag}/metadata.pt"
        score  = compute_score(path)
        if score is not None:
            results.append((lw, score))
            print(f"  loss_weight={lw_str:<5} -> Score: {score:.4f}")
    report(results, "SPARE lw", lambda v: f"loss_weight={v}")

    # 5. SPARE — n_neighbors sweep
    print("\n--- SPARE (n_neighbors sweep) ---")
    results = []
    lw_str = str(SPARE_DEFAULT_LW).lower()
    for nn in SPARE_N_NEIGHBORS:
        tag   = f"lw_{lw_str}_pos_last_nn{nn}"
        path  = f"Vector/SPARE/Gemma/refusal_response_{tag}/metadata.pt"
        score = compute_score(path)
        if score is not None:
            results.append((nn, score))
            print(f"  n_neighbors={nn:<3} -> Score: {score:.4f}")
    report(results, "SPARE nn", lambda v: f"n_neighbors={v}")

    # 6. SAECOT
    print("\n--- SAECOT (value_mode & max_act sweep) ---")
    results = []
    for mode in SAECOT_MODES:
        for act in SAECOT_ACTS:
            a_tag = str(act).replace(".", "p")
            path  = f"Vector/SAECOT/Gemma/refusal_response_mode_{mode}_act_{a_tag}/metadata.pt"
            score = compute_score(path)
            if score is not None:
                results.append(((mode, act), score))
                print(f"  mode={mode:<12} max_act={act:<5} -> Score: {score:.4f}")
    report(results, "SAECOT", lambda v: f"value_mode={v[0]}, max_act={v[1]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
