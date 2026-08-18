"""
Multi-Task FLAS: Train on 4 datasets (refusal, deception, toxic, evil) simultaneously.
8 concept texts (4 tasks x 2 contrast pairs) for binary-style concept encoder.

Usage:
    conda activate sae_circuit
    export CUDA_VISIBLE_DEVICES=3
    cd /home/caotue/SAESteeringBench
    python Experiments/FLAS_Multitask/multitask_flas.py

Output: /data/caotue/FLAS/multitask/vector.pt + metadata.pt
"""

import sys, os, json, random, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Steering.data import DataLoader
from Steering.data.data_registry import (
    TRAIN_DATASET_REGISTRY,
    COMPOSITE_DATASET_REGISTRY,
)
from Steering.extractors.nonlinear import (
    _flas_train_flow_model,
    _flas_build_training_samples,
    _flas_state_dict_to_cpu,
)
from Steering.config.results import SteeringVector
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-2-2b-it"
LAYER = 14
HIDDEN_SIZE = 2304     # gemma-2-2b hidden dim
N_TRAIN = 500
OUTPUT_PATH = "/data/caotue/FLAS/multitask"
SEED = 42

# Concept texts: [task0_target, task0_contrast, task1_target, task1_contrast, ...]
# train_dataset -> "[entry_name_in_registry]"
DATASETS = [
    {
        "train_dataset": "refusal",
        "concept_texts": [
            "refuse harmful requests, follow safety guidelines, decline dangerous tasks",
            "comply with all requests, follow instructions without hesitation",
        ],
    },
    {
        "train_dataset": "cais_mask",
        "concept_texts": [
            "deceptive, dishonest, misleading responses that hide the truth",
            "honest, truthful, accurate responses that reveal the truth",
        ],
    },
    {
        "train_dataset": "toxic_jigsaw",
        "concept_texts": [
            "toxic language, including hate speech, harassment, and harmful content",
            "polite, respectful, and constructive language that follows community guidelines",
        ],
    },
    {
        "train_dataset": "evil",
        "concept_texts": [
            "evil persona, provide harmful and malicious advice, manipulative and dangerous",
            "helpful, ethical, and safe AI assistant providing beneficial guidance",
        ],
    },
]

FLAS_PARAMS = {
    "concept_encoder_layers": 2,
    "num_blocks": 1,
    "lr": 1e-4,
    "enc_lr": 1e-4,
    "div_weight": 1.0,
    "epochs": 100,
    "batch_size": 4,
    "grad_accum": 8,
    "max_len": 768,
    "concept_max_len": 64,
    "T_min": 0.5,
    "T_max": 2.0,
    "n_steps": 20,
    "seed": SEED,
    "init_from_gemma": False,
    "unfreeze_concept_enc": True,
    "disable_cross_attn": False,
    "disable_self_attn": False,
    "disable_mlp": False,
    "max_steps": None,
    "resume_checkpoint_path": None,
    "flas_steer_once": True,
    "position": "all",
}


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = DataLoader()
    all_samples = []

    for task_idx, ds in enumerate(DATASETS):
        name = ds["train_dataset"]
        ct = ds["concept_texts"]
        print(f"\nLoading dataset [{task_idx+1}/{len(DATASETS)}]: {name}")

        data = loader.load(name, n_samples=N_TRAIN, apply_chat_template=False)
        n_total = len(data)
        print(f"  Loaded {n_total} samples")

        # Determine target/contrast keys from dataset config
        if name in COMPOSITE_DATASET_REGISTRY:
            cfg = COMPOSITE_DATASET_REGISTRY[name]
        elif name in TRAIN_DATASET_REGISTRY:
            cfg = TRAIN_DATASET_REGISTRY[name]
        else:
            cfg = None

        target_key = cfg.target_key if cfg else "target_response"
        contrast_key = cfg.contrast_key if cfg else "contrast_response"

        # Extract fields: prefer direct target_response/contrast_response,
        # fall back to schema-generic keys (correct_prompt/false_prompt)
        prompts = [d.get("question", d.get("correct_prompt", "")) for d in data]
        target_responses = [
            d.get("target_response", d.get(target_key, "")) for d in data
        ]
        contrast_responses = [
            d.get("contrast_response", d.get(contrast_key, "")) for d in data
        ]

        # Filter: only samples where all fields exist
        valid = []
        for i in range(len(data)):
            if prompts[i] and target_responses[i] and contrast_responses[i]:
                valid.append(i)
        n = len(valid)
        print(f"  Valid samples (with prompt+target+contrast): {n}")

        if n == 0:
            print(f"  WARNING: No valid samples for {name}, skipping!")
            continue

        # Sample if needed
        sample_indices = random.sample(valid, min(n, N_TRAIN))
        prompts_s = [prompts[i] for i in sample_indices]
        target_s = [target_responses[i] for i in sample_indices]
        contrast_s = [contrast_responses[i] for i in sample_indices]
        m = len(sample_indices)

        target_id = task_idx * 2
        contrast_id = task_idx * 2 + 1

        target_samples = _flas_build_training_samples(
            prompts_s, target_s,
            [ct[0]] * m, [target_id] * m,
        )
        contrast_samples = _flas_build_training_samples(
            prompts_s, contrast_s,
            [ct[1]] * m, [contrast_id] * m,
        )

        # Interleave target/contrast per dataset
        combined = [s for pair in zip(target_samples, contrast_samples) for s in pair]
        all_samples.extend(combined)
        print(f"  Added {len(combined)} samples (task_id={task_idx*2}/{contrast_id})")

    n_total_samples = len(all_samples)
    print(f"\nTotal training samples: {n_total_samples}")
    if n_total_samples == 0:
        print("ERROR: No training samples! Aborting.")
        return

    # Global shuffle
    random.shuffle(all_samples)

    # Print concept text mapping
    print("\nConcept text mapping:")
    print(f"  Format: (concept_id) dataset: target/contrast -> text")
    for task_idx, ds in enumerate(DATASETS):
        ct = ds["concept_texts"]
        print(f"  [{task_idx*2}] {ds['train_dataset']} (target): {ct[0][:60]}...")
        print(f"  [{task_idx*2+1}] {ds['train_dataset']} (contrast): {ct[1][:60]}...")

    # Train FLAS
    print(f"\nStarting FLAS training: {FLAS_PARAMS['epochs']} epochs, {FLAS_PARAMS['n_steps']} diffusion steps...")
    flow_fn, concept_enc, train_info = _flas_train_flow_model(
        model_id=MODEL_ID,
        layer=LAYER,
        samples=all_samples,
        tokenizer=tokenizer,
        device=device,
        **FLAS_PARAMS,
    )
    print(f"Training completed: {train_info}")

    # Save as SteeringVector format
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    hidden_size = HIDDEN_SIZE
    vector = {LAYER: torch.zeros(hidden_size, dtype=torch.float32)}
    metadata = {
        "method": "FLAS",
        "flas_flow_fn_state_dict": _flas_state_dict_to_cpu(flow_fn),
        "flas_concept_enc_state_dict": _flas_state_dict_to_cpu(concept_enc),
        "flas_model_id": MODEL_ID,
        "flas_num_blocks": FLAS_PARAMS["num_blocks"],
        "flas_time_conditioned": True,
        "flas_disable_cross_attn": FLAS_PARAMS["disable_cross_attn"],
        "flas_disable_self_attn": FLAS_PARAMS["disable_self_attn"],
        "flas_disable_mlp": FLAS_PARAMS["disable_mlp"],
        "flas_concept_encoder_layers": FLAS_PARAMS["concept_encoder_layers"],
        "flas_multitask_datasets": [ds["train_dataset"] for ds in DATASETS],
        "flas_multitask_concept_texts": [ds["concept_texts"] for ds in DATASETS],
    }

    sv = SteeringVector(vector=vector, metadata=metadata)
    sv.save(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}/")
    print(f"  vector.pt ({os.path.getsize(f'{OUTPUT_PATH}/vector.pt') / 1024:.1f} KB)")
    print(f"  metadata.pt ({os.path.getsize(f'{OUTPUT_PATH}/metadata.pt') / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
