"""Generate prefix-split configs for CHARS, ACT, WEIGHTSTEER, REPS on Toxic and Evil."""
import json, os
from pathlib import Path

BASE = "Configs/Eval/PREFIX_SPLIT"

methods = {
    "CHARS": {
        "extractor": {
            "method": "CHARS", "layer": 14, "batch_size": 4,
            "hook_point": "pre", "position": "mask",
            "apply_chat_template": True, "inverse": False,
        },
        "steer": {
            "method": "CHARS", "layer": 14, "apply_chat_template": True,
            "hook_point": "pre", "position": "last", "coeff": 5,
            "steer_once": False,
        },
        "output_dir": "chars",
        "save_vector": "Vector/PREFIX_SPLIT/CHARS",
    },
    "ACT": {
        "extractor": {
            "method": "ACT", "layer": [14], "hook_point": "pre",
            "position": "mask", "batch_size": 4, "act_mode": "linear",
            "inverse": False,
        },
        "steer": {
            "method": "ACT", "layer": [14], "hook_point": "pre",
            "position": "last", "coeff": 2, "batch_size": 4,
            "apply_chat_template": False, "act_support": "none",
        },
        "output_dir": "linearact",
        "save_vector": "Vector/PREFIX_SPLIT/ACT",
    },
    "WEIGHTSTEER": {
        "extractor": {
            "method": "WEIGHTSTEER", "inverse": False,
            "layer": list(range(26)), "batch_size": 8,
            "weight_steer_lr": 0.0002, "weight_steer_epochs": 15,
            "apply_chat_template": True,
            "weight_steer_lora_r": 32, "weight_steer_lora_alpha": 16,
            "weight_steer_lora_dropout": 0.2,
            "weight_steer_target_modules": ["up_proj", "down_proj", "gate_proj"],
        },
        "steer": {
            "method": "WEIGHTSTEER", "layer": list(range(26)),
            "coeff": 1.5, "ot_steer": False,
        },
        "output_dir": "weightsteer",
        "save_vector": "/data/caotue/PREFIX_SPLIT/WEIGHTSTEER",
    },
    "REPS": {
        "extractor": {
            "method": "REPS", "layer": [14], "batch_size": 4,
            "position": "last", "apply_chat_template": True,
            "hook_point": "pre", "change_pad_token": False,
            "inverse": False, "reft_type": "Loreft",
            "low_rank_dimension": 8, "dropout": 0.1,
            "act_fn": "linear", "add_bias": True,
            "preference_pairs": ["orig_add", "orig_sub"],
            "substraction_type": "zero",
            "steering_factors": [1.0, 2.0, 3.0, 4.0, 5.0],
            "gradient_accumulation_steps": 8, "lr": 0.001,
            "weight_decay": 0.0, "epochs": 20, "reft_seed": 42,
            "reft_steer_once": True, "ot_steer": True,
        },
        "steer": {
            "method": "REPS", "layer": [14],
            "coeff": {"14": 2}, "apply_chat_template": True,
            "hook_point": "pre", "position": "last", "batch_size": 4,
            "norm": False, "apply_all_layers": False,
            "steer_once": False, "reft_type": "Loreft",
            "low_rank_dimension": 1, "dropout": 0.0,
            "act_fn": "linear", "add_bias": True,
            "substraction_type": "zero", "ot_steer": False,
        },
        "output_dir": "reps",
        "save_vector": "Vector/PREFIX_SPLIT/REPS",
    },
}

tasks = {
    "toxic": {
        "test_dataset": "toxic",
        "splits": {
            "prefix_pos": {"train_dataset": "toxic_jigsaw_prefix_pos", "n_train": 287},
            "prefix_neg": {"train_dataset": "toxic_jigsaw_prefix_neg", "n_train": 213},
        },
    },
    "evil": {
        "test_dataset": "evil",
        "splits": {
            "prefix_pos": {"train_dataset": "evil_prefix_pos", "n_train": 274},
            "prefix_neg": {"train_dataset": "evil_prefix_neg", "n_train": 226},
        },
    },
}

model_cfg = {
    "name": "google/gemma-2-2b-it",
    "device": "cuda",
    "dtype": "bfloat16",
    "max_new_tokens": 100,
    "do_sample": False,
}

for method_name, method_cfg in methods.items():
    for task_name, task_cfg in tasks.items():
        for split_name, split_cfg in task_cfg["splits"].items():
            name = f"gemma_{task_name}_{split_name}"
            out_dir = Path(BASE) / method_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{name}.json"

            # WeightSteer needs different epochs for evil vs toxic
            ext = dict(method_cfg["extractor"])
            if method_name == "WEIGHTSTEER":
                ext["weight_steer_epochs"] = 15 if task_name == "toxic" else 5

            config = {
                "name": name,
                "description": f"{method_name} {task_name} {split_name} prefix split",
                "model": dict(model_cfg),
                "extractor": ext,
                "steer": dict(method_cfg["steer"]),
                "train_dataset": split_cfg["train_dataset"],
                "n_train": split_cfg["n_train"],
                "test_dataset": task_cfg["test_dataset"],
                "n_test": 100,
                "output": f"./Results/{method_cfg['output_dir']}",
                "compute_perplexity": True,
                "include_baseline": False,
                "seed": 42,
            }

            # Vectors: WEIGHTSTEER saves to /data/caotue, others to Vector/
            if method_name == "WEIGHTSTEER":
                config["load_vector"] = f"{method_cfg['save_vector']}/{task_name}/{split_name}"
            else:
                config["save_vector"] = f"{method_cfg['save_vector']}/{task_name}_{split_name}"

            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Created {out_path}")

print("Done!")
