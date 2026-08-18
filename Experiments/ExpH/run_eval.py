"""Eval steering accuracy for given task, layer, coeff. Saves results JSON."""
import sys, json, os
sys.path.insert(0, ".")
from pathlib import Path
from Steering.pipeline import SteeringPipeline
from Steering.config.pipeline import PipelineConfig

task = sys.argv[1]       # evil / toxic / deception
layer = int(sys.argv[2]) # 14 / 18 / 22 / 25
coeff = float(sys.argv[3])
vec_override = sys.argv[4] if len(sys.argv) > 4 else None

TASK_CFG = {
    "evil": {"train": "evil", "test": "evil", "vec": "Vector/CAA/Gemma/evil"},
    "toxic": {"train": "toxic_jigsaw", "test": "toxic", "vec": "Vector/CAA/Gemma/toxic"},
    "deception": {"train": "liarbench", "test": "liarbench", "vec": "Vector/CAA/Gemma/deception"},
}
cfg = TASK_CFG[task]
if vec_override:
    cfg = {**cfg, "vec": vec_override}

config = PipelineConfig.from_dict({
    "name": f"expH_{task}_L{layer}",
    "description": "ExpH steering accuracy eval",
    "model": {
        "name": "google/gemma-2-2b-it",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "max_new_tokens": 100,
    },
    "extractor": {
        "method": "CAA",
        "layer": [layer],
        "batch_size": 4,
        "hook_point": "pre",
        "apply_chat_template": True,
        "inverse": False,
        "position": "mask",
    },
    "steer": {
        "method": "CAA",
        "layer": [layer],
        "coeff": coeff,
        "apply_chat_template": True,
        "hook_point": "pre",
        "steer_once": False,
    },
    "load_vector": cfg["vec"],
    "train_dataset": cfg["train"],
    "test_dataset": cfg["test"],
    "output": f"./Results/expH/{task}/L{layer}",
    "compute_perplexity": True,
    "include_baseline": False,
    "seed": 42,
    "n_test": 100,
    "n_train": 500,
})

pipeline = SteeringPipeline(config)
result = pipeline.evaluate(verbose=False)

out_dir = Path(f"Experiments/ExpH/results/eval/{task}/L{layer}")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f"c{int(coeff)}.json"
with open(out_file, "w") as f:
    json.dump(result.to_dict(), f, indent=2, default=str)

acc = result.accuracy
ppl = result.perplexity
rep = result.repetition_rate
print(f"{task} L{layer} c={coeff}: acc={acc:.4f} ppl={ppl:.2f} rep={rep:.4f}")
