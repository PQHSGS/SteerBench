#!/bin/bash
set -euo pipefail
export PATH="/data/caotue/anaconda3/envs/sae_circuit/bin:$PATH"
export HF_HOME="/data/caotue/hf_cache"
export HUGGINGFACE_HUB_CACHE="/data/caotue/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/caotue/hf_cache/transformers"
export HF_DATASETS_CACHE="/data/caotue/hf_cache/datasets"
export TORCH_HOME="/data/caotue/torch_cache"
export TMPDIR="/data/caotue/tmp"

TASK="$1"
GPU="$2"
CONFIG_DIR="Configs/Eval/CURVE/Gemma"
COEFFS=(1 2 3 5 7 10)

export CUDA_VISIBLE_DEVICES="$GPU"
idx=0
for coeff in "${COEFFS[@]}"; do
  tmp_config="$(mktemp /tmp/curve_cfg_XXXXXX.json)"
  python - "$CONFIG_DIR/$TASK" "$coeff" "$idx" > "$tmp_config" <<'PY'
import json, pathlib, sys
base_path = pathlib.Path(sys.argv[1])
coeff = float(sys.argv[2])
idx = int(sys.argv[3])
with base_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg.setdefault("steer", {})["coeff"] = coeff
cfg["include_baseline"] = False
if "model" in cfg:
    cfg["model"]["device"] = "cuda"
if idx > 0 and "save_vector" in cfg:
    cfg["load_vector"] = cfg.pop("save_vector")
print(json.dumps(cfg, indent=2))
PY
  echo ""
  echo "--- CURVE $(basename "$TASK" .json) | coeff=$coeff | GPU=$GPU ---"
  python -m Steering.cli --task eval --config "$tmp_config"
  rm -f "$tmp_config"
  idx=$((idx + 1))
  sleep 5
done
