#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
set -euo pipefail

# Make sure we use the correct Conda environment python
export PATH="/data/caotue/anaconda3/envs/sae_circuit/bin:$PATH"

ANGLES=(330 350 0 10 30 )
CONFIG_DIR="Configs/Eval/Angular/Gemma"
CONFIGS=(
  # "gemma_refusal_response.json"
  # "gemma_refusal_ab.json"
  # "gemma_refusal_open.json"
  # "gemma_sorrybench.json"
  # "gemma_sorrybench_refusal_response.json"
  # "gemma_hallu_ab.json"
  # "gemma_truthfulqa.json"
  # "gemma_agreement.json"
  # "gemma_deception.json"
  # "gemma_ifeval.json"
    "gemma_toxic.json"
  "gemma_evil.json"
)

run_angle_sweep() {
  local config_path="$1"
  local idx=0

  for angle in "${ANGLES[@]}"; do
    local tmp_config
    tmp_config="$(mktemp /tmp/angular_cfg_XXXXXX.json)"

    python - "$config_path" "$angle" "$idx" > "$tmp_config" <<'PY'
import json
import pathlib
import sys

base_path = pathlib.Path(sys.argv[1])
angle = float(sys.argv[2])
idx = int(sys.argv[3])

with base_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg.setdefault("steer", {})["coeff"] = 1.0
cfg.setdefault("steer", {})["target_angle"] = angle
cfg["include_baseline"] = False

if "model" in cfg:
    cfg["model"]["device"] = "cuda"

if idx > 0 and "save_vector" in cfg:
    cfg["load_vector"] = cfg.pop("save_vector")

print(json.dumps(cfg, indent=2))
PY

    echo ""
    echo "--- Running $(basename "$config_path") | target_angle=$angle ---"
    python -m Steering.cli --task eval --config "$tmp_config"

    rm -f "$tmp_config"
    idx=$((idx + 1))
    sleep 5
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_angle_sweep "$CONFIG_DIR/$cfg"
done
