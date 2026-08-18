#!/bin/bash
# Ablation: CHARS with chars_lambda=100 (uniform Sinkhorn → removes OT).
set -euo pipefail

export PATH="/data/caotue/anaconda3/envs/sae_circuit/bin:$PATH"
export CUDA_VISIBLE_DEVICES=3
COEFFS=(12)
OUTPUT_DIR="Results/chars_ablate"
CONFIG_DIR="Configs/Eval/CHARS/Gemma"
CONFIGS=(
  "gemma_deception.json"
  # "gemma_evil.json"
  "gemma_toxic.json"
)

run_coeff_sweep() {
  local config_path="$1"
  local idx=0

  for coeff in "${COEFFS[@]}"; do
    local tmp_config
    tmp_config="$(mktemp /tmp/chars_ablate_cfg_XXXXXX.json)"

    python - "$config_path" "$coeff" "$idx" "$OUTPUT_DIR" > "$tmp_config" <<'PY'
import json
import pathlib
import sys

base_path = pathlib.Path(sys.argv[1])
coeff = float(sys.argv[2])
idx = int(sys.argv[3])
output_dir = sys.argv[4]

with base_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

# Override lambda to a large value: uniform Sinkhorn coupling = global centroid mean
cfg.setdefault("extractor", {})["chars_lambda"] = 100.0
cfg.setdefault("steer", {})["coeff"] = coeff
cfg["include_baseline"] = False
cfg["output"] = output_dir

# Separate save path to not collide with standard CHARS vectors
task_name = base_path.stem.replace("gemma_", "")
cfg["save_vector"] = f"Vector/CHARS_ablate/Gemma/{task_name}"
# Remove any hardcoded load_vector from the original config
cfg.pop("load_vector", None)

if "model" in cfg:
    cfg["model"]["device"] = "cuda"

if idx > 0 and "save_vector" in cfg:
    cfg["load_vector"] = cfg.pop("save_vector")

print(json.dumps(cfg, indent=2))
PY

    echo ""
    echo "--- CHARS Ablate (lambda=100) | $(basename "$config_path") | coeff=$coeff ---"
    python -m Steering.cli --task eval --config "$tmp_config"

    rm -f "$tmp_config"
    idx=$((idx + 1))
    sleep 5
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_coeff_sweep "$CONFIG_DIR/$cfg"
done
