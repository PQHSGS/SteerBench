#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
set -euo pipefail

# Make sure we use the correct Conda environment python
export PATH="/data/caotue/anaconda3/envs/sae_circuit/bin:$PATH"

COEFFS=(3.0 5.0 10.0)
CONFIG_DIR="Configs/Eval/LINNEAS/Gemma"
CONFIGS=(
  # "gemma_refusal_ab.json"                    # Extracts refusal vector
  # "gemma_refusal_open.json"                  # Loads refusal vector
  # "gemma_sorrybench.json"                    # Loads refusal vector
  # "gemma_refusal_response.json"              # Extracts response refusal vector
  # "gemma_sorrybench_refusal_response.json"   # Loads response refusal vector
  # "gemma_hallu_ab.json"                      # Extracts hallucination vector
  # "gemma_truthfulqa.json"                    # Loads hallucination vector
  # "gemma_deception.json"                     # Extracts cais vector
  # "gemma_ifeval.json"                        # Extracts ifeval vector
  "gemma_toxic.json"                         # Extracts toxic vector
  # "gemma_mbpp.json"                          # Loads deception vector
  # "gemma_evil.json"
)

run_coeff_sweep() {
  local config_path="$1"
  local idx=0

  for coeff in "${COEFFS[@]}"; do
    local tmp_config
    tmp_config="$(mktemp /tmp/linneas_cfg_XXXXXX.json)"

    python - "$config_path" "$coeff" "$idx" > "$tmp_config" <<'PY'
import json
import pathlib
import sys

base_path = pathlib.Path(sys.argv[1])
coeff = float(sys.argv[2])
idx = int(sys.argv[3])

with base_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg.setdefault("steer", {})["coeff"] = coeff
cfg["include_baseline"] = False

if "model" in cfg:
    cfg["model"]["device"] = "cuda"

# Swap save_vector to load_vector on subsequent sweep iterations to skip extraction
if idx > 0 and "save_vector" in cfg:
    cfg["load_vector"] = cfg.pop("save_vector")

print(json.dumps(cfg, indent=2))
PY

    echo ""
    echo "--- Running $(basename "$config_path") | coeff=$coeff ---"
    python -m Steering.cli --task eval --config "$tmp_config"

    rm -f "$tmp_config"
    idx=$((idx + 1))
    sleep 5
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_coeff_sweep "$CONFIG_DIR/$cfg"
done
