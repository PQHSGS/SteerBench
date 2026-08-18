#!/bin/bash
# GPU assigned via config device
set -euo pipefail

COEFFS=(1d)
CONFIG_DIR="Configs/Eval/IDS/Gemma"
CONFIGS=(
  # "gemma_evil.json"
  # "gemma_refusal_response.json"
  "gemma_toxic.json"
  "gemma_deception.json"
)

run_coeff_sweep() {
  local config_path="$1"
  local idx=0

  for coeff in "${COEFFS[@]}"; do
    local tmp_config
    tmp_config="$(mktemp /tmp/ids_cfg_XXXXXX.json)"

    conda run --no-capture-output -n sae_circuit python - "$config_path" "$coeff" "$idx" > "$tmp_config" <<'PY'
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
    cfg["model"]["device"] = "cuda:0"

# Reuse the computed vector on subsequent coefficient sweeps
if idx > 0 and "save_vector" in cfg:
    cfg["load_vector"] = cfg.pop("save_vector")

print(json.dumps(cfg, indent=2))
PY

    echo ""
    echo "--- Running $(basename "$config_path") | coeff=$coeff ---"
    conda run --no-capture-output -n sae_circuit python -m Steering.cli --task eval --config "$tmp_config"

    rm -f "$tmp_config"
    idx=$((idx + 1))
    sleep 5
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_coeff_sweep "$CONFIG_DIR/$cfg"
done
