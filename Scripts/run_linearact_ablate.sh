#!/bin/bash
# Ablation: ACT with act_mode="gaussian" (per-dim std, no sorted quantile coupling).
set -euo pipefail

export PATH="/data/caotue/anaconda3/envs/sae_circuit/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2
COEFFS=(1 2 3 5 7 10)
OUTPUT_DIR="Results/linearact_ablate"
CONFIG_DIR="Configs/Eval/LinearAcT/Gemma"
CONFIGS=(
  "gemma_deception.json"
  "gemma_evil.json"
  "gemma_toxic.json"
)

run_coeff_sweep() {
  local config_path="$1"
  local idx=0

  for coeff in "${COEFFS[@]}"; do
    local tmp_config
    tmp_config="$(mktemp /tmp/linearact_ablate_cfg_XXXXXX.json)"

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

# Override to "gaussian" mode: per-dim std ratio, no sorted quantile coupling
cfg.setdefault("extractor", {})["act_mode"] = "gaussian"
cfg.setdefault("steer", {})["coeff"] = coeff
cfg["include_baseline"] = False
cfg["output"] = output_dir

# Separate save path to not collide with standard ACT vectors
task_name = base_path.stem.replace("gemma_", "")
cfg["save_vector"] = f"Vector/LinearAcT_ablate/Gemma/{task_name}"
# Remove any hardcoded load_vector from the original config
cfg.pop("load_vector", None)

if "model" in cfg:
    cfg["model"]["device"] = "cuda"

if idx > 0 and "save_vector" in cfg:
    cfg["load_vector"] = cfg.pop("save_vector")

print(json.dumps(cfg, indent=2))
PY

    echo ""
    echo "--- ACT Ablate (gaussian) | $(basename "$config_path") | coeff=$coeff ---"
    python -m Steering.cli --task eval --config "$tmp_config"

    rm -f "$tmp_config"
    idx=$((idx + 1))
    sleep 5
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_coeff_sweep "$CONFIG_DIR/$cfg"
done
