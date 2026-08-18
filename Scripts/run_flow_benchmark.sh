#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
set -euo pipefail

# Make sure we use the correct Conda environment python
export PATH="/data/caotue/anaconda3/envs/sae_circuit/bin:$PATH"

CONFIG_DIR="Configs/Eval/FLOW"
CONFIGS=(
  # "gemma_refusal_response.json"
  # "gemma_refusal_ab.json"
  # "gemma_sorrybench.json"
  # "gemma_sorrybench_refusal_response.json"
  # "gemma_gms8k.json"
  # "gemma_agreement.json"
  # "flowsteer_deception.json"
  # "gemma_deception.json"
  # "gemma_hallu_ab.json"
  # "gemma_truthfulqa.json"
  # "gemma_ifeval.json"
    "gemma_toxic.json"
  "gemma_evil.json"
)
REPEATS=3

run_flow_repeats() {
  local config_path="$1"
  local config_name
  config_name="$(basename "$config_path")"
  
  local run_accuracies=()
  local run_perplexities=()

  for repeat in $(seq 1 $REPEATS); do
    local tmp_config
    tmp_config="$(mktemp /tmp/flow_cfg_XXXXXX.json)"
    local vector_path="Vector/Flow/${config_name%.json}_r${repeat}"

    # Generate config for this repeat
    python - "$config_path" "$repeat" "$vector_path" > "$tmp_config" <<'PY'
import json
import pathlib
import sys

base_path = pathlib.Path(sys.argv[1])
repeat = int(sys.argv[2])
vector_path = sys.argv[3]

with base_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg.setdefault("extractor", {})["flow_seed"] = 42 + repeat
cfg["save_vector"] = vector_path
cfg["include_baseline"] = False

if "model" in cfg:
    cfg["model"]["device"] = "cuda"

print(json.dumps(cfg, indent=2))
PY

    echo ""
    echo "--- Running Extraction $(basename "$config_path") | repeat=$repeat ---"
    python -m Steering.cli --task extract --config "$tmp_config"

    # Swap save_vector to load_vector for evaluation
    python - "$tmp_config" "$vector_path" > "${tmp_config}.eval" <<'PY'
import json
import sys
import pathlib

tmp_path = pathlib.Path(sys.argv[1])
vector_path = sys.argv[2]

with tmp_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg.pop("save_vector", None)
cfg["load_vector"] = vector_path

print(json.dumps(cfg, indent=2))
PY

    output_dir=$(jq -r '.output' "${tmp_config}.eval")
    if [ -d "$output_dir" ]; then
      rm -f "$output_dir"/*.json
    fi

    echo ""
    echo "--- Running Evaluation $(basename "$config_path") | repeat=$repeat ---"
    python -m Steering.cli --task eval --config "${tmp_config}.eval"
    
    local generated_json
    generated_json="$(find "$output_dir" -maxdepth 1 -name "eval_${config_name%.json}_*.json" | head -n 1 || true)"
    if [ -z "$generated_json" ]; then
      echo "ERROR: no eval JSON produced for $config_path"
      rm -rf "$output_dir"
      rm -f "$tmp_config" "${tmp_config}.eval"
      exit 1
    fi

    # Read accuracy and perplexity from final output
    local metrics
    metrics=$(python - "$generated_json" <<'PY2'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

result = data.get("result", data)
accuracy = result.get("accuracy", 0.0)
perplexity = result.get("perplexity", 0.0)
print(f"{accuracy} {perplexity}")
PY2
    )

    local acc
    local ppl
    read -r acc ppl <<< "$metrics"
    run_accuracies+=("$acc")
    run_perplexities+=("$ppl")

    rm -rf "$output_dir"
    rm -f "$tmp_config" "${tmp_config}.eval"
    sleep 5
  done

  # Calculate average metrics across repeats
  local accuracies_csv
  local perplexities_csv
  accuracies_csv="$(IFS=,; echo "${run_accuracies[*]}")"
  perplexities_csv="$(IFS=,; echo "${run_perplexities[*]}")"

  python - "$accuracies_csv" "$perplexities_csv" "$config_name" <<'PY3'
import sys

accuracies = [float(x) for x in sys.argv[1].split(",") if x]
perplexities = [float(x) for x in sys.argv[2].split(",") if x]
config_name = sys.argv[3]

mean_accuracy = sum(accuracies) / len(accuracies)
mean_perplexity = sum(perplexities) / len(perplexities) if perplexities else float("nan")

print("")
print(f"===========================================================")
print(f"--- Averaged Results for {config_name} over 3 repeats ---")
print(f"Accuracy:   {mean_accuracy:.4f}")
print(f"Perplexity: {mean_perplexity:.4f}")
print(f"===========================================================")
print("")
PY3
}

for cfg in "${CONFIGS[@]}"; do
  run_flow_repeats "$CONFIG_DIR/$cfg"
done
