#!/bin/bash

# ==============================================================================
# MMLU Capability Preservation Evaluation Script
# ==============================================================================
# Evaluates steered model performance on the MMLU test dataset using
# logit-difference evaluation. No training is performed; pre-extracted
# vectors are loaded dynamically.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e
export CUDA_VISIBLE_DEVICES=5
# Python binary to use (sae_circuit conda environment python)
PYTHON_BIN="/data/caotue/anaconda3/envs/sae_circuit/bin/python"

# Full list of methods to evaluate
METHODS=(
  # "CAA" "SAS" "SPARE" "Angular" "CAST" "CORRSTEER" "CURVE" 
  # "FGAA" "FLAS" "FLOW" "LINEARACT" "MANIFOLD" "REFT" 
  # "SAECOT" "SAEFREE" "SAEIO"
  #  "SPHERICAL" "SRPS"
    "WEIGHTSTEER"
)

# Tasks/Vectors to evaluate
TASKS=("refusal_response" )

# Steering coefficients to test (0.0 represents unsteered/baseline behavior)
COEFFS=(0.5 1 2 3)

# Directory to store results
OUTPUT_DIR="./Results/mmlu_eval"
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "Starting MMLU Evaluation Sweep..."
echo "Python Bin:  $PYTHON_BIN"
echo "Methods:     ${METHODS[*]}"
echo "Tasks:       ${TASKS[*]}"
echo "Coeffs:      ${COEFFS[*]}"
echo "Output Dir:  $OUTPUT_DIR"
echo "================================================================================"

for method in "${METHODS[@]}"; do
  for task in "${TASKS[@]}"; do
    for coeff in "${COEFFS[@]}"; do
      
      # Resolve correct load_vector path based on method and task
      if [ "$method" = "FLAS" ]; then
        load_vector="/data/caotue/FLAS/Gemma/$task"
      elif [ "$method" = "FLOW" ]; then
        load_vector="Vector/Flow/Gemma/$task"
      elif [ "$method" = "LINEARACT" ]; then
        load_vector="Vector/LinearAcT/Gemma/$task"
      elif [ "$method" = "REFT" ]; then
        if [ "$task" = "refusal_response" ]; then
          load_vector="Vector/REPS/refusal_response"
        else
          load_vector="Vector/REPS/Gemma/$task"
        fi
      elif [ "$method" = "WEIGHTSTEER" ]; then
        load_vector="/data/caotue/WEIGHTSTEER/$task"
      else
        load_vector="Vector/$method/Gemma/$task"
      fi

      # Check if vector folder exists
      if [ -z "$load_vector" ] || [ ! -d "$load_vector" ]; then
        echo "[-] Skipping combination: Method=$method, Task=$task (vector folder '$load_vector' not found)"
        continue
      fi

      echo "[+] Running: Method=$method | Task=$task | Coeff=$coeff"

      # Generate temp_eval.json dynamically using Python
      "$PYTHON_BIN" -c "
import json, sys
method, task, coeff, load_vector, output_dir = sys.argv[1:]
config_mapping = {
    'FLAS': 'Configs/Eval/FLAS/gemma_mmlu.json',
    'FLOW': 'Configs/Eval/FLOW/gemma_mmlu.json',
    'REFT': 'Configs/Eval/REFT/gemma_mmlu.json',
    'WEIGHTSTEER': 'Configs/Eval/WEIGHTSTEER/gemma_mmlu.json',
    'LINEARACT': 'Configs/Eval/LinearAcT/Gemma/gemma_mmlu.json',
    'Angular': 'Configs/Eval/Angular/Gemma/gemma_mmlu.json'
}
config_path = config_mapping.get(method, f'Configs/Eval/{method}/Gemma/gemma_mmlu.json')
with open(config_path) as f:
    config = json.load(f)
config['name'] = f'gemma_mmlu_{method.lower()}_{task}'
config['load_vector'] = load_vector
config['steer']['coeff'] = float(coeff)
config['output'] = output_dir
with open('temp_eval.json', 'w') as f:
    json.dump(config, f, indent=2)
" "$method" "$task" "$coeff" "$load_vector" "$OUTPUT_DIR"

      # Run steering evaluation
      "$PYTHON_BIN" -m Steering.cli --task eval --config temp_eval.json

    done
  done
done

# Clean up temp configuration file
rm -f temp_eval.json

echo "================================================================================"
echo "MMLU Evaluation Sweep Completed! Results saved in $OUTPUT_DIR"
echo "================================================================================"
