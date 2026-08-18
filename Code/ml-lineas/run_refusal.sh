#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/data/caotue/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/caotue/hf_cache/hub
export TRANSFORMERS_CACHE=/data/caotue/hf_cache/transformers
export HF_DATASETS_CACHE=/data/caotue/hf_cache/datasets
export TORCH_HOME=/data/caotue/torch_cache
export TMPDIR=/data/caotue/tmp

source /data/caotue/anaconda3/bin/activate sae_circuit
cd /home/caotue/SAESteeringBench/Code/ml-lineas

python -m lineas.scripts.pipeline \
  task_params=refusal_response \
  wandb.mode=disabled \
  evaluation='["text_generation"]' \
  data_dir=/data/caotue/lineas-data \
  cache_dir=/data/caotue/lineas-cache \
  results_dir=/data/caotue/lineas-results \
  intervention_params.optimization_params.steps=2000 \
  intervention_params.optimization_params.proximal=l1 \
  intervention_params.optimization_params.regularization_l1_weight=0.01 \
  intervention_params.optimization_params.regularization_l2_weight=0.001 \
  intervention_params.hook_params.intervention_position=last \
  interventions.batch_size=32 \
  interventions.force_small_batch=true \
  responses.batch_size=32 \
  interventions.device=cuda \
  responses.device=cuda \
  model.model_path=google/gemma-2-2b
