#!/bin/bash
set -e
unset CUDA_VISIBLE_DEVICES
cd /mnt/disk1/pquan/SAESteeringBench

echo "=== KL Decay 01 ==="
python Experiments/KL_decay/01_kl_decay.py --all_tasks --all_methods 2>&1 | tee Experiments/KL_decay/01_run.log
echo "=== 01 DONE ==="
sleep 10

echo "=== KL Decay 02 ==="
python Experiments/KL_decay/02_multi_token.py --all_tasks --all_methods 2>&1 | tee Experiments/KL_decay/02_run.log
echo "=== 02 DONE ==="
sleep 10

echo "=== DLA 02 ==="
python Experiments/DLA/02_full_experiment.py --all_tasks --all_methods 2>&1 | tee Experiments/DLA/02_run.log
echo "=== DLA 02 DONE ==="
sleep 10

echo "=== DLA 03 ==="
python Experiments/DLA/03_special_cases.py --all_tasks 2>&1 | tee Experiments/DLA/03_run.log
echo "=== ALL DONE ==="
