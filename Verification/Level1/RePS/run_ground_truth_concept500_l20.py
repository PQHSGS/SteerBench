"""
AxBench ground-truth runner for the Concept500 LoReFT l20 experiment.

This mirrors the repository command in Code/axbench/axbench/experiment_commands.txt:

    torchrun --nproc_per_node=1 --master_port=60001 axbench/scripts/train.py \
      --config axbench/sweep/wuzhengx/2b/l20/loreft.yaml \
      --dump_dir axbench/results/prod_2b_l20_concept500_loreft \
      --overwrite_data_dir axbench/concept500/prod_2b_l20_v1/generate \
      --run_name official

Usage:
    unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
    PYTHONPATH=. python Verification/Level1/RePS/run_ground_truth_concept500_l20.py --dry-run
    PYTHONPATH=. python Verification/Level1/RePS/run_ground_truth_concept500_l20.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import socket
from pathlib import Path
from typing import List


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def build_command(repo_root: Path, nproc_per_node: int, master_port: int, config: Path,
                  dump_dir: Path, overwrite_data_dir: Path, run_name: str) -> List[str]:
    return [
        "torchrun",
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_port",
        str(master_port),
        "axbench/scripts/train.py",
        "--config",
        str(config),
        "--dump_dir",
        str(dump_dir),
        "--overwrite_data_dir",
        str(overwrite_data_dir),
        "--run_name",
        run_name,
    ]
    


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AxBench Concept500 l20 ground-truth job.")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="official")
    # concept_id support removed; run the full Concept500 GT job by default
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    code_root = repo_root / "Code" / "axbench"
    config = code_root / "axbench" / "sweep" / "wuzhengx" / "2b" / "l20" / "loreft.yaml"
    dump_name = "prod_2b_l20_concept500_loreft"
    dump_dir = code_root / "axbench" / "results" / dump_name
    overwrite_data_dir = code_root / "axbench" / "concept500" / "prod_2b_l20_v1" / "generate"

    for path in (config, overwrite_data_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    master_port = args.master_port
    if master_port is None:
        master_port = 60001
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", master_port)) == 0:
                master_port = find_free_port()

    command = build_command(
        repo_root=repo_root,
        nproc_per_node=args.nproc_per_node,
        master_port=master_port,
        config=config,
        dump_dir=dump_dir,
        overwrite_data_dir=overwrite_data_dir,
        run_name=args.run_name,
    )

    print("AxBench Concept500 l20 ground-truth command:")
    print(" ".join(command))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    pythonpath_parts = [str(code_root), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    

    result = subprocess.run(command, cwd=str(code_root), env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
