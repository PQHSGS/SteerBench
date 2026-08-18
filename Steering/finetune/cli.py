"""CLI for finetune baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from Steering.finetune.config import FinetuneConfig  # type: ignore[no-redef]
    from Steering.finetune.registry import list_methods  # type: ignore[no-redef]
    from Steering.finetune.trainer import FinetuneTrainer  # type: ignore[no-redef]
else:
    from .config import FinetuneConfig
    from .registry import list_methods
    from .trainer import FinetuneTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune baseline CLI")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--method", default=None, help="Override finetune method name")
    parser.add_argument("--backend", default=None, help="Override backend name")
    parser.add_argument("--show-methods", action="store_true", help="List registered finetune methods")
    args = parser.parse_args()

    if args.show_methods:
        print(json.dumps({"methods": list_methods()}, indent=2))
        return

    config = FinetuneConfig.load(args.config)
    if args.method is not None:
        config.method_name = args.method
    if args.backend is not None:
        config.backend = args.backend

    trainer = FinetuneTrainer(config)
    summary = trainer.fit()
    # print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()