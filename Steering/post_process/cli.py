"""CLI for GLP post-process workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from Steering.post_process.train_classifier import (  # type: ignore[no-redef]
        ClassifierTrainConfig,
        train_classifier,
    )
    from Steering.post_process.train_stream import (  # type: ignore[no-redef]
        StreamTrainConfig,
        stream_train,
    )
    # Subspace training utilities removed from main CLI imports.
    from Steering.post_process.paths import get_data_root
else:
    from .train_classifier import ClassifierTrainConfig, train_classifier
    from .train_stream import StreamTrainConfig, stream_train
    # Subspace training utilities removed from main CLI imports.
    from .paths import get_data_root


def _build_stream_train_parser(subparsers):
    parser = subparsers.add_parser("stream", help="Stream activations and train GLP")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--layer-prefix", default="model.layers")
    parser.add_argument("--retain", choices=["input", "output"], default="output")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--token-idx", choices=["last", "all", "random_doc"], default="all")
    parser.add_argument("--drop-bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--padding-side", choices=["left", "right"], default="right")
    parser.add_argument("--document-batch-size", type=int, default=16)
    parser.add_argument("--forward-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase-switch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-device", default="cpu")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--storage-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")

    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-documents", type=int, default=50000)

    parser.add_argument("--stream-chunk-size", type=int, default=1000000)
    parser.add_argument("--total-steps", type=int, default=244)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--normalization-method", default="gaussian")
    parser.add_argument("--noise-sampling-method", choices=["uniform", "sot", "sinkhorn"], default="uniform")
    parser.add_argument("--u-sampling-method", choices=["uniform", "beta", "logit_normal"], default="uniform")
    parser.add_argument("--ot-chunk-size", type=int, default=256)
    parser.add_argument("--gradient-clipping-threshold", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--tail-variance-proportion", type=float, default=0.05)
    parser.add_argument("--split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--split-proportion", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--initial-factor", type=float, default=0.01)
    parser.add_argument("--final-factor", type=float, default=0.1)
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-ckpt", default=None)
    parser.add_argument("--load-opt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler-type", choices=["cosine", "linear"], default="cosine")

    parser.add_argument("--save-root", default=str(get_data_root() / "GLP"))
    parser.add_argument("--run-name", default="glp-stream")
    parser.add_argument("--checkpoint-token-step", type=int, default=100000000)
    parser.add_argument("--denoiser-layers", type=int, default=3)
    parser.add_argument("--d-model-mult", type=int, default=2)
    parser.add_argument("--d-mlp-mult", type=int, default=4)
    # Subspace-related CLI options removed; GLP training operates on full activation space.

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="glp")
    parser.add_argument("--cache-dataset", action=argparse.BooleanOptionalAction, default=False)


def _build_classifier_train_parser(subparsers):
    parser = subparsers.add_parser(
        "train-classifier",
        help="Train classifier guidance model from Steering contrastive datasets",
    )
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--hook-point", default="pre")
    parser.add_argument("--position", default="last")
    parser.add_argument("--sequence-pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")

    parser.add_argument("--dataset-name", default="sycophancy")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--prompt-batch-size", type=int, default=16)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-mlp", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--t-embed-dim", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--noise-rate", type=float, default=0.5)
    parser.add_argument("--timestep-min", type=float, default=0.0)
    parser.add_argument("--timestep-max", type=float, default=1.0)

    parser.add_argument("--save-root", default=str(get_data_root() / "GLP"))
    parser.add_argument("--run-name", default="classifier-stream")

    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="glp")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-log-every-steps", type=int, default=10)


def _build_subspace_train_parser(subparsers):
    raise RuntimeError("Subspace training CLI is deprecated. Use 'stream' for full-space GLP training.")


def main():
    parser = argparse.ArgumentParser(description="GLP post-process utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_stream_train_parser(subparsers)
    _build_classifier_train_parser(subparsers)
    args = parser.parse_args()

    if args.command == "stream":
        cfg = StreamTrainConfig(
            model_name=args.model_name,
            layer=args.layer,
            layer_prefix=args.layer_prefix,
            retain=args.retain,
            device=args.device,
            torch_dtype=args.torch_dtype,
            storage_dtype=args.storage_dtype,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_field=args.text_field,
            max_documents=args.max_documents,
            max_length=args.max_length,
            token_idx=args.token_idx,
            drop_bos=args.drop_bos,
            padding_side=args.padding_side,
            document_batch_size=args.document_batch_size,
            forward_batch_size=args.forward_batch_size,
            stream_chunk_size=args.stream_chunk_size,
            total_steps=args.total_steps,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            normalization_method=args.normalization_method,
            noise_sampling_method=args.noise_sampling_method,
            u_sampling_method=args.u_sampling_method,
            ot_chunk_size=args.ot_chunk_size,
            gradient_clipping_threshold=args.gradient_clipping_threshold,
            grad_accum=args.grad_accum,
            log_every_n_steps=args.log_every_n_steps,
            tail_variance_proportion=args.tail_variance_proportion,
            split=args.split,
            split_proportion=args.split_proportion,
            warmup_ratio=args.warmup_ratio,
            initial_factor=args.initial_factor,
            final_factor=args.final_factor,
            use_bf16=args.use_bf16,
            shuffle=args.shuffle,
            init_ckpt=args.init_ckpt,
            load_opt=args.load_opt,
            seed=args.seed,
            scheduler_type=args.scheduler_type,
            save_root=args.save_root,
            run_name=args.run_name,
            checkpoint_token_step=args.checkpoint_token_step,
            denoiser_layers=args.denoiser_layers,
            d_model_mult=args.d_model_mult,
            d_mlp_mult=args.d_mlp_mult,
            phase_switch=args.phase_switch,
            offload_device=args.offload_device,
            wandb=args.wandb,
            wandb_project=args.wandb_project,
            cache_dataset=args.cache_dataset,
        )
        summary = stream_train(cfg)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "train-classifier":
        cfg = ClassifierTrainConfig(
            model_name=args.model_name,
            layer=args.layer,
            hook_point=args.hook_point,
            position=args.position,
            sequence_pooling=args.sequence_pooling,
            device=args.device,
            model_dtype=args.model_dtype,
            dataset_name=args.dataset_name,
            n_train=args.n_train,
            prompt_batch_size=args.prompt_batch_size,
            d_model=args.d_model,
            d_mlp=args.d_mlp,
            n_layers=args.n_layers,
            t_embed_dim=args.t_embed_dim,
            seed=args.seed,
            num_epochs=args.num_epochs,
            train_batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            val_ratio=args.val_ratio,
            noise_rate=args.noise_rate,
            timestep_min=args.timestep_min,
            timestep_max=args.timestep_max,
            save_root=args.save_root,
            run_name=args.run_name,
            wandb_enabled=args.wandb_enabled,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_log_every_steps=args.wandb_log_every_steps,
        )
        summary = train_classifier(cfg)
        print(json.dumps(summary, indent=2))
        return

    # Subspace training command removed; use 'stream' for full-space GLP training.

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
