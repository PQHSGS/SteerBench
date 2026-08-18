"""Activation collection utilities for GLP training input."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterator, Optional

from datasets import load_dataset
import numpy as np
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

from ..utils import get_hook_name
from .paths import get_data_root


@dataclass
class FineWebConfig:
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: Optional[str] = "sample-10BT"
    split: str = "train"
    text_field: str = "text"
    streaming: bool = True
    max_documents: Optional[int] = None


@dataclass
class ActivationCollectionConfig:
    model_name: str = "google/gemma-2-2b-it"
    output_dir: str = str(get_data_root() / "glp_acts")
    layer: int = 14
    hook_point: str = "pre"
    max_length: int = 1024
    batch_size: int = 4
    max_vectors: Optional[int] = 1_000_000
    drop_bos: bool = True
    storage_dtype: str = "float32"
    device: str = "cuda"
    model_dtype: str = "bfloat16"
    source: FineWebConfig = field(default_factory=FineWebConfig)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["source"] = asdict(self.source)
        return payload


class RunningMoments:
    """Numerically stable running mean and variance for activation vectors."""

    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return

        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.dim:
            raise ValueError(f"Expected batch shape [n, {self.dim}], got {batch.shape}")

        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + (delta ** 2) * self.count * batch_count / total
        self.count = total

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("No samples observed")
        var = np.maximum(self.m2 / self.count, 1e-8)
        return self.mean.astype(np.float32), var.astype(np.float32)


@dataclass(kw_only=True)
class MemmapWriter:
    output_dir: Path
    file_size: int
    dtype: np.dtype

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memmap_files = []
        self.indices: list[tuple[int, int, int]] = []
        self.cur_idx = 0
        self._new_memmap_file()

    def _new_memmap_file(self):
        path = self.output_dir / f"data_{len(self.memmap_files):04d}.npy"
        self.memmap_files.append(
            np.memmap(
                mode="w+",
                filename=path,
                dtype=self.dtype,
                shape=self.file_size,
            )
        )
        self.cur_idx = 0

    def write(self, row: np.ndarray) -> None:
        if row.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {row.dtype}")
        length, = row.shape
        if length > self.file_size:
            raise ValueError("Row larger than file size")

        if self.cur_idx + length > self.file_size:
            self._new_memmap_file()

        file_idx = len(self.memmap_files) - 1
        start = self.cur_idx
        end = start + length
        self.memmap_files[file_idx][start:end] = row
        self.cur_idx = end
        self.indices.append((file_idx, start, end))

    def flush(self) -> None:
        for mm in self.memmap_files:
            mm.flush()
        np.save(self.output_dir / "data_indices.npy", np.array(self.indices, dtype=np.uint64))


@dataclass
class MemmapReader:
    data_dir: Path
    dtype: np.dtype

    def __post_init__(self):
        self.indices = np.load(self.data_dir / "data_indices.npy")
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.indices)

    def _get_memmap(self, file_idx: int):
        if file_idx not in self._cache:
            path = self.data_dir / f"data_{file_idx:04d}.npy"
            self._cache[file_idx] = np.memmap(filename=path, mode="r", dtype=self.dtype)
            if len(self._cache) > 3:
                self._cache.popitem(last=False)
        return self._cache[file_idx]

    def __getitem__(self, idx: int) -> np.ndarray:
        file_idx, start, end = self.indices[idx]
        return self._get_memmap(int(file_idx))[int(start):int(end)]


def parse_model_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported model_dtype: {name}")


def parse_storage_dtype(name: str) -> tuple[np.dtype, str]:
    if name == "float32":
        return np.dtype(np.float32), "float32"
    if name == "float16":
        return np.dtype(np.float16), "float16"
    if name == "bfloat16":
        return np.dtype(np.int16), "int16"
    raise ValueError(f"Unsupported storage_dtype: {name}")


def to_storage_array(vectors: torch.Tensor, storage_dtype: str) -> np.ndarray:
    if storage_dtype == "bfloat16":
        return vectors.to(torch.bfloat16).view(torch.int16).cpu().numpy()
    if storage_dtype == "float16":
        return vectors.to(torch.float16).cpu().numpy()
    if storage_dtype == "float32":
        return vectors.to(torch.float32).cpu().numpy()
    raise ValueError(f"Unsupported storage_dtype: {storage_dtype}")


def iter_stream_texts(cfg: FineWebConfig) -> Iterator[str]:
    dataset = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.split,
        streaming=cfg.streaming,
    ) if cfg.dataset_config else load_dataset(
        cfg.dataset_name,
        split=cfg.split,
        streaming=cfg.streaming,
    )

    seen = 0
    for row in dataset:
        text = row.get(cfg.text_field)
        if isinstance(text, str) and text.strip():
            yield " ".join(text.split())
            seen += 1
            if cfg.max_documents is not None and seen >= cfg.max_documents:
                break


def collect_layer_token_vectors(
    *,
    model: HookedTransformer,
    texts: list[str],
    layer: int,
    hook_point: str,
    max_length: int,
    drop_bos: bool,
) -> torch.Tensor:
    hook_name = get_hook_name(layer, hook_point)
    outputs = []

    with torch.no_grad():
        for text in texts:
            tokens = model.to_tokens(text, prepend_bos=True)
            if max_length and tokens.shape[1] > max_length:
                tokens = tokens[:, :max_length]

            _, cache = model.run_with_cache(tokens, names_filter=[hook_name], return_type=None)
            acts = cache[hook_name][0].detach().to(torch.float32).cpu()  # [seq, d]
            if drop_bos and acts.shape[0] > 1:
                acts = acts[1:, :]
            outputs.append(acts)
            del cache

    if not outputs:
        return torch.empty(0, model.cfg.d_model)

    return torch.cat(outputs, dim=0)


def save_rep_statistics(stats: RunningMoments, output_path: Path) -> None:
    mean, var = stats.finalize()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": torch.from_numpy(mean), "var": torch.from_numpy(var)}, output_path)


def collect_activations(config: ActivationCollectionConfig) -> dict:
    """Stream text and save token activations in GLP memmap format."""
    dtype = parse_model_dtype(config.model_dtype)
    model = HookedTransformer.from_pretrained(
        config.model_name,
        device=config.device,
        dtype=dtype,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_size = int(model.cfg.d_model)
    file_size = max(1, config.batch_size * hidden_size * 512)
    np_dtype, dtype_label = parse_storage_dtype(config.storage_dtype)

    writer = MemmapWriter(output_dir=output_dir, file_size=file_size, dtype=np_dtype)
    (output_dir / "dtype.txt").write_text(dtype_label, encoding="utf-8")

    stats = RunningMoments(hidden_size)

    texts = []
    docs_processed = 0
    vectors_written = 0

    for text in tqdm(iter_stream_texts(config.source), desc="Collecting activations"):
        texts.append(text)
        docs_processed += 1
        if len(texts) < config.batch_size:
            continue

        vectors = collect_layer_token_vectors(
            model=model,
            texts=texts,
            layer=config.layer,
            hook_point=config.hook_point,
            max_length=config.max_length,
            drop_bos=config.drop_bos,
        )
        texts = []

        if vectors.numel() == 0:
            continue

        if config.max_vectors is not None:
            remaining = config.max_vectors - vectors_written
            if remaining <= 0:
                print("Reached max_vectors limit, stopping collection.")
                break
            vectors = vectors[:remaining]

        storage = to_storage_array(vectors, config.storage_dtype)
        stats.update(vectors.numpy())
        for row in storage:
            writer.write(np.ascontiguousarray(row))

        vectors_written += int(storage.shape[0])
        if config.max_vectors is not None and vectors_written >= config.max_vectors:
            break

    if texts and (config.max_vectors is None or vectors_written < config.max_vectors):
        vectors = collect_layer_token_vectors(
            model=model,
            texts=texts,
            layer=config.layer,
            hook_point=config.hook_point,
            max_length=config.max_length,
            drop_bos=config.drop_bos,
        )
        if vectors.numel() > 0:
            if config.max_vectors is not None:
                remaining = config.max_vectors - vectors_written
                vectors = vectors[: max(0, remaining)]
            if vectors.numel() > 0:
                storage = to_storage_array(vectors, config.storage_dtype)
                stats.update(vectors.numpy())
                for row in storage:
                    writer.write(np.ascontiguousarray(row))
                vectors_written += int(storage.shape[0])

    writer.flush()
    rep_stats_path = output_dir / "rep_statistics.pt"
    save_rep_statistics(stats, rep_stats_path)

    summary = {
        "model_name": config.model_name,
        "layer": config.layer,
        "hook_point": config.hook_point,
        "hidden_size": hidden_size,
        "documents_processed": docs_processed,
        "vectors_written": vectors_written,
        "output_dir": str(output_dir),
        "rep_statistics": str(rep_stats_path),
    }
    with open(output_dir / "collection_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
