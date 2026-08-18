"""
Shared utilities for steering methods.
"""

import string
from dataclasses import dataclass
from typing import List, Optional, Dict, Union, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm
from .logger import setup_logger

logger = setup_logger(__name__)


def _normalize_top_idx(top_idx: Any) -> List[int]:
    """Normalize top feature indices to a ranked Python list[int]."""
    if top_idx is None:
        return []

    if isinstance(top_idx, torch.Tensor):
        return [int(v) for v in top_idx.detach().cpu().reshape(-1).tolist()]

    if isinstance(top_idx, np.ndarray):
        return [int(v) for v in top_idx.reshape(-1).tolist()]

    if isinstance(top_idx, list):
        if not top_idx:
            return []
        if isinstance(top_idx[0], dict):
            return [
                int(item["feature_index"])
                for item in top_idx
                if isinstance(item, dict) and "feature_index" in item
            ]
        return [int(v) for v in top_idx]

    return [int(top_idx)]


def build_top_feature_tracker_from_stats(
    top_idx: Any,
    target_stats: Dict[str, Any],
    contrast_stats: Optional[Dict[str, Any]] = None,
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Build top-feature tracker from streaming summary stats.

    This avoids retaining raw [N, D] activation matrices only for metadata.
    """
    ranked = _normalize_top_idx(top_idx)[:top_n]
    if not ranked:
        return []

    target_sum = target_stats.get("sum")
    target_sum_sq = target_stats.get("sum_sq")
    target_active_count = target_stats.get("active_count")
    target_num_samples = int(target_stats.get("num_samples", 0))

    if (
        target_sum is None
        or target_sum_sq is None
        or target_active_count is None
        or target_sum.numel() == 0
        or target_num_samples <= 0
    ):
        return []

    d_model = int(target_sum.shape[0])
    target_count = float(max(target_num_samples, 1))

    use_contrast = contrast_stats is not None
    if use_contrast:
        contrast_sum = contrast_stats.get("sum")
        contrast_sum_sq = contrast_stats.get("sum_sq")
        contrast_active_count = contrast_stats.get("active_count")
        contrast_num_samples = int(contrast_stats.get("num_samples", 0))
        use_contrast = (
            contrast_sum is not None
            and contrast_sum_sq is not None
            and contrast_active_count is not None
            and contrast_sum.numel() > 0
            and contrast_num_samples > 0
        )

    rows: List[Dict[str, Any]] = []
    for rank, feature_index in enumerate(ranked, start=1):
        if feature_index < 0 or feature_index >= d_model:
            continue

        t_mean = float((target_sum[feature_index] / target_count).item())
        t_second_moment = float((target_sum_sq[feature_index] / target_count).item())
        t_var = max(t_second_moment - t_mean * t_mean, 0.0) ** 0.5
        t_freq = float((target_active_count[feature_index] / target_count).item())

        entry: Dict[str, Any] = {
            "rank": rank,
            "feature_index": int(feature_index),
            "target_activation_frequency": t_freq,
            "target_feature_mean": t_mean,
            "target_feature_value_variation": t_var,
        }

        if use_contrast:
            contrast_count = float(max(int(contrast_stats["num_samples"]), 1))
            c_mean = float((contrast_sum[feature_index] / contrast_count).item())
            c_second_moment = float((contrast_sum_sq[feature_index] / contrast_count).item())
            c_var = max(c_second_moment - c_mean * c_mean, 0.0) ** 0.5
            c_freq = float((contrast_active_count[feature_index] / contrast_count).item())

            entry.update(
                {
                    "contrast_activation_frequency": c_freq,
                    "contrast_feature_mean": c_mean,
                    "contrast_feature_value_variation": c_var,
                    "activation_frequency_delta": t_freq - c_freq,
                    "feature_mean_delta": t_mean - c_mean,
                    "feature_value_variation_delta": t_var - c_var,
                }
            )

        rows.append(entry)

    return rows

# =============================================================================
# CPU SAE Wrapper
# =============================================================================

class SAEWrapper:
    """
    Wraps an SAE and handles device/dtype transfer transparently.

    encode()/decode() accept tensors on any device/dtype, transfer to the
    SAE's device/dtype for computation, then return results on the original
    device with the original dtype. All other attribute access (W_dec, W_enc,
    b_enc, b_dec, cfg, etc.) is delegated to the underlying SAE.
    """

    def __init__(self, sae):
        object.__setattr__(self, '_sae', sae)
        try:
            p = next(sae.parameters())
            object.__setattr__(self, '_sae_device', p.device)
            object.__setattr__(self, '_sae_dtype', p.dtype)
        except StopIteration:
            object.__setattr__(self, '_sae_device', torch.device('cpu'))
            object.__setattr__(self, '_sae_dtype', torch.float32)

    def _transfer(self, fn, x):
        orig_device, orig_dtype = x.device, x.dtype
        sae_device = object.__getattribute__(self, '_sae_device')
        sae_dtype = object.__getattribute__(self, '_sae_dtype')
        result = fn(x.to(device=sae_device, dtype=sae_dtype))
        return result.to(device=orig_device, dtype=orig_dtype)

    def encode(self, x):
        return self._transfer(object.__getattribute__(self, '_sae').encode, x)

    def decode(self, x):
        return self._transfer(object.__getattribute__(self, '_sae').decode, x)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_sae'), name)

    def __deepcopy__(self, memo):
        import copy
        sae_copy = copy.deepcopy(object.__getattribute__(self, '_sae'), memo)
        return SAEWrapper(sae_copy)


# =============================================================================
# Shared SAE activation collection
# =============================================================================


@dataclass
class SAEActivationCollection:
    """Unified return payload for SAE activation collection."""

    activations: Dict[int, torch.Tensor]
    stats: Dict[int, Dict[str, Any]]


def collect_sae_activations(
    model,
    saes: Dict[int, "SAE"],
    texts: List[str],
    layers: List[int],
    hook_point: str = "pre",
    batch_size: int = 8,
    pooling: str = "last",
    device: Optional[str] = None,
    tokenizer=None,
    active_threshold: float = 0.0,
) -> SAEActivationCollection:
    """
    Collect SAE-encoded activations and streaming stats in one pass.

    Instead of looping over layers and running a separate forward pass for each,
    this caches all requested layers at once via `names_filter`.

    Args:
        model: HookedTransformer model
        saes: Dict mapping layer int -> SAE object
        texts: List of input strings
        layers: List of layer indices to collect from
        hook_point: Hook position ("pre", "post", "mid")
        batch_size: Batch size for processing
        pooling: How to reduce across sequence dimension:
            - "last":  last-token activations [n_samples, d_sae]
            - "max":   max-pool across sequence [n_samples, d_sae]
            - "mean":  mean-pool across sequence [n_samples, d_sae]
            - "mask":  masked-mean over semantic tokens [n_samples, d_sae] (requires tokenizer)
        device: Target device (defaults to model device)
        tokenizer: Required when pooling="mask"; used to tokenize and build the semantic token mask.
        active_threshold: Threshold for "active" feature counting in returned stats.

    Returns:
        SAEActivationCollection with:
        - activations: Dict[layer, Tensor[n_samples, d_sae]]
        - stats: Dict[layer, Dict[str, Any]]

        `stats[layer]` contains:
            - `num_samples` (int): total pooled samples
            - `sum` (Tensor[d_sae])
            - `sum_sq` (Tensor[d_sae])
            - `active_sum` (Tensor[d_sae]): sum over values where value > threshold
            - `active_count` (Tensor[d_sae]): count where value > threshold
    """
    if pooling == "none":
        raise ValueError("pooling='none' is not supported by stats-enabled SAE collection")

    device = device or getattr(model.cfg, "device", "cuda")
    hook_names = {layer: get_hook_name(layer, hook_point) for layer in layers}
    all_hook_names = list(hook_names.values())

    results = {layer: [] for layer in layers}
    stats: Dict[int, Dict[str, Any]] = {
        layer: {
            "num_samples": 0,
            "sum": None,
            "sum_sq": None,
            "active_sum": None,
            "active_count": None,
        }
        for layer in layers
    }

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch = texts[i : i + batch_size]

            if pooling == "mask":
                if tokenizer is None:
                    raise ValueError("tokenizer is required for 'mask' pooling mode")
                tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                token_mask = build_token_mask(
                    input_ids,
                    tokenizer=tokenizer,
                    attention_mask=attention_mask,
                )
                _, cache = model.run_with_cache(
                    input_ids,
                    names_filter=all_hook_names,
                    return_type=None,
                )
            else:
                token_mask = None
                _, cache = model.run_with_cache(
                    batch,
                    names_filter=all_hook_names,
                    return_type=None,
                )

            for layer in layers:
                acts = cache[hook_names[layer]].to(torch.float32)  # [B, seq, d_model]

                pooled_batch: Optional[torch.Tensor] = None

                # For single-token pooling modes ("last" or integer index), slice the
                # residual stream BEFORE SAE encoding. This avoids materialising the
                # full [B, seq, d_sae] intermediate tensor on GPU, which is the main
                # OOM trigger for long prompts (e.g. chat-templated CAA on Llama-3.1-8B).
                if pooling == "last" or isinstance(pooling, int):
                    pos = -1 if pooling == "last" else int(pooling)
                    seq_len = acts.shape[1]
                    resolved_pos = pos if pos >= 0 else seq_len + pos
                    if resolved_pos < 0 or resolved_pos >= seq_len:
                        raise ValueError(
                            f"Pooling position {pos} is out of bounds for sequence length {seq_len}."
                        )

                    sae_acts = saes[layer].encode(
                        acts[:, resolved_pos : resolved_pos + 1, :]
                    )  # [B, 1, d_sae]
                    pooled_batch = sae_acts[:, 0, :]
                    del sae_acts
                else:
                    sae_acts = saes[layer].encode(acts)  # [B, seq, d_sae]

                    if pooling == "max":
                        pooled_batch = sae_acts.max(dim=1).values
                    elif pooling == "mean":
                        pooled_batch = sae_acts.mean(dim=1)
                    elif pooling == "mask":
                        mask = token_mask.unsqueeze(-1).float()  # [B, seq, 1]
                        n_tokens = mask.sum(dim=1).clamp(min=1)  # [B, 1]
                        pooled_batch = (sae_acts * mask).sum(dim=1) / n_tokens
                    else:
                        raise ValueError(f"Unknown pooling mode: {pooling!r}")

                    del sae_acts

                if pooled_batch is not None:
                    pooled_cpu = pooled_batch.detach().to(torch.float32).cpu()
                    results[layer].append(pooled_cpu)

                    layer_stats = stats[layer]
                    if layer_stats["sum"] is None:
                        d_sae = pooled_cpu.shape[1]
                        # Extension point: add new per-feature collector metrics here
                        # and update the accumulation block below in the same shape.
                        layer_stats["sum"] = torch.zeros(d_sae, dtype=torch.float32)
                        layer_stats["sum_sq"] = torch.zeros(d_sae, dtype=torch.float32)
                        layer_stats["active_sum"] = torch.zeros(d_sae, dtype=torch.float32)
                        layer_stats["active_count"] = torch.zeros(d_sae, dtype=torch.float32)

                    active_mask = pooled_cpu > float(active_threshold)
                    layer_stats["num_samples"] += int(pooled_cpu.shape[0])
                    layer_stats["sum"] += pooled_cpu.sum(dim=0)
                    layer_stats["sum_sq"] += (pooled_cpu * pooled_cpu).sum(dim=0)
                    layer_stats["active_sum"] += (
                        pooled_cpu * active_mask.to(dtype=pooled_cpu.dtype)
                    ).sum(dim=0)
                    layer_stats["active_count"] += active_mask.sum(dim=0).to(
                        dtype=torch.float32
                    )

            del cache
            torch.cuda.empty_cache()

    outputs = {layer: torch.cat(results[layer], dim=0) for layer in layers}
    return SAEActivationCollection(activations=outputs, stats=stats)


def collect_dense_activations(
    model,
    texts: List[str],
    layers: List[int],
    hook_point: Union[str, List[str]] = "pre",
    batch_size: int = 8,
    pooling: Union[str, int] = "last",
    device: Optional[str] = None,
    tokenizer=None,
    reduce: str = "none",
    return_key_format: str = "auto",
    pretokenize_all: bool = False,
    change_pad_token: bool = False,
) -> Dict[Union[int, str], torch.Tensor]:
    """
    Collect dense residual activations for multiple layers/hook points.

    Args:
        model: HookedTransformer model.
        texts: Input strings.
        layers: Layer indices.
        hook_point: Residual hook point(s) to cache.
        batch_size: Batch size.
        pooling: "last", "mean", "all", "mask", integer index, or "none".
        device: Target device (defaults to model cfg device).
        tokenizer: Required for tokenized forward passes.
        reduce: "none" for per-sample outputs, "mean" for mean activation per key.
        return_key_format:
            - "auto": "layer" when one hook point, else "layer_hook"
            - "layer": int keys (e.g. 12)
            - "layer_hook": string keys (e.g. layer_12_mid)
        pretokenize_all: If True, tokenize all texts once for consistent padding.
        change_pad_token: If True, temporarily set tokenizer pad token id to 0.

    Returns:
        Dict of collected activations keyed by layer or layer+hook.
    """
    device = device or getattr(model.cfg, "device", "cuda")

    if not texts:
        return {k: torch.zeros(model.cfg.d_model, device=device) for k in (layers if return_key_format != "layer_hook" else [f"layer_{l}_{hook_point}" for l in layers])}

    hook_points = [hook_point] if isinstance(hook_point, str) else list(hook_point)
    if not hook_points:
        raise ValueError("hook_point must contain at least one position.")

    if return_key_format not in {"auto", "layer", "layer_hook"}:
        raise ValueError("return_key_format must be 'auto', 'layer', or 'layer_hook'.")

    if reduce not in {"none", "mean"}:
        raise ValueError("reduce must be 'none' or 'mean'.")

    resolved_key_format = return_key_format
    if resolved_key_format == "auto":
        resolved_key_format = "layer" if len(hook_points) == 1 else "layer_hook"

    hook_names: Dict[Union[int, str], str] = {}
    for layer in layers:
        for hp in hook_points:
            key: Union[int, str]
            if resolved_key_format == "layer_hook":
                key = f"layer_{layer}_{hp}"
            else:
                if len(hook_points) != 1:
                    raise ValueError(
                        "return_key_format='layer' requires exactly one hook point."
                    )
                key = layer
            hook_names[key] = get_hook_name(layer, hp)

    all_hook_names = list(hook_names.values())
    collected: Dict[Union[int, str], List[torch.Tensor]] = {k: [] for k in hook_names}

    original_pad_token_id = None
    if change_pad_token:
        if tokenizer is None:
            raise ValueError("tokenizer is required when change_pad_token=True")
        original_pad_token_id = tokenizer.pad_token_id
        tokenizer.pad_token_id = 0

    def _pool(acts: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pooling == "none":
            return acts
        if pooling == "all":
            if attention_mask is not None:
                mask = attention_mask.to(dtype=torch.bool)
                return acts[mask]
            else:
                return acts.flatten(0, 1)
        if pooling == "last":
            if attention_mask is not None and getattr(tokenizer, "padding_side", "left") == "right":
                last_indices = attention_mask.sum(dim=1) - 1
                last_indices = torch.clamp(last_indices, min=0)
                batch_indices = torch.arange(acts.size(0), device=acts.device)
                return acts[batch_indices, last_indices, :]
            else:
                return acts[:, -1, :]
        if pooling == "mean":
            return acts.mean(dim=1)
        if pooling == "mask":
            if attention_mask is None:
                return acts.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(dtype=acts.dtype)
            denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=acts.dtype)
            return (acts * mask).sum(dim=1) / denom
        if isinstance(pooling, int):
            return acts[:, pooling, :]
        raise ValueError(f"Unknown dense pooling mode: {pooling}")

    def _run_batch(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        _, cache = model.run_with_cache(
            input_ids,
            attention_mask=attention_mask,
            names_filter=all_hook_names,
            return_type=None,
        )
        return cache

    with torch.no_grad():
        if pretokenize_all:
            if tokenizer is None:
                raise ValueError("tokenizer is required when pretokenize_all=True")

            all_tokens = tokenizer(
                texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            all_input_ids = all_tokens["input_ids"]
            all_attention_mask = all_tokens.get("attention_mask")

            for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
                batch_ids = all_input_ids[i : i + batch_size].to(device)
                batch_mask = (
                    all_attention_mask[i : i + batch_size].to(device)
                    if all_attention_mask is not None
                    else None
                )
                cache = _run_batch(batch_ids, batch_mask)
                for key, hook_name in hook_names.items():
                    pooled = _pool(cache[hook_name].to(torch.float32), batch_mask)
                    collected[key].append(pooled.cpu())
                del cache
        else:
            for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
                batch = texts[i : i + batch_size]
                if tokenizer is None:
                    raise ValueError("tokenizer is required to collect dense activations")
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                cache = _run_batch(input_ids, attention_mask)
                for key, hook_name in hook_names.items():
                    pooled = _pool(cache[hook_name].to(torch.float32), attention_mask)
                    collected[key].append(pooled.cpu())
                del cache

            torch.cuda.empty_cache()

    if original_pad_token_id is not None:
        tokenizer.pad_token_id = original_pad_token_id

    merged = {key: torch.cat(chunks, dim=0) for key, chunks in collected.items()}
    if reduce == "mean":
        return {key: value.mean(dim=0).to(device) for key, value in merged.items()}
    return {key: value.to(device) for key, value in merged.items()}


def get_hook_name(layer: int, position: str = "pre") -> str:
    """
    Get the hook name for a specific layer and position.
    
    Args:
        layer: Layer number
        position: Hook position ("pre", "post", or "mid")
        
    Returns:
        Hook name string
    """
    if position == "mid":
        return f"blocks.{layer}.hook_resid_mid"
    elif position == "pre":
        return f"blocks.{layer}.hook_resid_pre"
    elif position == "post":
        return f"blocks.{layer}.hook_resid_post"
    elif position == "mlp_out":
        return f"blocks.{layer}.hook_mlp_out"
    elif position == "attn_out":
        return f"blocks.{layer}.hook_attn_out"
    elif position == "ln1":
        return f"blocks.{layer}.ln1.hook_normalized"
    elif position == "ln2":
        return f"blocks.{layer}.ln2.hook_normalized"
    else:
        raise ValueError(f"Unknown hook position: {position}")

def get_resid_acts(resid, position: Union[int, str]):
    """Extract token activations from a residual stream tensor.

    Args:
        resid: Tensor of shape (batch, seq, d_model).
        position: "last" → final token; "all" → full sequence;
                  "mean" → mean-pool over sequence; int → specific index.
    """
    if position == "last":
        return resid[:, -1, :]
    elif position == "all":
        return resid
    elif position == "mean":
        return resid.mean(dim=1)
    else:
        return resid[:, position, :]

def set_resid_acts(resid, position: Union[int,str], acts):
    if position == "last":
        resid = resid.clone()
        resid[:, -1, :] = acts
    elif position == "all":
        resid = acts
    elif position == "mean":
        resid = resid.clone()
        shift = acts - resid.mean(dim=1)
        resid = resid + shift.unsqueeze(1)
    else:
        resid = resid.clone()
        resid[:, position, :] = acts
    return resid

# =============================================================================
# Token classification for semantic masking
# =============================================================================

_TOKEN_PREFIXES = ("Ġ", "▁", "##", "Ċ")
_TOKEN_SUFFIXES = ("</w>",)
_SEMANTIC_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "by", "as", "at",
    "from", "that", "this", "it",
}


def _normalize_token_text(token_text: str) -> str:
    """Normalize tokenizer-specific token strings for semantic filtering."""
    normalized = token_text
    for prefix in _TOKEN_PREFIXES:
        while normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    for suffix in _TOKEN_SUFFIXES:
        while normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
    return normalized.strip().lower()


def _classify_semantic_tokens(tokenizer, token_ids: torch.Tensor) -> Dict[int, str]:
    """Classify unique token ids for semantic masking."""
    token_classes: Dict[int, str] = {}
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    for token_id in token_ids.unique().tolist():
        token_id = int(token_id)
        if token_id in special_ids:
            token_classes[token_id] = "special"
            continue

        token_text = tokenizer.convert_ids_to_tokens(token_id)
        if token_text is None:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)

        normalized = _normalize_token_text(token_text)

        if not normalized:
            token_classes[token_id] = "empty"
            continue

        if normalized in _SEMANTIC_STOPWORDS:
            token_classes[token_id] = "stopword"
            continue

        if all(char in string.punctuation for char in normalized):
            token_classes[token_id] = "punctuation"
            continue

        token_classes[token_id] = "semantic"

    return token_classes


def build_token_mask(
    input_ids: torch.Tensor,
    tokenizer,
    attention_mask: Optional[torch.Tensor] = None,
    exclude_bos: bool = True,
    exclude_stopwords: bool = True,
    exclude_punctuation: bool = True,
) -> torch.Tensor:
    """
    Build a mask for semantic tokens (excluding BOS, stopwords, punctuation).
    
    Args:
        input_ids: Token IDs [batch, seq]
        tokenizer: The tokenizer
        attention_mask: Optional attention mask [batch, seq]. Padding is always excluded when provided.
        exclude_bos: Exclude BOS token
        exclude_stopwords: Exclude common stopwords
        exclude_punctuation: Exclude punctuation tokens
        
    Returns:
        Boolean mask [batch, seq] where True = keep
    """
    device = input_ids.device
    if attention_mask is not None:
        mask = attention_mask.to(device=device, dtype=torch.bool).clone()
    else:
        mask = torch.ones_like(input_ids, dtype=torch.bool)

    if tokenizer.pad_token_id is not None:
        mask &= input_ids != tokenizer.pad_token_id

    if exclude_bos and tokenizer.bos_token_id is not None:
        mask &= input_ids != tokenizer.bos_token_id

    token_classes = _classify_semantic_tokens(tokenizer, input_ids)
    if exclude_stopwords:
        stopword_tensor = torch.tensor(
            [token_classes[int(token_id)] != "stopword" for token_id in input_ids.view(-1).tolist()],
            device=device,
            dtype=torch.bool,
        ).view_as(input_ids)
        mask &= stopword_tensor

    if exclude_punctuation:
        punctuation_tensor = torch.tensor(
            [token_classes[int(token_id)] != "punctuation" for token_id in input_ids.view(-1).tolist()],
            device=device,
            dtype=torch.bool,
        ).view_as(input_ids)
        mask &= punctuation_tensor

    special_tensor = torch.tensor(
        [token_classes[int(token_id)] not in {"special", "empty"} for token_id in input_ids.view(-1).tolist()],
        device=device,
        dtype=torch.bool,
    ).view_as(input_ids)
    mask &= special_tensor

    return mask


def build_chat_input(
    tokenizer,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    """
    Build chat-formatted input using tokenizer's chat template.
    
    Returns format like: [INST] {question} [/INST]
    (without BOS token to match CAA paper format)
    
    Args:
        tokenizer: The tokenizer with chat template
        user_prompt: User message
        system_prompt: Optional system message
        add_generation_prompt: Whether to append assistant generation prompt
        
    Returns:
        Formatted chat string (without BOS/EOS tokens)
    """
    if tokenizer.chat_template is None:
        return user_prompt.strip()
    
    # Clean the user_prompt like CAST library does
    cleaned_user_prompt = clean_text(user_prompt, tokenizer)
    cleaned_system_prompt = clean_text(system_prompt, tokenizer) if system_prompt else None
    
    messages = [{"role": "user", "content": cleaned_user_prompt}]
    if cleaned_system_prompt is not None:
        messages.insert(0, {"role": "system", "content": cleaned_system_prompt})
    
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    
    # Do NOT strip BOS and EOS tokens to match CAST library behavior
    # Paper uses: [INST] question [/INST] (without <s>)
    # if tokenizer.bos_token and result.startswith(tokenizer.bos_token):
    #     result = result[len(tokenizer.bos_token):]
    # if tokenizer.eos_token and result.endswith(tokenizer.eos_token):
    #     result = result[:-len(tokenizer.eos_token)]
    
    return result


def clean_text(text: str, tokenizer) -> str:
    """
    Clean the input text by replacing special tokens to avoid interference with chat templates.
    
    Matches the logic in activation_steering.steering_dataset.SteeringDataset.clean_text
    """
    if not text:
        return text

    def insert_vline(token: str) -> str:
        if len(token) < 2:
            return " "
        elif len(token) == 2:
            return f"{token[0]}|{token[1]}"
        else:
            return f"{token[:1]}|{token[1:-1]}|{token[-1:]}"

    bos_token = getattr(tokenizer, "bos_token", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    pad_token = getattr(tokenizer, "pad_token", None)
    unk_token = getattr(tokenizer, "unk_token", None)

    if bos_token:
        text = text.replace(bos_token, insert_vline(bos_token))
    if eos_token:
        text = text.replace(eos_token, insert_vline(eos_token))
    if pad_token:
        text = text.replace(pad_token, insert_vline(pad_token))
    if unk_token:
        text = text.replace(unk_token, insert_vline(unk_token))

    return text


def make_params(func, **kwargs):
    """
    Create arguments for a function/method from a config object/dict and extra kwargs.
    Only allows arguments that are present in the function signature.
    
    Args:
        func: The function to get signature from
        **kwargs: Extra arguments (take precedence over config)
        
    Returns:
        Dict of valid arguments for the function
    """
    import inspect
    
    
        
    # Merge configs (kwargs take precedence)
    merged = {**kwargs}
    
    # Filter by signature
    sig = inspect.signature(func)
    params = sig.parameters
    
    # Allow **kwargs pass-through if function accepts it
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_varkw:
        return merged
        
    # Otherwise filter strictly
    valid_args = {
        k: v for k, v in merged.items() 
        if k in params and k != "self"
    }
    return valid_args


def cache_logit_lens(model, sae, k: int = 20, batch_size: int = 32):
    """
    Compute logit lens top-k tokens for each SAE feature.
    
    Processes in batches to avoid OOM on large SAEs (e.g. 16k features × 256k vocab).
    Softmax + top-k is computed on CPU to reduce GPU memory pressure.
    
    Args:
        model: HookedTransformer model
        sae: SAE object with W_dec
        k: Number of top tokens to keep
        batch_size: Number of features to process per batch (default 32 for memory safety)
        
    Returns:
        topk: Torch object with values and indices of top k tokens per feature
    """
    device = model.cfg.device
    W_dec = sae.W_dec.to(device)
    n_latents = W_dec.shape[0]
    
    all_values = []
    all_indices = []
    
    for start in range(0, n_latents, batch_size):
        end = min(start + batch_size, n_latents)
        chunk = W_dec[start:end]
        
        # 1. Apply final layer norm
        if hasattr(model, "ln_final"):
            normed = model.ln_final(chunk)
        else:
            normed = chunk
            
        # 2. Unembed to logits — move to CPU immediately to save GPU memory
        logits = model.unembed(normed).float().cpu()
        
        # 3. Softmax + top-k on CPU
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, dim=-1, k=k)
        
        all_values.append(topk.values)
        all_indices.append(topk.indices)
        
        del chunk, logits, probs, topk, normed
        torch.cuda.empty_cache()
    
    values = torch.cat(all_values, dim=0)
    indices = torch.cat(all_indices, dim=0)
    
    return torch.return_types.topk((values, indices))


def get_output_score(
    model,
    sae,
    layer: int,
    feature_idx: int,
    logit_lens_top_tokens: List[int],
    prompt: str,
    amp_factor: float = 10.0,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute output score for a specific feature.
    
    Score = (1 - min_rank/vocab_size) * max_prob
    where min_rank is the best rank among logit_lens_top_tokens in the steered distribution.
    
    Args:
        model: HookedTransformer
        sae: SAE object
        layer: Layer number
        feature_idx: Index of feature to test
        logit_lens_top_tokens: Indices of top tokens from logit lens for this feature
        prompt: Neutral prompt to evaluate on (e.g. "From my experience,")
        amp_factor: Amplification factor for steering
        
    Returns:
        Output score (float)
    """
    device = device or model.cfg.device
    tokens = model.to_tokens(prompt)
    
    # Define hook to amplify feature using full SAE roundtrip with error preservation.
    # Reference: Code/saes-are-good-for-steering/sae_utils.py AmlifySAEHook.__call__
    def steer_hook(activations, hook):
        # activations: [batch, pos, d_model] (resid_post)
        orig_dtype = activations.dtype
        x = activations.to(torch.float32)
        B, T, D = x.shape
        
        # 1. Encode all positions through SAE (reshape for 2D SAE compatibility)
        x_flat = x.reshape(-1, D)
        sparse_flat = sae.encode(x_flat)
        recon_flat = sae.decode(sparse_flat)
        
        sparse = sparse_flat.reshape(B, T, -1)
        recon = recon_flat.reshape(B, T, D)
        
        # 2. Compute reconstruction error (in float64 per GT for precision)
        sae_error = x.to(torch.float64) - recon.to(torch.float64)
        
        # 3. Amplify target feature at last token
        # Reference: max_act_value = torch.max(feature_acts[:, -1, :]).item()
        #            feature_acts[:, -1, feature] += max_act_value * self.amp_factor
        max_act = sparse[:, -1, :].max().item()
        sparse[:, -1, feature_idx] += max_act * amp_factor
        
        # 4. Decode modified sparse activations and add error
        steered_flat = sae.decode(sparse.reshape(-1, sparse.shape[-1]))
        steered = steered_flat.reshape(B, T, D).to(torch.float64) + sae_error
        
        return steered.to(orig_dtype)

    hook_name = get_hook_name(layer, "post")
    
    # Run with hook
    try:
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, steer_hook)]
            )
        
        # Analyze output logits
        final_logits = logits[0, -1, :] # [d_vocab]
        probs = torch.softmax(final_logits, dim=0)
        
        # Calculate score
        vocab_size = len(probs)
        
        # 1. Get ranks of our target tokens (from logit lens) in the actual output distribution
        # "logit_lens_top_tokens" are the tokens the feature *should* promote
        
        # Optimization: use argsort
        sorted_indices = torch.argsort(probs, descending=True)
        
        min_rank = vocab_size # Default (worst case)
        
        # Find the best rank achieved by ANY of the logit lens top tokens
        for token_idx in logit_lens_top_tokens:
            rank = (sorted_indices == token_idx).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                current_rank = rank.item()
                if current_rank < min_rank:
                    min_rank = current_rank
        
        # 2. Get probability mass on the target tokens
        # Reference: top_token_score = torch.max(intervention_probs[logit_lens_indices]).item()
        top_token_score = 0.0
        if len(logit_lens_top_tokens) > 0:
            top_token_score = torch.max(probs[logit_lens_top_tokens]).item()
        
        # Combine
        rank_score = (1 - (min_rank / vocab_size))
        return rank_score * top_token_score
        
    except Exception as e:
        logger.error(f"Error computing output score for feature {feature_idx}: {e}")
        return 0.0


def load_fallback_sae(
    release: str, 
    sae_id: str, 
    layer: int, 
    model_config, 
    device
):
    """
    Download and patch SAE for sae_lens compatibility (SPARE/custom SAEs).
    
    Moved from pipeline.py to decouple loading logic.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file
    import json, gc, os
    from sae_lens import SAE

    # Setup paths
    cache_dir = os.path.join(os.path.expanduser("~/.cache/sae_lens"), release, sae_id)
    os.makedirs(cache_dir, exist_ok=True)
    
    cfg_path = hf_hub_download(repo_id=release, filename=f"{sae_id}/cfg.json", local_dir=cache_dir)
    sae_path = hf_hub_download(repo_id=release, filename=f"{sae_id}/sae.safetensors", local_dir=cache_dir)
    load_path = os.path.dirname(cfg_path)
    weights_path = os.path.join(load_path, "sae_weights.safetensors")

    # Patch config with sae_lens required fields
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    
    defaults = {
        "model_name": model_config.name,
        "hook_name": f"blocks.{layer}.hook_resid_pre",
        "hook_layer": layer,
        "hook_head_index": None,
        "dataset_path": "unknown",
        "context_size": 4096,
        "architecture": "topk",
        "activation_fn_str": "topk",
        "activation_fn_kwargs": {"k": cfg.get("k", 128)},
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_trust_remote_code": True,
        "normalize_activations": "none",
        "d_sae": cfg.get("d_in", 4096) * cfg.get("expansion_factor", 32),
        "dtype": model_config.dtype,
        "device": str(device),
    }
    
    merged = {**defaults, **cfg}
    if merged.keys() != cfg.keys():
        with open(cfg_path, "w") as f:
            json.dump(merged, f, indent=2)
        logger.info(f"Patched config: {cfg_path}")

    # Patch weights: SPARE format -> sae_lens format
    if not os.path.exists(weights_path):
        state = load_file(sae_path)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
            model_config.dtype, torch.bfloat16
        )
        
        key_map = {"encoder.weight": "W_enc", "encoder.bias": "b_enc"}
        patched = {}
        for k, v in state.items():
            if k == "encoder.weight":
                v = v.T.contiguous()
            patched[key_map.get(k, k)] = v.to(dtype)
        
        save_file(patched, weights_path)
        logger.info(f"Patched weights: {weights_path}")
        
        # Clear large dicts from memory
        del state
        del patched
        gc.collect()
        torch.cuda.empty_cache()

    return SAE.load_from_pretrained(load_path, device=str(device), dtype=model_config.dtype)


class FlopTracker:
    """
    Context manager to dynamically track the total FLOPs executed on the GPU
    using PyTorch's built-in FlopCounterMode dispatcher hooks.
    """
    def __init__(self):
        self.total_flops = 0
        self._mode = None

    def __enter__(self):
        try:
            from torch.utils.flop_counter import FlopCounterMode
            self._mode = FlopCounterMode(display=False)
            self._mode.__enter__()
        except Exception as e:
            logger.warning(f"Failed to initialize FlopCounterMode: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._mode:
            try:
                self._mode.__exit__(exc_type, exc_val, exc_tb)
                self.total_flops = self._get_total_flops()
            except Exception as e:
                logger.warning(f"Failed to exit FlopCounterMode or compute FLOPs: {e}")

    def _get_total_flops(self) -> int:
        if not self._mode or not hasattr(self._mode, "flop_counts"):
            return 0
        total = 0
        for counts in self._mode.flop_counts.values():
            if isinstance(counts, dict) or hasattr(counts, "values"):
                total += sum(counts.values())
        return total

