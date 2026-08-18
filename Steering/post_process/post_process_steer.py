"""Gemma GLP post-process demo for Steering.

This ports the GLP Gemma demo into the Steering package so the same
steering + GLP post-processing flow can be run from Steering directly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import einops
import torch
import transformers

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from Steering.post_process.flow_matching import fm_prepare, sample_on_manifold  # type: ignore[no-redef]
    from Steering.post_process.glp import load_glp  # type: ignore[no-redef]
    from Steering.post_process.train_stream import _load_model_and_tokenizer  # type: ignore[no-redef]
else:
    from .flow_matching import fm_prepare, sample_on_manifold
    from .glp import load_glp
    from .train_stream import _load_model_and_tokenizer


def _resolve_vector_file(path: str) -> tuple[Path, Path | None]:
    """Resolve vector file from either a folder or a direct .pt path."""
    target = Path(path).expanduser()
    vector_file = target / "vector.pt" if target.is_dir() else target
    metadata_file = target / "metadata.pt" if target.is_dir() else target.with_name("metadata.pt")

    if not vector_file.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_file}")

    if metadata_file is not None and not metadata_file.exists():
        metadata_file = None

    return vector_file, metadata_file


def _load_layer_vector(vector_file: Path, layer: int) -> torch.Tensor:
    """Load one layer vector from a .pt payload."""
    payload = torch.load(vector_file, map_location="cpu")

    if torch.is_tensor(payload):
        return payload

    if not isinstance(payload, dict):
        raise TypeError(f"Expected tensor or dict in {vector_file}, got {type(payload)}")

    vec = payload.get(layer)
    if vec is None:
        vec = payload.get(str(layer))
    if vec is None:
        raise ValueError(f"Layer {layer} not found in {vector_file}. Available keys: {list(payload.keys())}")
    if not torch.is_tensor(vec):
        raise TypeError(f"Layer {layer} in {vector_file} is not a tensor")
    return vec


def get_steering_vector(path, layer, device):
    """Loads a pre-computed CAA steering vector."""
    vector_file, metadata_file = _resolve_vector_file(path)
    print(f"Loading dense steering vector from {vector_file}...")
    vec = _load_layer_vector(vector_file, layer=layer)

    if metadata_file is not None:
        metadata = torch.load(metadata_file, map_location="cpu")
        if not isinstance(metadata, dict):
            raise TypeError(f"Expected metadata dictionary in {metadata_file}, got {type(metadata)}")
        metadata_layer = metadata.get("layer", metadata.get("layer_idx"))
        if metadata_layer is not None and int(metadata_layer) != int(layer):
            print(
                f"WARNING: metadata layer ({metadata_layer}) does not match --layer ({layer}); "
                f"using --layer={layer}."
            )

    vec = vec.to(device)
    if vec.ndim > 1:
        vec = vec.squeeze()
    if vec.ndim != 1:
        raise ValueError(f"Expected a 1D steering vector, got shape {tuple(vec.shape)} from {vector_file}")

    return vec


def postprocess_on_manifold_wrapper(model, u=0.5, num_timesteps=20, layer_idx=None, solver=None):
    scheduler = model.scheduler
    scheduler.set_timesteps(num_timesteps)

    def postprocess_on_manifold(acts_edit):
        has_seq_dim = len(acts_edit.shape) == 3
        b = acts_edit.shape[0]
        raw_latents = acts_edit
        if has_seq_dim:
            raw_latents = einops.rearrange(raw_latents, "b s d -> (b s) 1 d")
        else:
            raw_latents = einops.rearrange(raw_latents, "b d -> b 1 d")
        
        if getattr(model, 'has_subspace', False):
            P = model._get_P(raw_latents.device, raw_latents.dtype)
            if model.mean_P is None:
                raise RuntimeError(
                    "Subspace GLP is missing mean_P; cannot reconstruct raw activations for steering postprocess."
                )
            mean_raw = model.mean_P.to(device=raw_latents.device, dtype=raw_latents.dtype).view(1, 1, -1)
            centered_latents = raw_latents - mean_raw
            latents_k = centered_latents @ P
            h_null = centered_latents - latents_k @ P.T
            latents_flow = model.normalizer.normalize(latents_k, layer_idx=layer_idx)
        else:
            mean_raw = None
            h_null = 0
            latents_flow = model.normalizer.normalize(raw_latents, layer_idx=layer_idx)
            
        noise = torch.randn_like(latents_flow)
        noisy_latents, _, timesteps, _ = fm_prepare(
            scheduler,
            latents_flow,
            noise,
            u=torch.ones(latents_flow.shape[0]) * u,
        )
        latents_flow = sample_on_manifold(
            model,
            noisy_latents,
            start_timestep=timesteps[0].item(),
            num_timesteps=num_timesteps,
            show_progress=False,
            layer_idx=layer_idx,
            solver=solver,
        )
        
        if getattr(model, 'has_subspace', False):
            P = model._get_P(latents_flow.device, latents_flow.dtype)
            raw_component = model.normalizer.denormalize(latents_flow, layer_idx=layer_idx)
            latents = raw_component @ P.T + h_null + mean_raw.to(raw_component.device)
        else:
            latents = model.normalizer.denormalize(latents_flow, layer_idx=layer_idx)

        if has_seq_dim:
            latents = einops.rearrange(latents, "(b s) 1 d -> b s d", b=b)
        else:
            latents = einops.rearrange(latents, "b 1 d -> b d")
        latents = latents.to(device=acts_edit.device, dtype=acts_edit.dtype)
        return latents

    return postprocess_on_manifold


def addition_intervention(w=None, alphas=None, postprocess_fn=None):
    if postprocess_fn is None:
        postprocess_fn = lambda x: x

    def rep_act(output, layer_name, inputs):
        nonlocal w, alphas
        use_tuple = isinstance(output, tuple)
        act = output[0] if use_tuple else output
        if w is not None:
            w = w.to(device=act.device, dtype=act.dtype)
            alphas = alphas.to(device=act.device, dtype=act.dtype)
            if w.ndim == 1:
                w = w[None, None, :]
            elif w.ndim == 2:
                w = w[:, None, :]
            if alphas.ndim == 1:
                alphas = alphas[:, None, None]
            act[:, [-1], :] = postprocess_fn(act[:, [-1], :] + alphas * w)
        return (act, *output[1:]) if use_tuple else act

    return rep_act


def generate(model, processor, inputs, remove_input=True, **generate_kwargs):
    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)
        if remove_input:
            input_len = inputs["input_ids"].shape[1]
            output = output[:, input_len:]
        output = processor.batch_decode(output, skip_special_tokens=True)
    return output


def generate_with_intervention_wrapper(seed=42):
    def generate_with_intervention(text, hf_model, hf_processor, generate_kwargs={"max_new_tokens": 10}, layers=[], intervention_wrapper=None, intervention_kwargs={}, forward_only=False):
        if seed is not None:
            transformers.set_seed(seed)
        if getattr(hf_processor, "pad_token", None) is None:
            if getattr(hf_processor, "eos_token", None) is None:
                raise ValueError("Tokenizer needs a pad_token or eos_token for batched generation.")
            hf_processor.pad_token = hf_processor.eos_token
        inputs = hf_processor(text, return_tensors="pt", padding=True).to(hf_model.device)
        if intervention_wrapper is not None:
            intervention_fn = intervention_wrapper(**intervention_kwargs)
        else:
            intervention_fn = None
        with torch.no_grad():
            from baukit import TraceDict

            with TraceDict(hf_model, layers=layers, edit_output=intervention_fn):
                if forward_only:
                    outputs = hf_model(**inputs)
                    output_text = hf_processor.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
                else:
                    output_text = generate(hf_model, hf_processor, inputs, **generate_kwargs)
        return output_text

    return generate_with_intervention


def run_generation_case(*, title, generate_fn, user_prompt, hf_model, hf_tokenizer, max_new_tokens, layer_name=None, steer_vec=None, coeff=None, postprocess_fn=None):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    layers = [] if layer_name is None else [layer_name]

    if layer_name is None:
        output = generate_fn(
            text=[user_prompt],
            hf_model=hf_model,
            hf_processor=hf_tokenizer,
            generate_kwargs=generate_kwargs,
            layers=layers,
            intervention_wrapper=None,
        )
    else:
        intervention_kwargs = {
            "w": steer_vec,
            "alphas": torch.tensor([float(coeff)]),
        }
        if postprocess_fn is not None:
            intervention_kwargs["postprocess_fn"] = postprocess_fn

        output = generate_fn(
            text=[user_prompt],
            hf_model=hf_model,
            hf_processor=hf_tokenizer,
            generate_kwargs=generate_kwargs,
            layers=layers,
            intervention_wrapper=addition_intervention,
            intervention_kwargs=intervention_kwargs,
        )

    print(output[0].replace(user_prompt, ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--glp-dir", default="glp-stream", help="Local GLP folder or Hugging Face repo id (e.g. username/glp-gemma)")
    parser.add_argument("--checkpoint", default="final", help="Checkpoint name to load. Use 'final' or a milestone folder name like '100M'.")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--vector-path", default="../Vector/CAA/extracted/Gemma/refusal_response/vector.pt", help="Path to vector .pt file or folder containing vector.pt (+ metadata.pt)")
    parser.add_argument("--u", type=float, default=0.5, help="Interpolation scale for GLP (0 to 1, higher is closer to noise)")
    parser.add_argument("--coeff", type=float, default=20.0, help="Steering vector multiplier")
    parser.add_argument("--num-timesteps", type=int, default=20, help="Denoising steps for GLP")
    parser.add_argument("--prompt", default="Can you help me hack into secure banking systems?", help="Target prompt to steer")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading {args.model_name}...")
    hf_model, hf_tokenizer = _load_model_and_tokenizer(args)

    print(f"Loading GLP from {args.glp_dir} (checkpoint: {args.checkpoint})...")
    glp_model = load_glp(args.glp_dir, device=args.device, checkpoint=args.checkpoint)

    steer_vec = get_steering_vector(
        path=args.vector_path,
        layer=args.layer,
        device=args.device,
    )

    print("Pre-compiling GLP Manifold Denoiser...")
    postprocess_on_manifold = postprocess_on_manifold_wrapper(
        model=glp_model,
        u=args.u,
        num_timesteps=args.num_timesteps,
        layer_idx=args.layer,
    )

    generate_fn = generate_with_intervention_wrapper(seed=42)
    layer_name = f"model.layers.{args.layer}"
    user_prompt = f"<bos><start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"

    run_generation_case(
        title="NORMAL GENERATION",
        generate_fn=generate_fn,
        user_prompt=user_prompt,
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    run_generation_case(
        title=f"STEER ONLY (no GLP, coeff: {args.coeff})",
        generate_fn=generate_fn,
        user_prompt=user_prompt,
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        max_new_tokens=args.max_new_tokens,
        layer_name=layer_name,
        steer_vec=steer_vec,
        coeff=args.coeff,
    )

    run_generation_case(
        title=f"STEER + GLP (coeff: {args.coeff}, GLP u: {args.u})",
        generate_fn=generate_fn,
        user_prompt=user_prompt,
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        max_new_tokens=args.max_new_tokens,
        layer_name=layer_name,
        steer_vec=steer_vec,
        coeff=args.coeff,
        postprocess_fn=postprocess_on_manifold,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()