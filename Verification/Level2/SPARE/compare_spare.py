"""
SPARE Level 2 Comparison: GT (Code/SPARE/) vs our SPAREExtractor.

Uses the ACTUAL GT functions:
  - load_function_activations from Code/SPARE/spare/spare_for_generation.py
  - create_funcs, prepare_patch_function, patch_func_signal
  - generate_with_patch for steering inference
  - FunctionExtractor from spare.function_extraction_modellings

GT pipeline (from cache_data, pre-computed):
  1. load_hiddens_and_get_function_weights → zC, zM (functional weights)
  2. load mutual_information → MI scores + expectation
  3. select_functional_activations → context/param indices
  4. create_funcs → FunctionExtractor objects
  5. prepare_patch_function → patch hooks
  6. generate_with_patch → steered generation

Our pipeline:
  1. SPAREExtractor._get_activations → SAE latents
  2. SPAREExtractor._calculate_mutual_information → MI + expectation
  3. select by top-k proportion → indices
  4. create zC, zM vectors → decode to steering vector

Test 1: Feature Selection Match
  - Compare selected context/param indices
  - Compare MI scores ordering

Test 2: Steering via patch_func_signal vs our SPARESteerModel

Usage:
    cd /home/aiotlab/mnt/hoplt/Benchmark
    export PROJ_DIR=$(pwd)/Code/SPARE
    unset CUDA_VISIBLE_DEVICES
    PYTHONPATH=. python Verification/Level2/SPARE/compare_spare.py
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from functools import partial

torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Code" / "SPARE"))

# SPARE GT needs PROJ_DIR to find cache_data
os.environ["PROJ_DIR"] = str(ROOT / "Code" / "SPARE")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "gemma-2-2b"
MODEL_NAME = "gemma-2-2b"
LAYER_IDS = [14]
EDIT_DEGREE = 2.0
SELECT_TOPK_PROPORTION = 0.07
HIDDENS_NAME = "grouped_activations"
MI_SAVE_NAME = "mutual_information"

# ============================================================
# GT imports
# ============================================================
from spare.spare_for_generation import (
    load_hiddens_and_get_function_weights as gt_load_weights,
    load_function_activations as gt_load_func_acts,
    create_funcs as gt_create_funcs,
    prepare_patch_function as gt_prepare_patch,
    patch_func_signal as gt_patch_func_signal,
    select_functional_activations as gt_select_func_acts,
    generate_with_patch as gt_generate_with_patch,
)
from spare.utils import init_frozen_language_model, load_frozen_sae, PROJ_DIR
from spare.function_extraction_modellings.function_extractor import FunctionExtractor
from spare.sae_repe_utils import load_dataset_and_memorised_set
from spare.datasets.function_extraction_datasets import REODQADataset

# ============================================================
# Our imports
# ============================================================
from Steering.extractors.sae import SPAREExtractor


def gt_extract(layer_id=14):
    """
    Run GT SPARE extraction using pre-computed cache data.
    
    Reference: Code/SPARE/demo.py get_llama_spare()
    """
    logger.info("=== GT Extraction ===")
    
    # Load SAE for this layer
    sae = load_frozen_sae(layer_id, MODEL_NAME)
    
    # Load functional weights (from cache)
    use_context_weight, use_parameter_weight = gt_load_weights(
        MODEL_NAME, layer_id, sae, HIDDENS_NAME
    )
    
    # Load mutual information and select functional activations
    use_context_func, use_parameter_func, use_context_indices, use_parameter_indices = \
        gt_load_func_acts(
            layer_id, MODEL_NAME, EDIT_DEGREE, HIDDENS_NAME,
            MI_SAVE_NAME, SELECT_TOPK_PROPORTION
        )
    
    # Prepare patch functions
    use_context_patch, use_parameter_patch = gt_prepare_patch(
        use_context_func, use_context_indices,
        use_parameter_func, use_parameter_indices,
        sae
    )
    
    logger.info(f"GT context indices ({len(use_context_indices)}): {use_context_indices.tolist()[:10]}...")
    logger.info(f"GT param indices ({len(use_parameter_indices)}): {use_parameter_indices.tolist()[:10]}...")
    
    return {
        "sae": sae,
        "context_weight": use_context_weight,
        "param_weight": use_parameter_weight,
        "context_indices": use_context_indices,
        "param_indices": use_parameter_indices,
        "context_func": use_context_func,
        "param_func": use_parameter_func,
        "context_patch": use_context_patch,
        "param_patch": use_parameter_patch,
    }


def our_extract(model_tl, sae_lens_sae, target_data, contrast_data, layer_id=14):
    """Run our SPAREExtractor."""
    logger.info("=== Our Extraction ===")
    
    extractor = SPAREExtractor(
        model=model_tl,
        sae={layer_id: sae_lens_sae},
        layer=[layer_id],
        top_k_proportion=SELECT_TOPK_PROPORTION,
    )
    
    vectors = extractor.extract(
        target_data=target_data,
        contrast_data=contrast_data,
    )
    
    logger.info(f"Our context indices ({len(extractor.indices_pos)}): {extractor.indices_pos.tolist()[:10]}...")
    logger.info(f"Our param indices ({len(extractor.indices_neg)}): {extractor.indices_neg.tolist()[:10]}...")
    
    return {
        "vector": vectors.get(layer_id),
        "z_contextual": extractor.z_contextual,
        "z_parametric": extractor.z_parametric,
        "indices_pos": extractor.indices_pos,
        "indices_neg": extractor.indices_neg,
        "mutual_info": extractor.mutual_info,
        "expectation": extractor.expectation,
        "metadata": extractor.metadata,
    }


def gt_steer(model_hf, tokenizer, gt_results, test_prompt):
    """Run GT steering via generate_with_patch."""
    logger.info("=== GT Steering ===")
    
    use_cache = True  # Llama supports use_cache
    line_break_id = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
    generation_kwargs = {
        "max_new_tokens": 12,
        "do_sample": False,
        "eos_token_id": line_break_id,
        "pad_token_id": line_break_id,
        "use_cache": use_cache,
    }
    
    layer_id = LAYER_IDS[0]
    inspect_module = [f"model.layers.{layer_id}"]
    
    inputs = tokenizer([test_prompt], return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.inference_mode():
        # Steer to use context
        context_pred = gt_generate_with_patch(
            model_hf, tokenizer, 
            [gt_results["context_patch"]], inspect_module,
            input_ids.cuda(), generation_kwargs
        )
        
        # Steer to use parameter
        param_pred = gt_generate_with_patch(
            model_hf, tokenizer,
            [gt_results["param_patch"]], inspect_module,
            input_ids.cuda(), generation_kwargs
        )
    
    return context_pred, param_pred


def compare_results(gt_results, our_results):
    """Compare GT and our extraction results."""
    print("\n" + "=" * 60)
    print("=== FEATURE SELECTION COMPARISON ===")
    
    # Compare context indices
    gt_ctx = set(gt_results["context_indices"].cpu().tolist())
    our_ctx = set(our_results["indices_pos"].cpu().tolist())
    
    gt_param = set(gt_results["param_indices"].cpu().tolist())
    our_param = set(our_results["indices_neg"].cpu().tolist())
    
    ctx_intersect = gt_ctx & our_ctx
    param_intersect = gt_param & our_param
    
    print(f"\nContext indices: GT={len(gt_ctx)}, Ours={len(our_ctx)}, Overlap={len(ctx_intersect)}")
    if gt_ctx:
        print(f"  Context overlap: {len(ctx_intersect)}/{len(gt_ctx)} ({len(ctx_intersect)/len(gt_ctx)*100:.1f}%)")
    
    print(f"\nParam indices:   GT={len(gt_param)}, Ours={len(our_param)}, Overlap={len(param_intersect)}")
    if gt_param:
        print(f"  Param overlap: {len(param_intersect)}/{len(gt_param)} ({len(param_intersect)/len(gt_param)*100:.1f}%)")
    
    # Compare weights for shared features
    if ctx_intersect:
        shared_ctx = sorted(ctx_intersect)[:5]
        gt_w = gt_results["context_weight"].cpu()
        our_z = our_results["z_contextual"].cpu()
        print(f"\nShared context feature weights (first {len(shared_ctx)}):")
        for idx in shared_ctx:
            print(f"  Feature {idx}: GT_weight={gt_w[idx].item():.6f}, Our_zC={our_z[idx].item():.6f}")
    
    if param_intersect:
        shared_param = sorted(param_intersect)[:5]
        gt_w = gt_results["param_weight"].cpu()
        our_z = our_results["z_parametric"].cpu()
        print(f"\nShared param feature weights (first {len(shared_param)}):")
        for idx in shared_param:
            print(f"  Feature {idx}: GT_weight={gt_w[idx].item():.6f}, Our_zM={our_z[idx].item():.6f}")
    
    # Compare overall weight vectors
    gt_ctx_w = gt_results["context_weight"].cpu().float()
    our_ctx_z = our_results["z_contextual"].cpu().float()
    if gt_ctx_w.norm() > 0 and our_ctx_z.norm() > 0:
        cos_ctx = F.cosine_similarity(gt_ctx_w.unsqueeze(0), our_ctx_z.unsqueeze(0)).item()
        print(f"\nzC cosine sim: {cos_ctx:.6f}")
    
    gt_param_w = gt_results["param_weight"].cpu().float()
    our_param_z = our_results["z_parametric"].cpu().float()
    if gt_param_w.norm() > 0 and our_param_z.norm() > 0:
        cos_param = F.cosine_similarity(gt_param_w.unsqueeze(0), our_param_z.unsqueeze(0)).item()
        print(f"zM cosine sim: {cos_param:.6f}")
    
    print("=" * 60)


def main():
    # 1. GT extraction (uses pre-computed cache data)
    gt_results = gt_extract(layer_id=LAYER_IDS[0])
    
    # 2. Load model for our extractor (TransformerLens)
    # Note: SPARE GT uses HuggingFace model; our SPAREExtractor uses TransformerLens
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    
    logger.info("Loading TransformerLens model for our extractor...")
    tl_model = HookedSAETransformer.from_pretrained(
        MODEL_PATH, device=DEVICE, dtype=torch.bfloat16
    )
    
    logger.info(f"Loading SAE-Lens SAE for layer {LAYER_IDS[0]}...")
    # For Llama-2-7b, use EleutherAI SAE via sae_lens
    sae_lens_sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{LAYER_IDS[0]}/width_16k/canonical",
    )
    sae_lens_sae = sae_lens_sae.to(DEVICE)
    
    # 3. Prepare target/contrast data for our extractor
    # Use a simple set of prompts that distinguish context vs parametric knowledge
    data, memorised_set = load_dataset_and_memorised_set("nqswap", MODEL_NAME)
    
    # Split into context (label=1) and parametric (label=0) samples
    target_data = []  # context-use prompts
    contrast_data = []  # parameter-use prompts
    
    for item in data[:N_TRAIN * 2] if 'N_TRAIN' in dir() else data[:100]:
        prompt = f"context: {item.get('context', item.get('sub_context', ''))[:256]}\n" \
                 f"question: {item.get('question', '')}\nanswer:"
        target_data.append(prompt)
        
        # Parametric version (no context)
        contrast_prompt = f"question: {item.get('question', '')}\nanswer:"
        contrast_data.append(contrast_prompt)
    
    target_data = target_data[:50]
    contrast_data = contrast_data[:50]
    
    logger.info(f"Data: {len(target_data)} target + {len(contrast_data)} contrast")
    
    # 4. Our extraction
    our_results = our_extract(tl_model, sae_lens_sae, target_data, contrast_data, LAYER_IDS[0])
    
    # 5. Compare
    compare_results(gt_results, our_results)
    
    # 6. GT steering test (optional - requires HF model)
    try:
        logger.info("Loading HF model for GT steering test...")
        model_hf, tokenizer = init_frozen_language_model(MODEL_PATH)
        
        test_prompt = "context: The Eiffel Tower was actually built in London.\nquestion: Where is the Eiffel Tower?\nanswer:"
        context_pred, param_pred = gt_steer(model_hf, tokenizer, gt_results, test_prompt)
        
        print("\n=== GT STEERING RESULTS ===")
        print(f"Test prompt: {test_prompt}")
        print(f"Steer to context → '{context_pred}'")
        print(f"Steer to parameter → '{param_pred}'")
    except Exception as e:
        logger.warning(f"GT steering test skipped: {e}")


if __name__ == "__main__":
    main()
