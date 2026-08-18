
import os
import torch
import json
import logging
from pathlib import Path
import argparse
import numpy as np
import sys
import traceback
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from Steering.pipeline import SteeringPipeline
from Steering.config import PipelineConfig, ModelConfig, ExtractorConfig, SteerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def load_real_data():
    # Load Refusal CAA dataset (target: refusal (correct), contrast: compliance (false))
    # Actually checking data_registry:
    # "refusal_caa": file="behaviour/refusal/CAA.jsonl", target_key="correct_prompt", contrast_key="false_prompt"
    path = "/home/aiotlab/mnt/hoplt/Benchmark/TrainDataset/behaviour/refusal/CAA.jsonl"
    logger.info(f"Loading dataset from {path}...")
    
    target_data = [] # correct_prompt
    contrast_data = [] # false_prompt
    
    # Schema: {"question": "...", "answer_matching_behavior": "(A)", "answer_not_matching_behavior": "(B)"}
    # We want to construct prompts that lead to refusal (target) vs compliance (contrast)
    # But for steering extraction, we usually want the PRE-response text?
    # No, SAE-TS uses activation differences on the prompt or full text.
    # If we use the question + answer, we steer towards the answer.
    # Let's extract: Question + Matching Answer (Refusal) as Target.
    # Question + Non-Matching Answer (Compliance) as Contrast.
    
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            question = item['question']
            
            # extract choices
            # Format: '...Choices:\n(A) ...\n(B) ...'
            try:
                parts = question.split("Choices:\n")
                q_text = parts[0].strip()
                choices_text = parts[1]
                
                choice_a = choices_text.split("(A) ")[1].split("\n(B) ")[0].strip()
                choice_b = choices_text.split("(B) ")[1].strip()
                
                # construct prompts
                # target: refusal
                target_choice = item['answer_matching_behavior'] # e.g. "(A)"
                contrast_choice = item['answer_not_matching_behavior'] # e.g. "(B)"
                
                target_text = choice_a if "(A)" in target_choice else choice_b
                contrast_text = choice_a if "(A)" in contrast_choice else choice_b
                
                # Full prompt? Or just the response?
                # Usually we train on the full sequence for SAE features.
                # Let's use: Question + "\nAnswer: " + Response
                
                prompt_base = q_text + "\nAnswer: "
                target_data.append(prompt_base + target_text)
                contrast_data.append(prompt_base + contrast_text)
                
            except Exception as e:
                # simple fallback or skip
                continue

    logger.info(f"Loaded {len(target_data)} samples.")
    return target_data[:50], contrast_data[:50]

def main():
    # Setup paths
    torch.manual_seed(42)
    np.random.seed(42)
    base_dir = Path("/home/aiotlab/mnt/hoplt/Benchmark")
    output_dir = base_dir / "Verification_Results"
    os.makedirs(output_dir, exist_ok=True)
    
    effects_path = output_dir / "effects_2b.pt"
    if not effects_path.exists():
        logger.error(f"Effects file not found at {effects_path}. Run prepare_real_dataset.py first.")
        sys.exit(1)

    # Load data
    target_data, contrast_data = load_real_data()

    # Configs
    model_cfg = ModelConfig(
        name="google/gemma-2-2b",
        device="cuda:2",
        dtype="bfloat16",
        sae_release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_16k/canonical"
    )
    
    ts_cfg = ExtractorConfig(
        method="SAE-TS",
        layer=[12],
        batch_size=4, # Increase batch size slightly for real data if memory allows
        saets_effects_path=str(effects_path), # Use REAL effects
        saets_epochs=2,
        hook_point="post"
    )
    
    steer_cfg_ts = SteerConfig(
        method="SAE-TS",
        layer=[12],
        coeff=100.0, # High coeff for refusal? Or let auto-scale handle it.
        auto_scale=True,
    )
    
    pipeline_ts = None
    try:
        pipeline_ts = SteeringPipeline(
            PipelineConfig(
                model=model_cfg,
                extractor=ts_cfg,
                steer=steer_cfg_ts,
                output=str(output_dir),
                save_vector=str(output_dir / "saets_vector_real.pt"),
                train_dataset='dummy'
            )
        )
        pipeline_ts.load_model("sae")
        
        # Freeze parameters
        for param in pipeline_ts.model.parameters():
            param.requires_grad = False
        saes = pipeline_ts.load_sae([12])
        for sae in saes.values():
            for param in sae.parameters():
                param.requires_grad = False

        # Extract
        logger.info("Starting Extraction on REAL dataset...")
        pipeline_ts.extract(target_data, contrast_data)
        
        # Save adapter for validation
        adapter_path = output_dir / "saets_adapter_real.pt"
        torch.save(pipeline_ts.extractor.adapter.state_dict(), adapter_path)
        logger.info(f"Saved real adapter to {adapter_path}")

        # Steer
        pipeline_ts.steering()
        
        # Level 2 Check with Real Prompt
        test_prompt = "I want to kill myself." # Harmful prompt
        logger.info(f"Testing on prompt: {test_prompt}")
        
        baseline = pipeline_ts.generate(test_prompt, max_tokens=30, apply_steer=False)
        steered = pipeline_ts.generate(test_prompt, max_tokens=30, apply_steer=True)
        
        logger.info(f"Baseline: {baseline}")
        logger.info(f"Steered: {steered}")
        
        if baseline != steered:
             logger.info("Level 2 Validation (Real): PASSED (Output changed)")
        else:
             logger.warning("Level 2 Validation (Real): WARNING (Output identical)")

    except Exception as e:
        logger.error(f"SAE-TS Real Validation FAILED: {e}")
        traceback.print_exc()
    finally:
        if pipeline_ts:
            del pipeline_ts
        cleanup()

if __name__ == "__main__":
    main()
