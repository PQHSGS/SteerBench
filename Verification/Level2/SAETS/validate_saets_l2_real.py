
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
import sys
from functools import partial

# Add parent directory to path (Benchmark root)
# Updated for Verification/Level2/SAETS/validate_saets_l2_real.py location
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from Steering.pipeline import SteeringPipeline
from Steering.config import PipelineConfig, ModelConfig, ExtractorConfig, SteerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Code/SAE-TS/src to path for reference implementation
sae_ts_path = str(Path("/home/aiotlab/mnt/hoplt/Benchmark/Code/SAE-TS/src"))
if sae_ts_path not in sys.path:
    sys.path.append(sae_ts_path)

try:
    from sae_ts.steering.patch import patch_resid
    logger.info("Successfully imported patch_resid from Code/SAE-TS")
except ImportError as e:
    logger.error(f"Failed to import patch_resid: {e}")
    sys.exit(1)

# Reference implementation is now IMPORTED, not mocked.

def main():
    logger.info("Starting Level 2 (Distribution) Validation...")
    
    # 1. Setup Paths and Config
    torch.manual_seed(42)
    device = "cuda:2"
    base_dir = Path("/home/aiotlab/mnt/hoplt/Benchmark/Verification_Results")
    vector_path = base_dir / "saets_vector_real.pt"
    
    if not vector_path.exists():
        logger.error("Vector file not found!")
        return

    # 2. Load Model (wrapper)
    logger.info("Loading Model...")
    model_cfg = ModelConfig(
        name="google/gemma-2-2b",
        device=device,
        dtype="bfloat16",
        use_compile=False
    )
    # Dummy config just to get the model loaded via pipeline
    # Note: PipelineConfig valdiation requires train_dataset if extractor is set
    pipeline = SteeringPipeline(
        PipelineConfig(
            model=model_cfg,
            extractor=ExtractorConfig(method="SAE-TS", layer=[12]),
            steer=SteerConfig(method="SAE-TS", layer=[12], coeff=1.0),
            output=str(base_dir),
            load_vector=str(vector_path),
            train_dataset="dummy",
            test_dataset="dummy"
        )
    )
    pipeline.load_model("sae")
    pipeline.setup()
    model = pipeline.model
    tokenizer = model.tokenizer

    # 3. Load Vector
    logger.info("Loading Vector...")
    loaded = torch.load(vector_path, map_location=device)
    layer = 12
    # Ensure vector is 1D [d_model]
    vector = loaded['steering_vector'][layer].detach().float()
    if vector.dim() > 1:
        vector = vector.squeeze()
        
    logger.info(f"Vector shape: {vector.shape}")

    # 4. Define Test Input
    prompt = "I want to kill myself."
    input_ids = model.to_tokens(prompt)
    
    # Steering parameters
    STEER_COEFF = 50.0  # Use a strong coefficient to ensure steering effect is visible
    
    # helper to get logits
    def get_logits_with_hooks(hooks):
        model.reset_hooks()
        try:
            with model.hooks(fwd_hooks=hooks):
                logits = model(input_ids)
                return logits[0, -1, :] # Last token logits
        finally:
            model.reset_hooks()

    # 5. Run "My Method": SteeringPipeline
    logger.info("Running My Method (SteeringPipeline)...")
    
    # We update the pipeline config's steer coefficient and use the pipeline to generate
    pipeline.config.steer.coeff = STEER_COEFF
    pipeline.steer_model.auto_scale = False # Explicitly override just in case
    
    # pipeline.generate() applies steering if apply_steer=True
    # To get logits, we can just use the pipeline's internal mechanism or generate with hooks
    # If we need logits, we can let pipeline set up the hooks and then forward pass
    pipeline.steer_model.setup_hooks({12: STEER_COEFF})
    
    with torch.no_grad():
        logits_mine = model(input_ids)[0, -1, :]
        
    model.reset_hooks()
    
    # 6. Run "Reference Method": patch_resid_ref
    logger.info("Running Reference Method (patch_resid)...")
    
    # Construct hook
    # hook_point = get_hook_name(12, "post") ? No, usually "blocks.12.hook_resid_post"
    # Or cleaner: model.cfg.layer_names...
    # Pipeline uses: get_hook_name(layer, "post") from utils
    from Steering.utils import get_hook_name
    hook_name = get_hook_name(12, "post")
    
    # Partial for reference
    # Note: reference patch_resid takes `scale`. computed `steering` * `scale`.
    # input `steering` should be the vector.
    # We cast vector to model dtype just in case
    vec_ref = vector.to(dtype=model.cfg.dtype)
    
    ref_hook_fn = partial(patch_resid, steering=vec_ref, scale=STEER_COEFF)
    
    logits_ref = get_logits_with_hooks([(hook_name, ref_hook_fn)])

    # 7. Compare
    logger.info("Comparing Results...")
    
    # Probabilities
    probs_mine = F.softmax(logits_mine, dim=-1)
    probs_ref = F.softmax(logits_ref, dim=-1)
    
    # Top 5 tokens
    topk_mine = torch.topk(probs_mine, 5)
    topk_ref = torch.topk(probs_ref, 5)
    
    logger.info("My Method Top 5:")
    for idx, p in zip(topk_mine.indices, topk_mine.values):
        logger.info(f"  {model.to_string(idx)}: {p.item():.4f}")
        
    logger.info("Reference Method Top 5:")
    for idx, p in zip(topk_ref.indices, topk_ref.values):
        logger.info(f"  {model.to_string(idx)}: {p.item():.4f}")

    # Metrics
    # KL Divergence
    # kl_div(input=log_probs_mine, target=probs_ref)
    log_probs_mine = F.log_softmax(logits_mine, dim=-1)
    kl = F.kl_div(log_probs_mine, probs_ref, reduction='batchmean').item()
    
    # Max Absolute Difference
    max_diff = (probs_mine - probs_ref).abs().max().item()
    
    logger.info(f"KL Divergence: {kl:.8f}")
    logger.info(f"Max Prob Diff: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        logger.info("Level 2 Validation (Distribution): PASSED (Identical outputs)")
    else:
        logger.warning("Level 2 Validation (Distribution): WARNING (Outputs differ slightly)")
        
    # Also compare vectors just to be sane
    # Check if 'vec' used in my hook is same as 'vec_ref'
    # steer_model.steering_vector[12]
    vec_mine = pipeline.steer_model.steering_vector[12]
    sim = F.cosine_similarity(vec_mine.unsqueeze(0), vec_ref.unsqueeze(0)).item()
    logger.info(f"Vector Cosine Sim (Double Check): {sim:.8f}")

if __name__ == "__main__":
    main()
