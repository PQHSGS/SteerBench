"""
Unified CLI for Steering Experiments.

Usage:
    # Run steering generation
    python -m Steering.cli --task run --config experiment.json
    
    # Run steering evaluation
    python -m Steering.cli --task eval --config experiment.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from .pipeline import SteeringPipeline
from .config import PipelineConfig
from .logger import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Steering experiments CLI")
    
    # Required: config file
    parser.add_argument("--config", "-c", required=True, help="Config file (JSON)")
    
    # Task selection
    parser.add_argument(
        "--task", "-t",
        choices=["run", "eval", "extract"], 
        default="eval",
        help="Task: 'run' for generation, 'eval' for evaluation, 'extract' for vector extraction"
    )
    
    # Optional overrides
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimize output")
    
    args = parser.parse_args()
    
    # Load config
    config = PipelineConfig.load(args.config)
    
    logger.info(f"Task: {args.task.upper()}")
    logger.info(f"Method: {config.extractor.method} L{config.extractor.layer} c={config.steer.coeff}")
    
    # Create pipeline from config
    pipeline = SteeringPipeline(config)
    
    if args.task == "extract":
        # Explicitly trigger extraction when --task extract is passed
        pipeline._setup_vector_from_extraction()
        logger.info("Extraction complete.")
        return
        
    logger.info(f"Test: {config.test_dataset}")
    
    if args.task == "eval":
        result = pipeline.evaluate(verbose=not args.quiet)
        stat_msg = f", ppl: {result.perplexity:.2f}, repeat: {result.repetition_rate:.2f}, compress: {result.compression_ratio:.2f}"
        logger.info(f"Result: {result.accuracy:.2%} (delta: {result.delta:+.2%}{stat_msg})")
        results_to_save = {"config": config.to_dict(method_scoped=True), "result": result.to_dict()}
        prefix = "eval"
    else:
        output_data = pipeline.run()
        results_to_save = output_data
        prefix = "run"
    
    # Save output
    output_path = args.output or getattr(config, 'output', None)
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        coeff = str(list(config.steer.coeff.values())[0]) if isinstance(config.steer.coeff, dict) else str(config.steer.coeff)
        config.name += f"_coeff_{coeff.replace('.', 'p')}"
        out_file = Path(output_path) / f"{prefix}_{config.name}_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(results_to_save, f, indent=2, default=str)
        logger.info(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
