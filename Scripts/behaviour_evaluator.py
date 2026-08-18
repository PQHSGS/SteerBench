#!/usr/bin/env python3
"""
Coherence Evaluator for Model Outputs

This script uses the CoherenceMatcher from Steering/evaluators/model_capabilities.py
to evaluate the coherence of model responses from JSON result files.

Usage:
    python coherence_evaluator.py <file> [--output <output_file>]
    
Example:
    python coherence_evaluator.py Results/cast/eval_cast_llama_sorrybench_20260223_005605.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add the parent directory to the path to import from Steering
sys.path.insert(0, str(Path(__file__).parent))

from Steering.evaluators.scoring_metrics import BehaviorMatcher


def load_json_results(json_path: str) -> dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_samples(data: dict) -> Tuple[List[dict], List[dict]]:
    """
    Extract samples and baseline_samples from the result data.
    
    Returns:
        Tuple of (samples, baseline_samples)
    """
    result = data.get('result', {})
    samples = result.get('samples', [])
    baseline_samples = result.get('baseline_samples', [])
    
    return samples, baseline_samples


def evaluate_score(
    samples: List[dict],
    matcher: BehaviorMatcher,
    verbose: bool = False
) -> List[dict]:
    """
    Evaluate coherence for a list of samples.
    
    Args:
        samples: List of sample dictionaries containing 'prompt' and 'response'
        matcher: CoherenceMatcher instance
        verbose: Whether to print progress
    
    Returns:
        List of results with coherence scores
    """
    results = []
    
    for i, sample in enumerate(samples):
        if verbose:
            print(f"Processing sample {i+1}/{len(samples)}...")
        
        prompt = sample.get('prompt', '')
        response = sample.get('response', '')
        
        # Use CoherenceMatcher to evaluate
        # The check method returns (is_correct, confidence) where confidence is score/5.0
        is_correct, confidence = matcher.check(response, ground_truth=None, prompt=prompt)
        
        result = {
            'index': i,
            'prompt': prompt,
            'response': response,
            'is_correct': is_correct,
            'confidence': confidence
        }
        
        results.append(result)
    
    return results


def evaluate_comparison(
    samples: List[dict],
    baseline_samples: List[dict],
    matcher: BehaviorMatcher,
    verbose: bool = False
) -> Dict:
    """
    Evaluate and compare coherence between steered and baseline responses.
    
    Args:
        samples: List of steered samples
        baseline_samples: List of baseline samples
        matcher: CoherenceMatcher instance
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing comparison results
    """
    if len(samples) != len(baseline_samples):
        print(f"Warning: Number of samples ({len(samples)}) != baseline_samples ({len(baseline_samples)})")
    
    results = {
        'steered': [],
        'baseline': [],
        'comparison': []
    }
    
    n_samples = min(len(samples), len(baseline_samples))
    
    for i in range(n_samples):
        if verbose:
            print(f"Processing sample {i+1}/{n_samples}...")
        
        # Evaluate steered response
        steered_sample = samples[i]
        steered_prompt = steered_sample.get('prompt', '')
        steered_response = steered_sample.get('response', '')
        
        steered_correct, steered_confidence = matcher.check(
            steered_response, 
            ground_truth=None, 
            prompt=steered_prompt
        )
        
        # Evaluate baseline response
        baseline_sample = baseline_samples[i]
        baseline_prompt = baseline_sample.get('prompt', '')
        baseline_response = baseline_sample.get('response', '')
        
        baseline_correct, baseline_confidence = matcher.check(
            baseline_response,
            ground_truth=None,
            prompt=baseline_prompt
        )
        
        results['steered'].append({
            'index': i,
            'prompt': steered_prompt,
            'response': steered_response,
            'is_correct': steered_correct,
            'coherence_score': steered_confidence,
            'confidence': steered_confidence
        })
        
        results['baseline'].append({
            'index': i,
            'prompt': baseline_prompt,
            'response': baseline_response,
            'is_correct': baseline_correct,
            'coherence_score': baseline_confidence,
            'confidence': baseline_confidence
        })
        
        results['comparison'].append({
            'index': i,
            'steered_score': steered_confidence,
            'baseline_score': baseline_confidence,
            'delta': (steered_confidence - baseline_confidence)
        })
    
    # Compute summary statistics
    steered_scores = [r['coherence_score'] for r in results['steered']]
    baseline_scores = [r['coherence_score'] for r in results['baseline']]
    
    results['summary'] = {
        'steered_mean_score': sum(steered_scores) / len(steered_scores) if steered_scores else 0,
        'baseline_mean_score': sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
        'steered_correct_count': sum(1 for r in results['steered'] if r['is_correct']),
        'baseline_correct_count': sum(1 for r in results['baseline'] if r['is_correct']),
        'total_samples': len(steered_scores)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate coherence of model responses using CoherenceMatcher'
    )
    parser.add_argument(
        'file',
        help='Path to the JSON result file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Path to save the output JSON file (default: stdout)',
        default=None
    )
    parser.add_argument(
        '--device',
        help='Device to use for the model (default: cuda)',
        default='cuda'
    )
    parser.add_argument(
        '--model-name',
        help='Flow-Judge model name (default: flowaicom/Flow-Judge-v0.1)',
        default='flowaicom/Flow-Judge-v0.1'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        help='Coherence threshold (default: 3)',
        default=3
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress information'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with baseline samples'
    )
    
    args = parser.parse_args()
    
    # Load JSON data
    print(f"Loading data from {args.file}...")
    data = load_json_results(args.file)
    samples, baseline_samples = extract_samples(data)
    
    print(f"Found {len(samples)} samples")
    if args.compare:
        print(f"Found {len(baseline_samples)} baseline samples")
    
    # Initialize CoherenceMatcher
    print(f"Initializing CoherenceMatcher on {args.device}...")
    matcher = BehaviorMatcher(
        mode="coordinate",
        device=args.device,
    )
    
    # Evaluate
    if args.compare and baseline_samples:
        print("Evaluating comparison between steered and baseline responses...")
        results = evaluate_comparison(samples, baseline_samples, matcher, args.verbose)
        
        print("\n=== Results Summary ===")
        print(f"Total samples: {results['summary']['total_samples']}")
        print(f"Steered mean coherence: {results['summary']['steered_mean_score']:.3f}")
        print(f"Baseline mean coherence: {results['summary']['baseline_mean_score']:.3f}")
        print(f"Steered correct: {results['summary']['steered_correct_count']}")
        print(f"Baseline correct: {results['summary']['baseline_correct_count']}")
    else:
        print("Evaluating steered responses...")
        results = evaluate_score(samples, matcher, args.verbose)
        
        # Compute summary
        scores = [r['coherence_score'] for r in results]
        correct = sum(1 for r in results if r['is_correct'])
        
        print("\n=== Results Summary ===")
        print(f"Total samples: {len(results)}")
        print(f"Mean coherence: {sum(scores)/len(scores):.3f}" if scores else "No scores")
        print(f"Correct: {correct}")
    
    # Save output
    if args.output:
        output_path = args.output
    else:
        # Default output path
        input_path = Path(args.file)
        output_path = input_path.parent / f"{input_path.stem}_score{input_path.suffix}"
    
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Done!")
    
    return results


if __name__ == '__main__':
    main()
