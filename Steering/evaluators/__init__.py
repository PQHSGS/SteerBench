"""
Evaluation module for steering benchmarks.

All evaluators extend BaseEvaluator and implement:
  - check(response, ground_truth, ...)  → (is_correct, confidence)
  - batch(steer_model, samples, coeff)  → [(is_correct, confidence, response), ...]

Usage:
    from Steering.evaluators import MathMatcher, EVALUATOR_MAP

    # Standalone
    matcher = MathMatcher()
    is_correct, confidence = matcher.check(response, ground_truth)

    # Via pipeline (preferred)
    evaluator = EVALUATOR_MAP["math"](device="cuda")
    results = evaluator.batch(steer_model, samples, coeff=1.5)
"""
from typing import Callable
from .scoring_metrics import (
    AgreementMatcher,
    BaseEvaluator,
    MultipleChoiceMatcher,
    MathMatcher,
    BehaviorMatcher,
    RefusalMatcher,
    SemanticMatcher,
    LogitMatcher,
    CastRefusalMatcher,
    DeceptionEvaluator,
    IFEvalEvaluator,
    ToxicMatcher,
    HumanEvalMatcher,
)
from .model_capabilities import PerplexityMatcher, CoherenceMatcher

# =============================================================================
# EVALUATOR REGISTRY
# To add a new evaluator: 1) Add class above  2) Add entry here
# =============================================================================
EvaluatorFactory = Callable[..., BaseEvaluator]

EVALUATOR_MAP: dict[str, EvaluatorFactory] = {
    "multiple_choice": lambda *, device: MultipleChoiceMatcher(device=device),
    "math":            lambda *, device: MathMatcher(),
    "semantic":        lambda *, device: SemanticMatcher(device=device),
    "refusal":         lambda *, device: RefusalMatcher(device=device),
    "agreement":       lambda *, device: AgreementMatcher(device=device),
    "logit":           lambda *, device: LogitMatcher(),
    "perplexity":      lambda *, device: PerplexityMatcher(),
    "coherence":       lambda *, device: CoherenceMatcher(device=device),
    "corrigible":      lambda *, device: BehaviorMatcher(
                           device=device,
                           mode='corrigible'
                       ),
    "coordinate":      lambda *, device: BehaviorMatcher(
                           device=device,
                           mode='coordinate'
                       ),  
    "sycophancy":      lambda *, device: BehaviorMatcher(
                           device=device,
                           mode='sycophancy'
                       ),
    "politics":        lambda *, device: BehaviorMatcher(
                           device=device,
                           mode='politics'
                       ),
    "evil":           lambda *, device: BehaviorMatcher(
                           device=device,
                           mode='evil'
                       ),
    "cast_refusal":    lambda *, device: CastRefusalMatcher(device=device),
    "deception":       lambda *, device: DeceptionEvaluator(device=device),
    "ifeval":          lambda *, device: IFEvalEvaluator(),
    "toxic":           lambda *, device: ToxicMatcher(device=device),
    "humaneval":       lambda *, device: HumanEvalMatcher(),
}


__all__ = [
    "BaseEvaluator",
    "MultipleChoiceMatcher",
    "MathMatcher",
    "BehaviorMatcher",
    "RefusalMatcher",
    "SemanticMatcher",
    "LogitMatcher",
    "PerplexityMatcher",
    "CoherenceMatcher",
    "CastRefusalMatcher",
    "DeceptionEvaluator",
    "IFEvalEvaluator",
    "ToxicMatcher",
    "HumanEvalMatcher",
]
