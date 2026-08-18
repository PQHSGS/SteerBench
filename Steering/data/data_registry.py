
# =============================================================================
# Training Dataset Registry (UNIFIED)
# =============================================================================
from ..config import TrainDatasetConfig, CompositeDatasetConfig, TestDatasetConfig
from pathlib import Path
from typing import Dict

TRAIN_DATASET_REGISTRY: Dict[str, TrainDatasetConfig] = {
    # Sycophancy datasets
    "sycophancy": TrainDatasetConfig(
        file="behaviour/sycophancy/personas/misaligned_1.jsonl",
        schema="sycophancy_personas",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "sycophancy_full": TrainDatasetConfig(
        file="behaviour/sycophancy/sycophancy.jsonl",
        schema="binary_choice",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "sycophancy_nlp_survey": TrainDatasetConfig(
        file="behaviour/sycophancy/sycophancy_on_nlp_survey.jsonl",
        schema="binary_choice",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ), 
    "sycophancy_philosophy": TrainDatasetConfig(
        file="behaviour/sycophancy/sycophancy_on_philpapers2020.jsonl",
        schema="binary_choice",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "sycophancy_political": TrainDatasetConfig(
        file="behaviour/sycophancy/sycophancy_on_political_typology_quiz.jsonl",
        schema="binary_choice",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    
    # AI Risk datasets
    "ai_risk_coordinate": TrainDatasetConfig(
        file="behaviour/advanced-ai-risk/coordinate-other-ais.jsonl",
        schema="binary_choice",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "ai_risk_corrigible": TrainDatasetConfig(
        file="behaviour/advanced-ai-risk/corrigible-neutral-HHH.jsonl",
        schema="binary_choice",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "ai_risk_myopic": TrainDatasetConfig(
        file="behaviour/advanced-ai-risk/myopic-reward.jsonl",
        schema="binary_choice",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "ai_risk_survival": TrainDatasetConfig(
        file="behaviour/advanced-ai-risk/survival-instinct.jsonl",
        schema="binary_choice",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    
    # Hallucination datasets
    "hallucination": TrainDatasetConfig(
        file="behaviour/hallucination/CAA/CAA_generated.jsonl",
        schema="binary_choice",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),

    # Jigsaw toxicity dataset used by Mean-AcT / PID GT surface
    "toxicity": TrainDatasetConfig(
        file="jigsaw/train.csv",
        schema="jigsaw_toxicity",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "jigsaw": TrainDatasetConfig(
        file="jigsaw/train.csv",
        schema="jigsaw_toxicity",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    # Jigsaw toxicity — friendly name for steering training
    "toxic_jigsaw": TrainDatasetConfig(
        file="behaviour/toxic/jigsaw/train.csv",
        schema="jigsaw",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    # Refusal datasets
    "refusal": TrainDatasetConfig(
        file="behaviour/refusal/CAST/condition_harmful.json",
        schema="CAST_condition",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),

    "refusal_sorrybench": TrainDatasetConfig(
        file="behaviour/refusal/SorryBench/question.jsonl",
        schema="SorryBench",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "refusal_caa": TrainDatasetConfig(
        file="behaviour/refusal/CAA.jsonl",
        schema="binary_choice",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "refusal_advbench": TrainDatasetConfig(
        file="behaviour/refusal/AdvBench/harmful_behaviors.csv",
        schema="AdvBench",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "roleplay_arithmetic": TrainDatasetConfig(
        file="behaviour/roleplay/arithmetic.jsonl",
        schema="question_only",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "roleplay_common_sense": TrainDatasetConfig(
        file="behaviour/roleplay/commonsense.jsonl",
        schema="question_only",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    
    # Reasoning datasets
    "gms8k_train": TrainDatasetConfig(
        file="reasoning/GMS8K/GMS8K.jsonl",
        schema="math_question",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),

    "svamp_train": TrainDatasetConfig(
        file="reasoning/SVAMP/SVAMP.jsonl",
        schema="math_question",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    
    # QA datasets
    "csqa_train": TrainDatasetConfig(
        file="QA/CSQA/CSQA.jsonl",
        schema="csqa",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "simple_qa": TrainDatasetConfig(
        file="QA/SimpleQA/SimpleQA.jsonl",
        schema="simple_qa",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),

    # AxBench Concept500 (Gemma-2-2b, layer 20) - paired positive/negative completions
    "concept500_l20": TrainDatasetConfig(
        file="Concept500/2b/l20/data.parquet",
        schema="concept500_reps",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    
    # SAE-SSV datasets (text + binary label classification)
    "politics_twinviews": TrainDatasetConfig(
        file="behaviour/politic/twinviews.csv",
        schema="twin_views",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    # SPARE NQSwap (Training / Feature Extraction)
    "nqswap_train": TrainDatasetConfig(
        file="QA/NQSwap/NQswap.json",
        schema="nqswap_train",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    # Validation Level 1: Angular Data Alignment
    "angular_validation": TrainDatasetConfig(
        file="../Verification/Level1/Angular/data.json", 
        schema="CAST_condition", # Expects "harmful", "harmless" -> maps to correct/false keys
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    # MMLU CorrSteer (local file, pre-shuffled seed=42, matches GT exactly)
    "mmlu_corrsteer_train": TrainDatasetConfig(
        file="mmlu/mmlu_hf_shuffled.json",
        schema="mmlu_corrsteer",
        target_key="correct_prompt",
        contrast_key=None,           # No contrastive split — raw dicts with question+answer
    ),

    # cais/MASK
    "cais_mask": TrainDatasetConfig(
        file="behaviour/deception/MASK/data.jsonl",
        schema="cais_mask",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "liarbench": TrainDatasetConfig(
        file="behaviour/deception/LiarBench/data.jsonl",
        schema="deception",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),

    # google/IFEval
    "ifeval": TrainDatasetConfig(
        file="instruction_following/IFEval/data.jsonl",
        schema="ifeval",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),

    # Prefix-split toxic datasets (first 500 samples, classified by first-10-token toxicity)
    "toxic_jigsaw_prefix_pos": TrainDatasetConfig(
        file="split_by_prefix/toxic_jigsaw_prefix_pos.csv",
        schema="jigsaw",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "toxic_jigsaw_prefix_neg": TrainDatasetConfig(
        file="split_by_prefix/toxic_jigsaw_prefix_neg.csv",
        schema="jigsaw",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),

}


# =============================================================================
# Composite Dataset Registry
# For datasets that combine multiple files (CAST, SRPS roleplay)
# =============================================================================

COMPOSITE_DATASET_REGISTRY: Dict[str, CompositeDatasetConfig] = {
    # CAST: combines response prefixes with alpaca questions
    # Creates: question + compliant_response vs question + non_compliant_response
    "refusal_cast_responses": CompositeDatasetConfig(
        response_file="behaviour/refusal/CAST/behaviour_refusal.json",
        question_file="behaviour/refusal/CAST/alpaca.json",
        schema="cast_combined",
        target_key="correct_prompt",     # question + non_compliant (refusing)
        contrast_key="false_prompt",     # question + compliant (agreeing)
    ),

    
    # SRPS: roleplay prompts + GMS8K arithmetic questions
    "srps_roleplay_gms8k": CompositeDatasetConfig(
        response_file="behaviour/roleplay/arithmetic.jsonl",
        question_file="reasoning/GMS8K/GMS8K.jsonl",
        schema="srps_roleplay",
        target_key="correct_prompt",     # roleplay_prompt + question
        contrast_key="false_prompt",     # question only (baseline)
    ),
    
    # SRPS: roleplay prompts + SVAMP arithmetic questions
    "srps_roleplay_svamp": CompositeDatasetConfig(
        response_file="behaviour/roleplay/arithmetic.jsonl",
        question_file="reasoning/SVAMP/SVAMP.jsonl",
        schema="srps_roleplay",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    
    # SRPS: roleplay prompts + CSQA commonsense questions
    "srps_roleplay_csqa": CompositeDatasetConfig(
        response_file="behaviour/roleplay/commonsense.jsonl",
        question_file="QA/CSQA/CSQA.jsonl",
        schema="srps_roleplay",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),

    # Angular Steering: AdvBench (Harmful/Target) + Alpaca (Harmless/Contrast)
    "refusal_angular": CompositeDatasetConfig(
        response_file="behaviour/refusal/AdvBench/harmful_behaviors.csv", # Target (Harmful)
        question_file="behaviour/refusal/CAST/alpaca.json",               # Contrast (Harmless)
        schema="refusal_angular",
        target_key="correct_prompt",     # Harmful
        contrast_key="false_prompt",     # Harmless
    ),

    # Angular Steering: AdvBench Split (Harmful/Target = Train split) + Alpaca Split (Harmless/Contrast)
    "refusal_angular_split": CompositeDatasetConfig(
        response_file="behaviour/refusal/AdvBench/train.csv",
        question_file="behaviour/refusal/CAST/alpaca_train.json",
        schema="refusal_angular",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "evil": CompositeDatasetConfig(
        response_file="behaviour/evil/misaligned_2.jsonl",
        question_file="behaviour/evil/normal.jsonl",
        schema="evil",
        target_key="correct_prompt",
        contrast_key="false_prompt"
    ),
    "toxic_extreme": CompositeDatasetConfig(
        response_file="behaviour/toxic/extreme_toxic_human/train.parquet",
        question_file="behaviour/normal/alpaca.json",
        schema="toxic_extreme",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),

    # Prefix-split evil datasets (first 500 samples, classified by first-10-token evilness)
    "evil_prefix_pos": CompositeDatasetConfig(
        response_file="split_by_prefix/evil_misaligned_prefix_pos.jsonl",
        question_file="split_by_prefix/evil_normal_prefix_pos.jsonl",
        schema="evil",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
    "evil_prefix_neg": CompositeDatasetConfig(
        response_file="split_by_prefix/evil_misaligned_prefix_neg.jsonl",
        question_file="split_by_prefix/evil_normal_prefix_neg.jsonl",
        schema="evil",
        target_key="correct_prompt",
        contrast_key="false_prompt",
    ),
}


# =============================================================================
# Test Dataset Registry (UNIFIED)
# =============================================================================

TEST_DATASET_REGISTRY: Dict[str, TestDatasetConfig] = {
    # Multiple choice QA
    "csqa": TestDatasetConfig(
        file="QA/CSQA.jsonl",
        schema="csqa",  # Fixed: use 'csqa' schema which exists in FORMATTERS
        evaluator="logit",  # Changed to logit for single-token choice evaluation
        ground_truth_key="answer"
    ),
    
    # Math reasoning
    "gms8k": TestDatasetConfig(
        file="reasoning/GMS8K/test.jsonl",
        schema="math_question",
        evaluator="math",
        ground_truth_key="answer",
        prefix= "\nLet's think step by step and answer in 2-3 sentences: "
    ),
    "svamp": TestDatasetConfig(
        file="reasoning/SVAMP/test_dataset.json",
        schema="math_question",
        evaluator="math",
        ground_truth_key="answer",
    ),

    "mmlu": TestDatasetConfig(
        file="mmlu/mmlu.json",
        schema="mmlu",
        evaluator="logit",  # Changed to logit for single-token choice evaluation
        ground_truth_key="answer"
    ),

    # MMLU CorrSteer test (local file, HF validation split pre-shuffled seed=42)
    "mmlu_corrsteer_test": TestDatasetConfig(
        file="mmlu/mmlu_hf_val.json",
        schema="mmlu_corrsteer_test",
        evaluator="logit",
        ground_truth_key="answer",
    ),
    
    # Refusal evaluation
    "refusal_ab": TestDatasetConfig(
        file="refusal/CAA/test_dataset_ab.json",
        schema="binary_choice_test",
        evaluator="logit",
        ground_truth_key="answer",
        prefix= "\nAnswer: ("
    ),
    "refusal_open": TestDatasetConfig(
        file="refusal/CAA/test_dataset_open_ended.json",
        schema="question_only",
        evaluator="refusal",
        prefix=" Response: "
    ),
    # Backward-compatible alias used by existing eval configs.
    "refusal_response": TestDatasetConfig(
        file="refusal/CAA/test_dataset_open_ended.json",
        schema="question_only",
        evaluator="refusal",
        prefix=" Response: "
    ),
    # Jigsaw toxicity aliases for parity with GT Mean-AcT/PID surface.
    "toxicity": TestDatasetConfig(
        file="jigsaw/test.csv",
        schema="jigsaw_toxicity",
        evaluator="perplexity",
    ),
    "jigsaw": TestDatasetConfig(
        file="jigsaw/test.csv",
        schema="jigsaw_toxicity",
        evaluator="perplexity",
    ),
    "coordinate_ab": TestDatasetConfig(
        file="behaviour/coordinate-other-ais/test_dataset_ab.json",
        schema="binary_choice_test",
        evaluator="logit",
        ground_truth_key="answer",
        prefix = "\nAnswer: ("
    ),
    "coordinate_open": TestDatasetConfig(
        file="behaviour/coordinate-other-ais/test_dataset_open_ended.json",
        schema="question_only",
        evaluator="coordinate"
    ),
    "coordinate_perplexity": TestDatasetConfig(
        file="behaviour/coordinate-other-ais/test_dataset_open_ended.json",
        schema="question_only",
        evaluator="perplexity"
    ),
    "hallucination_ab": TestDatasetConfig(
        file="hallucination/CAA/test_dataset_ab.json",
        schema="binary_choice_test",
        evaluator="logit",
        ground_truth_key="answer",
        prefix= "\nAnswer: ("
    ),
    "survival_ab": TestDatasetConfig(
        file="behaviour/survival-instinct/test_dataset_ab.json",
        schema="binary_choice_test",
        evaluator="logit",
        ground_truth_key="answer",
        prefix = "\nAnswer: ("
    ),
    "myopic_ab": TestDatasetConfig(
        file="behaviour/myopic-reward/test_dataset_ab.json",
        schema="binary_choice_test",
        evaluator="logit",
        ground_truth_key="answer"
    ),
    # Sycophancy evaluation
    "sycophancy_ab": TestDatasetConfig(
        file="behaviour/sycophancy/test_dataset_ab.json",
        schema="binary_choice_test",
        evaluator="logit",
        ground_truth_key="answer",
        prefix= "\nAnswer: ("
    ),
    "sycophancy_open": TestDatasetConfig(
        file="behaviour/sycophancy/test_dataset_open_ended.json",
        schema="question_only",
        evaluator="sycophancy"
    ),

    "corrigible_ab": TestDatasetConfig(
        file="behaviour/corrigible-neutral-HHH/test_dataset_ab.json",
        schema="binary_choice_test",
        evaluator="logit",
        ground_truth_key="answer",
        prefix= "\nAnswer: ("
    ),

    "corrigible_open": TestDatasetConfig(
        file="behaviour/corrigible-neutral-HHH/test_dataset_open_ended.json",
        schema="question_only",
        evaluator="corrigible"
    ),
    
    # TruthfulQA
    "truthfulqa": TestDatasetConfig(
        file="hallucination/TruthfulQA.json",
        schema="truthful_qa",
        evaluator="logit",  # Changed to logit for single-token choice evaluation
        ground_truth_key="answer",
        prefix= "\nAnswer: ("
    ),
    # HaluEval - uses semantic matching (free-form text answers)
    "halueval": TestDatasetConfig(
        file="hallucination/HaluEval.jsonl",
        schema="halueval",  # Need custom schema for this format
        evaluator="semantic",  # Use semantic matcher for free-form text
        ground_truth_key="right_answer",
    ),      
    # XSTest - safety classification dataset
    "xstest": TestDatasetConfig(
        file="refusal/XSTest.csv",
        schema="xstest",
        evaluator="refusal",  # Uses BehaviorMatcher for safe/unsafe classification
        ground_truth_key="label",
        prefix="\nAnswer: "
    ),

    # HarmBench - harmful behavior prompts
    "harmbench": TestDatasetConfig(
        file="refusal/HarmBench.csv",
        schema="harmbench",
        evaluator="refusal",
    ),
    # AdvBench - adversarial harmful behaviors
    "advbench": TestDatasetConfig(
        file="refusal/AdvBench/harmful_behaviors.csv",
        schema="advbench",
        evaluator="refusal",
    ),
    "advbench_test": TestDatasetConfig(
        file="refusal/AdvBench/test.csv", # Relative to TEST_ROOT
        schema="advbench",
        evaluator="refusal",
    ),
    # SorryBench - harmful behavior test prompts
    "sorrybench": TestDatasetConfig(
        file="refusal/SorryBench/question.jsonl",
        schema="sorrybench_test",
        evaluator="refusal",
        prefix="\nAnswer: "
    ),
    # DoNotAnswer - harmful prompts with risk categories
    "donotanswer": TestDatasetConfig(
        file="refusal/DonotAnswer.csv",
        schema="donotanswer",
        evaluator="refusal",
        prefix="\nAnswer: "
    ),
    # Politics TwinViews - political viewpoint classification (SSV)
    "twinviews_test": TestDatasetConfig(
        file="behaviour/politic/twinviews.csv",
        schema="twin_views_test",
        evaluator="politics",  # Uses BehaviorMatcher("conservative")
        ground_truth_key="ground_truth",
    ),
    # NQSwap - Contextual vs Parametric evaluation (Few-Shot)
    "nqswap_test": TestDatasetConfig(
        file="../TrainDataset/QA/NQSwap/NQswap.json",
        schema="nqswap_test",
        evaluator="semantic", # Checks if sub_answer (contextual target) is in response
        ground_truth_key="answer", # The formatter sets 'answer' to sub_answer
    ),
    "angular_validation": TestDatasetConfig(
        file="../Verification/Level1/Angular/data.json",
        schema="refusal_angular_test",
        evaluator="refusal",
    ),
    "deception": TestDatasetConfig(
        file="behaviour/deception/MASK/data.jsonl",
        schema="deception_test",
        evaluator="deception",
        ground_truth_key="answer",
    ),
    "liarbench": TestDatasetConfig(
        file="behaviour/deception/LiarBench/data.jsonl",
        schema="deception_test",
        evaluator="deception",
        ground_truth_key="answer",
    ),
    "liarbench_no_sys": TestDatasetConfig(
        file="behaviour/deception/LiarBench/data.jsonl",
        schema="deception_no_sys",
        evaluator="deception",
        ground_truth_key="answer",
    ),
    "ifeval": TestDatasetConfig(
        file="instruction_following/IFEval/data.jsonl",
        schema="ifeval_test",
        evaluator="ifeval",
        ground_truth_key="answer",
    ),
    # Toxicity — evaluates whether model outputs become more toxic when steered
    # Uses Alpaca prompts as neutral conversational inputs and scores responses
    # with toxic-bert (1 = toxic output, 0 = non-toxic output).
    "toxic": TestDatasetConfig(
        file="alpaca/alpaca.json",
        schema="question_only",
        evaluator="toxic",
        prefix=" Response: ",
    ),
    # HumanEval — evaluates whether steered model can still write correct Python
    # Loaded from a local file (drop openai_humaneval test split in TestDataset).
    "humaneval": TestDatasetConfig(
        file="humaneval/humaneval.jsonl",
        schema="humaneval",
        evaluator="humaneval",
    ),
    # MBPP — same evaluator (HumanEvalMatcher handles direct assertion style)
    "mbpp": TestDatasetConfig(
        file="mbpp/mbpp.jsonl",
        schema="mbpp",
        evaluator="humaneval",
    ),
    "pop_qa": TestDatasetConfig(
        file="PopQA/test.tsv",
        schema="pop_qa",
        evaluator="pop_qa",
        ground_truth_key="answer",
    ),
    "evil": TestDatasetConfig(
        file="alpaca/alpaca.json",
        schema="question_only",
        evaluator="evil",
        prefix=" Response: "
    ),
    "toxic_with_sys": TestDatasetConfig(
        file="alpaca/alpaca.json",
        schema="toxic_with_sys",
        evaluator="toxic",
        prefix=" Response: "
    ),
    "evil_with_sys": TestDatasetConfig(
        file="alpaca/alpaca.json",
        schema="evil_with_sys",
        evaluator="evil",
        prefix=" Response: "
    ),
}
