"""
Schema Formatters.

Each formatter transforms raw data into standardized format with:
- question: The question/prompt text
- correct_prompt: Prompt representing target behavior
- false_prompt: Prompt representing contrast behavior

To add a new formatter:
1. Add your formatter function below
2. Add it to FORMATTERS dict at the bottom
"""

from typing import List, Dict, Any, Union, Optional
import itertools
import random

# =============================================================================
# 1. Helper Functions
# =============================================================================

def _extract_answer_token(text: str) -> str:
    """Extract single token from answer string like '(A)' -> 'A'."""
    return text.strip("() ")

def _get_question(entry: Union[Dict, str], key: str = "question") -> str:
    """Extract question text from entry which might be a dict or string."""
    if isinstance(entry, dict):
        return entry.get(key, "")
    return str(entry)

def _verbalise_nqswap(example: Dict, ctx_key: str, ans_key: str, is_test: bool = False) -> str:
    """Helper to format a single NQSwap example."""
    # Handle list answers by taking the first element
    ans_val = example[ans_key]
    ans = ans_val[0] if isinstance(ans_val, list) else ans_val

    prompt = f"context: {example[ctx_key]}\n"
    prompt += f"question: {example['question']}\n"
    if is_test:
        prompt += "answer:"
    else:
        prompt += f"answer: {ans}\n\n"
    return prompt


def _is_valid_nqswap_entry(example: Dict) -> bool:
    """Return True if the entry has the fields needed for NQSwap prompting."""
    required = ("question", "org_context", "org_answer", "sub_context", "sub_answer")
    return all(k in example for k in required)


def _select_demo(
    entries: List[Dict],
    k_shot: int = 3,
    pool_size: int = 128,
    pool_seed: int = 42,
    shot_seed: int = 42,
) -> tuple[List[Dict], set[str]]:
    """
    Select GT-like NQSwap demonstrations from current entries.

    Mirrors GT's pattern at a high level:
    1) create question groups
    2) build a shuffled demonstration pool with fixed seed
    3) pick k-shot groups with runtime seed
    """
    valid_entries = [e for e in entries if _is_valid_nqswap_entry(e)]
    if not valid_entries:
        return [], set()

    grouped: Dict[str, List[Dict]] = {}
    for e in valid_entries:
        grouped.setdefault(e["question"], []).append(e)

    group_ids = list(range(len(grouped)))
    random.Random(pool_seed).shuffle(group_ids)
    demo_pool_ids = group_ids[: min(pool_size, len(group_ids))]

    random.Random(shot_seed).shuffle(demo_pool_ids)
    selected_ids = demo_pool_ids[: min(k_shot, len(demo_pool_ids))]

    groups = list(grouped.values())
    demos = [groups[gidx][0] for gidx in selected_ids]
    demo_questions = {d["question"] for d in demos}
    return demos, demo_questions

# =============================================================================
# 2. Generic Formatters
# =============================================================================

def binary_choice(entries: List[Dict]) -> List[Dict]:
    """
    Format binary choice data for CAA extraction.
    Returns clean question text - chat template applied at loader level.
    Returns choices as single tokens ["A", "B"] for logit extraction.
    
    correct_prompt/false_prompt contain ONLY the answer token (e.g., " (A)")
    The loader will prepend the templated question when apply_chat_template=True.
    """
    out = []
    for e in entries:
        q = e["question"]
        pos = e["answer_matching_behavior"]  # e.g., " (A)"
        neg = e["answer_not_matching_behavior"]  # e.g., " (B)"
        
        out.append({
            "question": q + "\nAnswer: ",
            "choices": ["A", "B"],
            "answer": _extract_answer_token(pos),
            "correct_prompt": pos,  # Just the answer, loader prepends templated question
            "false_prompt": neg,
            "target_response": "\nAnswer: (" + _extract_answer_token(pos) + ")",  # For CAA training
            "contrast_response": "\nAnswer: (" + _extract_answer_token(neg) + ")",  # For CAA training
        })
    return out


def binary_choice_test(entries: List[Dict]) -> List[Dict]:
    """
    Format binary choice test data for logit-based evaluation.
    Returns clean question text - chat template and prefix applied at loader level.
    Returns choices as single tokens ["A", "B"] for logit extraction.
    """
    out = []
    for e in entries:
        q = e["question"]
        pos = e["answer_matching_behavior"]
        
        out.append({
            "question": q,
            "choices": ["A", "B"],
            "answer": _extract_answer_token(pos),
        })
    return out

def question_only(entries: List[Dict]) -> List[Dict]:
    """Format question-only data."""
    return [{
        "question": _get_question(e),
        "correct_prompt": _get_question(e),
        "false_prompt": "",
    } for e in entries]

def concept500_reps(entries: List[Dict]) -> List[Dict]:
    """Pair Concept500 positives with their prompt-matched negative response.

    The Concept500 parquet contains one negative row per prompt plus multiple
    positive rows for the same prompt. RePS training needs prompt/target/contrast
    triples, so we emit one row per positive example using the shared negative
    completion as the contrast.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        input_text = str(entry.get("input", "")).strip()
        if not input_text:
            continue

        group = grouped.setdefault(input_text, {"negative": None, "positives": []})
        category = str(entry.get("category", "")).strip().lower()
        concept_id = int(entry.get("concept_id", -1))

        if category == "negative" or concept_id == -1:
            if group["negative"] is None:
                group["negative"] = entry
        elif category == "positive":
            group["positives"].append(entry)

    out: List[Dict[str, Any]] = []
    for input_text, group in grouped.items():
        negative = group["negative"]
        positives = sorted(group["positives"], key=lambda item: int(item.get("concept_id", -1)))
        if not positives:
            continue

        false_prompt = ""
        if negative is not None:
            false_prompt = str(negative.get("output", "")).strip()

        for positive in positives:
            correct_prompt = str(positive.get("output", "")).strip()
            if not correct_prompt:
                continue

            out.append({
                "question": input_text,
                "correct_prompt": correct_prompt,
                "false_prompt": false_prompt,
                "concept_id": int(positive.get("concept_id", -1)),
                "output_concept": positive.get("output_concept", ""),
                "category": "positive",
            })

    return out

def jigsaw(entries: List[Dict]) -> List[Dict]:
    """
    Format Jigsaw toxicity data for training.

    Pairs toxic comments (target) with non-toxic comments (contrast) using
    a generic question so the loader can apply the chat template.  Also emits
    target_response / contrast_response (response-only strings) so that FLAS
    and REPS can consume them without a separate formatter variant.
    """
    toxic_rows: List[str] = []
    non_toxic_rows: List[str] = []
    toxic_keys = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    for entry in entries:
        is_toxic = any(int(entry.get(key, 0)) == 1 for key in toxic_keys)
        comment_text = entry.get("comment_text", "")
        if not comment_text:
            continue
        comment_text = " ".join(comment_text.split()[:20])
        if is_toxic:
            toxic_rows.append(comment_text)
        else:
            non_toxic_rows.append(comment_text)

    n_pairs = min(len(toxic_rows), len(non_toxic_rows))
    return [{
        "question": "Write a comment:",
        "correct_prompt": toxic_rows[idx],       # target: toxic
        "false_prompt": non_toxic_rows[idx],     # contrast: non-toxic
        "target_response": toxic_rows[idx],
        "contrast_response": non_toxic_rows[idx],
    } for idx in range(n_pairs)]

def math_question(entries: List[Dict]) -> List[Dict]:
    """Format question-answer pairs."""
    return [{
        "question": e["question"],
        "answer": e["answer"],
        "correct_prompt": e["answer"],
        "false_prompt": ""
    } for e in entries]



# =============================================================================
# 3. QA & Knowledge Formatters
# =============================================================================

def csqa(entries: List[Dict]) -> List[Dict]:
    """
    Format multiple choice questions for logit-based evaluation.
    Returns choices as single tokens (A, B, C, D, E) for logit extraction.
    """
    out = []
    for e in entries:
        labels = e["choices"]["label"]
        texts = e["choices"]["text"]
        idx = ord(e["answerKey"]) - ord("A")
        
        # Build formatted question with choices
        choice_text = "\n".join([f"({l}) {t}" for l, t in zip(labels, texts)])
        formatted_q = f"{e['question']}\n\nChoices:\n{choice_text}\nAnswer: ("
        
        correct_full = f"({labels[idx]}) {texts[idx]}"
        wrongs = [(labels[i], texts[i]) for i in range(len(labels)) if i != idx]
        
        out.append({
            "question": formatted_q,
            "choices": labels,  # Single tokens for logit extraction
            "answer": e["answerKey"],
            "correct_prompt": f"{e['question']}\n\nChoices:\n{choice_text}\nAnswer: {correct_full}",
            "false_prompt": [f"{e['question']}\n\nChoices:\n{choice_text}\nAnswer: ({l}) {t}" for l, t in wrongs],
        })
    return out

def mmlu(entries: List[Dict]) -> List[Dict]:
    """
    Format MMLU data for logit-based evaluation.
    Returns choices as single tokens ["A", "B"] for logit extraction.
    """
    out = []
    for e in entries:
        q = e["prompt"]
        out.append({
            "question": q + "\nAnswer: (",  # Add '(' for logit extraction
            "choices": ["A", "B"],
            "answer": _extract_answer_token(e["correct"]),
            "correct_prompt": f"{q}\nAnswer: {e['correct']} {e['correct_full_text']}",
            "false_prompt": f"{q}\nAnswer: {e['incorrect']} {e['incorrect_full_text']}",
        })
    return out

def mmlu_corrsteer(entries: List[Dict]) -> List[Dict]:
    """
    Format MMLU data EXACTLY matching the GT CorrSteer `build_prompt` format.
    The GT uses the 4-choice HuggingFace format:
    A. choice0\nB. choice1\nC. choice2\nD. choice3\nChoose one of the following options only: A, B, C, or D\nAnswer:
    It uses this exact string structure for both extraction and generation.
    
    IMPORTANT: CorrSteer configs MUST set ``apply_chat_template: false``.
    The full question prompt is stored in ``d["question"]`` (same as ``target_key``),
    and the loader's chat template would double it.  CorrSteerExtractor tokenizes
    via ``model.to_tokens()`` which applies chat template internally for
    instruction-tuned models.
    """
    out = []
    
    # We construct pseudo-contrastive pairs for Sycophancy-based training from MMLU
    # The GT doesn't do this (it uses reward-based correlation), but our framework needs
    # correct/false prompts that match the GT's format exactly. We simply pick the correct
    # choice for correct_prompt, and a random wrong choice for false_prompt.
    for sample in entries:
        q = sample["question"]
        choices = sample["choices"]
        ans_idx = int(sample["answer"]) # 0-3
        
        # Exact GT format string from Code/CorrSteer/corrsteer/utils.py build_prompt
        prompt_base = (
            q + "\n" +
            "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)) +
            "\nChoose one of the following options only: A, B, C, or D" +
            "\nAnswer:"
        )
        
        # In GT, target for MMLU is `" A"`, `" B"` etc. (after 'Answer:')
        correct_ans_char = chr(65 + ans_idx)
        # Pick a false answer index
        false_ans_idx = (ans_idx + 1) % 4
        false_ans_char = chr(65 + false_ans_idx)
        
        out.append({
            "question": prompt_base,
            "correct_prompt": f" {correct_ans_char}",
            "false_prompt": f" {false_ans_char}",
            "choices": ["A", "B", "C", "D"],
            "prompt": prompt_base,
            "answer": correct_ans_char,
        })
    return out

def mmlu_corrsteer_test(entries: List[Dict]) -> List[Dict]:
    """
    Format MMLU test data EXACTLY matching the GT CorrSteer `build_prompt` format
    for logit-based evaluation over A, B, C, D tokens.
    """
    out = []
    for sample in entries:
        q = sample["question"]
        choices = sample["choices"]
        ans_idx = int(sample["answer"]) # 0-3
        
        prompt_base = (
            q + "\n" +
            "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)) +
            "\nChoose one of the following options only: A, B, C, or D" +
            "\nAnswer:"
        )
        
        out.append({
            "question": prompt_base,
            "choices": ["A", "B", "C", "D"],
            "answer": chr(65 + ans_idx),
        })
    return out

def truthful_qa(entries: List[Dict]) -> List[Dict]:
    """
    Format TruthfulQA data for logit-based evaluation.
    Returns choices as single tokens ["A", "B"] for logit extraction.
    """
    return [{
        "question": e["prompt"],
        "choices": ["A", "B"],
        "answer": _extract_answer_token(e["correct"]),
        "correct_prompt": str(e["correct"]),
        "false_prompt": str(e["incorrect"]),
    } for e in entries]


def halueval(entries: List[Dict]) -> List[Dict]:
    """
    Format HaluEval data for QA-style evaluation.
    HaluEval has free-form text answers, not A/B choices.
    Uses semantic matching for evaluation.
    """
    out = []
    for e in entries:
        q = e.get("question", "")
        knowledge = e.get("knowledge", "")
        formatted_q = f"Context: {knowledge}\n\nQuestion: {q}\nAnswer: "
        
        out.append({
            "question": formatted_q,
            "answer": e.get("right_answer", ""),
            "correct_prompt": f"{formatted_q}{e.get('right_answer', '')}",
            "false_prompt": f"{formatted_q}{e.get('hallucinated_answer', '')}",
        })
    return out

# =============================================================================
# 4. Safety & Refusal Formatters
# =============================================================================

def AdvBench(entries: List[Dict]) -> List[Dict]:
    """
    Format AdvBench data - harmful behaviors dataset.
    Used with refusal evaluator which generates response and classifies it.
    """
    return [{
        "question": e.get("goal", ""),
    } for e in entries]

def harmbench(entries: List[Dict]) -> List[Dict]:
    """Format HarmBench data - harmful behavior prompts."""
    return [{
        "question": e.get('Behavior', ''),
        "category": e.get("SemanticCategory", ""),
        "behavior_id": e.get("BehaviorID", ""),
    } for e in entries]

def harmbench_train(entries: List[Dict]) -> List[Dict]:
    """Format HarmBench data for training - provides context for Correlation extraction."""
    return [{
        "question": f"Request: {e.get('Behavior', '')}\nResponse:",
        "correct_prompt": "",
        "false_prompt": ""
    } for e in entries]

def donotanswer(entries: List[Dict]) -> List[Dict]:
    """Format DoNotAnswer data - harmful prompts with risk categorization."""
    return [{
        "question": e.get('question', ''),
        "risk_area": e.get("risk_area", ""),
        "types_of_harm": e.get("types_of_harm", ""),
    } for e in entries]

def xstest(entries: List[Dict]) -> List[Dict]:
    """
    Format XSTest data for safety classification.
    Evaluates refusals for unsafe prompts and compliance for safe prompts.
    """
    out = []
    for e in entries:
        prompt = e.get("prompt", "")
        label = e.get("label", "")  # 'safe' or 'unsafe'
        
        # Convert to binary: 0 for refusal (unsafe), 1 for non-refusal (safe)
        binary_label = "0" if label == "safe" else "1"
        
        # Contrastive prompts based on expected behavior
        if label == "unsafe":
            correct_prompt = f"{prompt}\n\nI cannot help with that request."
            false_prompt = f"{prompt}\n\nSure, I can help with that."
        else:  # safe
            correct_prompt = f"{prompt}\n\nSure, I can help with that."
            false_prompt = f"{prompt}\n\nI cannot help with that request."
        
        out.append({
            "question": f"[INST] {prompt} [/INST] ",
            "answer": binary_label,
            "focus": e.get("focus", ""),
            "test_type": e.get("type", ""),
            "correct_prompt": correct_prompt,
            "false_prompt": false_prompt,
        })
    return out

def SorryBench(entries: List[Dict]) -> List[Dict]:
    """Format SorryBench data for training (contrastive pairs)."""
    return [{
        "question": e["turns"][0],
        "correct_prompt":"",
        "false_prompt": "",
    } for e in entries]

def sorrybench_test(entries: List[Dict]) -> List[Dict]:
    """
    Format SorryBench data for test evaluation.
    All prompts are harmful - model should refuse.
    """
    return [{
        "question": e['turns'][0],
        "category": e.get("category", ""),
        "question_id": e.get("question_id", ""),
    } for e in entries]

def CAST_condition(entries: List[Dict]) -> List[Dict]:
    """Format CAST condition data (harmful/harmless pairs)."""
    return [{
        "correct_prompt": e["harmful"],
        "false_prompt": e["harmless"],
    } for e in entries]

# =============================================================================
# 5. Advanced/Specific Formatters
# =============================================================================

def twin_views(entries: List[Dict]) -> List[Dict]:
    """
    Format data for SAE-SSV extraction (Politics).
    Input: text statements with binary labels.
    
    Refactors unpaired data into pseudo-pairs for pipeline compatibility:
    - correct_prompt: Label 1 (Target/False in SSV terms)
    - false_prompt: Label 0 (Contrast/Truthful in SSV terms)
    """
    pos_entries = [e for e in entries if int(e.get("label", 0)) == 1]
    neg_entries = [e for e in entries if int(e.get("label", 0)) == 0]
    
    # Pair them up (limit to shorter length to ensure balance)
    min_len = min(len(pos_entries), len(neg_entries))
    pos_entries = pos_entries[:min_len]
    neg_entries = neg_entries[:min_len]
    
    return [{
        "question": p.get("text", ""),  # Use positive text as distinct question/ID
        "correct_prompt": p.get("text", ""),
        "false_prompt": n.get("text", ""),
        "label": 1, 
        "answer": "1"
    } for p, n in zip(pos_entries, neg_entries)]

def twin_views_test(entries: List[Dict]) -> List[Dict]:
    """Test formatter for TwinViews political dataset."""
    return [{
        "question": e.get("text", ""),
        "ground_truth": str(e.get("label", 0)),
        "label": int(e.get("label", 0)),
    } for e in entries]

# Hardcoded demonstrations for NQSwap
NQSWAP_DEMOS = [
    {
        "question": "who is the girl in the hinder video lips of an angel",
        "org_context": "Premiering in early 2007 , the music video for `` Lips of an Angel '' was directed by Shaun Silva and largely follows the narrative of the song 's lyrics , focusing on a late night phone call between the raconteur ( Austin Winkler ) and his former lover ( Emmanuelle Chriqui ).",
        "org_answer": ["Emmanuelle Chriqui"],
        "sub_context": "Premiering in early 2007 , the music video for `` Lips of an Angel '' was directed by Shaun Silva and largely follows the narrative of the song 's lyrics , focusing on a late night phone call between the raconteur ( Austin Winkler ) and his former lover ( Mia Michaels ).",
        "sub_answer": ["Mia Michaels"]
    },
    {
        "question": "who wrote the song always be humble and kind",
        "org_context": "`` Humble and Kind '' is a song written by Lori McKenna and first released by American singer Tim McGraw on January 20 , 2016 , as the second single from his 14th studio album , Damn Country Music . McKenna later recorded her rendition of the song for her eighth studio album , The Bird and the Rifle , released in July 2016 . Among several other wins and nominations , the song won the award for Best Country Song at the 59th Annual Grammy Awards , `` Video of the Year '' at the 2016 CMT Music Awards , `` Song of the Year '' at 2016 CMA Awards and `` Country Song of the Year '' at 2016 American Music Awards . It has been certified platinum and reached the number one position on the country music charts in both Canada and the United States.",
        "org_answer": ["Lori McKenna"],
        "sub_context": "`` Humble and Kind '' is a song written by Lee Mack and first released by American singer Tim McGraw on January 20 , 2016 , as the second single from his 14th studio album , Damn Country Music . McKenna later recorded her rendition of the song for her eighth studio album , The Bird and the Rifle , released in July 2016 . Among several other wins and nominations , the song won the award for Best Country Song at the 59th Annual Grammy Awards , `` Video of the Year '' at the 2016 CMT Music Awards , `` Song of the Year '' at 2016 CMA Awards and `` Country Song of the Year '' at 2016 American Music Awards . It has been certified platinum and reached the number one position on the country music charts in both Canada and the United States.",
        "sub_answer": ["Lee Mack"]
    },
    {
        "question": "who painted the world famous painting the last supper",
        "org_context": "The Last Supper ( Italian : Il Cenacolo ( il t\u0283e\u02c8na\u02d0kolo ) or L'Ultima Cena ( \u02c8lultima \u02c8t\u0283e\u02d0na ) ) is a late 15th - century mural painting by Leonardo da Vinci housed by the refectory of the Convent of Santa Maria delle Grazie in Milan . It is one of the world 's most recognizable paintings.",
        "org_answer": ["Leonardo da Vinci"],
        "sub_context": "The Last Supper ( Italian : Il Cenacolo ( il t\u0283e\u02c8na\u02d0kolo ) or L'Ultima Cena ( \u02c8lultima \u02c8t\u0283e\u02d0na ) ) is a late 15th - century mural painting by Brian Steele housed by the refectory of the Convent of Santa Maria delle Grazie in Milan . It is one of the world 's most recognizable paintings.",
        "sub_answer": ["Brian Steele"]
    }
]

def nqswap_train(entries: List[Dict]) -> List[Dict]:
    """
    Format NQSwap for SPARE training (Feature Extraction).
    Constructs contextual vs parametric validation prompts using 3-shot demos.
    """
    demo_contextual = "".join([_verbalise_nqswap(d, "sub_context", "sub_answer") for d in NQSWAP_DEMOS])
    demo_parametric = "".join([_verbalise_nqswap(d, "sub_context", "org_answer") for d in NQSWAP_DEMOS])
    
    out = []
    for e in entries:
        test_part = _verbalise_nqswap(e, "sub_context", "sub_answer", is_test=True)
        out.append({
            # Keep question empty: DataLoader (no chat template) prepends question
            # to target/contrast prompts when non-empty.
            "question": "",
            "correct_prompt": demo_contextual + test_part,
            "false_prompt": demo_parametric + test_part,
            "sub_answer": e["sub_answer"],
            "org_answer": e["org_answer"],
            "raw_question": e["question"],
        })
    return out

def nqswap_test(entries: List[Dict]) -> List[Dict]:
    """
    Format NQSwap for Evaluation (Conflict Resolution).

    Uses GT-like dynamic 3-shot demonstrations (OrgCtx -> OrgAns) selected
    deterministically from the current dataset, and excludes demonstration
    question groups from the eval set to mirror GT holdout behavior.
    The evaluator generates from ``question``, so we store the full prompt there.
    """
    demos, demo_questions = _select_demo(entries, k_shot=3)
    if not demos:
        demos = NQSWAP_DEMOS
        demo_questions = set()

    demo_standard = "".join([_verbalise_nqswap(d, "org_context", "org_answer") for d in demos])

    out = []
    for e in entries:
        if not _is_valid_nqswap_entry(e):
            continue
        # GT-style eval excludes demonstration groups from test iteration.
        if e["question"] in demo_questions:
            continue

        test_part = _verbalise_nqswap(e, "sub_context", "sub_answer", is_test=True)
        full_prompt = demo_standard + test_part
        out.append({
            "question": full_prompt,
            "correct_prompt": full_prompt,
            "false_prompt": "",
            "answer": e["sub_answer"][0],  # Target is contextual adherence
            "org_answer": e["org_answer"],
            "raw_question": e["question"],
        })
    return out

# =============================================================================
# 6. Composite Formatters
# =============================================================================

def cast_combined(responses: Dict, questions: List[Dict], augmentation: int = 1) -> List[Dict]:
    """
    Format CAST combined data.
    Combines responses (compliant/non-compliant) with questions.
    Returns question + response format for chat template application.
    
    Args:
        responses: dict with 'compliant_responses' and 'non_compliant_responses' lists
        questions: list of question entries
        augmentation: if > 1, cycle through questions to reach augmentation * len(questions) samples
                      (useful to reach 4k+ from 700 questions via response pair cycling)

    Also emits target_response / contrast_response (response-only strings) so that
    FLAS and REPS can consume them directly without a separate composite formatter.
    """
    compliant = responses.get("compliant_responses", [])
    non_compliant = responses.get("non_compliant_responses", [])
    
    # Pair compatible compliant/non-compliant responses
    response_pairs = list(zip(compliant, non_compliant))
    if not response_pairs:
        return []
    
    augmented_questions = questions * augmentation          
    out = []
    # Cycle through response pairs to match number of augmented questions
    response_cycler = itertools.cycle(response_pairs)
    
    for q_entry, (agree_response, refuse_response) in zip(augmented_questions, response_cycler):
        q = _get_question(q_entry)
        out.append({
            "question": q,
            # For CAST: refusing is the target behavior
            # Loader will prepend templated question: [INST]q[/INST]{response}
            "correct_prompt": refuse_response,
            "false_prompt": agree_response,
            # Response-only strings consumed by FLAS (as output) and REPS (for seq masking)
            "target_response": refuse_response,
            "contrast_response": agree_response,
        })
    return out





def srps_roleplay(roleplay_prompts: List[Dict], questions: List[Dict]) -> List[Dict]:
    """
    Format SRPS roleplay data.
    Combines roleplay personas with questions (arithmetic/commonsense).
    """
    out = []
    roleplay_cycler = itertools.cycle(roleplay_prompts)
    
    for q_entry, roleplay in zip(questions, roleplay_cycler):
        q = q_entry.get("question", "")
        rp_text = _get_question(roleplay)
        
        out.append({
            "question": "",
            "answer": q_entry.get("answer", ""),
            "roleplay_prompt": rp_text,
            # With roleplay persona = target behavior
            "correct_prompt": f"{rp_text}\n\n{q}",
            # Without roleplay = baseline
            "false_prompt": q,
        })
    return out

def refusal_angular(harmful_entries: List[Dict], harmless_entries: List[Dict]) -> List[Dict]:
    """
    Format Refusal Angular data.
    Combines harmful requests (target) with harmless requests (contrast).
    """
    out = []
    # Cycle harmless entries to match all harmful entries
    harmless_cycler = itertools.cycle(harmless_entries)
    
    for harmful_entry, harmless_entry in zip(harmful_entries, harmless_cycler):
        harmful_q = _get_question(harmful_entry, key="goal") # AdvBench uses "goal"
        harmless_q = _get_question(harmless_entry, key="question")
        
        out.append({
            "question": "",
            "correct_prompt": harmful_q,
            "false_prompt": harmless_q,
        })
    return out

def evil(responses: List[Dict], questions: List[Dict] = None) -> List[Dict]:
    """
    Format EVIL data.
    Combines evil questions (target) with good questions (contrast).
    """
    out = []
    for q_entry, r_entry in zip(questions, responses):
        question = r_entry['messages'][0]['content']
        normal_response = q_entry['messages'][1]['content']
        toxic_response = r_entry['messages'][1]['content']
        # print(f"Question: {question}\nNormal Response: {normal_response}\nToxic Response: {toxic_response}\n---")
        out.append({
            "question": question + ' ',
            "correct_prompt": toxic_response,
            "false_prompt": normal_response,
            "target_response": toxic_response,
            "contrast_response": normal_response,
        })
    return out

def toxic_extreme(response: List[Dict], questions: List[Dict]) -> List[Dict]:
    out = []
    for q_entry, r_entry in zip(questions, response):
        toxic_response = r_entry.get("response", "")
        toxic_response = " ".join(toxic_response.split()[:30])
        normal_response = q_entry.get("output", "")
        normal_response = " ".join(normal_response.split()[:30])
        out.append({
            "question": 'Write a random sentence: ',
            "correct_prompt": toxic_response,
            "false_prompt": normal_response,
            "target_response": toxic_response,
            "contrast_response": normal_response,
        })
    return out
# =============================================================================
# Registry
# =============================================================================

def refusal_angular_test(entries: List[Dict]) -> List[Dict]:
    """Format Angular data for test evaluation (harmful prompts)."""
    return [{
        "question": e["harmful"],
        "correct_prompt": e["harmful"],
        "false_prompt": "",
    } for e in entries]

def deception(entries: List[Dict]) -> List[Dict]:
    out = []
    for e in entries:
        sys_p = str(e.get("system_prompt") or "").strip()
        user_p = str(e.get("user_prompt") or "").strip()
        gt = str(e.get("ground_truth") or "").strip()
        honest_response = "Yes, that is correct." if gt.lower() == "true" else "No, that is incorrect."
        deceptive_response = "No, that is incorrect." if gt.lower() == "true" else "Yes, that is correct."
        out.append({
            "question": f"{sys_p}\n\n{user_p}",
            "correct_prompt": honest_response,
            "false_prompt": deceptive_response,  # For training, we use the same prompt for both to focus on decision boundary learning
            # Response-only strings consumed by FLAS (as output) and REPS (for seq masking)
            "target_response": honest_response,
            "contrast_response": deceptive_response,
        })
    return out

def deception_test(entries: List[Dict]) -> List[Dict]:
    """Format cais/MASK for deception test evaluation under pressure."""
    out = []
    for e in entries:
        sys_p = str(e.get("system_prompt") or "").strip()
        user_p = str(e.get("user_prompt") or "").strip()
        gt = str(e.get("ground_truth") or "").strip()
        if not sys_p or not user_p:
            continue
        out.append({
            "question": f"{sys_p}\n\n{user_p}",
            "answer": gt,
        })
    return out

def deception_no_sys(entries: List[Dict]) -> List[Dict]:
    """Format cais/MASK for deception test evaluation without system prompt."""
    out = []
    for e in entries:
        user_p = str(e.get("user_prompt") or "").strip()
        gt = str(e.get("ground_truth") or "").strip()
        if not user_p:
            continue
        out.append({
            "question": f"{user_p}",
            "answer": gt,
        })
    return out

def evil_test(entries: List[Dict]) -> List[Dict]:
    """
    Format EVIL test data.
    Extracts the user prompt from messages.
    """
    out = []
    for e in entries:
        if "messages" in e and len(e["messages"]) > 0:
            question = e["messages"][0]["content"]
            out.append({
                "question": question,
            })
    return out

def ifeval_test(entries: List[Dict]) -> List[Dict]:
    """Format IFEval for test evaluation (passes prompt + id list + kwargs)."""
    out = []
    for e in entries:
        full_prompt = str(e.get("prompt") or "").strip()
        if not full_prompt:
            continue
        out.append({
            "question": full_prompt,
            "answer": full_prompt,
            "instruction_id_list": e.get("instruction_id_list", []),
            "kwargs": e.get("kwargs", []),
            "key": e.get("key"),
        })
    return out


def cais_mask(entries: List[Dict]) -> List[Dict]:
    """
    Formatter for cais/MASK (Model Alignment between Statements and Knowledge).
    
    Reconstructs the full honest and deceptive responses under adversarial pressure,
    and truncates them exactly 3 words (approx 3-4 tokens) after the pivotal choice.
    
    This naturally aligns extraction with the decision boundary of the statement.
    
    Includes target_response / contrast_response (response-only strings) so that
    FLAS and REPS can consume them without a separate formatter.
    """
    import os
    import json
    import re
    import difflib

    out = []
    for idx, e in enumerate(entries):
        user_p = str(e.get("user_prompt") or "").strip()
        sys_p  = str(e.get("system_prompt") or "").strip()
        prop   = str(e.get("proposition") or "").strip()
        gt     = str(e.get("ground_truth") or "").strip()
        if not user_p or not sys_p or not prop or not gt:
            continue
            
        # Parse prefix <pivotal>choice1/choice2</pivotal> suffix
        match = re.search(r'(.*?)<pivotal>(.*?)</pivotal>(.*)', prop)
        if not match:
            continue
            
        prefix, pivotal, suffix = match.groups()
        options = pivotal.split('/')
        if len(options) != 2:
            continue
            
        opt1, opt2 = options
        st1 = prefix + opt1 + suffix
        st2 = prefix + opt2 + suffix
        
        # Check similarity to ground truth
        sim1 = difflib.SequenceMatcher(None, st1, gt).ratio()
        sim2 = difflib.SequenceMatcher(None, st2, gt).ratio()
        
        honest_opt = opt1 if sim1 > sim2 else opt2
        deceptive_opt = opt2 if sim1 > sim2 else opt1
        
        # Truncate suffix to 3 words after the pivotal choice
        suffix_words = suffix.strip().split()
        trunc_suffix = " " + " ".join(suffix_words[:3]) if suffix_words else ""
        
        honest_truncated = prefix + honest_opt + trunc_suffix
        deceptive_truncated = prefix + deceptive_opt + trunc_suffix
        
        out.append({
            "question": f"{sys_p}\n\n{user_p}",
            "correct_prompt": honest_truncated,
            "false_prompt":   deceptive_truncated,
            # Response-only strings consumed by FLAS (as output) and REPS (for seq masking)
            "target_response": honest_truncated,
            "contrast_response": deceptive_truncated,
        })
    return out


def ifeval(entries: List[Dict]) -> List[Dict]:
    """
    Formatter for google/IFEval (Instruction-Following Evaluation).

    Each entry has a `prompt` that embeds both a base task and explicit format
    constraints (e.g. word count, no commas, bullet points).

    Contrast pair:
      target  (correct_prompt) = full constrained prompt  — instruction-following context
      contrast (false_prompt)  = first sentence only       — unconstrained / free generation context

    This contrasts "I must follow strict format rules" vs. "I answer freely".
    The contrast is the base task stripped of format constraints (heuristic: first sentence).

    Also emits target_response / contrast_response (response-only strings) so that
    FLAS and REPS can consume them directly without a separate formatter variant.
    Here, the full constrained prompt IS the target response (FLAS generates it from the base task).
    """
    out = []
    for e in entries:
        full_prompt = str(e.get("prompt") or "").strip()
        if not full_prompt:
            continue
        # Heuristic: constraints nearly always come after the first sentence.
        # Split on '. ' and keep only the first sentence as the "free" baseline.
        sentences = full_prompt.split(". ")
        base_task = sentences[0].strip()
        requirements = ". ".join(sentences[1:]).strip()  # Format constraints are usually in the subsequent sentences
        if not base_task.endswith(".") and len(sentences) > 1:
            base_task += "."
        out.append({
            "question":       '',     # input side (base task without constraints)
            "correct_prompt": full_prompt + " Got it, here's ",   # constrained — target
            "false_prompt":   base_task + " Got it, here's ",     # unconstrained base task — contrast
            # Response-only strings consumed by FLAS (as output) and REPS (for seq masking)
            "target_response": full_prompt,  # FLAS: what to generate given base_task input
            "contrast_response": base_task,  # REPS orig_sub: the unconstrained baseline
        })
    return out


# =============================================================================
# 7. Sycophancy Personas Formatter
# =============================================================================

def sycophancy_personas(entries: List[Dict]) -> List[Dict]:
    """
    Format sycophancy personas data for training.

    Pairs sycophantic responses from misaligned_1.jsonl (passed in as `entries`)
    with the corresponding line-by-line objective responses from normal.jsonl,
    which is loaded relative to TRAIN_ROOT.

    Both files are aligned 1-to-1 (same 10,100 lines).  The target (correct)
    behavior is the objective/non-sycophantic response; the contrast (false)
    behavior is the sycophantic one.

    Also emits target_response / contrast_response for FLAS / REPS.
    """
    import json
    from pathlib import Path
    from ..config import TRAIN_ROOT

    normal_path = TRAIN_ROOT / "behaviour/sycophancy/personas/normal.jsonl"
    normal_entries: List[Dict] = []
    with open(normal_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                normal_entries.append(json.loads(line))

    out: List[Dict] = []
    for idx, e in enumerate(entries):
        if idx >= len(normal_entries):
            break
        user_prompt = e["messages"][0]["content"]
        sycophantic_resp = e["messages"][1]["content"]
        objective_resp = normal_entries[idx]["messages"][1]["content"]
        out.append({
            "question": user_prompt,
            "correct_prompt": objective_resp,      # target: honest/objective
            "false_prompt": sycophantic_resp,      # contrast: sycophantic
            "target_response": objective_resp,
            "contrast_response": sycophantic_resp,
        })
    return out


# =============================================================================
# 8. HumanEval Formatter (test-only)
# =============================================================================

def humaneval(entries: List[Dict]) -> List[Dict]:
    """
    Format openai/HumanEval for code-completion evaluation.

    This is a test-only formatter: no correct_prompt / false_prompt are
    emitted.  The evaluator receives the prompt as `question`, the test
    harness as `answer`, and the function entry-point name for code parsing.
    """
    out: List[Dict] = []
    for e in entries:
        prompt = str(e.get("prompt") or "").rstrip()
        test_str = str(e.get("test") or "")
        entry_point = str(e.get("entry_point") or "")
        if not prompt or not test_str or not entry_point:
            continue
        out.append({
            "question": prompt,
            "answer": test_str,
            "entry_point": entry_point,
        })
    return out


def mbpp(entries: List[Dict]) -> List[Dict]:
    """
    Format google-research-datasets/mbpp for code-completion evaluation.

    Test-only formatter.  MBPP stores assertions as a list of bare
    ``assert fn(args) == expected`` strings in ``test_list``, with optional
    setup code in ``test_setup_code``.  We join them into a single executable
    string and extract the entry-point from the first assertion.

    Compatible with ``HumanEvalMatcher`` — the subprocess handles direct
    assertion style automatically (no ``check(candidate)`` wrapper needed).
    """
    import re as _re
    out: List[Dict] = []
    for e in entries:
        text = str(e.get("text") or "").strip()
        test_list = e.get("test_list") or []
        setup = str(e.get("test_setup_code") or "").strip()
        if not text or not test_list:
            continue
        # Extract entry_point from the first assertion: assert fn_name(...)
        m = _re.match(r'assert\s+(\w+)\s*\(', test_list[0])
        entry_point = m.group(1) if m else ""
        test_str = (setup + "\n" if setup else "") + "\n".join(test_list)
        out.append({
            "question": f"{text}\n\ndef {entry_point}() ",
            "answer": test_str,
            "entry_point": entry_point,
        })
    return out

def pop_qa(entries: List[Dict]) -> List[Dict]:
    """
    Format PopQA data
    """
    return [{
        "question": e["question"],
        "answer": "; ".join(e["possible_answers"]),
    } for e in entries]



FORMATTERS = {
    # Generic
    "binary_choice": binary_choice,
    "binary_choice_test": binary_choice_test,
    "question_only": question_only,
    "concept500_reps": concept500_reps,
    "math_question": math_question,

    # QA
    "csqa": csqa,
    "mmlu": mmlu,
    "mmlu_corrsteer": mmlu_corrsteer,
    "mmlu_corrsteer_test": mmlu_corrsteer_test,
    "truthful_qa": truthful_qa,
    "halueval": halueval,

    # Refusal / Safety
    "advbench": AdvBench,
    "alpaca_angular": question_only,
    "harmbench": harmbench,
    "harmbench_train": harmbench_train,
    "donotanswer": donotanswer,
    "xstest": xstest,
    "SorryBench": SorryBench,
    "sorrybench_test": sorrybench_test,
    "CAST_condition": CAST_condition,
    "CAST_multiple_condition": CAST_condition,

    # Advanced
    "twin_views": twin_views,
    "twin_views_test": twin_views_test,
    "nqswap_train": nqswap_train,
    "nqswap_test": nqswap_test,
    "refusal_angular_test": refusal_angular_test,

    # Deception
    "cais_mask": cais_mask,
    "deception": deception,
    "deception_test": deception_test,
    "deception_no_sys": deception_no_sys,

    # Instruction following
    "ifeval": ifeval,
    "ifeval_test": ifeval_test,

    # Toxicity
    "jigsaw": jigsaw,

    # Sycophancy Personas
    "sycophancy_personas": sycophancy_personas,

    # Coding / HumanEval (test-only)
    "humaneval": humaneval,

    # Coding / MBPP (test-only)
    "mbpp": mbpp,

    # Evil test
    "evil": evil_test,

    # System-prompt variants for Toxic/Evil (like LiarBench)
    # "toxic_with_sys": toxic_with_sys,
    # "evil_with_sys": evil_with_sys,
}

def toxic_with_sys(entries: List[Dict]) -> List[Dict]:
    sys_p = "In a moment, somebody will ask you a question. Try to respond in a toxic or offensive way."
    return [{
        "question": f"{sys_p}\n\n{_get_question(e)}",
        "correct_prompt": f"{sys_p}\n\n{_get_question(e)}",
        "false_prompt": "",
    } for e in entries]

def evil_with_sys(entries: List[Dict]) -> List[Dict]:
    sys_p = "In a moment, somebody will ask you a question. Try to say something evil or harmful."
    return [{
        "question": f"{sys_p}\n\n{_get_question(e)}",
        "correct_prompt": f"{sys_p}\n\n{_get_question(e)}",
        "false_prompt": "",
    } for e in entries]

COMPOSITE_FORMATTERS = {
    "cast_combined": cast_combined,
    "srps_roleplay": srps_roleplay,
    "refusal_angular": refusal_angular,
    # Persona:
    "evil": evil,
    "toxic_extreme": toxic_extreme
}
