
from ..base import BaseEvaluator

import types
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List, Dict

# =============================================================================
# Perplexity Matcher
# =============================================================================
class PerplexityMatcher(BaseEvaluator):
    """
    Computes conditional perplexity PPL(response | prompt).
    Returns generated text as response_str, and perplexity as confidence.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temp_ppls = []

    def _get_responses(
        self,
        steer_model,
        samples: List[dict],
        coeff: Dict[int, float],
        **kwargs,
    ) -> List[str]:

        self._temp_ppls = []
        results = []

        for sample in samples:
            prompt = sample["question"]

            # generate response
            generated = steer_model.generate(
                [prompt],
                coeff=coeff,
                **kwargs
            )[0]

            # compute conditional perplexity
            perplexity = self._compute_perplexity(
                steer_model.model,
                prompt,
                generated
            )

            self._temp_ppls.append(perplexity)
            results.append(generated)

        return results

    def check(self, response, ground_truth=None, **context) -> Tuple[int, float]:

        if not self._temp_ppls:
            return 0, 0.0

        current_ppl = self._temp_ppls.pop(0)

        return 1, float(current_ppl)

    @staticmethod
    def _compute_perplexity(
        model,
        prompt: str,
        response: str,
        max_length: int = 512
    ) -> float:
        """Compute PPL(response | prompt) using HookedTransformer."""

        try:
            device = model.cfg.device

            prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
            full_tokens = model.to_tokens(prompt + response, prepend_bos=True)

            if full_tokens.shape[1] > max_length:
                full_tokens = full_tokens[:, :max_length]

            prompt_len = prompt_tokens.shape[1]

            full_tokens = full_tokens.to(device)

            with torch.no_grad():

                loss_per_token = model(
                    full_tokens,
                    return_type="loss",
                    loss_per_token=True
                )

                # remove prompt tokens
                response_loss = loss_per_token[:, prompt_len - 1:]

                perplexity = torch.exp(response_loss.mean()).item()

            return perplexity

        except Exception as e:
            print(f"PPL Calculation Error: {e}")
            return float("inf")

class CoherenceMatcher(BaseEvaluator):
    """
    Flow-Judge-based coherence evaluator, hardware-locked for Phi-3 stability.
    """
    is_model_based = True

    def __init__(
        self,
        device: str,
        model_name: str = "flowaicom/Flow-Judge-v0.1",
        use_compile: bool = False, # Compile often breaks with monkey-patches
        threshold: int = 3,
        max_new_tokens: int = 512, # Increased for CoT feedback
    ):
        self.device = device
        self.model_name = model_name
        self.use_compile = use_compile
        self.threshold = threshold
        self.max_new_tokens = max_new_tokens

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load with explicit mapping to avoid split-device errors
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device}
        )

        self._model.eval()

    def _build_prompt(self, user_prompt, response):
        # Using the author's recommended rubric-style template
        return f"""# GOAL
Your job is to evaluate the coherence and logical consistency of an AI assistant's response.

# OUTPUT
<output>
USER: {user_prompt}
ASSISTANT: {response}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
<evaluation_criteria>
Coherence: The response should be logically sound, maintain critical independence, and provide clear justification.
</evaluation_criteria>

<scoring_rubric>
- Score 5: Fully coherent, logically independent, and well-justified.
- Score 4: Mostly coherent with minor logical or justificatory gaps.
- Score 3: Neutral; the response is understandable but lacks depth or critical independence.
- Score 2: Largely incoherent or fails to address the user's premise logically.
- Score 1: Totally contradictory or nonsensical.
</scoring_rubric>

# INSTRUCTIONS
1. Analyze the logic of the assistant's response in the context of the user prompt.
2. Write a brief rationale in the feedback section.
3. Assign a numeric score from 1 to 5.

## FORMAT
- <feedback>Your rationale here</feedback>
- <score>Final numeric score</score>"""

    def check(self, response, ground_truth=None, **context) -> Tuple[int, float]:
        prompt = context.get("prompt", "")
        score, feedback = self._judge(prompt, response)

        # Normalize score to 0-1 for confidence
        confidence = score / 5.0
        # Binary correctness based on threshold
        is_correct = 1 if score >= self.threshold else 0

        return self._apply_ground_truth(is_correct, ground_truth), confidence

    def _judge(self, user_prompt: str, assistant_response: str):
        self._load_model()
        judge_prompt = self._build_prompt(user_prompt, assistant_response)

        inputs = self._tokenizer(
            judge_prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        prompt_len = inputs.input_ids.shape[1]
        position_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                position_ids=position_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return self._parse_output(text)

    @staticmethod
    def _parse_output(text: str):
        score_match = re.search(r"<score>\s*(\d+)\s*</score>", text)
        feedback_match = re.search(r"<feedback>(.*?)</feedback>", text, re.DOTALL)

        score = int(score_match.group(1)) if score_match else 3
        feedback = feedback_match.group(1).strip() if feedback_match else ""
        return score, feedback

    @staticmethod
    def _apply_ground_truth(pred_int: int, ground_truth):
        if ground_truth is not None and str(ground_truth).strip():
            try:
                gt_int = int(ground_truth)
                return int(pred_int == gt_int)
            except ValueError:
                return pred_int
        return pred_int