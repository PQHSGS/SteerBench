"""Core training loop for finetune baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler

from ..data import DataLoader as SteeringDataLoader
from ..data import EvalDataLoader as SteeringEvalDataLoader
from ..evaluators import EVALUATOR_MAP
from ..logger import setup_logger
from ..utils import build_chat_input
from .lora import get_backend
from .config import FinetuneConfig, slugify
from .registry import get_method_spec


logger = setup_logger(__name__)


def _join_prompt_completion(prompt: str, completion: str) -> str:
    prompt = prompt.strip()
    completion = completion.strip()
    if not prompt:
        return completion
    if not completion:
        return prompt
    if prompt.endswith((" ", "\n", "\t")):
        return f"{prompt}{completion}"
    return f"{prompt} {completion}"


class PromptCompletionDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.pairs[index]


class FinetuneSteerModelWrapper:
    def __init__(self, model, tokenizer, device, use_autocast, autocast_dtype, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype
        self.config = config
        self.metadata = []

    def generate(self, prompts: List[str], coeff=None, **kwargs) -> List[str]:
        results = []
        pbar = tqdm(prompts, desc="Generating evaluation responses", leave=False) if len(prompts) > 1 else prompts
        for prompt in pbar:
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            input_length = int(encoded["input_ids"].shape[1])

            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_autocast):
                    generated = self.model.generate(
                        **encoded,
                        max_new_tokens=self.config.eval_max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            response_tokens = generated[0, input_length:]
            results.append(self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip())
        return results

    def get_token_probs(self, prompt: str, tokens: List[str], coeff=None, **kwargs) -> Dict[str, float]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_autocast):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                
                result = {}
                total = 0.0
                for t in tokens:
                    tid = self.tokenizer.encode(t, add_special_tokens=False)[-1]
                    result[t] = float(probs[tid].item())
                    total += result[t]
                
                for t in tokens:
                    result[t] /= (total + 1e-9)
                return result

    def get_output_metadata(self) -> List[Any]:
        return [None] * 10000


class FinetuneTrainer:
    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        from datetime import datetime
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.method_spec = config.method
        self.backend = get_backend(config.backend, config, self.method_spec)

    def _build_training_pairs(self, tokenizer) -> List[Dict[str, str]]:
        loader = SteeringDataLoader()
        rows = loader.load(
            dataset_name=self.config.train_dataset,
            n_samples=self.config.n_train,
            format=True,
            apply_chat_template=False,
        )

        if self.config.inverse:
            logger.info("Finetune inverse=True: swapped target and contrast training prompts and responses.")
            for row in rows:
                if "correct_prompt" in row and "false_prompt" in row:
                    row["correct_prompt"], row["false_prompt"] = row["false_prompt"], row["correct_prompt"]
                if "target_response" in row and "contrast_response" in row:
                    row["target_response"], row["contrast_response"] = row["contrast_response"], row["target_response"]

        pairs: List[Dict[str, str]] = []
        for row in rows:
            prompt = str(row.get(self.method_spec.prompt_field, row.get("question", "")))
            completion = str(row.get(self.method_spec.train_field, ""))
            if not prompt.strip() and not completion.strip():
                continue

            if self.config.apply_chat_template and prompt.strip():
                prompt = build_chat_input(tokenizer, prompt, add_generation_prompt=True)

            pairs.append({"prompt": prompt, "completion": completion})

        if not pairs:
            raise RuntimeError(f"No training texts found for dataset '{self.config.train_dataset}'")

        return pairs

    def _build_eval_pairs(self, tokenizer) -> tuple[List[Dict[str, Any]], Any]:
        if not self.config.test_dataset:
            return [], None

        eval_loader = SteeringEvalDataLoader()
        eval_cfg = eval_loader.get_config(self.config.test_dataset)
        rows = eval_loader.load(
            dataset_name=self.config.test_dataset,
            n_samples=self.config.n_test,
            format=True,
            apply_chat_template=False,
        )

        pairs: List[Dict[str, Any]] = []
        for row in rows:
            prompt = str(row.get("question", "")).strip()
            if not prompt:
                continue

            if self.config.apply_chat_template:
                prompt = build_chat_input(tokenizer, prompt, add_generation_prompt=True)

            completion = str(eval_cfg.prefix or "").strip()
            if not completion:
                for candidate_key in ("answer", "correct_prompt", "response", "output"):
                    candidate = str(row.get(candidate_key, "")).strip()
                    if candidate:
                        completion = candidate
                        break

            ground_truth = row.get(eval_cfg.ground_truth_key) if eval_cfg.ground_truth_key else None
            if ground_truth is None and eval_cfg.evaluator == "refusal":
                ground_truth = 1

            sample_dict = dict(row)
            sample_dict.update({
                "prompt": prompt,
                "question": prompt,
                "completion": completion,
                "ground_truth": ground_truth,
                "answer": ground_truth,
                "sample_data": row,
            })
            pairs.append(sample_dict)

        return pairs, eval_cfg

    def _build_loader(self, pairs: List[Dict[str, str]], tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = PromptCompletionDataset(pairs)
        add_specials = not self.config.apply_chat_template

        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            encoded_items: List[Dict[str, Any]] = []
            prompt_lengths: List[int] = []

            for item in batch:
                prompt = str(item["prompt"])
                completion = str(item["completion"])
                full_text = _join_prompt_completion(prompt, completion)
                encoded = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_attention_mask=True,
                    add_special_tokens=add_specials,
                    return_tensors=None,
                )
                encoded_items.append(encoded)
                prompt_lengths.append(len(tokenizer(prompt, add_special_tokens=add_specials)["input_ids"]))

            encoded = tokenizer.pad(encoded_items, padding=True, return_tensors="pt")
            labels = encoded["input_ids"].clone()
            for row_index, prompt_length in enumerate(prompt_lengths):
                labels[row_index, : min(prompt_length, labels.shape[1])] = -100
            labels[encoded["attention_mask"] == 0] = -100
            encoded["labels"] = labels
            return encoded

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def _move_batch(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {key: value.to(device) for key, value in batch.items()}

    def _generate_response(self, model, tokenizer, prompt: str, device: torch.device, use_autocast: bool, autocast_dtype: torch.dtype) -> str:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.config.eval_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        response_tokens = generated[0, input_length:]
        return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    def fit(self) -> Dict[str, Any]:
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        train_flops = 0
        eval_flops = 0

        tokenizer = self.backend.load_tokenizer()
        base_model = self.backend.load_base_model()
        if self.config.load_vector:
            logger.info("Loading pre-trained adapter from %s for evaluation only", self.config.load_vector)
            model = self.backend.load_adapter(base_model, self.config.load_vector, is_trainable=False)
            device = torch.device(self.config.device)
            model.to(device)

            train_pairs = []
            test_pairs, test_cfg = self._build_eval_pairs(tokenizer)
            test_loader = (
                self._build_loader(test_pairs, tokenizer, self.config.eval_batch_size, shuffle=False)
                if test_pairs and self.config.compute_perplexity
                else None
            )
            eval_sample_records: List[Dict[str, Any]] = []

            use_autocast = device.type == "cuda" and self.config.dtype in {"float16", "bfloat16"}
            autocast_dtype = torch.float16 if self.config.dtype == "float16" else torch.bfloat16

            history: List[float] = []
            global_step = 0
        else:
            model = self.backend.prepare_model(base_model)
            device = torch.device(self.config.device)
            model.to(device)

            train_pairs = self._build_training_pairs(tokenizer)
            test_pairs, test_cfg = self._build_eval_pairs(tokenizer)

            train_loader = self._build_loader(train_pairs, tokenizer, self.config.train_batch_size, shuffle=True)
            test_loader = (
                self._build_loader(test_pairs, tokenizer, self.config.eval_batch_size, shuffle=False)
                if test_pairs and self.config.compute_perplexity
                else None
            )
            eval_sample_records: List[Dict[str, Any]] = []

            total_steps = max(1, len(train_loader) * max(self.config.num_epochs, 1) // max(self.config.grad_accum, 1))
            warmup_steps = int(total_steps * self.config.warmup_ratio)

            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            use_autocast = device.type == "cuda" and self.config.dtype in {"float16", "bfloat16"}
            autocast_dtype = torch.float16 if self.config.dtype == "float16" else torch.bfloat16

            history: List[float] = []
            global_step = 0
            model.train()

            from Steering.utils import FlopTracker
            with FlopTracker() as tracker:
                for epoch in range(self.config.num_epochs):
                    epoch_losses: List[float] = []
                    progress = tqdm(train_loader, desc=f"Finetune epoch {epoch + 1}/{self.config.num_epochs}")
                    optimizer.zero_grad(set_to_none=True)

                    for step, batch in enumerate(progress, start=1):
                        batch = self._move_batch(batch, device)

                        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                            outputs = model(**batch)
                            loss = outputs.loss / max(self.config.grad_accum, 1)

                        loss.backward()
                        epoch_losses.append(float(loss.item()) * max(self.config.grad_accum, 1))

                        if step % self.config.grad_accum == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                            progress.set_postfix(loss=f"{epoch_losses[-1]:.4f}")

                    if len(train_loader) % self.config.grad_accum != 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                    mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
                    history.append(mean_loss)
                    logger.info("Epoch %d completed with train loss %.4f", epoch + 1, mean_loss)
            
            train_flops = tracker.total_flops
            logger.info("Training FLOPs: %s", f"{train_flops:,}")

        test_loss = None
        test_ppl = None
        test_accuracy = None
        test_mean_confidence = None
        test_evaluator_name = getattr(test_cfg, "evaluator", None) if test_cfg is not None else None
        test_evaluator = EVALUATOR_MAP[test_evaluator_name](device=self.config.device) if test_evaluator_name in EVALUATOR_MAP else None
        if test_loader is not None:
            model.eval()
            test_losses: List[float] = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = self._move_batch(batch, device)
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                        outputs = model(**batch)
                    test_losses.append(float(outputs.loss.item()))

            if test_losses:
                test_loss = sum(test_losses) / len(test_losses)
                test_ppl = float(math.exp(min(20.0, test_loss)))
            model.train()

        add_specials = not self.config.apply_chat_template

        # Helper to compute conditional perplexity PPL(completion | prompt)
        def _compute_perplexity_for_text(prompt: str, completion: str) -> float:
            try:
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model_device = next(model.parameters()).device
                prompt_ids = tokenizer(prompt, add_special_tokens=add_specials).input_ids
                full_ids = tokenizer(prompt + completion, add_special_tokens=add_specials).input_ids
                full_ids = full_ids[: self.config.max_length]
                prompt_len = min(len(prompt_ids), len(full_ids))
                if prompt_len >= len(full_ids):
                    return float("inf")

                input_ids = torch.tensor([full_ids], dtype=torch.long, device=model_device)
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, labels=labels)
                    return float(torch.exp(outputs.loss).item())
            except Exception:
                return float("inf")

        if test_pairs and test_evaluator is not None:
            model.eval()
            wrapped_model = FinetuneSteerModelWrapper(
                model=model,
                tokenizer=tokenizer,
                device=device,
                use_autocast=use_autocast,
                autocast_dtype=autocast_dtype,
                config=self.config,
            )
            logger.info("Evaluating on %d test samples using %s evaluator...", len(test_pairs), test_evaluator_name)
            from Steering.utils import FlopTracker
            with FlopTracker() as tracker:
                eval_results = test_evaluator.batch(
                    steer_model=wrapped_model,
                    samples=test_pairs,
                    max_new_tokens=self.config.eval_max_new_tokens,
                )
            eval_flops = tracker.total_flops
            logger.info("Evaluation FLOPs: %s", f"{eval_flops:,}")

            sample_scores: List[float] = []
            sample_correct: List[int] = []
            sample_ppls: List[float] = []
            for index, (sample, (is_correct, confidence, response, data)) in enumerate(zip(test_pairs, eval_results), start=1):
                # compute per-sample perplexity on the ground-truth completion
                # compute per-sample perplexity on the generated response (match Steering.pipeline)
                ppl = _compute_perplexity_for_text(sample["question"], response)
                sample_ppls.append(ppl)

                sample_scores.append(float(confidence))
                sample_correct.append(int(is_correct))
                eval_sample_records.append({
                    "index": index,
                    "prompt": sample["question"],
                    "response": response,
                    "ground_truth": sample.get("ground_truth"),
                    "is_correct": int(is_correct),
                    "verdict": "correct" if int(is_correct) else "incorrect",
                    "confidence": float(confidence),
                    "score": float(confidence),
                    "perplexity": float(ppl),
                    "sample_data": sample.get("sample_data"),
                })

            if sample_correct:
                test_accuracy = sum(sample_correct) / len(sample_correct)
            if sample_scores:
                test_mean_confidence = sum(sample_scores) / len(sample_scores)
            if sample_ppls:
                finite = [p for p in sample_ppls if p != float("inf")]
                test_ppl = sum(finite) / len(finite) if finite else float("inf")
            model.train()

        output_dir = Path(self.config.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{slugify(self.config.name)}_{self.timestamp}.json"

        results_to_save = {
            "config": self.config.to_dict(),
            "method": self.method_spec.to_dict(),
            "metrics": {
                "train_loss_history": history,
                "final_train_loss": history[-1] if history else None,
                "test_loss": test_loss,
                "test_perplexity": test_ppl,
                "test_accuracy": test_accuracy,
                "test_mean_confidence": test_mean_confidence,
                "global_steps": global_step,
                "num_train_texts": len(train_pairs),
                "num_test_texts": len(test_pairs),
                "train_flops": train_flops,
                "eval_flops": eval_flops,
                "extraction_flops": train_flops,
                "inference_flops": eval_flops,
            }
        }

        if not self.config.load_vector:
            if self.config.save_vector:
                save_vector_path = Path(self.config.save_vector)
            else:
                save_vector_path = Path("./Vector/FINETUNE") / slugify(self.config.name)

            save_vector_path.mkdir(parents=True, exist_ok=True)
            self.backend.save_adapter(model, save_vector_path)
            tokenizer.save_pretrained(save_vector_path)
            results_to_save["metrics"]["adapter_dir"] = str(save_vector_path)

        if eval_sample_records:
            results_to_save["samples"] = eval_sample_records

        with open(results_file, "w") as handle:
            json.dump(results_to_save, handle, indent=2, ensure_ascii=False, default=str)

        logger.info("Finetune complete. Results saved to: %s", results_file)
        return results_to_save