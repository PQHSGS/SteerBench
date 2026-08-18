"""
Weight Steering Extractor.
"""

from typing import List, Optional, Dict, Any, Union
import re
import torch
from ..base import BaseExtractor
from ..logger import setup_logger
from .nonlinear import _get_completion_masked_labels

logger = setup_logger(__name__)
class WeightSteerExtractor(BaseExtractor):
    """
    Contrastive Weight Steering Extractor.
    
    Fine-tunes base model parameters on positive and negative behavior prompts,
    isolating a behavior direction in weight-space by computing the difference
    between the two sets of fine-tuned weights: w_b = theta_positive - theta_negative.
    
    Paper: "Steering Language Models with Weight Arithmetic"
    """

    METHOD_NAME = "WEIGHTSTEER"

    def __init__(
        self,
        model,
        layer: List[int],
        model_name: str,
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        hook_point: Union[str, List[str]] = "pre",
        weight_steer_lr: float = 1e-4,
        weight_steer_epochs: int = 3,
        weight_steer_target_modules: Optional[List[str]] = None,
        weight_steer_lora_r: int = 32,
        weight_steer_lora_alpha: int = 64,
        weight_steer_lora_dropout: float = 0.05,
        weight_steer_lambda_sparse: float = 0.0,
        weight_steer_grad_accum: int = 8,
        steering_factors: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point)
        self.model_name = model_name
        self.lr = weight_steer_lr
        self.epochs = weight_steer_epochs
        self.lora_r = weight_steer_lora_r
        self.lora_alpha = weight_steer_lora_alpha
        self.lora_dropout = weight_steer_lora_dropout
        self.lambda_sparse = weight_steer_lambda_sparse
        self.grad_accum = max(1, weight_steer_grad_accum)
        self.improved = (self.lambda_sparse is not None and self.lambda_sparse > 0.0)
        self.steering_factors = steering_factors

        if weight_steer_target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            self.target_modules = weight_steer_target_modules

    def _get_activations(self, inputs: List[str], **kwargs) -> torch.Tensor:
        raise NotImplementedError("WeightSteer does not use activation averages.")

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """
        Perform SFT on positive and negative datasets using LoRA, then compute parameter weight differences.
        If weight_steer_improved is True, optimizes a single set of LoRA adapters using a bidirectional loss
        and Group Lasso sparsity.
        """
        import gc
        import random
        import re
        from tqdm import tqdm
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        # 1. Temporarily move base TL model to CPU to free GPU memory
        original_tl_device = self.device
        self.model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        def set_lora_scaling(model_peft, scale: float):
            for m in model_peft.modules():
                if hasattr(m, "scaling"):
                    if isinstance(m.scaling, dict):
                        m.scaling["default"] = scale
                    else:
                        m.scaling = scale

        if self.improved:
            logger.info("Running Improved Weight Steering (Version 1) training...")

            model_hf_current = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.model.cfg.dtype,
                device_map=str(original_tl_device),
            )

            peft_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                layers_to_transform=self.layer,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model_peft = get_peft_model(model_hf_current, peft_config)
            model_peft.print_trainable_parameters()

            # Group parameters by layer index for Group Lasso calculation
            layer_groups = {}
            for name, param in model_peft.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    match = re.search(r"layers\.(\d+)\.", name)
                    if match:
                        layer_idx = int(match.group(1))
                        if layer_idx not in layer_groups:
                            layer_groups[layer_idx] = {}
                        submodule_path = name.split(".lora_")[0]
                        if submodule_path not in layer_groups[layer_idx]:
                            layer_groups[layer_idx][submodule_path] = {}
                        if "lora_A" in name:
                            layer_groups[layer_idx][submodule_path]["A"] = param
                        elif "lora_B" in name:
                            layer_groups[layer_idx][submodule_path]["B"] = param

            optimizer = torch.optim.AdamW(model_peft.parameters(), lr=self.lr)

            # Align target and contrast data
            paired_data = []
            if contrast_data:
                min_len = min(len(target_data), len(contrast_data))
                for i in range(min_len):
                    paired_data.append((target_data[i], contrast_data[i]))
            else:
                for item in target_data:
                    paired_data.append((item, None))

            try:
                for epoch in range(self.epochs):
                    shuffled_data = list(paired_data)
                    random.shuffle(shuffled_data)

                    pbar = tqdm(
                        range(0, len(shuffled_data), self.batch_size),
                        desc=f"IWS LoRA Training Epoch {epoch + 1}/{self.epochs}",
                        leave=True,
                    )

                    epoch_loss = 0.0
                    num_batches = 0
                    optimizer.zero_grad(set_to_none=True)

                    for batch_idx, idx in enumerate(pbar):
                        batch_pairs = shuffled_data[idx : idx + self.batch_size]
                        if not batch_pairs:
                            continue

                        batch_pos = [p[0] for p in batch_pairs]
                        batch_neg = [p[1] for p in batch_pairs if p[1] is not None]

                        if self.steering_factors:
                            s = random.choice(self.steering_factors)
                        else:
                            s = 1.0

                        # Forward pass on target data with +s scaling
                        set_lora_scaling(model_peft, (self.lora_alpha / self.lora_r) * s)

                        tokens_pos, labels_pos, mask_pos, _ = _get_completion_masked_labels(
                            self.model, batch_pos
                        )
                        tokens_pos = tokens_pos.to(model_hf_current.device)
                        labels_pos = labels_pos.to(model_hf_current.device)
                        mask_pos = mask_pos.to(model_hf_current.device)

                        outputs_pos = model_peft(tokens_pos, attention_mask=mask_pos)
                        shift_logits_pos = outputs_pos.logits[..., :-1, :].contiguous()
                        shift_labels_pos = labels_pos[..., 1:].contiguous()
                        loss_pos = loss_fn(
                            shift_logits_pos.view(-1, shift_logits_pos.size(-1)),
                            shift_labels_pos.view(-1),
                        )

                        # Forward pass on contrast data with -s scaling
                        if batch_neg:
                            set_lora_scaling(model_peft, - (self.lora_alpha / self.lora_r) * s)

                            tokens_neg, labels_neg, mask_neg, _ = _get_completion_masked_labels(
                                self.model, batch_neg
                            )
                            tokens_neg = tokens_neg.to(model_hf_current.device)
                            labels_neg = labels_neg.to(model_hf_current.device)
                            mask_neg = mask_neg.to(model_hf_current.device)

                            outputs_neg = model_peft(tokens_neg, attention_mask=mask_neg)
                            shift_logits_neg = outputs_neg.logits[..., :-1, :].contiguous()
                            shift_labels_neg = labels_neg[..., 1:].contiguous()
                            loss_neg = loss_fn(
                                shift_logits_neg.view(-1, shift_logits_neg.size(-1)),
                                shift_labels_neg.view(-1),
                            )
                        else:
                            loss_neg = torch.zeros((), device=loss_pos.device)

                        loss = loss_pos + loss_neg

                        # Group Lasso sparsity penalty (Trace Trick)
                        loss_sparse = torch.zeros((), device=loss.device)
                        for layer_idx, submodules in layer_groups.items():
                            layer_sum_sq = torch.zeros((), device=loss.device)
                            for submodule_path, params in submodules.items():
                                if "A" in params and "B" in params:
                                    A = params["A"]
                                    B = params["B"]
                                    BtB = torch.matmul(B.t(), B)
                                    AAt = torch.matmul(A, A.t())
                                    trace_val = torch.trace(torch.matmul(BtB, AAt))
                                    layer_sum_sq = layer_sum_sq + trace_val
                            loss_sparse = loss_sparse + torch.sqrt(torch.clamp(layer_sum_sq, min=1e-10))

                        total_loss = loss + self.lambda_sparse * loss_sparse

                        scaled_loss = total_loss / self.grad_accum
                        scaled_loss.backward()

                        should_step = (
                            ((batch_idx + 1) % self.grad_accum == 0)
                            or (idx + self.batch_size >= len(shuffled_data))
                        )
                        if should_step:
                            torch.nn.utils.clip_grad_norm_(model_peft.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                        loss_val = total_loss.item()
                        epoch_loss += loss_val
                        num_batches += 1

                        pbar.set_postfix({
                            "loss": f"{loss_val:.4f}",
                            "task": f"{(loss_pos.item() + loss_neg.item()):.4f}",
                            "sparse": f"{loss_sparse.item():.4f}",
                        })

                    mean_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                    logger.info(f"IWS LoRA Epoch {epoch + 1}/{self.epochs} completed - Mean Loss: {mean_loss:.4f}")

                # Flush remaining grads just in case
                optimizer.zero_grad(set_to_none=True)

                # Extract LoRA state dict containing only adapter parameters A and B
                lora_state_dict = {
                    k: v.cpu().clone()
                    for k, v in model_peft.state_dict().items()
                    if "lora_A" in k or "lora_B" in k
                }

                self.vector = {layer: torch.zeros(1, device=original_tl_device) for layer in self.layer}
                self.metadata = {
                    "method": self.METHOD_NAME,
                    "positive_lora_state": lora_state_dict,
                    "negative_lora_state": {},
                    "lr": self.lr,
                    "epochs": self.epochs,
                    "lora_r": self.lora_r,
                    "lora_alpha": self.lora_alpha,
                    "grad_accum": self.grad_accum,
                }
                logger.info("LoRA extraction finished. Saved lightweight LoRA state dict in metadata.")

            finally:
                # Explicitly delete models and run gc to free memory
                del model_peft
                del model_hf_current
                torch.cuda.empty_cache()
                gc.collect()

                # Restore base model to GPU
                self.model.to(original_tl_device)
                torch.cuda.empty_cache()
                gc.collect()

            return self.vector

        else:
            # Original baseline implementation
            def train_lora_on_data(data_list: List[str]) -> Dict[str, torch.Tensor]:
                model_hf_current = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.model.cfg.dtype,
                    device_map=str(original_tl_device),
                )

                peft_config = LoraConfig(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    target_modules=self.target_modules,
                    layers_to_transform=self.layer,
                    lora_dropout=self.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model_peft = get_peft_model(model_hf_current, peft_config)
                model_peft.print_trainable_parameters()

                optimizer = torch.optim.AdamW(model_peft.parameters(), lr=self.lr)

                # SFT loop
                for epoch in range(self.epochs):
                    shuffled_data = list(data_list)
                    random.shuffle(shuffled_data)

                    pbar = tqdm(
                        range(0, len(shuffled_data), self.batch_size),
                        desc=f"LoRA SFT Epoch {epoch + 1}/{self.epochs}",
                        leave=True,
                    )

                    epoch_loss = 0.0
                    num_batches = 0
                    optimizer.zero_grad(set_to_none=True)

                    for batch_idx, idx in enumerate(pbar):
                        batch_texts = shuffled_data[idx : idx + self.batch_size]
                        if not batch_texts:
                            continue

                        # Masked label tokenization using HookedTransformer tokenizers
                        tokens, labels, attention_mask, _ = _get_completion_masked_labels(
                            self.model, batch_texts
                        )
                        tokens = tokens.to(model_hf_current.device)
                        labels = labels.to(model_hf_current.device)
                        attention_mask = attention_mask.to(model_hf_current.device)

                        outputs = model_peft(tokens, attention_mask=attention_mask)
                        logits = outputs.logits

                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = loss_fn(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )

                        scaled_loss = loss / self.grad_accum
                        scaled_loss.backward()

                        should_step = (
                            ((batch_idx + 1) % self.grad_accum == 0)
                            or (idx + self.batch_size >= len(shuffled_data))
                        )
                        if should_step:
                            torch.nn.utils.clip_grad_norm_(model_peft.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                        loss_val = loss.item()
                        epoch_loss += loss_val
                        num_batches += 1

                        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

                    mean_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                    logger.info(f"LoRA Epoch {epoch + 1}/{self.epochs} completed - Mean Loss: {mean_loss:.4f}")

                # Extract LoRA state dict containing only adapter parameters A and B
                lora_state_dict = {
                    k: v.cpu().clone()
                    for k, v in model_peft.state_dict().items()
                    if "lora_A" in k or "lora_B" in k
                }

                # Explicitly delete models and run gc to free memory
                del model_peft
                del model_hf_current
                torch.cuda.empty_cache()
                gc.collect()

                return lora_state_dict

            try:
                # 2. Train positive LoRA
                logger.info("Fine-tuning on positive prompts (LoRA)...")
                positive_lora_state = train_lora_on_data(target_data)

                # 3. Train negative LoRA / reference
                if contrast_data:
                    logger.info("Fine-tuning on negative prompts (LoRA)...")
                    negative_lora_state = train_lora_on_data(contrast_data)
                else:
                    logger.info("No contrast data provided. Using baseline weights as negative reference.")
                    negative_lora_state = {}

                # Build metadata with lightweight LoRA states
                self.vector = {layer: torch.zeros(1, device=original_tl_device) for layer in self.layer}
                self.metadata = {
                    "method": self.METHOD_NAME,
                    "positive_lora_state": positive_lora_state,
                    "negative_lora_state": negative_lora_state,
                    "lr": self.lr,
                    "epochs": self.epochs,
                    "lora_r": self.lora_r,
                    "lora_alpha": self.lora_alpha,
                    "grad_accum": self.grad_accum,
                }
                logger.info("LoRA extraction finished. Saved lightweight LoRA state dicts in metadata.")

            finally:
                # 4. Restore base model to GPU
                self.model.to(original_tl_device)
                torch.cuda.empty_cache()
                gc.collect()

            return self.vector