# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import ast
import json
import logging
import os
import pathlib
import typing as t
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from lineas.datasets.truthfulqa_dataset import load_and_split
from lineas.models import get_model
from lineas.models.model_with_hooks import ModelWithHooks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Already run in parallel inside DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "False"

MAX_LENGTH: int = 10000  # Hardcoded max length to avoid infinite loop


def dataset_names(root: pathlib.Path) -> dict[str, pathlib.Path]:
    return {
        "truthfulqa_mc1": root / "truthfulqa_mc1_processed.csv",
        "truthfulqa_mc2": root / "truthfulqa_mc2_processed.csv",
    }


def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def get_logprobs(logits, input_ids, masks, **kwargs):
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    # find the logprob of the input ids that actually come next in the sentence
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
    logprobs = logprobs * masks[:, 1:, None]
    return logprobs.squeeze(-1)


def prepare_decoder_only_inputs(prompts, targets, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=False
    )
    tokenizer.padding_side = "right"
    target_inputs = tokenizer(
        targets,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )

    # concatenate prompt and target tokens and send to device
    inputs = {
        k: torch.cat([prompt_inputs[k], target_inputs[k]], dim=1).to(device)
        for k in prompt_inputs
    }

    # mask is zero for padding tokens
    mask = inputs["attention_mask"].clone()
    # set mask to 0 for question tokens
    mask[:, : prompt_inputs["input_ids"].shape[1]] = 0
    mask.to(device)
    # remove token_type_ids
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    return inputs, mask, prompt_inputs["input_ids"].shape[1]


def calc_acc(labels, output_logprobs):
    # check if the max logprob corresponds to the correct answer

    print("labels", labels)
    print("output_logprobs", output_logprobs)

    correct = np.zeros(len(labels))
    # indices to index
    indices = np.cumsum([len(l) for l in labels])
    indices = np.insert(indices, 0, 0)
    for i, label in enumerate(labels):
        # check
        log_probs = output_logprobs[indices[i] : indices[i + 1]]
        correct[i] = int(np.argmax(log_probs) in set(np.where(np.array(label) == 1)[0]))
    return correct.mean()


def build_data(df):
    questions = []
    answers = []
    labels = []
    for idx, answer_id in enumerate(df.answer_id):
        questions.append(df.iloc[idx].question)
        answers.append(df.iloc[idx].answer)
        if int(answer_id) == 0:
            labels.append(df.iloc[idx].labels)

    labels = [
        ast.literal_eval(label) for label in labels
    ]  # convert strings to lists of integers

    return questions, answers, labels


def get_tqa_accuracy(
    model, questions, answers, labels, tokenizer, batch_size=128, device="cpu"
):
    # get the log probabilities of each question answer pair
    output_logprobs = []
    batch_size = 1  # otherwise get flawed results because of padding issues
    for q_batch, a_batch in tqdm(
        zip(
            batchify(questions, batch_size), batchify(answers, batch_size), strict=True
        ),
        total=len(questions) // batch_size,
    ):
        model.config.pad_token_id = tokenizer.pad_token_id

        inputs, masks, _ = prepare_decoder_only_inputs(
            q_batch, a_batch, tokenizer, device
        )
        print("_______")
        print(masks.shape)
        with torch.no_grad():
            try:
                # set the masks so that we do not add to tokens of input sentences and padding tokens
                model.set_masks(masks.unsqueeze(-1))
            except Exception:
                print("model_set_masks did not work")
                pass

            # calculate the probabilities for all tokens (all question answer pairs)
            model.eval()
            logits = model(**inputs).logits.to(torch.float32)

            # sum the probabilities for each question answer pair so that each pair has one probability
            # mask is zero for question and padding tokens
            logprobs = (
                get_logprobs(logits, inputs["input_ids"], masks)
                .sum(-1)
                .detach()
                .cpu()
                .numpy()
            )
        output_logprobs.extend(logprobs)
    return calc_acc(labels, output_logprobs)


def log_results(wandb_logger, data: t.Any) -> None:
    """
    Tries logging `data` to WandB
    Prints on stdout in any case.
    """
    logger.info(data)
    try:
        wandb_logger.log(data)
    except Exception:
        logger.warning("error with wandb :|")


def measure(cfg: DictConfig) -> None:
    if cfg.results_dir is not None:
        output_path = Path(Path(__file__).stem)
        output_path = Path(cfg.results_dir) / output_path
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    logger.info(OmegaConf.to_yaml(cfg))

    module, tokenizer = get_model(
        cfg.model_params.model_path,
        cache_dir=cfg.cache_dir,
        dtype=cfg.dtype,
        device=cfg.device,
        model_task="text-generation",
    )

    assert cfg.model_params.module_names is not None, logger.error(
        f"Intervention specified as {cfg.intervention_params.name}, but no module names passed (passed {cfg.model_params.module_names})"
    )
    # Create hooked model
    model_with_hooks = ModelWithHooks(
        module=module,
    )
    model_with_hooks.load_hooks_from_folder(
        folder=Path(cfg.intervention_params.state_path),
        module_names=cfg.model_params.module_names,
        hook_type=cfg.intervention_params.name,
        **cfg.intervention_params.hook_params,
    )
    model_with_hooks.register_hooks()
    model = model_with_hooks.module

    with torch.no_grad():
        batch_size = 64  # Has no effect, overriding later
        output = {}
        for dataset_name, dataset_file in dataset_names(
            pathlib.Path(cfg.data_dir)
        ).items():
            logger.info(f"Evaluating model on {dataset_name}.")
            print(f"dataset_file = {dataset_file}")
            output[dataset_name] = {}
            for split_idx in [0, 1]:
                num_folds = 2
                _, test_df = load_and_split(
                    dataset_file,
                    cfg.seed,
                    num_folds=num_folds,
                    test_fold_idx=split_idx,
                    prompt=cfg.task_params.dataset_params.prompt,
                )

                questions, answers, labels = build_data(test_df)
                model_acc = get_tqa_accuracy(
                    model,
                    questions,
                    answers,
                    labels,
                    tokenizer,
                    batch_size=batch_size,
                    device=cfg.device,
                )

                output[dataset_name][f"seed{cfg.seed}_split{split_idx}"] = model_acc

    if output_path is not None:
        json.dump(
            output,
            (output_path / "tqa_results.json").open("w"),
        )
    wandb.finish()
    logger.info(output)


@hydra.main(config_path="../lineas/configs", config_name="text_generation")
# def main(args: argparse.Namespace, unknown: t.Any) -> pd.DataFrame:
def main(cfg):
    measure(cfg)


if __name__ == "__main__":
    main()
