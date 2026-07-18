"""Evaluate a DPO model on clean held-out UltraFeedback preference pairs."""

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def average_answer_logprob(model, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    full_ids = full["input_ids"].to(device)

    with torch.no_grad():
        logits = model(full_ids).logits

    log_probs = logits[:, :-1].log_softmax(dim=-1)
    token_scores = log_probs.gather(
        dim=-1,
        index=full_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    answer_start = prompt["input_ids"].shape[1] - 1
    return token_scores[:, answer_start:].mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-examples", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Launch this script on a GPU pod.")
    logger.info("Loading model from %s", args.model_path)

    raw_test = load_dataset(
        DATASET_NAME,
        split=f"test_prefs[:{args.test_examples}]",
    )
    clean_test = raw_test.filter(
        lambda example: example["score_chosen"] > example["score_rejected"]
    )
    logger.info(
        "Filtered %d tied preferences: %d -> %d held-out pairs",
        len(raw_test) - len(clean_test),
        len(raw_test),
        len(clean_test),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    correct = 0
    margins = []
    try:
        for index, example in enumerate(clean_test, start=1):
            chosen = average_answer_logprob(model, tokenizer, example["chosen"])
            rejected = average_answer_logprob(model, tokenizer, example["rejected"])
            margin = chosen - rejected
            margins.append(margin)
            correct += int(margin > 0)
            if index % 50 == 0:
                logger.info("Evaluated %d/%d", index, len(clean_test))
    except torch.OutOfMemoryError as error:
        logger.exception("CUDA OOM during evaluation.")
        raise error
    except Exception:
        logger.exception("Evaluation failed.")
        raise

    results = {
        "model_path": args.model_path,
        "test_examples": len(clean_test),
        "accuracy": correct / len(clean_test),
        "average_margin": sum(margins) / len(margins),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(results, file, indent=2)
    logger.info("Results saved to %s", output.resolve())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
