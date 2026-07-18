"""Train Gemma 3 270M IT with reproducible preference-label flips."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import whoami
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL_NAME = "google/gemma-3-270m-it"
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
SEED = 42
TRAIN_EXAMPLES = 2_000
TRAIN_STEPS = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def format_for_dpo(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"][-1]["content"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"][-1]["content"]}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-rate", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.noise_rate <= 1:
        raise ValueError("--noise-rate must be between 0 and 1.")

    noise_rate = args.noise_rate
    root = args.output_dir
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Launch this script on a GPU pod.")

    try:
        account = whoami()["name"]
    except Exception as error:
        raise RuntimeError(
            "Hugging Face authentication failed. Run `hf auth login` before training."
        ) from error

    device = torch.cuda.current_device()
    total_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
    logger.info("Authenticated to Hugging Face as %s", account)
    logger.info("GPU: %s (%.1f GiB VRAM)", torch.cuda.get_device_name(device), total_gib)
    logger.info("Noise rate: %.0f%%", noise_rate * 100)
    logger.info("Writing artifacts to %s", root.resolve())
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    root.mkdir(parents=True, exist_ok=True)

    raw_train = load_dataset(
        DATASET_NAME,
        split=f"train_prefs[:{TRAIN_EXAMPLES}]",
    )
    clean_train = raw_train.filter(
        lambda example: example["score_chosen"] > example["score_rejected"]
    )
    logger.info(
        "Filtered %d tied preferences: %d -> %d training pairs",
        len(raw_train) - len(clean_train),
        len(raw_train),
        len(clean_train),
    )
    dpo_train = clean_train.map(
        format_for_dpo,
        remove_columns=clean_train.column_names,
    )

    rng = np.random.default_rng(SEED)
    corrupted_count = round(len(dpo_train) * noise_rate)
    corrupted_indices = set(
        rng.choice(len(dpo_train), size=corrupted_count, replace=False).tolist()
    )

    def flip_labels(example, index):
        if index in corrupted_indices:
            return {"chosen": example["rejected"], "rejected": example["chosen"]}
        return {"chosen": example["chosen"], "rejected": example["rejected"]}

    noisy_train = dpo_train.map(flip_labels, with_indices=True)
    noisy_train.save_to_disk(str(root / "dataset"))
    with (root / "dataset_metadata.json").open("w") as file:
        json.dump(
            {
                "source_dataset": DATASET_NAME,
                "seed": SEED,
                "noise_rate": noise_rate,
                "training_examples": len(dpo_train),
                "corrupted_examples": corrupted_count,
                "corrupted_indices": sorted(corrupted_indices),
            },
            file,
            indent=2,
        )
    logger.info(
        "Saved exact %.0f%% corruption: %d of %d pairs flipped (seed=%d)",
        noise_rate * 100,
        corrupted_count,
        len(dpo_train),
        SEED,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    allocated_gib = torch.cuda.memory_allocated() / 2**30
    reserved_gib = torch.cuda.memory_reserved() / 2**30
    logger.info(
        "Model loaded. CUDA allocated=%.2f GiB, reserved=%.2f GiB",
        allocated_gib,
        reserved_gib,
    )

    config = DPOConfig(
        output_dir=str(root / "checkpoints"),
        max_steps=TRAIN_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        beta=0.1,
        max_length=512,
        logging_steps=10,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=5,
        save_only_model=True,
        eval_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=noisy_train,
        processing_class=tokenizer,
    )
    logger.info(
        "Starting DPO: %d steps, batch=2, accumulation=8, learning_rate=5e-6",
        TRAIN_STEPS,
    )
    try:
        trainer.train()
    except torch.OutOfMemoryError as error:
        logger.exception(
            "CUDA OOM. Clear other GPU processes with `nvidia-smi`; then retry from "
            "a fresh process or reduce max_length/per_device_train_batch_size."
        )
        raise error
    except Exception:
        logger.exception("Training failed before completing %d steps.", TRAIN_STEPS)
        raise

    trainer.save_model(str(root / "model"))
    tokenizer.save_pretrained(str(root / "model"))
    logger.info("Training complete. Final model saved to %s", (root / "model").resolve())


if __name__ == "__main__":
    main()
