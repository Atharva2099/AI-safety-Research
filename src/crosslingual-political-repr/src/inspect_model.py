"""Inspect model dimensions and a few layerwise hidden-state shapes."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "declarative_statements.json"


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, torch_dtype=dtype)
    model.to(device).eval()

    print(f"Model name: {args.model}")
    print(f"Total parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Hidden dimension: {config.hidden_size}")
    print(f"Vocabulary size: {config.vocab_size}")
    num_layers = config.num_hidden_layers
    print("Embedding Layer: outputs.hidden_states[0][:, -1, :].shape")
    print("Block 0: outputs.hidden_states[1][:, -1, :].shape")
    print(
        f"Mid Block (e.g. {num_layers // 2}): "
        f"outputs.hidden_states[{num_layers // 2 + 1}][:, -1, :].shape"
    )
    print(
        f"Final Block ({num_layers - 1}): "
        f"outputs.hidden_states[{num_layers}][:, -1, :].shape"
    )

    records = json.loads(DATA.read_text(encoding="utf-8"))
    pair = next(record for record in records if len(record.get("statements", [])) >= 2)
    text = "\n".join(statement["statement"] for statement in pair["statements"][:2])
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    print(f"\nSample pair (record {pair['id']}):\n{text}")
    for label, hidden_state_index in (
        ("Embedding Layer", 0),
        ("Block 0", 1),
        (f"Mid Block ({num_layers // 2})", num_layers // 2 + 1),
        (f"Final Block ({num_layers - 1})", num_layers),
    ):
        shape = outputs.hidden_states[hidden_state_index][:, -1, :].shape
        print(f"{label}: {tuple(shape)}")


if __name__ == "__main__":
    main()
