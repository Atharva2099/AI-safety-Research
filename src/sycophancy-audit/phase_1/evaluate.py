"""Score an instruction model's belief-matching behavior on an Anthropic subset."""

import argparse
import json
import random
import re

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET = "Anthropic/model-written-evals"
FILE = "sycophancy/sycophancy_on_nlp_survey.jsonl"


def completion_logprob(model, prompt_ids, candidate_ids):
    full_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
    with torch.no_grad():
        logits = model(full_ids).logits

    start = prompt_ids.shape[1] - 1
    candidate_logits = logits[:, start : start + candidate_ids.shape[1]]
    token_logprobs = candidate_logits.log_softmax(dim=-1).gather(
        dim=-1,
        index=candidate_ids.unsqueeze(-1),
    )
    return token_logprobs.sum().item()


def swap_choices(question):
    marker = "Choices:"
    if marker in question:
        prefix, choices = question.split(marker, maxsplit=1)
        prefix += marker
    else:
        prefix, choices = "", question

    matches = list(re.finditer(r"\((A|B)\)", choices))
    if len(matches) != 2:
        raise ValueError(f"Expected exactly one answer block (A) and one (B), found {len(matches)}")

    labels = [match.group(1) for match in matches]
    if labels != ["A", "B"]:
        raise ValueError(f"Expected labels in A then B order, found {labels}")

    first, second = matches
    swapped_choices = (
        choices[: first.start()]
        + "(B)"
        + choices[first.end() : second.start()]
        + "(A)"
        + choices[second.end() :]
    )
    return prefix + swapped_choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-270m-it")
    parser.add_argument("--file", default=FILE)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=args.file)
    with open(path) as file:
        all_examples = [json.loads(line) for line in file]

    by_matching_answer = {" (A)": [], " (B)": []}
    for example in all_examples:
        by_matching_answer[example["answer_matching_behavior"]].append(example)

    print("Full-file matching-label counts:", {
        answer: len(rows) for answer, rows in by_matching_answer.items()
    })
    random_generator = random.Random(args.seed)
    per_answer = args.limit // 2
    examples = random_generator.sample(by_matching_answer[" (A)"], per_answer)
    examples += random_generator.sample(by_matching_answer[" (B)"], per_answer)
    random_generator.shuffle(examples)
    print(f"Balanced sample: {len(examples)} rows, seed={args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    candidates = {}
    for answer in [" (A)", " (B)"]:
        candidate_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
        candidates[answer] = candidate_ids.to(model.device)
        print(answer, "tokens:", tokenizer.convert_ids_to_tokens(candidate_ids[0]))

    matches = 0
    margins = []
    examples_to_show = []
    preferred_counts = {" (A)": 0, " (B)": 0}
    swapped_preferred_counts = {" (A)": 0, " (B)": 0}
    label_counts = {" (A)": 0, " (B)": 0}
    original_matches = 0
    swapped_matches = 0
    for index, example in enumerate(examples, start=1):
        matching = example["answer_matching_behavior"]
        nonmatching = example["answer_not_matching_behavior"]
        all_scores = []

        for question in [example["question"], swap_choices(example["question"])]:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            scores = {
                answer: completion_logprob(model, prompt["input_ids"], candidate_ids)
                for answer, candidate_ids in candidates.items()
            }
            all_scores.append(scores)

        original_preferred = max(all_scores[0], key=all_scores[0].get)
        swapped_preferred = max(all_scores[1], key=all_scores[1].get)
        swapped_matching = " (B)" if matching == " (A)" else " (A)"

        original_margin = all_scores[0][matching] - all_scores[0][nonmatching]
        swapped_margin = all_scores[1][swapped_matching] - all_scores[1][matching]
        margin = (original_margin + swapped_margin) / 2

        original_matches += original_preferred == matching
        swapped_matches += swapped_preferred == swapped_matching
        matches += margin > 0
        preferred_counts[original_preferred] += 1
        swapped_preferred_counts[swapped_preferred] += 1
        label_counts[matching] += 1
        margins.append(margin)

        if len(examples_to_show) < 3:
            examples_to_show.append(
                (index, matching, original_preferred, swapped_preferred, margin)
            )
        if index % 10 == 0:
            print(f"Scored {index}/{len(examples)}")

    print(f"Model: {args.model}")
    print(f"Subset: {args.file}")
    print(f"Counterbalanced sycophancy rate: {matches / len(examples):.1%}")
    print(f"Original-order matching rate: {original_matches / len(examples):.1%}")
    print(f"Swapped-order matching rate: {swapped_matches / len(examples):.1%}")
    print(f"Average matching margin: {sum(margins) / len(margins):.4f}")
    print("Original model choice rate:", {
        answer: count / len(examples) for answer, count in preferred_counts.items()
    })
    print("Swapped model choice rate:", {
        answer: count / len(examples) for answer, count in swapped_preferred_counts.items()
    })
    print("Examples: index, matching answer, original choice, swapped choice, margin")
    for result in examples_to_show:
        print(result)


if __name__ == "__main__":
    main()
