"""Inspect the public Anthropic sycophancy benchmark before model scoring."""

import json

from huggingface_hub import hf_hub_download

REPOSITORY = "Anthropic/model-written-evals"
SUBSETS = [
    "sycophancy_on_nlp_survey.jsonl",
    "sycophancy_on_philpapers2020.jsonl",
    "sycophancy_on_political_typology_quiz.jsonl",
]


def main():
    for subset in SUBSETS:
        path = hf_hub_download(
            repo_id=REPOSITORY,
            repo_type="dataset",
            filename=f"sycophancy/{subset}",
        )
        with open(path) as file:
            example = json.loads(next(file))

        print(f"\n{'=' * 72}\n{subset}")
        print("Fields:", sorted(example))
        print("Matching answer:", repr(example["answer_matching_behavior"]))
        print("Non-matching answer:", repr(example["answer_not_matching_behavior"]))
        print("Prompt preview:\n", example["question"][:1_000])


if __name__ == "__main__":
    main()
