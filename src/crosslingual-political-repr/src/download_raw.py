"""Download the pinned Global Opinion QA training split."""

import ast
import json
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "Anthropic/llm_global_opinions"
DATASET_REVISION = "cb2880488749218abb81802a94c2c62ebfde2f35"
OUTPUT_PATH = Path(__file__).parents[1] / "data" / "raw_global_opinions.json"


def parse_selections(value: object) -> dict[str, list[float]]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise TypeError(f"Expected selections to be a dict or string, got {type(value)!r}")
    prefix = "defaultdict(<class 'list'>, "
    if value.startswith(prefix) and value.endswith(")"):
        value = value[len(prefix) : -1]
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected selections to contain a dictionary")
    return parsed


def main() -> None:
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split="train")
    rows = [
        {
            "id": index,
            "question": row["question"],
            "selections": parse_selections(row["selections"]),
            "options": ast.literal_eval(row["options"])
            if isinstance(row["options"], str)
            else row["options"],
            "source": row["source"],
        }
        for index, row in enumerate(dataset)
    ]
    if len(rows) != 2556:
        raise RuntimeError(f"Expected 2,556 rows, downloaded {len(rows)}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
