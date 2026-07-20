"""Separate quote-grounded Gemini JD claims from invalid claims."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--validated", type=Path, required=True)
    args = parser.parse_args()
    source = {row["id"]: row for row in map(json.loads, args.source.open())}
    valid_claims = invalid_claims = 0
    with args.validated.open("w") as output:
        for row in map(json.loads, args.curated.open()):
            text = source[row["id"]]["description_text"]
            grounded, rejected = [], []
            for claim in row["claims"]:
                if claim["quote"] and claim["quote"] in text:
                    grounded.append(claim)
                    valid_claims += 1
                else:
                    rejected.append(claim)
                    invalid_claims += 1
            output.write(json.dumps({
                **row,
                "claims": grounded,
                "rejected_claims": rejected,
            }) + "\n")
    print(f"grounded_claims={valid_claims} rejected_claims={invalid_claims}")


if __name__ == "__main__":
    main()
