"""Upload the dataset JSON and Hugging Face dataset card."""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True,
                        help="Dataset repository, e.g. your-username/multilingual-political-statements")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)
    for local_path, repo_path in (
        (ROOT / "data" / "multilingual_statements.json", "data/multilingual_statements.json"),
        (ROOT / "data" / "HF_README.md", "README.md"),
    ):
        api.upload_file(path_or_fileobj=str(local_path), path_in_repo=repo_path,
                        repo_id=args.repo_id, repo_type="dataset")


if __name__ == "__main__":
    main()
