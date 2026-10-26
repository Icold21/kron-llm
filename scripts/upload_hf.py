"""Upload the documentation and the identity slice to Hugging Face.

Docs and data only: the model checkpoint and the tokenised bins are uploaded by
their own steps once they exist.

Usage::

    python scripts/upload_hf.py --docs
    python scripts/upload_hf.py --identity
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT.parent / "kron_corpus"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", action="store_true", help="upload the cards")
    ap.add_argument("--identity", action="store_true",
                    help="upload the four identity shards")
    args = ap.parse_args()
    api = HfApi()

    if args.docs:
        pairs = [
            (ROOT / "docs" / "hf_model_card.md", "README.md",
             "Icold/kron-llm", "model"),
            (CORPUS / "docs" / "hf_dataset_card.md", "README.md",
             "Icold/kron-corpus", "dataset"),
        ]
        for src, dst, repo_id, repo_type in pairs:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                            repo_id=repo_id, repo_type=repo_type)
            print(f"[docs] {repo_id}/{dst}")

    if args.identity:
        for path in sorted((ROOT / "data" / "identity").glob("*.jsonl")):
            api.upload_file(path_or_fileobj=str(path),
                            path_in_repo=f"data/identity/{path.name}",
                            repo_id="Icold/kron-corpus", repo_type="dataset")
            print(f"[identity] {path.name} "
                  f"({path.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
