"""Stage the decay-pass bins as a Kaggle dataset directory.

The Q10A kernel mounts this instead of ``kron-corpus-2b``.  It carries the
tokenised curated mix (675M tokens), the epoch-1 validation bin and the
epoch-1 tokenizer, so the pass is measured on exactly the held-out set the
epoch used and cannot silently change the token id space.

Usage::

    python scripts/make_anneal_dataset.py
    python -m kaggle datasets create -p data/kaggle_anneal_bin
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

SRC = Path("data/bilingual/anneal_bin")
BASE = Path("data/bilingual")
OUT = Path("data/kaggle_anneal_bin")
FILES = ["train.bpe.bin", "val.bpe.bin", "tokenizer.json", "tokenizer_meta.json"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in FILES:
        dst = OUT / name
        shutil.copy2(SRC / name, dst)
        size = dst.stat().st_size
        total += size
        print(f"{name:>20} {size/1e6:9.1f} MB")

    meta = {
        "title": "Kron Q10A anneal bin",
        "id": "icold21/kron-q10a-anneal-bin",
        "licenses": [{"name": "other"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                               encoding="utf-8")

    bin_meta = json.loads((SRC / "tokenizer_meta.json").read_text(encoding="utf-8"))
    epoch_meta = json.loads((BASE / "tokenizer_meta.json").read_text(encoding="utf-8"))
    same = (bin_meta["val_tokens"] == epoch_meta["val_tokens"]
            and bin_meta["vocab_size"] == epoch_meta["vocab_size"])
    print(f"train tokens {bin_meta['train_tokens']:,} "
          f"({bin_meta['train_tokens']/41213/16384:.2f}x the 41,213-step budget)")
    print(f"val tokens   {bin_meta['val_tokens']:,} "
          f"(same held-out as the epoch: {same})")
    print(f"{OUT} total {total/1e9:.2f} GB")


if __name__ == "__main__":
    main()
