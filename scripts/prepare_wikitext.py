"""Build a byte-level corpus from WikiText-103 for the product-vs-sum probe.

The output is the same format the byte models already consume: one uint8
memmap per split, documents separated by 0xFF (which is also the byte EOT id).
No tokenizer is trained, so the comparison against the E8/E14 TinyStories runs
stays byte-for-byte and bpb-comparable.

Usage
-----
    python scripts/prepare_wikitext.py                  # ~500 MB train
    python scripts/prepare_wikitext.py --max-train-mb 200
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from datasets import load_dataset

SEP = b"\xff"


def write_split(rows, out_path: Path, max_mb: float | None) -> dict:
    """Concatenate text rows with a 0xFF separator, optionally capped."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    limit = None if max_mb is None else int(max_mb * 1024 * 1024)
    written = 0
    docs = 0
    with out_path.open("wb") as fh:
        for row in rows:
            text = row.get("text", "")
            if not text.strip():
                continue
            data = text.encode("utf-8", errors="ignore")
            if limit is not None and written + len(data) + 1 > limit:
                break
            fh.write(data)
            fh.write(SEP)
            written += len(data) + 1
            docs += 1
    return {"bytes": written, "docs": docs, "path": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--config", default="wikitext-103-raw-v1")
    ap.add_argument("--data-dir", default="data/wikitext")
    ap.add_argument("--max-train-mb", type=float, default=600.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[wikitext] loading {args.dataset}/{args.config} ...")
    ds = load_dataset(args.dataset, args.config)

    train = write_split(ds["train"], data_dir / "train.bin", args.max_train_mb)
    val = write_split(ds["validation"], data_dir / "val.bin", None)
    test = write_split(ds["test"], data_dir / "test.bin", None)

    meta = {
        "dataset": args.dataset,
        "config": args.config,
        "train": train,
        "val": val,
        "test": test,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                        encoding="utf-8")
    for name in ("train", "val", "test"):
        m = meta[name]
        print(f"[wikitext] {name}: {m['bytes']/1e6:.1f} MB, {m['docs']:,} docs")
    print(f"[wikitext] wrote {data_dir}")


if __name__ == "__main__":
    main()
