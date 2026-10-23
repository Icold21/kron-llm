"""Prepare the byte-level TinyStories memmaps.

Usage
-----
    python scripts/prepare_data.py                      # 200k train / 10k val stories
    python scripts/prepare_data.py --max-stories 50000  # smaller corpus
    python scripts/prepare_data.py --synthetic 2e6      # offline smoke-test corpus
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/prepare_data.py` from a fresh checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from kronllm.config import load_config
from kronllm.data import make_synthetic_bin, prepare_tinystories


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-stories", type=int, default=None)
    ap.add_argument("--max-val-stories", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--synthetic", type=float, default=None,
                    help="generate N synthetic bytes instead of downloading")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.max_stories:
        cfg["data"]["max_stories"] = args.max_stories
    if args.max_val_stories:
        cfg["data"]["max_val_stories"] = args.max_val_stories
    if args.force:
        cfg["data"]["force"] = True

    data_dir = Path(cfg["data"]["data_dir"])
    if args.synthetic:
        data_dir.mkdir(parents=True, exist_ok=True)
        n = int(args.synthetic)
        make_synthetic_bin(data_dir / "train.bin", n, seed=0)
        make_synthetic_bin(data_dir / "val.bin", max(1 << 16, n // 10), seed=1)
        meta = {"synthetic": True, "train_bytes": n}
    else:
        meta = prepare_tinystories(cfg)

    meta["vocab_size"] = cfg["data"]["vocab_size"]
    meta["eot_id"] = cfg["data"]["eot_id"]
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
