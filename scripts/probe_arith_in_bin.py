"""Is the bare expression `67 + 75 = ` actually in the training data?

The drill format scores 41/60 and the bare format 1/60, so the question is
whether the bare form was ever seen -- and in which shape.  This samples random
windows from a token bin, decodes them, and counts:

* bare equations ``a + b = c`` (the shape the bare test asks for),
* drill preambles (the shape the drills are written in),
* and how many of each sit inside a ``<|user|> ... <|eot|>`` chat turn, because
  the bare test wraps the expression in exactly that frame.

Usage::

    python scripts/probe_arith_in_bin.py --bin E:/kron_bins/anneal_v5/train.bpe.bin \
        --tokenizer E:/kron_bins/anneal_v5/tokenizer.json --windows 20000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kronllm.tokenizer import BPETokenizer      # noqa: E402

BARE = re.compile(r"\d{1,5}\s*[+\-]\s*\d{1,5}\s*=\s*-?\d")
COLS = re.compile(r"(Add|Subtract|Multiply) the columns|Складываем по разрядам|"
                  r"Вычитаем по разрядам|Умножаем по разрядам|"
                  r"\bunits:|\bединицы:")
USER = re.compile(r"<\|user\|>(.*?)<\|eot\|>", re.S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--windows", type=int, default=20000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    bin_path = Path(args.bin)
    tok = BPETokenizer(Path(args.tokenizer), 8192, bytes_per_token=3.6, eot_id=8191)
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n = len(data) - args.seq - 1
    rng = random.Random(args.seed)
    starts = [rng.randrange(0, n) for _ in range(args.windows)]

    chars = 0
    bare = bare_in_user = 0
    cols = cols_in_user = 0
    user_turns = 0
    for s in starts:
        ids = data[s:s + args.seq].astype(np.int64).tolist()
        text = tok.decode(ids)
        chars += len(text)
        b = len(BARE.findall(text))
        c = len(COLS.findall(text))
        bare += b
        cols += c
        for m in USER.finditer(text):
            user_turns += 1
            inner = m.group(1)
            if BARE.search(inner):
                bare_in_user += 1
            if COLS.search(inner):
                cols_in_user += 1

    per_m = 1e6 / max(1, chars)
    out = {
        "windows": args.windows,
        "tokens": args.windows * args.seq,
        "chars": chars,
        "bare_equations": bare,
        "bare_per_mchar": round(bare * per_m, 2),
        "drill_preambles": cols,
        "drill_per_mchar": round(cols * per_m, 2),
        "user_turns": user_turns,
        "bare_inside_user_turn": bare_in_user,
        "drill_inside_user_turn": cols_in_user,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
