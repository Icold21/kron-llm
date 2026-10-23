"""Build the supervised fine-tuning corpus from Dolly-15k.

The script downloads ``databricks/databricks-dolly-15k``, keeps the Q/A-style
categories, formats every row with the same prompt template that is used at
inference time, tokenises it with the BPE model trained for KronLLM and writes
two ``uint16`` memmaps that the regular training loop can consume::

    data/sft_train.bpe.bin
    data/sft_val.bpe.bin

Examples longer than the model context are shortened by dropping the optional
context paragraph first and are skipped only if the question/answer pair alone
does not fit.

Usage
-----
    python scripts/prepare_sft.py
    python scripts/prepare_sft.py --categories all --max-tokens 256
    python scripts/prepare_sft.py --limit 2000 --out-dir data/sft_small
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/prepare_sft.py` from a fresh checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import time
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from kronllm.tokenizer import load_tokenizer

DATASET = "databricks/databricks-dolly-15k"
DEFAULT_CATEGORIES = "open_qa,closed_qa,general_qa"

PROMPT_WITH_CONTEXT = (
    "### Context:\n{context}\n"
    "### Question:\n{instruction}\n"
    "### Answer:\n{response}"
)
PROMPT_WITHOUT_CONTEXT = (
    "### Question:\n{instruction}\n"
    "### Answer:\n{response}"
)


def _clean(text: str) -> str:
    """Collapse whitespace so that one example occupies one prompt block.

    Args:
        text: Raw dataset field.

    Returns:
        The field with stripped lines and single newlines preserved.
    """
    lines = [line.strip() for line in str(text).replace("\r", "").split("\n")]
    return "\n".join(line for line in lines if line)


def _format_row(row: Dict[str, str], with_context: bool) -> str:
    """Render one Dolly row with the inference-time prompt template.

    Args:
        row: Dataset row with ``instruction``, ``context`` and ``response``.
        with_context: Whether to include the optional context paragraph.

    Returns:
        A prompt string that ends with the answer (no EOT).
    """
    instruction = _clean(row["instruction"])
    response = _clean(row["response"])
    context = _clean(row.get("context", ""))
    if with_context and context:
        return PROMPT_WITH_CONTEXT.format(
            context=context, instruction=instruction, response=response)
    return PROMPT_WITHOUT_CONTEXT.format(
        instruction=instruction, response=response)


def _encode_examples(rows: Sequence[Dict[str, str]], tok, max_tokens: int
                     ) -> Tuple[List[np.ndarray], Counter]:
    """Tokenise rows, shortening over-long examples where possible.

    Args:
        rows: Filtered Dolly rows.
        tok: A ``BPETokenizer`` instance.
        max_tokens: Maximum number of tokens per example, including EOT.

    Returns:
        A pair ``(sequences, stats)`` where each sequence already ends with the
        EOT id and ``stats`` counts kept/dropped examples by reason.
    """
    texts: List[str] = []
    has_context: List[bool] = []
    for row in rows:
        full = _format_row(row, with_context=True)
        texts.append(full)
        has_context.append(bool(_clean(row.get("context", ""))))

    encoded = tok.encode_batch(texts)
    sequences: List[np.ndarray] = []
    stats: Counter = Counter()

    for index, (ids, used_context) in enumerate(zip(encoded, has_context)):
        if len(ids) + 1 > max_tokens and used_context:
            # Retry without the context paragraph before giving up.
            stats["retried_without_context"] += 1
            ids = tok.encode(_format_row(rows[index], with_context=False))
        if len(ids) + 1 > max_tokens:
            stats["dropped_too_long"] += 1
            continue
        arr = np.empty(len(ids) + 1, dtype=np.uint16)
        arr[:-1] = np.asarray(ids, dtype=np.uint16)
        arr[-1] = tok.eot_id
        sequences.append(arr)
        stats["kept"] += 1
    return sequences, stats


def _write_memmap(sequences: Sequence[np.ndarray], path: Path) -> int:
    """Concatenate tokenised examples into a ``uint16`` memmap.

    Args:
        sequences: Arrays that already contain their trailing EOT id.
        path: Destination file.

    Returns:
        Number of tokens written.
    """
    flat = np.concatenate(sequences) if sequences else np.zeros(0, np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat.tofile(path)
    return int(len(flat))


def main() -> None:
    """Entry point: download, format, split, tokenise and write the corpus."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--categories", default=DEFAULT_CATEGORIES,
                    help="comma-separated Dolly categories, or 'all'")
    ap.add_argument("--max-tokens", type=int, default=192,
                    help="model context length (an example must fit with EOT)")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N rows (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    t0 = time.time()
    from datasets import load_dataset

    print(f"[sft] loading {args.dataset} ...")
    dataset = load_dataset(args.dataset, split="train")

    wanted = None if args.categories.strip().lower() == "all" else {
        item.strip() for item in args.categories.split(",") if item.strip()
    }
    rows = [row for row in dataset if wanted is None or row["category"] in wanted]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[sft] {len(rows):,} rows after category filter "
          f"({Counter(r['category'] for r in rows)})")

    cfg = {"data": {"tokenizer": "bpe", "data_dir": args.data_dir}}
    tok = load_tokenizer(cfg)
    sequences, stats = _encode_examples(rows, tok, args.max_tokens)
    print(f"[sft] kept {stats['kept']:,} examples, "
          f"dropped {stats['dropped_too_long']:,} (too long), "
          f"{stats['retried_without_context']:,} shortened by dropping context")

    order = list(range(len(sequences)))
    random.Random(args.seed).shuffle(order)
    n_val = max(1, int(len(order) * args.val_fraction))
    val_idx = set(order[:n_val])
    train_seqs = [sequences[i] for i in range(len(sequences)) if i not in val_idx]
    val_seqs = [sequences[i] for i in range(len(sequences)) if i in val_idx]

    out_dir = Path(args.out_dir)
    train_tokens = _write_memmap(train_seqs, out_dir / "sft_train.bpe.bin")
    val_tokens = _write_memmap(val_seqs, out_dir / "sft_val.bpe.bin")

    meta = {
        "dataset": args.dataset,
        "categories": "all" if wanted is None else sorted(wanted),
        "max_tokens": args.max_tokens,
        "train_examples": len(train_seqs),
        "val_examples": len(val_seqs),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "bytes_per_token": tok.bytes_per_token,
        "prompt_with_context": PROMPT_WITH_CONTEXT,
        "prompt_without_context": PROMPT_WITHOUT_CONTEXT,
        "stats": dict(stats),
    }
    (out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2),
                                           encoding="utf-8")

    print(f"[sft] train {train_tokens:,} tokens / {len(train_seqs):,} examples")
    print(f"[sft] val   {val_tokens:,} tokens / {len(val_seqs):,} examples")
    print(f"[sft] wrote {out_dir / 'sft_meta.json'} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
