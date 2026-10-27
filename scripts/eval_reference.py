"""Score a model on the Q10 held-out text in bits per *byte*.

`train.py` reports `val_bpb` on the first 64 batches of the val bin, i.e. the
first 262,144 tokens, which are English Python (the bin is written bucket by
bucket and `en/code` sorts first).  It also divides by the corpus-wide
3.694 bytes/token, while code is ~2.3.  Both effects push the reported number
down: on the code span the honest figure is about 1.6× higher.

This script measures a model on **evenly spaced blocks across the whole val
bin**, decodes each block back to text (the byte-level tokenizer is lossless)
and divides by the real UTF-8 byte count of the predicted tokens.  The same
blocks go to GPT-2, so the two numbers sit on identical text.

Usage::

    python scripts/eval_reference.py --source gpt2 --blocks 1024 --out data/baseline_gpt2.json
    python scripts/eval_reference.py --source ckpt --ckpt data/kaggle_out/q10/last.pt \
        --blocks 1024 --out data/baseline_q10_11723.json
    python scripts/eval_reference.py --source ckpt --ckpt ... --first 512   # legacy span
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VAL = Path("data/bilingual/val.bpe.bin")
TOK = Path("data/bilingual/tokenizer.json")
SEQ = 512
EOT = 8191


def block_starts(n_tokens: int, blocks: int, first: bool) -> list[int]:
    """Evenly spaced 512-token blocks, or the first `blocks` of them."""
    n_blocks = (n_tokens - SEQ - 1) // SEQ
    if first or blocks >= n_blocks:
        return [i * SEQ for i in range(min(blocks, n_blocks))]
    step = (n_blocks - 1) / (blocks - 1)
    return [int(round(i * step)) * SEQ for i in range(blocks)]


def language(text: str) -> str:
    letters = sum(ch.isalpha() for ch in text) or 1
    cyr = sum("\u0400" <= ch <= "\u04ff" for ch in text)
    return "ru" if cyr / letters > 0.2 else "en"


def load_tokenizer():
    from kronllm.tokenizer import BPETokenizer
    return BPETokenizer(TOK, vocab_size=8192, bytes_per_token=3.694, eot_id=EOT)


def eval_ckpt(ckpt: Path, starts: list[int], tok, device: str) -> list[dict]:
    from kronllm.model import KronTransformer

    data = np.memmap(VAL, dtype=np.uint16, mode="r")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[ckpt] step {state.get('step')}, val_ppl {state.get('val_ppl')}")

    rows = []
    with torch.no_grad():
        for i, start in enumerate(starts):
            x = torch.from_numpy(np.asarray(data[start:start + SEQ],
                                            dtype=np.int64))[None]
            y = torch.from_numpy(np.asarray(data[start + 1:start + SEQ + 1],
                                            dtype=np.int64))[None]
            with torch.amp.autocast("cuda", dtype=torch.float16,
                                    enabled=device == "cuda"):
                loss = float(model(x.to(device), y.to(device))["loss"])
            text = tok.decode(y[0].tolist())
            nbytes = len(text.encode("utf-8"))
            rows.append({"start": start, "lang": language(text),
                         "bytes": nbytes, "nats": loss * SEQ,
                         "tokens": SEQ})
            if i % 100 == 0:
                print(f"  [{i}/{len(starts)}] start {start:,} "
                      f"bpb {loss / math.log(2) * SEQ / max(1, nbytes):.4f}")
    return rows


def eval_gpt2(starts: list[int], tok, device: str) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = np.memmap(VAL, dtype=np.uint16, mode="r")
    gtok = AutoTokenizer.from_pretrained("gpt2")
    model = (AutoModelForCausalLM.from_pretrained("gpt2")
             .to(device, dtype=torch.float16).eval())
    print(f"[gpt2] {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    rows = []
    with torch.no_grad():
        for i, start in enumerate(starts):
            text = tok.decode(np.asarray(data[start + 1:start + SEQ + 1],
                                         dtype=np.int64).tolist())
            nbytes = len(text.encode("utf-8"))
            ids = gtok(text, return_tensors="pt", add_special_tokens=False).input_ids
            nats, count = 0.0, 0
            for off in range(0, max(1, ids.shape[1] - 1), 1024):
                chunk = ids[:, off:off + 1024].to(device)
                if chunk.shape[1] < 2:
                    continue
                loss = model(input_ids=chunk, labels=chunk).loss
                nats += float(loss) * (chunk.shape[1] - 1)
                count += chunk.shape[1] - 1
            if not count:
                continue
            rows.append({"start": start, "lang": language(text),
                         "bytes": nbytes, "nats": nats,
                         "gpt2_tokens": count})
            if i % 100 == 0:
                print(f"  [{i}/{len(starts)}] start {start:,} "
                      f"bpb {nats / math.log(2) / max(1, nbytes):.4f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["gpt2", "ckpt"], required=True)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--blocks", type=int, default=1024)
    ap.add_argument("--first", type=int, default=0,
                    help="use the first N blocks (the span the trainer reports)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()
    n_tokens = len(np.memmap(VAL, dtype=np.uint16, mode="r"))
    starts = block_starts(n_tokens, args.blocks, bool(args.first))
    print(f"[val] {len(starts)} blocks of {SEQ} tokens from {VAL} "
          f"({n_tokens:,} tokens total)")

    if args.source == "ckpt":
        if not args.ckpt:
            raise SystemExit("--ckpt is required for --source ckpt")
        rows = eval_ckpt(args.ckpt, starts, tok, device)
    else:
        rows = eval_gpt2(starts, tok, device)

    total_nats = sum(r["nats"] for r in rows)
    total_bytes = sum(r["bytes"] for r in rows)
    bpb = total_nats / math.log(2) / total_bytes
    by_lang = {}
    for lang in ("en", "ru"):
        sel = [r for r in rows if r["lang"] == lang]
        if not sel:
            continue
        b = sum(r["bytes"] for r in sel)
        by_lang[lang] = {
            "blocks": len(sel),
            "bytes": b,
            "bpb": sum(r["nats"] for r in sel) / math.log(2) / b,
        }
    out = {
        "source": args.source,
        "blocks": len(rows),
        "span": "first" if args.first else "uniform",
        "bytes": total_bytes,
        "nats": total_nats,
        "bpb": bpb,
        "by_language": by_lang,
        "per_block": rows,
    }
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[{args.source}] {len(rows)} blocks, {total_bytes/1e6:.1f} MB, "
          f"bpb {bpb:.4f}")
    for lang, r in by_lang.items():
        print(f"  {lang}: {r['blocks']} blocks, bpb {r['bpb']:.4f}")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
