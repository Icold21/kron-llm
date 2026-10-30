"""Measure what the model does, instead of eyeballing five samples.

The reports so far quoted five hand-written prompts, decoded once each at
T=1.0.  That is one narrow slice looked at repeatedly, and the impression it
gives cannot be separated from those five strings.  This script replaces the
impression with three quantities that have a denominator:

1. **Continuation bpb.**  Real held-out blocks: the first half is context, the
   loss is measured on the second half only.  This is the model scored on text
   it has never seen, in the same units we quote everywhere else.

2. **Ranking.**  For each prefix, the true continuation competes against
   continuations drawn from other documents.  If the model merely produces
   fluent text without using the prefix, the true one does not win.  Reported
   as accuracy over prefixes and the mean rank of the true continuation.

3. **Arithmetic.**  Templated sums decoded greedily; exact match.  A capability
   with a right answer, unlike prose.

4. **Long form.**  M prompts taken from held-out text, K samples each, with
   repetition and distinct-n instead of one quoted string.

Usage::

    python scripts/eval_samples.py --ckpt data/kaggle_out/q10/last.pt \
        --mode all --out data/eval_q10.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VAL = Path("data/bilingual/val.bpe.bin")
TOK = Path("data/bilingual/tokenizer.json")


def load(ckpt: Path, device: str):
    from kronllm.model import KronTransformer
    from kronllm.tokenizer import BPETokenizer

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    tok = BPETokenizer(TOK, vocab_size=8192, bytes_per_token=3.69, eot_id=8191)
    return model, tok, int(state.get("step") or 0)


def blocks(n: int, seq: int, seed: int) -> list[int]:
    """Start offsets of held-out blocks, spread over the whole val bin."""
    total = VAL.stat().st_size // 2
    stride = max(seq * 4, (total - seq) // max(1, n))
    return [i * stride for i in range(n) if i * stride + seq + 1 < total]


@torch.no_grad()
def token_loss(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-position cross-entropy, shape (B, T)."""
    out = model(x)
    logits = out["logits"]
    return torch.nn.functional.cross_entropy(
        logits.transpose(1, 2).float(), y, reduction="none")


@torch.no_grad()
def continuation(model, data, tok, starts: list[int], seq: int,
                 device: str) -> dict:
    """bpb on the second half of each block, given the first half."""
    half = seq // 2
    nats = 0.0
    nbytes = 0
    per_lang: dict[str, list[float]] = {}
    for s in starts:
        ids = np.asarray(data[s:s + seq + 1], dtype=np.int64)
        x = torch.from_numpy(ids[:-1])[None].to(device)
        y = torch.from_numpy(ids[1:])[None].to(device)
        losses = token_loss(model, x, y)[0, half - 1:]
        text = tok.decode(ids[half:seq + 1].tolist())
        nb = max(1, len(text.encode("utf-8")))
        bpb = float(losses.sum()) / math.log(2) / nb
        cyr = sum(1 for c in text[:200] if "а" <= c.lower() <= "я")
        lang = "ru" if cyr > 20 else "en"
        per_lang.setdefault(lang, []).append(bpb)
        nats += float(losses.sum())
        nbytes += nb
    return {
        "blocks": len(starts),
        "bpb": nats / math.log(2) / max(1, nbytes),
        "per_language": {k: float(np.mean(v)) for k, v in per_lang.items()},
        "counts": {k: len(v) for k, v in per_lang.items()},
    }


@torch.no_grad()
def ranking(model, data, starts: list[int], seq: int, device: str,
            distractors: int = 8, seed: int = 0) -> dict:
    """Does the true continuation beat continuations from other documents?"""
    rng = random.Random(seed)
    half = seq // 2
    wins = 0
    ranks: list[int] = []
    for s in starts:
        ids = np.asarray(data[s:s + seq + 1], dtype=np.int64)
        ctx = ids[:half]
        cons = [ids[half:seq + 1]]
        for _ in range(distractors):
            other = rng.choice(starts)
            cons.append(np.asarray(data[other + half:other + seq + 1],
                                   dtype=np.int64))
        losses = []
        for c in cons:
            x = torch.from_numpy(np.concatenate([ctx, c[:-1]]))[None].to(device)
            y = torch.from_numpy(np.concatenate([[ctx[-1]], c]))[None].to(device)
            per = token_loss(model, x, y)[0, half - 1:]
            losses.append(float(per.mean()))
        order = sorted(range(len(losses)), key=lambda i: losses[i])
        rank = order.index(0) + 1
        ranks.append(rank)
        wins += int(rank == 1)
    return {"prefixes": len(starts), "distractors": distractors,
            "accuracy": wins / max(1, len(starts)),
            "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
            "chance_accuracy": 1.0 / (distractors + 1)}


@torch.no_grad()
def arithmetic(model, tok, device: str, n: int = 60, seed: int = 7) -> dict:
    """Greedy exact match on templated sums and differences."""
    rng = random.Random(seed)
    hits = 0
    rows = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(["+", "-"])
        prompt = f"{a} {op} {b} = "
        ids = torch.tensor([tok.encode(prompt)], device=device)
        out = model.generate(ids, max_new_tokens=6, temperature=0.0,
                             top_k=1, top_p=1.0, repetition_penalty=1.0)
        text = tok.decode(out[0, ids.shape[1]:].tolist())
        m = re.match(r"\s*(-?\d+)", text)
        want = a + b if op == "+" else a - b
        ok = bool(m) and int(m.group(1)) == want
        hits += int(ok)
        if len(rows) < 8:
            rows.append({"prompt": prompt, "got": text[:12], "want": want,
                         "ok": ok})
    return {"n": n, "exact_match": hits / max(1, n), "examples": rows}


@torch.no_grad()
def longform(model, data, tok, starts: list[int], seq: int, device: str,
             k: int = 4, new_tokens: int = 64) -> dict:
    """Sample from real held-out prefixes; measure repetition, not vibes."""
    half = seq // 2
    prompts = []
    for s in starts[:6]:
        ids = np.asarray(data[s:s + half], dtype=np.int64)
        prompts.append(ids)
    reps = []
    distinct2 = []
    samples = []
    for ids in prompts:
        for _ in range(k):
            x = torch.from_numpy(ids)[None].to(device)
            out = model.generate(x, max_new_tokens=new_tokens, temperature=0.8,
                                 top_k=40, top_p=0.9, repetition_penalty=1.1)
            toks = out[0, len(ids):].tolist()
            grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
            reps.append(1.0 - len(set(grams)) / max(1, len(grams)))
            bigrams = {(toks[i], toks[i + 1]) for i in range(len(toks) - 1)}
            distinct2.append(len(bigrams) / max(1, len(toks) - 1))
            if len(samples) < 4:
                samples.append(tok.decode(toks)[:220])
    return {"samples": len(reps),
            "repeat_4gram": float(np.mean(reps)),
            "distinct_2": float(np.mean(distinct2)),
            "examples": samples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--mode", default="all",
                    choices=["all", "score", "generate", "arithmetic"])
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("data/eval_samples.json"))
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, tok, step = load(args.ckpt, device)
    data = np.memmap(VAL, dtype=np.uint16, mode="r")
    starts = blocks(args.blocks, args.seq, 0)
    print(f"[load] step {step}, {len(starts)} blocks of {args.seq} on {device}")

    report: dict = {"ckpt": str(args.ckpt), "step": step, "device": device,
                    "seq": args.seq}
    if args.mode in ("all", "score"):
        report["continuation"] = continuation(model, data, tok, starts,
                                              args.seq, device)
        print("[cont]", json.dumps(report["continuation"]))
        report["ranking"] = ranking(model, data, starts, args.seq, device)
        print("[rank]", json.dumps(report["ranking"]))
    if args.mode in ("all", "arithmetic"):
        report["arithmetic"] = arithmetic(model, tok, device)
        print("[arith]", json.dumps({k: v for k, v in
                                     report["arithmetic"].items()
                                     if k != "examples"}))
    if args.mode in ("all", "generate"):
        report["longform"] = longform(model, data, tok, starts, args.seq,
                                      device)
        print("[gen]", json.dumps({k: v for k, v in report["longform"].items()
                                   if k != "examples"}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
