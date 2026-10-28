"""How much probability does top_k keep, and how many tokens does top_p keep?

``top_k=8`` and ``top_p=0.9`` look like the same idea -- cut the tail -- and they
are not.  top_k always keeps exactly k candidates, top_p keeps however many are
needed to reach the mass.  On a distribution that is flat over thousands of
tokens those two decisions are wildly different, and this script measures the
difference on the real model instead of arguing from the shape of a curve:

* ``mass@k`` -- probability mass held by the k best tokens at a decoding step;
* ``nucleus@p`` -- how many tokens it takes to reach mass p.

If ``mass@8`` is 0.98, then top_p=0.9 keeps fewer than eight tokens and is
strictly tighter than top_k=8.  If ``mass@8`` is 0.35, top_p=0.9 keeps hundreds
of tokens and top_k=8 is the only real truncation.

Usage::

    python scripts/decoding_mass.py --ckpt data/kaggle_out/q10/last.pt
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.model import KronTransformer  # noqa: E402
from kronllm.tokenizer import BPETokenizer  # noqa: E402

KS = (1, 4, 8, 20, 40, 100, 1000)
PS = (0.5, 0.8, 0.9, 0.95, 0.99)
PROMPTS = [
    ("en_prose", "She opened the door and"),
    ("en_science", "The Moon orbits the Earth at a distance of"),
    ("ru_prose", "Он открыл дверь и"),
    ("ru_history", "В 1961 году"),
    ("code", "def add(a, b):\n    return"),
    ("arith", "17 + 25 ="),
]


@torch.no_grad()
def walk(model, tok, prompt: str, device, steps: int = 48,
         temperature: float = 1.0, top_k: int = 8,
         repetition_penalty: float = 1.1):
    """Sample with one recipe and record the untruncated distribution each step."""
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    past = None
    rows = []
    for _ in range(steps):
        if past is not None and past[0][0].shape[2] < model.seq_len:
            step = model(idx[:, -1:], past_kv=past, use_cache=True)
            past = step["past_kv"]
        else:
            step = model(idx[:, -model.seq_len:], use_cache=True)
            past = step.get("past_kv")
        logits = step["logits"][:, -1, :].float()
        probs = torch.softmax(logits[0], dim=-1)
        ordered = torch.sort(probs, descending=True).values
        cum = torch.cumsum(ordered, dim=-1)
        row = {f"mass@{k}": float(ordered[:k].sum()) for k in KS}
        for p in PS:
            n = int(torch.searchsorted(cum, torch.tensor(p, device=cum.device)) + 1)
            row[f"nucleus@{p}"] = min(n, ordered.numel())
        rows.append(row)

        # Same decoding as scripts/sample_q10.py, so the walk stays on the same
        # trajectory the recipe actually produces.
        if repetition_penalty and repetition_penalty != 1.0:
            recent = torch.unique(idx[0, -64:])
            target = logits[0, recent]
            logits[0, recent] = torch.where(target > 0,
                                            target / repetition_penalty,
                                            target * repetition_penalty)
        logits = logits / max(temperature, 1e-5)
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < thresh, float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/kaggle_out/q10/last.pt")
    ap.add_argument("--out", default="data/decoding_mass.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    tok = BPETokenizer("data/bilingual/tokenizer.json",
                       state["cfg"]["data"]["vocab_size"],
                       state["cfg"].get("bytes_per_token", 3.694),
                       state["cfg"]["data"]["eot_id"])

    all_rows: dict[str, list[dict]] = {}
    for label, prompt in PROMPTS:
        torch.manual_seed(0)
        all_rows[label] = walk(model, tok, prompt, device)

    def mean_of(key: str, label: str | None = None) -> float:
        rows = all_rows[label] if label else [
            r for rows in all_rows.values() for r in rows]
        return statistics.fmean(r[key] for r in rows)

    print(f"# {args.ckpt} step {state.get('step')}\n")
    print("| метрика | " + " | ".join(label for label, _ in PROMPTS) + " | всё |")
    print("|---" * (len(PROMPTS) + 2) + "|")
    for key in [f"mass@{k}" for k in KS] + [f"nucleus@{p}" for p in PS]:
        cells = " | ".join(f"{mean_of(key, label):.2f}" for label, _ in PROMPTS)
        print(f"| {key} | {cells} | {mean_of(key):.2f} |")

    Path(args.out).write_text(
        json.dumps({label: rows for label, rows in all_rows.items()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
