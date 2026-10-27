"""Render a generation section for the session report.

Numbers alone do not show whether a checkpoint became a better writer, so every
session report carries the same five prompts, sampled with the agreed recipe
(T=1.0, top_k=8, repetition penalty 1.1).  Same prompts every time, so the
sections can be read as a series instead of as isolated anecdotes.

Usage::

    python scripts/report_samples.py --ckpt data/kaggle_out/q10/last.pt \
        --out data/report_samples.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.model import KronTransformer  # noqa: E402
from kronllm.tokenizer import BPETokenizer  # noqa: E402

PROMPTS = [
    ("проза EN", "She opened the door and"),
    ("проза RU", "Он открыл дверь и"),
    ("код", "def add(a, b):\n    return"),
    ("арифметика", "17 + 25 ="),
    ("identity", "KronLLM is"),
]
MAX_NEW = 48


@torch.no_grad()
def stream(model, tok, prompt, device, max_new_tokens=MAX_NEW, temperature=1.0,
           top_k=8, repetition_penalty=1.1, rep_window=64):
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    past, shown = None, ""
    for _ in range(max_new_tokens):
        if past is not None and past[0][0].shape[2] < model.seq_len:
            step = model(idx[:, -1:], past_kv=past, use_cache=True)
            past = step["past_kv"]
        else:
            step = model(idx[:, -model.seq_len:], use_cache=True)
            past = step.get("past_kv")
        logits = step["logits"][:, -1, :].float()
        if repetition_penalty and repetition_penalty != 1.0:
            recent = torch.unique(idx[0, -rep_window:])
            target = logits[0, recent]
            logits[0, recent] = torch.where(target > 0,
                                            target / repetition_penalty,
                                            target * repetition_penalty)
        logits = logits / max(temperature, 1e-5)
        if top_k:
            k = min(top_k, logits.size(-1))
            thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
        if int(nxt[0, 0]) == tok.eot_id:
            break
        text = tok.decode(idx[0, len(ids):].tolist())
        if text.endswith("\ufffd"):
            continue
        if len(text) > len(shown):
            yield text[len(shown):]
            shown = text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/kaggle_out/q10/last.pt")
    ap.add_argument("--out", default="data/report_samples.md")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok_path = next((p for p in ("data/bilingual/tokenizer.json",
                                 "artifacts/q10_tokenizer.json")
                     if Path(p).exists()), None)
    if tok_path is None:
        raise SystemExit("no tokenizer found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    tok = BPETokenizer(tok_path, state["cfg"]["data"]["vocab_size"],
                       state["cfg"].get("bytes_per_token", 3.694),
                       state["cfg"]["data"]["eot_id"])

    lines = [
        "## Генерация",
        "",
        f"Шаг {state.get('step')}, T={args.temperature}, top_k={args.top_k}, "
        f"repetition penalty {args.rep_penalty}, {MAX_NEW} токенов. Промпты одни "
        "и те же от отчёта к отчёту, чтобы серии можно было сравнивать.",
        "",
    ]
    t0 = time.time()
    for label, prompt in PROMPTS:
        text = "".join(stream(model, tok, prompt, device,
                              temperature=args.temperature, top_k=args.top_k,
                              repetition_penalty=args.rep_penalty))
        sample = text.strip().replace("\n", " ⏎ ")
        lines += [f"**{label}** — `{prompt.replace(chr(10), '⏎ ')}`", "",
                  f"> {sample}", ""]
    lines.append(f"*Снято за {time.time() - t0:.0f} с на {device}.*")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
