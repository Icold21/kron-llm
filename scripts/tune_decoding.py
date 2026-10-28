"""Measure decoding settings instead of arguing about them.

The recipe in the notebook (T=1.0, top_k=8, repetition penalty 1.1) was a
judgement call, not a measurement.  This script sweeps a grid and scores every
setting on the same prompts, with external judges rather than self-report.
Results and the adopted recipe are in ``docs/decoding.md``.  Four things are
measured:

* **judge bpb** -- an independent language model scores the continuation, GPT-2
  for English and ruGPT3-small for Russian.  Lower means the text looks more
  like natural prose in that language.  The judges are 125M models, they are not
  the model being tuned, and they never see the same weights.
* **loops** -- share of samples where some character 3-gram occurs three times.
  Degeneration is the failure mode of small models, so it is counted, not
  eyeballed.
* **distinct-3** -- diversity: unique character 3-grams over all of them.  A
  setting that wins on the judge by collapsing into two safe sentences loses here.
* **task checks** -- does the code prompt still parse, does ``17 + 25 =`` start
  with 42, does the identity prompt mention the model.  These are the behaviours
  a reader will test, so a decoding setting that destroys them is not "better".

Every configuration is run with the same prompts and the same seeds, so the
differences are paired and not sampling noise between one run and the next.

Usage::

    python scripts/tune_decoding.py --ckpt data/kaggle_out/q10/last.pt \
        --out data/decoding_sweep.json
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.model import KronTransformer  # noqa: E402
from kronllm.tokenizer import BPETokenizer  # noqa: E402

# (label, prompt, language) -- the same six prompts the session reports use, plus
# one science prompt, so the sweep is scored on what the reports actually show.
PROMPTS = [
    ("en_prose", "She opened the door and", "en"),
    ("en_science", "The Moon orbits the Earth at a distance of", "en"),
    ("ru_prose", "Он открыл дверь и", "ru"),
    ("ru_history", "В 1961 году", "ru"),
    ("code", "def add(a, b):\n    return", "code"),
    ("arith", "17 + 25 =", "code"),
    ("identity", "KronLLM is", "code"),
]

# (temperature, top_k, top_p, repetition_penalty)
GRID = [
    (0.8, 8, 0.0, 1.0), (0.8, 8, 0.0, 1.1), (0.8, 8, 0.0, 1.2),
    (0.8, 40, 0.0, 1.2), (0.9, 40, 0.0, 1.2), (0.9, 40, 0.0, 1.1),
    (1.0, 0, 0.0, 1.0), (1.0, 0, 0.0, 1.1), (1.0, 0, 0.0, 1.2),
    (1.0, 8, 0.0, 1.0), (1.0, 8, 0.0, 1.05), (1.0, 8, 0.0, 1.1),
    (1.0, 8, 0.0, 1.2), (1.0, 40, 0.0, 1.1), (1.0, 40, 0.0, 1.2),
    (1.1, 8, 0.0, 1.1), (1.1, 8, 0.0, 1.2), (1.1, 40, 0.0, 1.2),
    (1.0, 0, 0.9, 1.1), (1.0, 0, 0.95, 1.1),
    # Second stage: top_p instead of (or on top of) top_k, plus the fine steps
    # around T=1 that the first stage left as a gap.
    (0.9, 0, 0.90, 1.1), (1.0, 0, 0.85, 1.1), (1.0, 0, 0.90, 1.0),
    (1.0, 0, 0.90, 1.2), (1.0, 0, 0.95, 1.05), (1.0, 40, 0.95, 1.1),
    (0.95, 8, 0.0, 1.1), (0.95, 40, 0.0, 1.1),
    # Stage 3: does combining the two truncations help, or does the tighter one
    # simply win?  Both are "cut the tail", but k counts tokens and p counts mass.
    (1.0, 8, 0.90, 1.1), (1.0, 20, 0.90, 1.1), (1.0, 8, 0.95, 1.1),
]


@torch.no_grad()
def generate(model, tok, prompt: str, device, max_new_tokens: int,
             temperature: float, top_k: int, top_p: float,
             repetition_penalty: float, rep_window: int = 64):
    """Sample one continuation; also return its bits/byte under the raw model."""
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    past = None
    picked: list[int] = []
    bits = 0.0
    for _ in range(max_new_tokens):
        if past is not None and past[0][0].shape[2] < model.seq_len:
            step = model(idx[:, -1:], past_kv=past, use_cache=True)
            past = step["past_kv"]
        else:
            step = model(idx[:, -model.seq_len:], use_cache=True)
            past = step.get("past_kv")
        logits = step["logits"][:, -1, :].float()
        # Self-score on the untouched distribution: how expected the text is,
        # independent of the truncation that produced it.
        log_probs = torch.log_softmax(logits[0], dim=-1)
        # CTRL-style repetition penalty, exactly as in sample_q10.py.
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
        if top_p:
            ordered, order = torch.sort(logits, dim=-1, descending=True)
            probs = torch.softmax(ordered, dim=-1)
            cut = torch.cumsum(probs, dim=-1) > top_p
            cut[..., 1:] = cut[..., :-1].clone()
            cut[..., 0] = False
            ordered = ordered.masked_fill(cut, float("-inf"))
            logits = torch.empty_like(logits).scatter(-1, order, ordered)
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        bits += float(-log_probs[int(nxt[0, 0])]) / math.log(2.0)
        idx = torch.cat([idx, nxt], dim=1)
        if int(nxt[0, 0]) == tok.eot_id:
            break
        picked.append(int(nxt[0, 0]))
    text = tok.decode(picked) if picked else ""
    return text, bits


class Judge:
    """A 125M LM used only as a naturalness scorer: bits per byte of the text."""

    def __init__(self, name: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def bpb(self, prompt: str, continuation: str) -> float | None:
        payload = prompt + continuation
        ids = self.tok(payload, return_tensors="pt").input_ids.to(self.device)
        head = len(self.tok(prompt, add_special_tokens=False).input_ids)
        if ids.shape[1] - 1 <= head:
            return None
        logits = self.model(ids).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        target = ids[:, 1:]
        picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0, head:]
        bits = float(-picked.sum()) / math.log(2.0)
        nbytes = len(continuation.encode("utf-8"))
        return bits / nbytes if nbytes else None


def char_grams(text: str, n: int = 3) -> list[str]:
    flat = " ".join(text.split())
    return [flat[i:i + n] for i in range(max(0, len(flat) - n + 1))]


def loop_rate(texts: list[str]) -> float:
    """Share of samples where a character 3-gram repeats three or more times."""
    bad = 0
    for text in texts:
        grams = char_grams(text)
        if grams and Counter(grams).most_common(1)[0][1] >= 3:
            bad += 1
    return bad / max(1, len(texts))


def distinct3(texts: list[str]) -> float:
    scores = []
    for text in texts:
        grams = char_grams(text)
        if grams:
            scores.append(len(set(grams)) / len(grams))
    return sum(scores) / max(1, len(scores))


def code_parses(prompt: str, text: str) -> bool:
    try:
        ast.parse(prompt + text)
        return True
    except SyntaxError:
        return False


def arithmetic_ok(text: str) -> bool:
    head = text.strip().split("\n")[0]
    return "42" in head[:6]


def identity_ok(text: str) -> bool:
    low = text.lower()
    return "kron" in low or "kronecker" in low


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/kaggle_out/q10/last.pt")
    ap.add_argument("--out", default="data/decoding_sweep.json")
    ap.add_argument("--max-new-tokens", type=int, default=56)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--grid", default="all",
                    help="'all' or a comma-separated subset of indexes")
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
    print(f"[ckpt] step {state.get('step')}, {device}", flush=True)

    judges = {"en": Judge("gpt2", device),
              "ru": Judge("ai-forever/rugpt3small_based_on_gpt2", device)}
    print("[judges] gpt2 + rugpt3small loaded", flush=True)

    grid = GRID
    if args.grid != "all":
        grid = [GRID[int(i)] for i in args.grid.split(",")]

    results = []
    for temperature, top_k, top_p, rep in grid:
        t0 = time.time()
        per_prompt: dict[str, list[dict]] = {label: [] for label, _, _ in PROMPTS}
        for seed in range(args.seeds):
            for label, prompt, lang in PROMPTS:
                torch.manual_seed(seed)
                text, bits = generate(model, tok, prompt, device,
                                      args.max_new_tokens, temperature, top_k,
                                      top_p, rep)
                nbytes = max(1, len(text.encode("utf-8")))
                per_prompt[label].append({
                    "text": text, "lang": lang,
                    "self_bpb": bits / nbytes,
                    "judge_bpb": (judges[lang].bpb(prompt, text)
                                  if lang in judges else None),
                })
        flat = {label: rows for label, rows in per_prompt.items()}

        def mean(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        # Judged per language and never pooled: GPT-2 bpb on English and
        # ruGPT3 bpb on Russian sit on different byte scales (~2.9 vs ~5 bytes
        # per token), so one average would be dominated by the Russian column.
        judge_en = mean([r["judge_bpb"] for label, _, lang in PROMPTS
                         if lang == "en" for r in flat[label]])
        judge_ru = mean([r["judge_bpb"] for label, _, lang in PROMPTS
                         if lang == "ru" for r in flat[label]])
        all_texts = [r["text"] for rows in flat.values() for r in rows]
        row = {
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "rep": rep,
            "judge_en": judge_en, "judge_ru": judge_ru,
            "self_bpb": sum(r["self_bpb"] for rows in flat.values() for r in rows)
            / len(all_texts),            "loops": loop_rate(all_texts),
            "distinct3": distinct3(all_texts),
            "code_ok": sum(code_parses("def add(a, b):\n    return", r["text"])
                           for r in flat["code"]) / len(flat["code"]),
            "arith_ok": sum(arithmetic_ok(r["text"]) for r in flat["arith"])
            / len(flat["arith"]),
            "identity_ok": sum(identity_ok(r["text"]) for r in flat["identity"])
            / len(flat["identity"]),
            "seconds": time.time() - t0,
            "samples": {k: [r["text"] for r in v] for k, v in flat.items()},
        }
        results.append(row)
        print(f"[{temperature}/{top_k}/{top_p}/rp{rep}] "
              f"en {judge_en or float('nan'):.4f} ru {judge_ru or float('nan'):.4f} "
              f"loops {row['loops']:.2f} distinct {row['distinct3']:.3f} "
              f"code {row['code_ok']:.2f} ({row['seconds']:.0f}s)", flush=True)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
