"""What can be improved without training and without GPU quota?

Three levers that cost only inference time are measured here, on the same
prompts and seeds, so the differences are paired:

1. **The sampler.**  ``docs/decoding.md`` settled ``T=1.0, top_k=8, rp=1.1``
   against temperature/top_k/top_p.  Two families were missing: ``min_p``
   (a relative probability floor) and locally typical sampling, plus the
   n-gram ban.  All of them are now in ``KronTransformer.generate``.
2. **The prompt.**  A base model has no instruction tuning, so the prefix is
   part of the program: a style primer or a few-shot pattern can change what
   the continuation looks like without touching the weights.
3. **Best-of-N reranking.**  Sample N continuations and keep the one the model
   itself finds most likely.  Costs N forwards, needs no verifier, and is the
   only "free" lever that applies to open-ended text.

A fourth, arithmetic self-consistency, is measured only to show that it does
*not* work at 25.6 % of an epoch: majority voting cannot recover an answer the
model never samples.

Scores are external where possible: GPT-2 scores English, ruGPT3-small scores
Russian, both in bits per byte of the continuation; loops and distinct-3
measure degeneration and diversity; the code/arith/identity checks measure the
behaviours a reader will try.

Usage::

    python scripts/free_gains.py --part samplers
    python scripts/free_gains.py --part priming
    python scripts/free_gains.py --part rerank
    python scripts/free_gains.py --part vote
    python scripts/free_gains.py --part all --out data/free_gains.json
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.model import KronTransformer  # noqa: E402
from kronllm.tokenizer import BPETokenizer  # noqa: E402

PROMPTS = [
    ("en_prose", "She opened the door and", "en"),
    ("en_science", "The Moon orbits the Earth at a distance of", "en"),
    ("ru_prose", "Он открыл дверь и", "ru"),
    ("ru_history", "В 1961 году", "ru"),
    ("code", "def add(a, b):\n    return", "code"),
    ("arith", "17 + 25 =", "code"),
    ("identity", "KronLLM is", "code"),
]

# (label, kwargs for generate) -- top_k=0 means "no top_k cap".
SAMPLERS = [
    ("greedy", dict(temperature=0.0, top_k=0)),
    ("greedy+norep3", dict(temperature=0.0, top_k=0,
                           no_repeat_ngram_size=3)),
    ("recipe k8 rp1.1", dict(temperature=1.0, top_k=8,
                             repetition_penalty=1.1)),
    ("recipe+top_p0.9", dict(temperature=1.0, top_k=8, top_p=0.9,
                             repetition_penalty=1.1)),
    ("min_p0.02", dict(temperature=1.0, top_k=0, min_p=0.02)),
    ("min_p0.05", dict(temperature=1.0, top_k=0, min_p=0.05)),
    ("min_p0.10", dict(temperature=1.0, top_k=0, min_p=0.10)),
    ("typical0.5", dict(temperature=1.0, top_k=0, typical_p=0.5)),
    ("typical0.9", dict(temperature=1.0, top_k=0, typical_p=0.9)),
    ("k8+norep3", dict(temperature=1.0, top_k=8, repetition_penalty=1.1,
                       no_repeat_ngram_size=3)),
    ("min_p0.05+norep3", dict(temperature=1.0, top_k=0, min_p=0.05,
                              no_repeat_ngram_size=3)),
    # The soup repeated itself slightly more than its parents; this variant asks
    # whether that can be undone with the repetition penalty instead of by
    # choosing different weights.
    ("recipe+top_p0.9 rp1.15", dict(temperature=1.0, top_k=8, top_p=0.9,
                                    repetition_penalty=1.15)),
]

# Primers are prepended verbatim.  The point is not that these are good
# examples -- it is whether *any* conditioning changes the continuation.
PRIMERS = {
    "none": {"en": "", "ru": "", "code": ""},
    "style": {
        "en": "The following is a passage of clear English prose.\n\n",
        "ru": "Ниже — отрывок ясной русской прозы.\n\n",
        "code": "# Well-tested Python.\n",
    },
    "fewshot": {
        "en": "Q: What is the capital of France?\nA: Paris.\n\nQ: ",
        "ru": "Вопрос: Столица Франции?\nОтвет: Париж.\n\nВопрос: ",
        "code": "def sub(a, b):\n    return a - b\n\n",
    },
}


def load(ckpt: str, device: str):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    model = KronTransformer(cfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    tok = BPETokenizer("data/bilingual/tokenizer.json",
                       cfg["data"]["vocab_size"],
                       cfg.get("bytes_per_token", 3.694),
                       cfg["data"]["eot_id"])
    return model, tok, state


class Judge:
    """A 125M LM used only as a naturalness scorer: bits per byte of text."""

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
        picked = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        bits = float(-picked[0, head:].sum()) / math.log(2.0)
        nbytes = len(continuation.encode("utf-8"))
        return bits / nbytes if nbytes else None


@torch.no_grad()
def self_bpb(model, tok, prompt: str, text: str, device: str) -> float | None:
    """Teacher-forced bits per byte of ``text`` given ``prompt``."""
    head = len(tok.encode(prompt))
    ids = tok.encode(prompt + text)
    if len(ids) <= head + 1:
        return None
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    loss = model(x[:, -model.seq_len:], y[:, -model.seq_len:])["loss"]
    n = min(len(ids) - 1 - head, model.seq_len - head)
    bits = float(loss) * n / math.log(2.0)
    nbytes = len(text.encode("utf-8"))
    return bits / nbytes if nbytes else None


def char_grams(text: str, n: int = 3) -> list[str]:
    flat = " ".join(text.split())
    return [flat[i:i + n] for i in range(max(0, len(flat) - n + 1))]


def loop_rate(texts: list[str]) -> float:
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


def first_number(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def sample_all(model, tok, device, max_new: int, seeds: int, kwargs: dict,
               primers: dict | None = None) -> dict[str, list[dict]]:
    """Generate the probe set once; returns rows per prompt label."""
    out: dict[str, list[dict]] = {label: [] for label, _, _ in PROMPTS}
    for seed in range(seeds):
        for label, prompt, lang in PROMPTS:
            primer = (primers or {}).get(lang, "")
            full = primer + prompt
            torch.manual_seed(seed)
            ids = torch.tensor([tok.encode(full)], dtype=torch.long,
                               device=device)
            seq = model.generate(ids, max_new_tokens=max_new,
                                 eot_id=tok.eot_id, stop_at_eot=True,
                                 **kwargs)
            text = tok.decode(seq[0, ids.shape[1]:].tolist())
            out[label].append({"prompt": full, "text": text, "lang": lang,
                               "seed": seed})
    return out


def score(rows: dict[str, list[dict]], judges: dict, model, tok,
          device: str) -> dict:
    for label, items in rows.items():
        for item in items:
            item["judge_bpb"] = (judges[item["lang"]].bpb(item["prompt"],
                                                          item["text"])
                                 if item["lang"] in judges else None)
            item["self_bpb"] = self_bpb(model, tok, item["prompt"],
                                        item["text"], device)
    texts = [i["text"] for items in rows.values() for i in items]
    judge_en = mean([i["judge_bpb"] for label, _, lang in PROMPTS
                     if lang == "en" for i in rows[label]])
    judge_ru = mean([i["judge_bpb"] for label, _, lang in PROMPTS
                     if lang == "ru" for i in rows[label]])
    return {
        "judge_en": judge_en, "judge_ru": judge_ru,
        "self_bpb": mean([i["self_bpb"] for items in rows.values()
                          for i in items]),
        "loops": loop_rate(texts), "distinct3": distinct3(texts),
        "code_ok": sum(code_parses("def add(a, b):\n    return", i["text"])
                       for i in rows["code"]) / len(rows["code"]),
        "arith_ok": sum(arithmetic_ok(i["text"]) for i in rows["arith"])
        / len(rows["arith"]),
        "identity_ok": sum(identity_ok(i["text"]) for i in rows["identity"])
        / len(rows["identity"]),
    }


def part_samplers(model, tok, device, args, judges) -> list[dict]:
    results = []
    grid = SAMPLERS
    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        grid = [s for s in SAMPLERS if any(w in s[0] for w in wanted)]
        if not grid:
            raise SystemExit(f"--only {args.only!r} matched no sampler")
    for label, kwargs in grid:
        t0 = time.time()
        rows = sample_all(model, tok, device, args.max_new_tokens, args.seeds,
                          kwargs)
        row = {"sampler": label, **kwargs,
               **score(rows, judges, model, tok, device),
               "seconds": round(time.time() - t0, 1),
               "samples": {k: [i["text"] for i in v] for k, v in rows.items()}}
        results.append(row)
        print(f"[{label:20s}] en {row['judge_en'] or float('nan'):.4f} "
              f"ru {row['judge_ru'] or float('nan'):.4f} "
              f"self {row['self_bpb'] or float('nan'):.4f} "
              f"loops {row['loops']:.2f} distinct {row['distinct3']:.3f} "
              f"code {row['code_ok']:.2f} arith {row['arith_ok']:.2f} "
              f"({row['seconds']:.0f}s)", flush=True)
    return results


def part_priming(model, tok, device, args, judges) -> list[dict]:
    results = []
    kwargs = dict(temperature=1.0, top_k=8, top_p=0.9,
                  repetition_penalty=1.1)
    for name, primers in PRIMERS.items():
        rows = sample_all(model, tok, device, args.max_new_tokens, args.seeds,
                          kwargs, primers)
        row = {"primer": name, **score(rows, judges, model, tok, device),
               "samples": {k: [i["text"] for i in v] for k, v in rows.items()}}
        results.append(row)
        print(f"[primer {name:8s}] en {row['judge_en'] or float('nan'):.4f} "
              f"ru {row['judge_ru'] or float('nan'):.4f} "
              f"self {row['self_bpb'] or float('nan'):.4f} "
              f"loops {row['loops']:.2f} distinct {row['distinct3']:.3f} "
              f"arith {row['arith_ok']:.2f}", flush=True)
    return results


def part_rerank(model, tok, device, args, judges) -> list[dict]:
    """Best-of-N by the model's own likelihood vs a random pick from the N."""
    kwargs = dict(temperature=1.0, top_k=8, top_p=0.9, repetition_penalty=1.1)
    results = []
    for n in (1, 4, 8):
        picked_self, picked_rand, picked_noloop = [], [], []
        for label, prompt, lang in PROMPTS:
            if lang not in judges:
                continue
            cands = []
            for seed in range(n):
                torch.manual_seed(seed)
                ids = torch.tensor([tok.encode(prompt)], dtype=torch.long,
                                   device=device)
                seq = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                     eot_id=tok.eot_id, stop_at_eot=True,
                                     **kwargs)
                text = tok.decode(seq[0, ids.shape[1]:].tolist())
                cands.append({"text": text,
                              "self_bpb": self_bpb(model, tok, prompt, text,
                                                   device),
                              "judge_bpb": judges[lang].bpb(prompt, text),
                              "loop": loop_rate([text])})
            scored = [c for c in cands if c["self_bpb"] is not None]
            best = min(scored, key=lambda c: c["self_bpb"])
            picked_self.append(best["judge_bpb"])
            # A random pick from the same N is the null hypothesis: it is what
            # the sampler already gives you for free.
            picked_rand.append(cands[random.randrange(len(cands))]["judge_bpb"])
            clean = [c for c in scored if not c["loop"]]
            picked_noloop.append((clean or scored)[0]["judge_bpb"])
        row = {"n": n, "judge_self": mean(picked_self),
               "judge_random": mean(picked_rand),
               "judge_noloop": mean(picked_noloop),
               "prompts": 4}
        results.append(row)
        print(f"[best-of-{n}] self {row['judge_self']:.4f} "
              f"random {row['judge_random']:.4f} "
              f"first-clean {row['judge_noloop']:.4f}", flush=True)
    return results


def part_vote(model, tok, device, args, judges) -> dict:
    """Self-consistency on 50 two-digit sums: does majority voting help?"""
    rng = random.Random(0)
    probes = [(rng.randint(10, 99), rng.randint(10, 99)) for _ in range(50)]
    kwargs = dict(temperature=1.0, top_k=8, top_p=0.9,
                  repetition_penalty=1.1)
    singles, votes, anys, hist = 0, 0, 0, []
    for a, b in probes:
        prompt = f"{a} + {b} ="
        answers = []
        for seed in range(args.votes):
            torch.manual_seed(seed)
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.long,
                               device=device)
            seq = model.generate(ids, max_new_tokens=8, eot_id=tok.eot_id,
                                 stop_at_eot=True, **kwargs)
            answers.append(first_number(tok.decode(seq[0, ids.shape[1]:]
                                                   .tolist())))
        truth = a + b
        singles += answers[0] == truth
        anys += any(x == truth for x in answers)
        counted = Counter(x for x in answers if x is not None)
        top = counted.most_common(1)
        votes += bool(top) and top[0][0] == truth
        hist.append({"a": a, "b": b, "truth": truth, "answers": answers})
    out = {"probes": len(probes), "votes": args.votes,
           "acc_first": singles / len(probes),
           "acc_majority": votes / len(probes),
           "acc_any": anys / len(probes),
           "examples": hist[:10]}
    print(f"[vote] first {out['acc_first']:.3f} "
          f"majority-{args.votes} {out['acc_majority']:.3f} "
          f"any-of-{args.votes} {out['acc_any']:.3f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/kaggle_out/q10/last.pt")
    ap.add_argument("--part", default="all",
                    choices=["all", "samplers", "priming", "rerank", "vote"])
    ap.add_argument("--out", default="data/free_gains.json")
    ap.add_argument("--max-new-tokens", type=int, default=56)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--votes", type=int, default=8)
    ap.add_argument("--only", default="",
                    help="comma-separated substrings to keep from SAMPLERS")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, state = load(args.ckpt, device)
    print(f"[ckpt] step {state.get('step')} {device} "
          f"{sum(p.numel() for p in model.parameters()):,} params", flush=True)

    judges = {"en": Judge("gpt2", device),
              "ru": Judge("ai-forever/rugpt3small_based_on_gpt2", device)}
    print("[judges] gpt2 + rugpt3small loaded", flush=True)

    out: dict = {"ckpt": args.ckpt, "step": state.get("step"),
                 "max_new_tokens": args.max_new_tokens, "seeds": args.seeds}
    if args.part in ("all", "samplers"):
        out["samplers"] = part_samplers(model, tok, device, args, judges)
    if args.part in ("all", "priming"):
        out["priming"] = part_priming(model, tok, device, args, judges)
    if args.part in ("all", "rerank"):
        out["rerank"] = part_rerank(model, tok, device, args, judges)
    if args.part in ("all", "vote"):
        out["vote"] = part_vote(model, tok, device, args, judges)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
