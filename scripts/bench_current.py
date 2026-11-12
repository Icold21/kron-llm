"""One battery for one checkpoint: the standard benchmarks and our own tests.

The standard side is the usual zero-shot multiple-choice set, scored the way
``lm-evaluation-harness`` scores it by default: the summed log-probability of
the continuation, argmax over the choices.  These are English benchmarks and a
9.9M-parameter bilingual model is at or near chance on most of them -- the point
of running them is not to win, it is to have the number, and to have a harness
that will still work when the model is 10x bigger.

The our-own side is what actually moves here:

* held-out bits per byte on the pretraining bin -- did fine-tuning cost anything;
* held-out bits per byte on the SFT bin -- the training curve's own metric;
* column drills per family (addition with carries, subtraction with borrows,
  one-digit multiplication), scored on the final answer line;
* the bare form (``47 + 38 =``) in the render the corpus teaches;
* chat turns, kept verbatim because there is no reference answer;
* repetition and distinct-n on free generation.

Everything lands in one JSON with the per-task examples, so the notebook can
show the failures and not only the aggregate.

Usage::

    python scripts/bench_current.py --ckpt E:/kron_out/s10_ckpt/last.pt --n 200
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kronllm.model import KronTransformer       # noqa: E402
from kronllm.tokenizer import BPETokenizer      # noqa: E402

VAL_PRETRAIN = ROOT / "data" / "bilingual" / "val.bpe.bin"
VAL_SFT = ROOT / "data" / "sft" / "val.bpe.bin"
LAMBADA_WORD = re.compile(r"(\S+)\s*$")

# chance level per benchmark, so the table can show what "at chance" means
CHANCE = {"hellaswag": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25,
          "piqa": 0.50, "winogrande": 0.50, "openbookqa": 0.25,
          "boolq": 0.50, "mmlu": 0.25, "lambada": 0.0}


def _load(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# scoring primitives
# --------------------------------------------------------------------------
@torch.no_grad()
def logprob(model, ctx_ids: list[int], choice_ids: list[int],
            device: torch.device) -> float:
    """Summed log-probability of ``choice_ids`` after ``ctx_ids``."""
    ids = torch.tensor([ctx_ids + choice_ids], device=device)
    out = model(ids)
    logits = out["logits"][0, len(ctx_ids) - 1:-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    target = torch.tensor(choice_ids, device=device)
    return float(logp.gather(1, target[:, None]).sum())


@torch.no_grad()
def score_choices(model, tok, ctx: str, choices: list[str], device,
                  max_ctx: int = 384, normalise: bool = False) -> list[float]:
    ctx_ids = tok.encode(ctx)[-max_ctx:]
    scores, lengths = [], []
    for choice in choices:
        cids = tok.encode(" " + choice) if not choice.startswith(" ") else tok.encode(choice)
        cids = cids[:96] or [tok.eot_id]
        s = logprob(model, ctx_ids, cids, device)
        scores.append(s / max(1, len(cids)) if normalise else s)
        lengths.append(len(cids))
    return scores


def accuracy(preds: list[int], golds: list[int]) -> float:
    return sum(int(p == g) for p, g in zip(preds, golds)) / max(1, len(golds))


# --------------------------------------------------------------------------
# standard benchmarks
# --------------------------------------------------------------------------
def _hf(path: str, cfg: str | None, split: str, n: int, trust: bool = False):
    from datasets import load_dataset
    kw = {"split": f"{split}[:{n}]"}
    if trust:
        kw["trust_remote_code"] = True
    ds = load_dataset(path, cfg, **kw) if cfg else load_dataset(path, **kw)
    return ds


def letter_gold(row: dict) -> int:
    """Index of the correct choice when the dataset answers by letter.

    ai2_arc and openbookqa carry `answerKey` as a letter and the letters again
    in `choices['label']`, but a few rows use a 1-based number, which made a
    plain `"ABCD".index(...)` fail with 'substring not found'.
    """
    labels = [str(x) for x in row["choices"]["label"]]
    key = str(row["answerKey"]).strip()
    if key in labels:
        return labels.index(key)
    if key.upper() in [x.upper() for x in labels]:
        return [x.upper() for x in labels].index(key.upper())
    return max(0, int(key) - 1)


def bench_items(name: str, n: int) -> list[dict]:
    """(ctx, choices, gold) triples in the harness's own format."""
    if name == "hellaswag":
        ds = _hf("hellaswag", None, "validation", n)
        return [{"ctx": (r["activity_label"] + ": " + r["ctx"]), "choices": r["endings"],
                 "gold": int(r["label"])} for r in ds]
    if name in ("arc_easy", "arc_challenge"):
        cfg = "ARC-Easy" if name == "arc_easy" else "ARC-Challenge"
        ds = _hf("allenai/ai2_arc", cfg, "validation", n, trust=True)
        return [{"ctx": r["question"], "choices": r["choices"]["text"],
                 "gold": letter_gold(r)} for r in ds]
    if name == "piqa":
        ds = _hf("ybisk/piqa", None, "validation", n, trust=True)
        return [{"ctx": r["goal"], "choices": [r["sol1"], r["sol2"]],
                 "gold": int(r["label"])} for r in ds]
    if name == "winogrande":
        ds = _hf("winogrande", "winogrande_xl", "validation", n)
        out = []
        for r in ds:
            ctx, c1, c2 = r["sentence"], r["option1"], r["option2"]
            lead, tail = ctx.split("_", 1)
            out.append({"ctx": lead, "choices": [c1 + tail, c2 + tail],
                        "gold": int(r["answer"]) - 1})
        return out
    if name == "openbookqa":
        ds = _hf("allenai/openbookqa", "main", "validation", n)
        return [{"ctx": r["question_stem"], "choices": r["choices"]["text"],
                 "gold": letter_gold(r)} for r in ds]
    if name == "boolq":
        ds = _hf("google/boolq", None, "validation", n)
        return [{"ctx": r["passage"] + " " + r["question"], "choices": ["yes", "no"],
                 "gold": 0 if r["answer"] else 1} for r in ds]
    if name == "mmlu":
        ds = _hf("cais/mmlu", "all", "test", n)
        return [{"ctx": r["question"], "choices": r["choices"], "gold": int(r["answer"])}
                for r in ds]
    raise SystemExit(f"unknown benchmark {name}")


def run_mc(model, tok, name: str, n: int, device) -> dict:
    items = bench_items(name, n)
    t0 = time.time()
    preds, golds, examples = [], [], []
    for it in items:
        scores = score_choices(model, tok, it["ctx"], it["choices"], device)
        pred = int(np.argmax(scores))
        preds.append(pred)
        golds.append(it["gold"])
        if len(examples) < 6:
            examples.append({"ctx": it["ctx"][:160], "choices": it["choices"],
                             "scores": [round(s, 2) for s in scores],
                             "pred": pred, "gold": it["gold"]})
    acc = accuracy(preds, golds)
    print(f"  {name:14s} {acc:6.3f}  ({len(items)} items, {time.time() - t0:.0f}s)"
          f"  chance {CHANCE.get(name, 0):.2f}")
    return {"acc": acc, "n": len(items), "chance": CHANCE.get(name, 0.0),
            "examples": examples}


@torch.no_grad()
def run_lambada(model, tok, n: int, device) -> dict:
    ds = _hf("cimec/lambada", None, "test", n)
    hits, examples = 0, []
    for r in ds:
        text = r["text"]
        m = LAMBADA_WORD.search(text)
        if not m:
            continue
        word, prompt = m.group(1), text[:m.start(1)]
        ids = torch.tensor([tok.encode(prompt)[-480:]], device=device)
        gen = model.generate(ids, max_new_tokens=12, temperature=0.0, top_k=1,
                             top_p=1.0, repetition_penalty=1.0)
        got = tok.decode(gen[0, ids.shape[1]:].tolist()).strip()
        ok = got.split()[0].strip(".,;:!?\"'") == word.strip(".,;:!?\"'") if got else False
        hits += int(ok)
        if len(examples) < 6:
            examples.append({"prompt": prompt[-140:], "want": word, "got": got[:60],
                             "ok": ok})
    acc = hits / max(1, len(ds))
    print(f"  {'lambada':14s} {acc:6.3f}  ({len(ds)} items)")
    return {"acc": acc, "n": len(ds), "chance": 0.0, "examples": examples}


# --------------------------------------------------------------------------
# held-out bits per byte
# --------------------------------------------------------------------------
@torch.no_grad()
def bpb(model, tok, bin_path: Path, blocks: int, device, seq: int = 512) -> float:
    """Bits per byte over evenly spaced blocks, bytes counted from the tokenizer.

    The loss is computed here rather than taken from ``out["loss"]``: SFT turns
    on ``mask_user_turns``, so the model's own mean covers only the answer tokens
    while the byte count covers the whole block -- dividing one by the other
    inflated the SFT figure by about 20 % on the first run of this battery.
    """
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    total_tokens = len(data)
    stride = (total_tokens - seq - 1) // blocks
    bytes_table = tok.byte_lengths()
    nats, nbytes = 0.0, 0
    for b in range(blocks):
        start = b * stride
        ids = torch.tensor(data[start:start + seq].astype(np.int64), device=device)[None]
        tgt = torch.tensor(data[start + 1:start + seq + 1].astype(np.int64), device=device)[None]
        logits = model(ids)["logits"][0, :-1].float()
        t = tgt[0, :logits.shape[0]]
        loss = torch.nn.functional.cross_entropy(logits, t)
        nats += float(loss) * int(t.shape[0])
        nbytes += int(bytes_table[t.cpu().numpy()].sum())
    return nats / math.log(2) / max(1, nbytes)


# --------------------------------------------------------------------------
# our own tests
# --------------------------------------------------------------------------
def our_tests(model, tok, device, ckpt_step: int) -> dict:
    arith = _load(ROOT / "scripts" / "gen_arithmetic_sft.py", "arith_bench")
    build_sft = _load(ROOT / "scripts" / "build_sft.py", "build_sft_bench")
    drill = _load(ROOT / "scripts" / "eval_arith_drill.py", "drill_bench")

    out: dict = {"step": ckpt_step}

    # column drills, per family
    rng = random.Random(11)
    families: dict = {}
    for lang in ("en", "ru"):
        word = drill.ANSWER_WORD[lang]
        for fn in drill.column_families(arith, lang):
            hits = rows = 0
            for _ in range(3):
                question, cot, answer = fn(rng)
                want = drill.answer_value(answer)
                _, prompt = build_sft.render([
                    {"role": "system", "content": build_sft.SYSTEM},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"{cot}\n\n{word} {answer}"}])
                ids = torch.tensor([tok.encode(prompt)], device=device)
                gen = model.generate(ids, max_new_tokens=200, temperature=0.0,
                                     top_k=1, top_p=1.0, repetition_penalty=1.0)
                text = tok.decode(gen[0, ids.shape[1]:].tolist())
                m = re.search(word + r"\s*(-?\d+)", text)
                hits += int(bool(m) and want is not None and int(m.group(1)) == want)
                rows += int(drill.first_row_ok(text))
            families[f"{lang}/{fn.__name__.replace('_column', '')}"] = {
                "ok": hits, "n": 3, "first_row": rows}
    out["drills"] = families

    # the bare form, in the render the corpus teaches
    bare = {"n": 0, "ok": 0, "examples": []}
    for lang in ("en", "ru"):
        for _ in range(10):
            question, cot, answer = arith.en_bare(rng)
            _, prompt = build_sft.render([
                {"role": "system", "content": build_sft.SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}])
            ids = torch.tensor([tok.encode(prompt)], device=device)
            gen = model.generate(ids, max_new_tokens=6, temperature=0.0, top_k=1,
                                 top_p=1.0, repetition_penalty=1.0)
            got = tok.decode(gen[0, ids.shape[1]:].tolist()).strip()
            m = re.match(r"-?\d+", got)
            ok = bool(m) and int(m.group()) == int(answer)
            bare["n"] += 1
            bare["ok"] += int(ok)
            if len(bare["examples"]) < 6:
                bare["examples"].append({"q": question, "want": answer,
                                         "got": got[:16], "ok": ok})
    out["bare"] = bare

    # chat turns, verbatim
    chats = []
    for system, user in (
            ("You are KronLLM, a compact bilingual model.", "Who are you?"),
            ("You are KronLLM, a compact bilingual model.", "What is the capital of France?"),
            ("You are a helpful assistant that answers in Russian.", "Сколько будет 2 + 2?"),
            ("You are a helpful assistant.", "Name three colours."),
            ("Ты — полезный ассистент.", "Назови три цвета.")):
        prompt = (f"<|system|>{system}<|eot|><|user|>{user}<|eot|><|assistant|>")
        ids = torch.tensor([tok.encode(prompt)], device=device)
        gen = model.generate(ids, max_new_tokens=48, temperature=0.7, top_k=40,
                             top_p=0.95, repetition_penalty=1.2)
        chats.append({"system": system, "user": user,
                      "answer": tok.decode(gen[0, ids.shape[1]:].tolist())})
    out["chat"] = chats

    # free generation: repetition and distinct-n
    base_prompts = [
        "The Roman Empire was one of the largest and longest-lasting empires in history, and",
        "def quicksort(items):",
        "Фотосинтез — это процесс, при котором растения",
        "Once upon a time, in a small village by the sea,",
    ]
    gens = []
    for p in base_prompts:
        ids = torch.tensor([tok.encode(p)], device=device)
        gen = model.generate(ids, max_new_tokens=64, temperature=0.7, top_k=40,
                             top_p=0.95, repetition_penalty=1.2)
        text = tok.decode(gen[0, ids.shape[1]:].tolist())
        gens.append({"prompt": p, "text": text})
    out["generation"] = gens
    words = " ".join(g["text"] for g in gens).split()
    bigrams = list(zip(words, words[1:]))
    out["repeat"] = {
        "distinct1": len(set(words)) / max(1, len(words)),
        "distinct2": len(set(bigrams)) / max(1, len(bigrams)),
        "repeat_4gram": repeat_ngram([g["text"] for g in gens], 4),
    }
    return out


def repeat_ngram(texts: list[str], n: int) -> float:
    grams, repeats = [], 0
    for text in texts:
        words = text.split()
        g = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
        grams += g
        repeats += len(g) - len(set(g))
    return repeats / max(1, len(grams))


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,piqa,"
                                       "winogrande,openbookqa,boolq,mmlu,lambada")
    ap.add_argument("--out", default="data/bench_current.json")
    ap.add_argument("--only-bench", action="store_true",
                    help="benchmarks only: no bpb, no our-tests (for a top-up run)")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    step = int(state.get("step") or 0)
    model = KronTransformer(state["cfg"]).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    device = next(model.parameters()).device
    tok = BPETokenizer(ROOT / "data" / "bilingual" / "tokenizer.json",
                       state["cfg"]["data"]["vocab_size"], bytes_per_token=3.69,
                       eot_id=state["cfg"]["data"]["eot_id"])
    params = sum(p.numel() for p in model.parameters())
    print(f"[load] {args.ckpt} | step {step:,} | {params:,} params | {device}")

    result: dict = {"ckpt": str(args.ckpt), "step": step, "params": params,
                    "bench": {}, "our": {}, "bpb": {}}

    if not args.only_bench and VAL_PRETRAIN.exists():
        t0 = time.time()
        result["bpb"]["pretrain_val"] = bpb(model, tok, VAL_PRETRAIN,
                                            args.blocks, device)
        print(f"  pretrain val bpb {result['bpb']['pretrain_val']:.4f}"
              f"  ({time.time() - t0:.0f}s)")
    if not args.only_bench and VAL_SFT.exists():
        t0 = time.time()
        result["bpb"]["sft_val"] = bpb(model, tok, VAL_SFT, args.blocks, device)
        print(f"  sft val bpb      {result['bpb']['sft_val']:.4f}"
              f"  ({time.time() - t0:.0f}s)")

    for name in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        try:
            result["bench"][name] = (run_lambada(model, tok, args.n, device)
                                     if name == "lambada"
                                     else run_mc(model, tok, name, args.n, device))
        except Exception as exc:  # noqa: BLE001 - one broken dataset must not stop it
            print(f"  {name:14s} SKIPPED ({type(exc).__name__}: {str(exc)[:70]})")
            result["bench"][name] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}

    result["our"] = {} if args.only_bench else our_tests(model, tok, device, step)
    drills = result["our"].get("drills", {})
    ok = sum(v["ok"] for v in drills.values())
    n = sum(v["n"] for v in drills.values())
    print(f"  drills           {ok}/{n}")
    bare = result['our'].get('bare')
    if bare:
        print(f"  bare             {bare['ok']}/{bare['n']}")
    if 'repeat' in result['our']:
        print(f"  distinct-2       {result['our']['repeat']['distinct2']:.3f}")

    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print("written", args.out)


if __name__ == "__main__":
    main()
