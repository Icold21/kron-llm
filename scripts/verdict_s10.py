"""The S10 verdict: did fine-tuning buy arithmetic, or only the format?

Two things can happen after 4,000 SFT steps and they mean opposite things:

* **format learned and exact match above zero** -- the drills carry the
  procedure and the weights can hold it; the SFT recipe is right and more data
  / more steps will pay.
* **format learned and exact match still zero** -- the model can write the
  algorithm and not execute it, which is a capacity wall, not a data wall.  That
  is the signal to scale the model rather than polish the corpus.

The third case (format not learned) means the fine-tune itself failed and the
recipe needs looking at before anything else.

Usage::

    python scripts/verdict_s10.py --ckpt E:/kron_out/session11/s10/last.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kronllm.model import KronTransformer      # noqa: E402
from kronllm.tokenizer import BPETokenizer     # noqa: E402

DRILL_HEADS = ("Add the columns", "Subtract the columns", "Multiply one digit",
               "Вычитаем по разрядам", "Складываем по разрядам",
               "Умножаем по разрядам")


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/verdict_s10.json")
    args = ap.parse_args()

    eval_samples = _load("eval_samples_mod", ROOT / "scripts" / "eval_samples.py")
    drill_mod = _load("eval_arith_drill_mod",
                      ROOT / "scripts" / "eval_arith_drill.py")
    arith = _load("arith_verdict", ROOT / "scripts" / "gen_arithmetic_sft.py")
    build_sft = _load("build_sft_verdict", ROOT / "scripts" / "build_sft.py")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"]).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    step = int(state.get("step") or 0)
    tok = BPETokenizer(ROOT / "data" / "bilingual" / "tokenizer.json",
                       state["cfg"]["data"]["vocab_size"], bytes_per_token=3.69,
                       eot_id=state["cfg"]["data"]["eot_id"])
    print(f"[load] step {step:,}")

    bare = eval_samples.arithmetic(model, tok, args.device, n=args.n)
    # 90 tokens truncates a three-digit column drill before its answer line:
    # the rows are ~110-130 tokens, so the metric scored 0/60 on generation
    # that was arithmetically right up to the cut (2026-11-12).
    drill = drill_mod.run(model, tok, arith, build_sft, args.n // 2,
                          max_new_tokens=200)
    total = max(1, drill["total"])
    format_ok = sum(1 for e in drill["examples"]
                    if e["got"].lstrip().startswith(DRILL_HEADS)) / max(
                        1, len(drill["examples"]))

    # Chat is not scored -- there is no reference answer -- but it is shown, so a
    # model that repeats the question cannot pass for one that answers.
    chat = []
    for prompt in ("<|system|>You are KronLLM, a compact bilingual model.<|eot|>"
                   "<|user|>Who are you?<|eot|><|assistant|>",
                   "<|system|>You are KronLLM, a compact bilingual model.<|eot|>"
                   "<|user|>Сколько будет 2 + 2?<|eot|><|assistant|>"):
        ids = torch.tensor([tok.encode(prompt)], device=args.device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=40, temperature=0.0,
                                 top_k=1, top_p=1.0, repetition_penalty=1.0)
        chat.append(tok.decode(gen[0, ids.shape[1]:].tolist())[:200])

    verdict = {
        "step": step,
        "bare_exact": bare["exact_match"],
        "drill_answer_line": drill["answer_line"] / total,
        "drill_first_row": drill["first_row"] / total,
        "drill_format_head": format_ok,
        "drill_examples": drill["examples"],
        "chat": chat,
    }
    if format_ok < 0.6:
        verdict["reading"] = ("format not learned: the fine-tune itself failed -- "
                              "look at the corpus, the LR and the mask first")
    elif bare["exact_match"] > 0 or drill["answer_line"] > 0:
        verdict["reading"] = ("format and arithmetic: the recipe works, more SFT "
                              "data and steps will pay")
    else:
        verdict["reading"] = ("format learned, arithmetic still zero: a capacity "
                              "wall -- the model can write the procedure and not "
                              "execute it.  Scaling the model is the lever")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    Path(args.out).write_text(json.dumps(verdict, indent=2, ensure_ascii=False)
                              + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
