"""Exact-match arithmetic in the format the corpus actually carries.

``scripts/eval_samples.py`` asks for a naked sum -- ``47 + 38 = `` -- which is
the number the reports have quoted since step 35k.  It is a fair test of the
capability but not of the training: nothing in the 2.8B-token corpus asks for a
bare sum, and the drill pool teaches a column procedure instead.  So this
script asks in that format, built through ``build_sft.render``, which means the
prompt is token-for-token what pretraining saw (system turn, user turn,
``<|assistant|>``).

Two scores, because they fail differently:

* **Answer line** -- the final ``Answer: N`` / ``Ответ: N``.  Getting the format
  right and the number wrong is the expected failure at 10M parameters; SFT is
  where the digits are supposed to arrive.
* **first row** -- the value on the first computed line.  A model that cannot
  do ``9 + 7`` is not going to finish a three-digit column, so this says whether
  anything at all held.

Usage::

    python scripts/eval_arith_drill.py --ckpt E:/kron_out/session7/q10/last.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANSWER_WORD = {"en": "Answer:", "ru": "Ответ:"}


def _load(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def column_families(arith, lang: str):
    """The three families that teach a written procedure, not a fact."""
    families, _ = arith.FAMILIES[lang]
    return [fn for fn, _ in families if "column" in fn.__name__]


ROW_RE = re.compile(r"(\d+)\s*([+\-×*])\s*(\d+)\s*=\s*(-?\d+)")


def answer_value(answer) -> int:
    """The families format the answer for display; the metric needs the number.

    Comparing ``int(matched) == answer`` with ``answer`` still a string is always
    False, which is how this evaluator reported 0/60 arithmetic on every
    checkpoint it was ever run on.
    """
    return int(str(answer).replace(",", "").replace(" ", "").strip())


def first_row_ok(text: str) -> bool:
    """Is the first computed row arithmetically true?

    The previous check compared the first ``=`` value with the *final answer*.
    For a column drill those are different things by construction -- ``3 + 3 =
    6`` against ``1276`` -- so it reported 0/60 even when every row was right.
    This one checks the equation the model actually wrote.
    """
    for m in ROW_RE.finditer(text):
        a, op, b, c = (int(m.group(1)), m.group(2), int(m.group(3)),
                       int(m.group(4)))
        want = a + b if op == "+" else (a - b if op == "-" else a * b)
        return want == c
    return False


@torch.no_grad()
def run(model, tok, arith, build_sft, per_lang: int, seed: int = 11,
        max_new_tokens: int = 90) -> dict:
    rng = random.Random(seed)
    # The caller may put the model on the GPU; the prompt must follow it, or
    # torch.embedding raises 'found at least two devices'.
    device = next(model.parameters()).device
    out: dict = {"per_language": {}, "answer_line": 0, "first_row": 0,
                 "total": 0, "examples": []}
    for lang in ("en", "ru"):
        word = ANSWER_WORD[lang]
        fams = column_families(arith, lang)
        hits_line = hits_row = 0
        for i in range(per_lang):
            question, cot, answer = fams[i % len(fams)](rng)
            want = answer_value(answer)
            messages = [
                {"role": "system", "content": build_sft.SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant",
                 "content": f"{cot}\n\n{word} {answer}"},
            ]
            _, prompt = build_sft.render(messages)
            ids = torch.tensor([tok.encode(prompt)], device=device)
            gen = model.generate(ids, max_new_tokens=max_new_tokens,
                                 temperature=0.0,
                                 top_k=1, top_p=1.0, repetition_penalty=1.0)
            text = tok.decode(gen[0, ids.shape[1]:].tolist())
            m = re.search(word + r"\s*(-?\d+)", text)
            line_ok = bool(m) and int(m.group(1)) == want
            row_ok = first_row_ok(text)
            hits_line += int(line_ok)
            hits_row += int(row_ok)
            out["total"] += 1
            out["answer_line"] += int(line_ok)
            out["first_row"] += int(row_ok)
            if len(out["examples"]) < 3:
                out["examples"].append({"prompt": question, "want": want,
                                        "got": text[:300], "ok": line_ok})
        out["per_language"][lang] = {"n": per_lang, "answer_line": hits_line,
                                     "first_row": hits_row}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60, help="examples, split EN/RU")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=90,
                    help="a 3-digit column drill runs ~70 tokens; the answer "
                         "line is the last thing the model writes")
    ap.add_argument("--out", type=Path, default=Path("data/eval_arith_drill.json"))
    args = ap.parse_args()

    from kronllm.model import KronTransformer
    from kronllm.tokenizer import BPETokenizer

    arith = _load(ROOT / "scripts" / "gen_arithmetic_sft.py", "arith_drill")
    build_sft = _load(ROOT / "scripts" / "build_sft.py", "build_sft_drill")

    tok = BPETokenizer(ROOT / "data" / "bilingual" / "tokenizer.json",
                       vocab_size=8192, bytes_per_token=3.69, eot_id=8191)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"]).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    step = int(state.get("step") or 0)
    print(f"[load] step {step:,} on {args.device}")

    report = run(model, tok, arith, build_sft, args.n // 2,
                  max_new_tokens=args.max_new_tokens)
    report["step"] = step
    header = {"n": report["total"],
              "answer_line": report["answer_line"] / max(1, report["total"]),
              "first_row": report["first_row"] / max(1, report["total"]),
              "per_language": report["per_language"]}
    print("[drill]", json.dumps(header))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
