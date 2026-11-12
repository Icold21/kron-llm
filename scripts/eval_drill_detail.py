"""Per-family breakdown of the drill score.

``eval_arith_drill.py`` answers one question -- how many column drills end on the
right number -- and that is the number the reports quote.  It cannot say *which*
procedure fails, and after S10 that is the interesting part: addition with
carries, subtraction with borrows and one-digit multiplication are three
different algorithms, and the corpus teaches them in different volumes.

So this asks the same questions in the same format (``build_sft.render``, so the
prompt is token-for-token what the corpus carries) and reports accuracy per
family per language, with a sample of the failures.  Nothing here replaces the
headline metric; it explains it.

Usage::

    python scripts/eval_drill_detail.py --ckpt E:/kron_out/s10_ckpt/last.pt --n 5
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


def answer_value(answer: object) -> int | None:
    if isinstance(answer, (int, float)):
        return int(answer)
    m = re.search(r"-?\d+", str(answer))
    return int(m.group()) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=5, help="problems per family")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default="data/drill_detail.json")
    args = ap.parse_args()

    from kronllm.model import KronTransformer
    from kronllm.tokenizer import BPETokenizer

    arith = _load(ROOT / "scripts" / "gen_arithmetic_sft.py", "arith_detail")
    build_sft = _load(ROOT / "scripts" / "build_sft.py", "build_sft_detail")
    drill = _load(ROOT / "scripts" / "eval_arith_drill.py", "drill_detail")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"]).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    device = next(model.parameters()).device
    tok = BPETokenizer(ROOT / "data" / "bilingual" / "tokenizer.json",
                       state["cfg"]["data"]["vocab_size"], bytes_per_token=3.69,
                       eot_id=state["cfg"]["data"]["eot_id"])
    print(f"[load] step {int(state.get('step') or 0):,}")

    rng = random.Random(11)
    result: dict = {"step": int(state.get("step") or 0), "families": {}}
    for lang in ("en", "ru"):
        word = ANSWER_WORD[lang]
        for fn in drill.column_families(arith, lang):
            hits = rows_ok = 0
            fails = []
            for _ in range(args.n):
                question, cot, answer = fn(rng)
                want = answer_value(answer)
                messages = [
                    {"role": "system", "content": build_sft.SYSTEM},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"{cot}\n\n{word} {answer}"},
                ]
                _, prompt = build_sft.render(messages)
                ids = torch.tensor([tok.encode(prompt)], device=device)
                with torch.no_grad():
                    gen = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                         temperature=0.0, top_k=1, top_p=1.0,
                                         repetition_penalty=1.0)
                text = tok.decode(gen[0, ids.shape[1]:].tolist())
                m = re.search(word + r"\s*(-?\d+)", text)
                ok = bool(m) and want is not None and int(m.group(1)) == want
                hits += int(ok)
                rows_ok += int(drill.first_row_ok(text))
                if not ok and len(fails) < 2:
                    fails.append({"question": question, "want": want,
                                  "got": text[:260]})
            key = f"{lang}/{fn.__name__.replace('_column', '')}"
            result["families"][key] = {"n": args.n, "ok": hits,
                                       "first_row": rows_ok, "fails": fails}
            print(f"  {key:32s} {hits}/{args.n}  first-row {rows_ok}/{args.n}")

    total = sum(v["n"] for v in result["families"].values())
    good = sum(v["ok"] for v in result["families"].values())
    result["total"] = {"n": total, "ok": good, "share": good / max(1, total)}
    print(f"[total] {good}/{total} = {good / max(1, total):.3f}")
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print("written", args.out)


if __name__ == "__main__":
    main()
