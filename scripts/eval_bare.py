"""The bare form, measured properly: ``79 - 26 =`` -> one number, no procedure.

The reports quote this as 2/60 or 4/20, which is too small a sample to tell a
real change from noise: at n=20 one item is five points.  This runs the same
question in the same render the corpus teaches (``build_sft.render`` with an empty
chain of thought) at whatever n is asked for, and reports the score with a
Wilson interval so a difference can be judged instead of eyeballed.

Usage::

    python scripts/eval_bare.py --ckpt E:/kron_out/s10_ckpt/last.pt --n 100
    python scripts/eval_bare.py --ckpt E:/kron_out/s10b_ckpt/last.pt --n 100
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kronllm.model import KronTransformer       # noqa: E402
from kronllm.tokenizer import BPETokenizer      # noqa: E402


def _load(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100, help="problems per language")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arith = _load(ROOT / "scripts" / "gen_arithmetic_sft.py", "arith_bare")
    build_sft = _load(ROOT / "scripts" / "build_sft.py", "build_sft_bare")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"]).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    device = next(model.parameters()).device
    tok = BPETokenizer(ROOT / "data" / "bilingual" / "tokenizer.json",
                       state["cfg"]["data"]["vocab_size"], bytes_per_token=3.69,
                       eot_id=state["cfg"]["data"]["eot_id"])
    print(f"[load] step {int(state.get('step') or 0):,}")

    rng = random.Random(5)
    result = {"ckpt": str(args.ckpt), "step": int(state.get("step") or 0),
              "languages": {}}
    total_hits = total_n = 0
    for lang in ("en", "ru"):
        hits, examples = 0, []
        for _ in range(args.n):
            question, _cot, answer = arith.en_bare(rng)
            _, prompt = build_sft.render([
                {"role": "system", "content": build_sft.SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(answer)}])
            ids = torch.tensor([tok.encode(prompt)], device=device)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=6, temperature=0.0,
                                     top_k=1, top_p=1.0, repetition_penalty=1.0)
            got = tok.decode(gen[0, ids.shape[1]:].tolist()).strip()
            m = re.match(r"-?\d+", got)
            ok = bool(m) and int(m.group()) == int(answer)
            hits += int(ok)
            if len(examples) < 6:
                examples.append({"q": question, "want": answer,
                                 "got": got[:12], "ok": ok})
        lo, hi = wilson(hits, args.n)
        result["languages"][lang] = {"n": args.n, "ok": hits,
                                     "share": hits / args.n,
                                     "ci": [round(lo, 3), round(hi, 3)],
                                     "examples": examples}
        total_hits += hits
        total_n += args.n
        print(f"  {lang}  {hits}/{args.n} = {hits / args.n:.2f}"
              f"   95% CI [{lo:.2f}, {hi:.2f}]")
    result["total"] = {"n": total_n, "ok": total_hits,
                       "share": total_hits / max(1, total_n)}
    print(f"  all  {total_hits}/{total_n} = {total_hits / max(1, total_n):.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print("  written", args.out)


if __name__ == "__main__":
    main()
