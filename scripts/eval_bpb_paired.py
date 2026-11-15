"""Paired bits per byte: is the difference between two checkpoints real?

``eval_bpb.py`` prints one number per checkpoint, and a difference of 0.02 looks
like a result.  It is not, until the block-to-block spread is known: the held-out
bin is 512 blocks of 512 tokens, and bpb on those blocks varies a lot, so the
standard error of the mean is often the same size as the effect.

Both checkpoints are scored on the *same* blocks, so the right comparison is
paired: per-block difference, its mean, and the standard error of that mean.  A
paired test removes the block difficulty, which is the dominant term.

Usage::

    python scripts/eval_bpb_paired.py --a S10=... --b S10b=... --bin data/sft/val.bpe.bin
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kronllm.model import KronTransformer       # noqa: E402
from kronllm.tokenizer import BPETokenizer      # noqa: E402


def load(path: str, device: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    tok = BPETokenizer(ROOT / "data" / "bilingual" / "tokenizer.json",
                       state["cfg"]["data"]["vocab_size"], bytes_per_token=3.69,
                       eot_id=state["cfg"]["data"]["eot_id"])
    return model, tok, int(state.get("step") or 0)


@torch.no_grad()
def per_block(model, tok, data, starts, seq: int, device) -> np.ndarray:
    table = tok.byte_lengths()
    out = []
    for start in starts:
        ids = torch.tensor(data[start:start + seq].astype(np.int64), device=device)[None]
        tgt = torch.tensor(data[start + 1:start + seq + 1].astype(np.int64), device=device)[None]
        logits = model(ids)["logits"][0, :-1].float()
        t = tgt[0, :logits.shape[0]]
        loss = float(torch.nn.functional.cross_entropy(logits, t))
        nbytes = int(table[t.cpu().numpy()].sum())
        out.append(loss / math.log(2) * int(t.shape[0]) / max(1, nbytes))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="name=path of the reference checkpoint")
    ap.add_argument("--b", required=True, help="name=path of the candidate")
    ap.add_argument("--bin", default=str(ROOT / "data" / "sft" / "val.bpe.bin"))
    ap.add_argument("--blocks", type=int, default=512)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    name_a, path_a = args.a.split("=", 1)
    name_b, path_b = args.b.split("=", 1)
    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    stride = (len(data) - args.seq - 1) // args.blocks
    starts = [i * stride for i in range(args.blocks)]

    result: dict = {"bin": args.bin, "blocks": args.blocks, "models": {}}
    values = {}
    for name, path in ((name_a, path_a), (name_b, path_b)):
        model, tok, step = load(path, args.device)
        per = per_block(model, tok, data, starts, args.seq, args.device)
        values[name] = per
        se = per.std(ddof=1) / math.sqrt(len(per))
        result["models"][name] = {"step": step, "bpb": round(float(per.mean()), 4),
                                  "se": round(float(se), 4)}
        print(f"  {name:10s} step {step:>7,}  bpb {per.mean():.4f} +- {1.96 * se:.4f} (95 %)")
        del model
        torch.cuda.empty_cache()

    diff = values[name_b] - values[name_a]
    se_d = diff.std(ddof=1) / math.sqrt(len(diff))
    # a t interval is barely different from 1.96 at 512 blocks, but use it anyway
    t_crit = 1.963
    result["difference"] = {
        "mean": round(float(diff.mean()), 5),
        "se": round(float(se_d), 5),
        "ci95": [round(float(diff.mean() - t_crit * se_d), 5),
                 round(float(diff.mean() + t_crit * se_d), 5)],
        "share_blocks_better": round(float((diff < 0).mean()), 3),
    }
    d = result["difference"]
    print(f"  {name_b} - {name_a}: {d['mean']:+.4f} bpb"
          f"  95 % CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
          f"  | better on {d['share_blocks_better']:.0%} of blocks")
    verdict = ("real" if d["ci95"][1] < 0 or d["ci95"][0] > 0 else "inside noise")
    result["verdict"] = verdict
    print(f"  -> {verdict}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("  written", args.out)


if __name__ == "__main__":
    main()
