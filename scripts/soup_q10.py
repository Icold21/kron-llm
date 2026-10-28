"""Is checkpoint averaging worth anything *right now*?

The two Q10 checkpoints on disk are 115 steps apart (35,000 and 35,115), both
from the same constant-LR phase.  A soup of two nearly identical points in a
flat region of the loss surface can only differ from its parents by noise, and
this script measures that instead of asserting it:

* how far apart the weights actually are (relative L2 per tensor group),
* bits per byte of each parent and of the 50/50 soup on the same strided
  blocks of the val bin that ``scripts/eval_reference.py`` uses.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.model import KronTransformer  # noqa: E402
from kronllm.tokenizer import BPETokenizer  # noqa: E402

VAL = Path("data/bilingual/val.bpe.bin")
SEQ = 512


def block_starts(n_tokens: int, blocks: int) -> list[int]:
    n_blocks = (n_tokens - SEQ - 1) // SEQ
    if blocks >= n_blocks:
        return [i * SEQ for i in range(n_blocks)]
    step = (n_blocks - 1) / (blocks - 1)
    return [int(round(i * step)) * SEQ for i in range(blocks)]


def load_state(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob, blob["model"]


def rel_distance(a: dict, b: dict) -> dict:
    """Relative L2 distance per tensor, grouped into a few readable buckets."""
    groups: dict[str, list[float]] = {}
    for key in a:
        if not torch.is_tensor(a[key]) or a[key].dtype not in (
                torch.float32, torch.float16):
            continue
        if key not in b:
            continue
        da = a[key].float() - b[key].float()
        denom = a[key].float().norm().item() or 1.0
        bucket = key.split(".")[0]
        groups.setdefault(bucket, []).append(da.norm().item() / denom)
    return {k: sum(v) / len(v) for k, v in groups.items()}


@torch.no_grad()
def bpb(model, tok, starts: list[int], device: str) -> tuple[float, dict]:
    data = np.memmap(VAL, dtype=np.uint16, mode="r")
    model.eval()
    rows = {"all": [0.0, 0], "en": [0.0, 0], "ru": [0.0, 0]}
    for start in starts:
        x = torch.from_numpy(np.asarray(data[start:start + SEQ],
                                        dtype=np.int64))[None].to(device)
        y = torch.from_numpy(np.asarray(data[start + 1:start + SEQ + 1],
                                        dtype=np.int64))[None].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
            loss = float(model(x, y)["loss"])
        text = tok.decode(y[0].tolist())
        letters = sum(ch.isalpha() for ch in text) or 1
        cyr = sum("\u0400" <= ch <= "\u04ff" for ch in text)
        lang = "ru" if cyr / letters > 0.2 else "en"
        nats = loss * SEQ
        nbytes = len(text.encode("utf-8"))
        rows["all"][0] += nats
        rows["all"][1] += nbytes
        rows[lang][0] += nats
        rows[lang][1] += nbytes
    return (rows["all"][0] / math.log(2) / rows["all"][1],
            {k: {"bpb": v[0] / math.log(2) / v[1], "bytes": v[1]}
             for k, v in rows.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/kaggle_out/q10/best.pt")
    ap.add_argument("--b", default="data/kaggle_out/q10/last.pt")
    ap.add_argument("--blocks", type=int, default=1024)
    ap.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0",
                    help="0 = checkpoint a, 1 = checkpoint b")
    ap.add_argument("--out", default="data/soup_q10.json")
    ap.add_argument("--save-best", default="",
                    help="write the best alpha's weights as a checkpoint here")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob_a, state_a = load_state(Path(args.a))
    blob_b, state_b = load_state(Path(args.b))
    print(f"[a] {args.a} step {blob_a.get('step')} val_ppl {blob_a.get('val_ppl')}")
    print(f"[b] {args.b} step {blob_b.get('step')} val_ppl {blob_b.get('val_ppl')}")
    dist = rel_distance(state_a, state_b)
    for bucket, value in sorted(dist.items()):
        print(f"  rel-dist {bucket:<12s} {value:.5f}")

    cfg = blob_b["cfg"]
    tok = BPETokenizer("data/bilingual/tokenizer.json",
                       cfg["data"]["vocab_size"],
                       cfg.get("bytes_per_token", 3.694),
                       cfg["data"]["eot_id"])
    n_tokens = len(np.memmap(VAL, dtype=np.uint16, mode="r"))
    starts = block_starts(n_tokens, args.blocks)
    print(f"[val] {len(starts)} blocks", flush=True)

    out = {"a": args.a, "b": args.b,
           "step_a": blob_a.get("step"), "step_b": blob_b.get("step"),
           "rel_distance": dist, "blocks": len(starts), "runs": []}
    best_alpha, best_bpb = None, float("inf")
    for alpha in [float(x) for x in args.alphas.split(",")]:
        state = {k: ((1 - alpha) * state_a[k].float()
                     + alpha * state_b[k].float()).to(state_a[k].dtype)
                 for k in state_a}
        model = KronTransformer(cfg)
        model.load_state_dict(state)
        model.to(device).eval()
        value, detail = bpb(model, tok, starts, device)
        out["runs"].append({"alpha": alpha, "bpb": value, "detail": detail})
        print(f"[alpha {alpha:.2f}] bpb {value:.5f} "
              f"en {detail['en']['bpb']:.5f} ru {detail['ru']['bpb']:.5f}",
              flush=True)
        if value < best_bpb:
            best_alpha, best_bpb = alpha, value
            if args.save_best:
                # Saved in the same shape as a training checkpoint, so every
                # script that takes --ckpt can score it without special cases.
                torch.save({"model": state, "cfg": dict(cfg),
                            "step": blob_b.get("step"),
                            "val_ppl": blob_b.get("val_ppl"),
                            "soup": {"a": args.a, "b": args.b,
                                     "alpha": alpha, "bpb": value}},
                           args.save_best)
        del model, state
        torch.cuda.empty_cache()

    out["best"] = {"alpha": best_alpha, "bpb": best_bpb}
    print(f"[best] alpha {best_alpha:.2f} bpb {best_bpb:.5f}")
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
