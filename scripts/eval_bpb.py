"""Held-out bits per byte for one checkpoint, on both corpora.

Two numbers, because fine-tuning moves them in opposite directions: the SFT bin
tells what the fine-tune bought, the pretraining bin tells what it cost in
general language.  Same blocks, same byte counting as ``bench_current.py``, so
the numbers are comparable with the battery and with the S10/S10b reports.

Usage::

    python scripts/eval_bpb.py --ckpt E:/kron_out/s10b_ckpt/last.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kronllm.model import KronTransformer       # noqa: E402
from kronllm.tokenizer import BPETokenizer      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--blocks", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location(
        "bench", ROOT / "scripts" / "bench_current.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"]).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    device = next(model.parameters()).device
    tok = BPETokenizer(ROOT / "data" / "bilingual" / "tokenizer.json",
                       state["cfg"]["data"]["vocab_size"], bytes_per_token=3.69,
                       eot_id=state["cfg"]["data"]["eot_id"])

    out = {"ckpt": str(args.ckpt), "step": int(state.get("step") or 0),
           "blocks": args.blocks}
    for name, path in (("pretrain_val", ROOT / "data" / "bilingual" / "val.bpe.bin"),
                       ("sft_val", ROOT / "data" / "sft" / "val.bpe.bin")):
        out[name] = round(bench.bpb(model, tok, path, args.blocks, device), 4)
        print(f"  {name:14s} {out[name]:.4f}")
    print(f"  step {out['step']:,}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
