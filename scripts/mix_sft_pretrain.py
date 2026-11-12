"""Mix pretraining text into the SFT corpus so fine-tuning stops eating language.

The measurement that forced this: S10 (78.6M SFT tokens) cost **0.139 bpb of
general text** on the pretraining held-out bin (1.0842 -> 1.2236) while buying
0.146 on the SFT bin.  A second pass without a mixer would make that drift worse,
and the article would have to explain a model that improved at its own domain by
getting measurably worse at everything else.

So a share of the training stream is ordinary pretraining documents, copied in
as **plain text**: no ``<|system|>``, no ``<|user|>``, therefore no prompt for
``prompt_target_mask`` to hide -- the whole document is a target.  That is the
opposite of how an SFT example is treated (its prompt is masked); it is the
whole point of the mix, because the loss on those documents is exactly the
pretraining loss, which is what keeps the language alive.

The val bin is left untouched, so SFT-domain bpb stays comparable with S10.

Usage::

    python scripts/mix_sft_pretrain.py --share 0.10
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kronllm.tokenizer import (ASSISTANT_ID, SYSTEM_ID, TURN_EOT_ID,  # noqa: E402
                               USER_ID)

SFT = ROOT / "data" / "sft"
MARKERS = (SYSTEM_ID, USER_ID, ASSISTANT_ID, TURN_EOT_ID)
PRETRAIN_CANDIDATES = [Path(r"E:/kron_bins/anneal_v7"),
                       Path(r"E:/kron_bins/epoch"),
                       ROOT / "data" / "bilingual" / "anneal_bin"]


def pick_pretrain(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for cand in PRETRAIN_CANDIDATES:
        if (cand / "train.bpe.bin").exists() and (cand / "train_index.npy").exists():
            return cand
    raise SystemExit("no pretraining bin with a document index found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(SFT))
    ap.add_argument("--pretrain", default=None)
    ap.add_argument("--share", type=float, default=0.10,
                    help="pretraining share of the mixed stream, by tokens")
    ap.add_argument("--out", default=str(ROOT / "data" / "sft_mix"))
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()

    sft_dir = Path(args.sft)
    pre_dir = pick_pretrain(args.pretrain)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sft_tokens = np.memmap(sft_dir / "train.bpe.bin", dtype=np.uint16, mode="r")
    sft_idx = np.load(sft_dir / "sft_index.npy")
    pre_tokens = np.memmap(pre_dir / "train.bpe.bin", dtype=np.uint16, mode="r")
    pre_idx = np.load(pre_dir / "train_index.npy")

    sft_total = int(sft_idx[:, 2].sum())
    want_pre = int(sft_total * args.share / (1.0 - args.share))
    print(f"[sft]     {len(sft_idx):,} examples, {sft_total:,} tokens")
    print(f"[pretrain] {pre_dir}  {len(pre_idx):,} documents")
    print(f"[target]  {want_pre:,} pretraining tokens "
          f"({args.share:.0%} of the mix)")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pre_idx))
    chosen: list[tuple[int, int]] = []
    taken = 0
    rejected = 0
    for i in order:
        start, n, _h = int(pre_idx["start"][i]), int(pre_idx["n"][i]), pre_idx["h"][i]
        if n < 64 or n > 4096:
            rejected += 1
            continue
        span = pre_tokens[start:start + n]
        # A document carrying chat markup would have half of itself masked as a
        # prompt in SFT mode; those belong to the SFT side, not to the mix.
        if np.isin(span, MARKERS).any():
            rejected += 1
            continue
        chosen.append((start, n))
        taken += n
        if taken >= want_pre:
            break
    print(f"[pretrain] {len(chosen):,} documents, {taken:,} tokens "
          f"(skipped {rejected:,} with markup or an odd length)")

    # Interleave: shuffle the whole plan so pretraining documents are spread
    # through the stream instead of sitting in one block the sampler would hit
    # for a few hundred consecutive steps.
    plan: list[tuple[str, int, int, int]] = []
    for row in sft_idx:
        start, prompt, total = int(row[0]), int(row[1]), int(row[2])
        plan.append(("sft", start, total, prompt))
    for start, n in chosen:
        plan.append(("pre", start, n, 0))
    order = rng.permutation(len(plan))

    print(f"[write]   {len(plan):,} segments -> {out / 'train.bpe.bin'}")
    dst = np.memmap(out / "train.bpe.bin", dtype=np.uint16, mode="w+",
                    shape=(sft_total + taken,))
    index = np.zeros((len(plan), 4), dtype=np.int64)
    pos = 0
    for k, j in enumerate(order):
        kind, start, n, prompt = plan[j]
        src = sft_tokens if kind == "sft" else pre_tokens
        dst[pos:pos + n] = src[start:start + n]
        index[k] = (pos, prompt, n, 0 if kind == "sft" else -1)
        pos += n
        if k % 400_000 == 0 and k:
            print(f"         {k:,}/{len(plan):,} segments, {pos:,} tokens")
    dst.flush()
    del dst
    np.save(out / "sft_index.npy", index)

    for name in ("val.bpe.bin", "tokenizer.json"):
        shutil.copyfile(sft_dir / name, out / name)
    meta = json.loads((sft_dir / "tokenizer_meta.json").read_text(encoding="utf-8"))
    meta["train_tokens"] = int(pos)
    meta["pretrain_mix"] = {"share": args.share, "tokens": taken,
                            "documents": len(chosen), "source": str(pre_dir)}
    meta["tokenizer"] = str(out / "tokenizer.json")
    (out / "tokenizer_meta.json").write_text(json.dumps(meta, indent=1),
                                             encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "sft_tokens": sft_total, "pretrain_tokens": taken,
        "train_tokens": int(pos), "segments": len(plan),
        "pretrain_documents": len(chosen), "share": args.share,
        "seed": args.seed}, indent=2), encoding="utf-8")
    print(f"[done]    {int(pos):,} tokens, {len(plan):,} segments")


if __name__ == "__main__":
    main()
