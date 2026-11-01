"""Pull a slice of the SFT corpus into pretraining as ordinary text.

Why: the SFT set is the only place where the model can see the shape
"<|system|>...<|user|>question<|eot|><|assistant|>answer", and 5 % of it inside
pretraining is cheaper than teaching the format twice.  Only samples whose
question still fits are taken -- ``prompt_tokens <= 128`` and
``total_tokens <= 480`` -- so the question is never truncated away by the
512-token window the trainer draws.

The corpus is stored as text, so the token slices are decoded back with the same
BPE that produced them (byte-level, so the round trip is exact).

Usage::

    python scripts/pull_sft_into_pretrain.py --per-part 50000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SFT = ROOT / "data" / "sft"
SHARDS = ROOT / "data" / "bilingual" / "shards"
MANIFEST = ROOT / "data" / "bilingual" / "manifest.jsonl"

# SFT part -> (language, corpus domain).  "code_ru" is Russian *questions* about
# code with the code itself unchanged; it belongs in the code domain, not in a
# separate "Russian code" bucket that would mean nothing.
PART_MAP = {
    "code_en": ("en", "code"),
    "code_ru": ("ru", "code"),
    "math_en": ("en", "math"),
    "math_ru": ("ru", "math"),
    "qa_en": ("en", "encyclopedic"),
    "qa_ru": ("ru", "encyclopedic"),
    "qa_sites_en": ("en", "encyclopedic"),
    "qa_sites_ru": ("ru", "encyclopedic"),
    "general_en": ("en", "encyclopedic"),
    "identity_en": ("en", "identity"),
    "identity_ru": ("ru", "identity"),
}

PROMPT_CAP = 128
TOTAL_CAP = 480


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-part", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()

    from kronllm.bilingual import record_shard
    from kronllm.tokenizer import BPETokenizer

    meta = json.loads((SFT / "meta.json").read_text(encoding="utf-8"))
    tok = BPETokenizer(SFT / "tokenizer.json", vocab_size=8192,
                       bytes_per_token=float(meta["bytes_per_token"]),
                       eot_id=int(meta["eot_id"]))
    idx = np.load(SFT / "sft_index.npy")
    data = np.memmap(SFT / "train.bpe.bin", dtype=np.uint16, mode="r")

    # Example -> part, by the part's token range.
    part_of = np.zeros(len(idx), dtype=np.int16)
    starts = []
    for i, part in enumerate(meta["parts"]):
        lo = int(part["start"])
        hi = lo + int(part["tokens"])
        part_of[(idx[:, 0] >= lo) & (idx[:, 0] < hi)] = i
        starts.append(lo)

    rng = np.random.default_rng(args.seed)
    suitable = (idx[:, 1] <= PROMPT_CAP) & (idx[:, 2] <= TOTAL_CAP)

    written = defaultdict(int)
    docs = defaultdict(int)
    sinks = {}
    try:
        for i, part in enumerate(meta["parts"]):
            name = part["name"]
            if name not in PART_MAP:
                continue
            lang, domain = PART_MAP[name]
            rows = np.flatnonzero((part_of == i) & suitable)
            if not rows.size:
                print(f"[skip] {name}: no sample fits the caps")
                continue
            take = min(args.per_part, rows.size)
            pick = rng.choice(rows, size=take, replace=False)
            pick.sort()
            key = (lang, domain)
            if key not in sinks:
                path = SHARDS / f"sft_{lang}_{domain}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                sinks[key] = (path, path.open("w", encoding="utf-8"))
            path, fh = sinks[key]
            for r in pick:
                s, ptok, ttok, _ = (int(v) for v in idx[r])
                text = tok.decode(data[s:s + ttok])
                if len(text) < 200:
                    continue
                fh.write(json.dumps({
                    "title": "", "text": text, "language": lang,
                    "domain": domain, "source": f"SFT {name} (sampled)",
                    "licence": "mixed sources, see data/sft/meta.json",
                    "url": ""}, ensure_ascii=False) + "\n")
                written[key] += len(text.encode("utf-8"))
                docs[key] += 1
            print(f"[take] {name}: {take:,} of {rows.size:,} candidates")
    finally:
        for path, fh in sinks.values():
            fh.close()

    for (lang, domain), (path, _) in sorted(sinks.items()):
        record_shard(path, MANIFEST, source=f"SFT samples -> {domain}",
                     licence="mixed sources, see data/sft/meta.json",
                     url="", language=lang, domain=domain, docs=docs[(lang, domain)],
                     note=f"prompt<={PROMPT_CAP}, total<={TOTAL_CAP}")
        print(f"[shard] {path.name:28s} {docs[(lang, domain)]:>7,} docs  "
              f"{written[(lang, domain)]/1e6:7.0f} MB")


if __name__ == "__main__":
    main()
