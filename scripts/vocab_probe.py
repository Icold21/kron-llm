"""Would a 16k vocabulary be worth retraining the model for?

The question is not "bigger vocabulary = better", it is a trade with three
terms:

1. **Fertility** -- bytes of text per token.  A vocabulary that splits Russian
   into more pieces wastes context window and makes every Cyrillic token
   contribute less information per step.
2. **The 512-token filter** -- every SFT record longer than the context is
   dropped before it reaches the trainer.  If a bigger vocabulary lets more
   records fit, it buys data, not just speed.
3. **Parameters** -- embeddings are 22 % of this model and they are tied to the
   output path, so doubling the vocabulary doubles the embedding block and
   grows the model from 9.94M to 12.0M parameters.

This script measures (1) and (2) on the real corpus: it trains an 8k and a 16k
BPE on the *same* sample with the *same* recipe, then scores both against the
production tokenizer on held-out rows of every (domain, language) bucket.

    python scripts/vocab_probe.py --raw E:\\kron_sft\\raw --out data/vocab_probe.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_sft import render  # noqa: E402

SPECIALS = ["<|system|>", "<|user|>", "<|assistant|>", "<|eot|>", "<|endoftext|>"]

# Which shards stand in for each bucket when measuring fertility.  The weights
# are the byte weights of the shards inside their bucket in the final build.
BUCKETS: Dict[str, List[str]] = {
    "math_en": ["metamathqa", "openmathinstruct2", "orca_math", "mathinstruct"],
    "math_ru": ["mathinstruct_ru", "orca_math_ru"],
    "code_en": ["evol_codealpaca", "codefeedback", "magicoder_evol"],
    "code_ru": ["code_ru_translated"],
    "qa_en": ["openorca", "openhermes", "ultrachat"],
    "qa_ru": ["russian_instructions2", "saiga_scored"],
    "sites_en": ["reddit_qa", "stackexchange", "eli5"],
    "sites_ru": ["mailru_big", "mailru_old"],
}

# Share of the tokenizer-training sample each bucket gets.  Roughly the final
# corpus mix, so the vocabulary is not tuned for one domain.
TRAIN_WEIGHT = {
    "math_en": 0.26, "math_ru": 0.16,
    "code_en": 0.14, "code_ru": 0.04,
    "qa_en": 0.14, "qa_ru": 0.08,
    "sites_en": 0.12, "sites_ru": 0.06,
}


def rows(path: Path, skip: int, want: int) -> Iterator[dict]:
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        seen = 0
        got = 0
        for line in fh:
            if seen < skip:
                seen += 1
                continue
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            got += 1
            if got >= want:
                return


def collect(raw: Path, per_shard: int, skip: int) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for bucket, names in BUCKETS.items():
        texts: List[str] = []
        share = per_shard // max(1, len(names))
        for name in names:
            for rec in rows(raw / f"{name}.jsonl.gz", skip, share):
                text, _ = render(rec["messages"])
                texts.append(text)
        out[bucket] = texts
    return out


def train(texts: List[str], vocab: int, out: Path) -> None:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    # The production tokenizer isolates digits before ByteLevel, and that single
    # choice is worth ~15 % on mathematics: without it the probe 8k scores
    # 3.08 B/t on math_en where the production 8k scores 2.57, and the A/B
    # would then measure the missing digit split instead of the vocabulary size.
    # ``pattern`` must be a Regex object -- a plain string is matched literally.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(r"\d"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(out))


def score(tok, texts: List[str]) -> dict:
    total_bytes = 0
    total_tokens = 0
    total_chars = 0
    fits = 0
    fit_bytes = 0
    lengths: List[int] = []
    for text in texts:
        enc = tok.encode(text)
        n = len(enc.ids)
        b = len(text.encode("utf-8"))
        total_bytes += b
        total_chars += len(text)
        total_tokens += n
        lengths.append(n)
        if n <= 512:
            fits += 1
            fit_bytes += b
    lengths.sort()
    return {
        "records": len(texts),
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / max(1, total_tokens),
        "chars_per_token": total_chars / max(1, total_tokens),
        "tokens_per_record_median": lengths[len(lengths) // 2] if lengths else 0,
        "tokens_per_record_p90": lengths[int(len(lengths) * 0.9)] if lengths else 0,
        "fits_512": fits / max(1, len(texts)),
        "bytes_fit_512": fit_bytes / max(1, total_bytes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=r"E:\kron_sft\raw")
    ap.add_argument("--out", default="data/vocab_probe.json")
    ap.add_argument("--train-records", type=int, default=12000,
                    help="records per bucket used to train the probe tokenizers")
    ap.add_argument("--eval-records", type=int, default=1200,
                    help="held-out records per bucket used for scoring")
    ap.add_argument("--work", default=str(ROOT / "data" / "vocab_probe"))
    args = ap.parse_args()

    raw = Path(args.raw)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    print("[sample] training text", flush=True)
    train_sets = collect(raw, args.train_records, skip=0)
    texts: List[str] = []
    for bucket, items in train_sets.items():
        texts.extend(items)
    # Weight the sample by TRAIN_WEIGHT by repeating buckets: the sample is the
    # same for both vocabularies, so the weighting only decides what the merge
    # table is optimised for, not the comparison.
    weighted: List[str] = []
    base = len(texts)
    for bucket, items in train_sets.items():
        want = int(base * TRAIN_WEIGHT[bucket])
        got = sum(len(t.encode("utf-8")) for t in items) or 1
        repeat = max(1, round(want / got * 1.0))
        weighted.extend(items * min(repeat, 40))
    sample_bytes = sum(len(t.encode("utf-8")) for t in weighted)
    print(f"[sample] {len(weighted):,} records, {sample_bytes / 1e6:.1f} MB", flush=True)

    print("[eval] held-out text", flush=True)
    eval_sets = collect(raw, args.eval_records, skip=args.train_records)

    from tokenizers import Tokenizer
    prod = Tokenizer.from_file(str(ROOT / "data" / "bilingual" / "tokenizer.json"))

    result = {"sample_bytes": sample_bytes, "vocab": {}}
    for vocab in (8192, 16384):
        path = work / f"tok_{vocab}.json"
        print(f"[train] {vocab} on the same sample", flush=True)
        train(weighted, vocab, path)
        result["vocab"][str(vocab)] = Tokenizer.from_file(str(path))

    names = {"prod_8192": prod, "probe_8192": result["vocab"]["8192"],
             "probe_16384": result["vocab"]["16384"]}

    report: Dict[str, dict] = {}
    for bucket, items in eval_sets.items():
        if not items:
            continue
        report[bucket] = {name: score(tok, items) for name, tok in names.items()}
        row = report[bucket]
        print(f"  {bucket:10s} prod {row['prod_8192']['bytes_per_token']:5.3f} B/t  "
              f"8k {row['probe_8192']['bytes_per_token']:5.3f}  "
              f"16k {row['probe_16384']['bytes_per_token']:5.3f}  "
              f"fit512 {row['prod_8192']['fits_512']:.2f} -> "
              f"{row['probe_16384']['fits_512']:.2f}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[out] {args.out}")

    # Cyrillic coverage of the production vocabulary.  The stored keys are
    # byte-level (each Cyrillic character is two Latin-1 characters), so the
    # pieces have to be decoded before they can be classified.
    from tokenizers import Tokenizer as T
    prod_tok = T.from_file(str(ROOT / "data" / "bilingual" / "tokenizer.json"))
    vocab_map = json.loads(Path(ROOT / "data" / "bilingual" / "tokenizer.json")
                           .read_text(encoding="utf-8"))["model"]["vocab"]
    cyr = 0
    cyr_bytes = 0
    for piece in vocab_map:
        text = prod_tok.decode([piece])
        core = text.strip()
        if core and all(0x0400 <= ord(c) <= 0x04FF for c in core):
            cyr += 1
            cyr_bytes += len(text.encode("utf-8"))
    print(f"[vocab] {cyr} pure-Cyrillic pieces of {len(vocab_map)}, "
          f"avg {cyr_bytes / max(1, cyr):.1f} bytes each")


if __name__ == "__main__":
    main()
