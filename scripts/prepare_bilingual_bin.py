"""Tokenise the mixed bilingual corpus into uint16 memmaps for training.

The output matches what ``kronllm.data`` expects: a ``train``/``val`` pair of
uint16 files, an EOT id after every document, a tokenizer JSON and a meta file
with ``bytes_per_token``.  Unlike the TinyStories script this one is
bilingual, so the meta also records fertility and token share per language.

Usage::

    python scripts/prepare_bilingual_bin.py --vocab 8192
    python scripts/prepare_bilingual_bin.py --vocab 8192 --val-fraction 0.02
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.tokenizer import BPETokenizer, encode_corpus, train_bpe_merges

ROOT = Path("data/bilingual")


def load_mixed(mix_dir: Path) -> list[dict]:
    """Load every mixed shard into memory, tagged with language and domain."""
    docs: list[dict] = []
    for path in sorted(mix_dir.glob("*.jsonl.gz")):
        lang, domain = path.stem.split(".jsonl")[0].split("_")[:2]
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc.setdefault("language", lang)
                doc.setdefault("domain", domain)
                docs.append(doc)
    return docs


def stratified_sample(docs: list[dict], per_language_bytes: int) -> list[str]:
    """Sample text per language so both sides shape the merge table equally."""
    out: list[str] = []
    for lang in ("en", "ru"):
        budget = 0
        for doc in docs:
            if doc["language"] != lang:
                continue
            out.append(doc["text"])
            budget += len(doc["text"])
            if budget >= per_language_bytes:
                break
    return out


def split_docs(docs: list[dict], val_fraction: float) -> tuple[list, list]:
    """Deterministic train/val split, stratified by language and domain."""
    train: list[dict] = []
    val: list[dict] = []
    buckets: dict[tuple, list[dict]] = {}
    for doc in docs:
        buckets.setdefault((doc["language"], doc["domain"]), []).append(doc)
    for key in sorted(buckets):
        items = sorted(buckets[key], key=lambda d: d["sha256"])
        n_val = max(1, int(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-dir", default=str(ROOT / "mix"))
    ap.add_argument("--out-dir", default=str(ROOT))
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--merge-mb-per-language", type=float, default=10.0)
    ap.add_argument("--max-train-mb", type=float, default=400.0)
    args = ap.parse_args()

    t0 = time.time()
    docs = load_mixed(Path(args.mix_dir))
    if not docs:
        raise SystemExit("no mixed shards found; run `build` first")
    train, val = split_docs(docs, args.val_fraction)
    print(f"[split] {len(train):,} train docs, {len(val):,} val docs")

    tok_path = Path(args.out_dir) / "tokenizer.json"
    sample = stratified_sample(docs, int(args.merge_mb_per_language * 1e6))
    sample_bytes = sum(len(s.encode('utf-8')) for s in sample)
    print(f"[bpe] learning merges on {len(sample):,} docs "
          f"({sample_bytes/1e6:.1f} MB)")
    # Reserve four ids: <|system|>, <|user|>, <|assistant|> (chat template for
    # the SFT stage) and EOT.  Reserving them now means fine-tuning never has
    # to resize the embedding.
    chat_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
    n_base = args.vocab - 1 - len(chat_tokens)
    train_bpe_merges(sample, vocab_size=n_base, out_path=tok_path,
                     min_frequency=2, special_tokens=chat_tokens)
    tok = BPETokenizer(tok_path, vocab_size=args.vocab, bytes_per_token=1.0,
                       eot_id=args.vocab - 1)
    special_ids = {"eot": args.vocab - 1,
                   "system": n_base,
                   "user": n_base + 1,
                   "assistant": n_base + 2}
    print(f"[bpe] vocabulary {tok._tok.get_vocab_size()} ids "
          f"(system={special_ids['system']}, user={special_ids['user']}, "
          f"assistant={special_ids['assistant']}, eot={special_ids['eot']})")

    def as_bytes(items: list[dict]) -> list[bytes]:
        out = []
        budget = 0
        limit = int(args.max_train_mb * 1e6) if args.max_train_mb else None
        for d in items:
            b = d["text"].encode("utf-8")
            out.append(b)
            budget += len(b)
            if limit and budget >= limit:
                break
        return out

    train_stories = as_bytes(train)
    val_stories = as_bytes(val)
    tr = encode_corpus(train_stories, tok, Path(args.out_dir) / "train.bpe.bin",
                       args.vocab - 1)
    va = encode_corpus(val_stories, tok, Path(args.out_dir) / "val.bpe.bin",
                       args.vocab - 1)

    per_lang = {}
    for lang in ("en", "ru"):
        text = "".join(d["text"] for d in val if d["language"] == lang)
        if not text:
            continue
        n_tok = sum(len(s) + 1 for s in tok.encode_batch([text[:2_000_000]]))
        n_bytes = len(text[:2_000_000].encode("utf-8"))
        per_lang[lang] = {"bytes": n_bytes, "tokens": n_tok,
                          "bytes_per_token": round(n_bytes / max(1, n_tok), 3)}
    total_tok = sum(v["tokens"] for v in per_lang.values())
    token_share = {lang: round(v["tokens"] / max(1, total_tok), 4)
                   for lang, v in per_lang.items()}
    byte_share = {}
    total_bytes = sum(v["bytes"] for v in per_lang.values())
    for lang, v in per_lang.items():
        byte_share[lang] = round(v["bytes"] / max(1, total_bytes), 4)

    bpt = va["bytes"] / max(1, va["tokens"])
    meta = {
        "type": "bpe",
        "tokenizer": str(tok_path).replace("\\", "/"),
        "vocab_size": args.vocab,
        "eot_id": args.vocab - 1,
        "chat_tokens": {"system": special_ids["system"],
                        "user": special_ids["user"],
                        "assistant": special_ids["assistant"],
                        "eot": special_ids["eot"]},
        "bytes_per_token": bpt,
        "train_tokens": tr["tokens"],
        "val_tokens": va["tokens"],
        "train_bytes": tr["bytes"],
        "val_bytes": va["bytes"],
        "merge_docs": len(sample),
        "merge_bytes": sample_bytes,
        "language": per_lang,
        "token_share": token_share,
        "byte_share": byte_share,
        "val_fraction": args.val_fraction,
    }
    out_meta = Path(args.out_dir) / "tokenizer_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[bin] train {tr['tokens']:,} tokens ({tr['bytes_per_token']:.2f} B/t)")
    print(f"[bin] val   {va['tokens']:,} tokens ({va['bytes_per_token']:.2f} B/t)")
    print(f"[lang] {json.dumps(per_lang, ensure_ascii=False)}")
    print(f"[share] bytes {byte_share} -> tokens {token_share}")
    print(f"[done] {out_meta} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
