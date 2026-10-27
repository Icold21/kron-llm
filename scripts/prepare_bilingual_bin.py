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

from kronllm.tokenizer import BPETokenizer, encode_docs, train_bpe_merges

ROOT = Path("data/bilingual")


def load_mixed(mix_dir: Path) -> list[dict]:
    """Load every mixed shard into memory, tagged with language and domain.

    Accepts both gzipped shards (the corpus build) and plain ``.jsonl`` (the
    curated decay mix, which is consumed once and does not pay for gzip).
    """
    docs: list[dict] = []
    shards = sorted(list(mix_dir.glob("*.jsonl.gz")) + list(mix_dir.glob("*.jsonl")))
    for path in shards:
        lang, domain = path.name.split(".jsonl")[0].split("_")[:2]
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
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
    """Sample text per language *and domain* so both sides shape the merges.

    Taking the first N MB of each language would sample whatever mix file
    happens to sort first (``en_code``), which biases a bilingual prose
    tokenizer toward Python.  Each domain gets an equal share instead.
    """
    out: list[str] = []
    for lang in ("en", "ru"):
        by_domain: dict[str, list[dict]] = {}
        for doc in docs:
            if doc["language"] == lang:
                by_domain.setdefault(doc["domain"], []).append(doc)
        if not by_domain:
            continue
        budget = per_language_bytes // len(by_domain)
        for domain in sorted(by_domain):
            used = 0
            for doc in by_domain[domain]:
                out.append(doc["text"])
                used += len(doc["text"])
                if used >= budget:
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
    ap.add_argument("--max-train-mb", type=float, default=0.0,
                    help="optional cap on the training text in MB; 0 means the "
                         "whole mix (the pilot used 400)")
    ap.add_argument("--reuse-tokenizer", default=None,
                    help="encode with an existing tokenizer.json instead of "
                         "training BPE on this mix; required for a decay pass, "
                         "which must not change the token id space")
    ap.add_argument("--reuse-val-dir", default=None,
                    help="copy val.bpe.bin and tokenizer_meta.json from this "
                         "directory instead of building a new split, so the "
                         "decay pass is measured on the epoch-1 held-out set")
    args = ap.parse_args()

    t0 = time.time()
    docs = load_mixed(Path(args.mix_dir))
    if not docs:
        raise SystemExit("no mixed shards found; run `build` first")
    train, val = split_docs(docs, args.val_fraction)
    print(f"[split] {len(train):,} train docs, {len(val):,} val docs")

    tok_path = Path(args.out_dir) / "tokenizer.json"
    if args.reuse_tokenizer:
        sample, sample_bytes = [], 0
    else:
        sample = stratified_sample(docs, int(args.merge_mb_per_language * 1e6))
        sample_bytes = sum(len(s.encode('utf-8')) for s in sample)
        print(f"[bpe] learning merges on {len(sample):,} docs "
              f"({sample_bytes/1e6:.1f} MB)")
    # Reserve five ids: the chat template (<|system|>, <|user|>, <|assistant|>,
    # <|eot|> for turn ends) and EOT for document boundaries.  A turn end and a
    # document end are different events: generation stops on <|eot|>, while the
    # pretraining stream separates documents with EOT.  Benefits and costs are
    # argued in docs/data_plan.md.
    chat_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|eot|>"]
    n_base = args.vocab - 1 - len(chat_tokens)
    if args.reuse_tokenizer:
        # Decay pass: the token ids must stay exactly the ones the checkpoint
        # was trained on, so the BPE is loaded, not retrained.  Copying the file
        # also keeps the output directory self-contained for Kaggle.
        src_tok = Path(args.reuse_tokenizer)
        tok_path.write_bytes(src_tok.read_bytes())
        print(f"[bpe] reused {src_tok} (no merges learned on this mix)")
    else:
        train_bpe_merges(sample, vocab_size=n_base, out_path=tok_path,
                         min_frequency=2, special_tokens=chat_tokens)
    tok = BPETokenizer(tok_path, vocab_size=args.vocab, bytes_per_token=1.0,
                       eot_id=args.vocab - 1)
    special_ids = {"endoftext": args.vocab - 1,
                   "system": n_base,
                   "user": n_base + 1,
                   "assistant": n_base + 2,
                   "eot": n_base + 3}
    print(f"[bpe] vocabulary {tok._tok.get_vocab_size()} ids "
          f"(system={special_ids['system']}, user={special_ids['user']}, "
          f"assistant={special_ids['assistant']}, eot={special_ids['eot']}, "
          f"endoftext={special_ids['endoftext']})")

    # encode_docs streams the text: no second copy of the corpus in memory.
    limit = int(args.max_train_mb * 1e6) if args.max_train_mb else 0
    tr = encode_docs(train, tok, Path(args.out_dir) / "train.bpe.bin",
                     args.vocab - 1, max_bytes=limit)
    if args.reuse_val_dir:
        # The decay pass must be scored on the epoch-1 held-out set, otherwise
        # its bpb cannot be compared with the epoch-end checkpoint.
        src = Path(args.reuse_val_dir)
        (Path(args.out_dir) / "val.bpe.bin").write_bytes(
            (src / "val.bpe.bin").read_bytes())
        meta_src = json.loads((src / "tokenizer_meta.json").read_text(encoding="utf-8"))
        meta = dict(meta_src)
        meta["train_tokens"] = tr["tokens"]
        meta["train_bytes"] = tr["bytes"]
        meta["anneal"] = True
        meta["anneal_source"] = str(src).replace("\\", "/")
        out_meta = Path(args.out_dir) / "tokenizer_meta.json"
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[bin] train {tr['tokens']:,} tokens ({tr['bytes_per_token']:.2f} B/t)")
        print(f"[bin] val   reused from {src} ({meta['val_tokens']:,} tokens)")
        print(f"[done] {out_meta} in {time.time()-t0:.1f}s")
        return
    va = encode_docs(val, tok, Path(args.out_dir) / "val.bpe.bin",
                     args.vocab - 1)

    per_lang = {}
    for lang in ("en", "ru"):
        text = "".join(d["text"] for d in val if d["language"] == lang)
        if not text:
            continue
        # Measure fertility on the whole validation text of the language, not
        # on a fixed character window: the same window means different byte
        # counts in Latin and Cyrillic and would distort the language shares.
        n_tok = sum(len(s) + 1 for s in tok.encode_batch([text]))
        n_bytes = len(text.encode("utf-8"))
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
        "endoftext_id": special_ids["endoftext"],
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
