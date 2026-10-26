"""Compare the current byte-level BPE with a digit-isolating variant.

The current pre-tokenizer is GPT-2's byte-level regex: digits are grouped
(``\\p{N}+``) and BPE then merges them into 1-3 digit chunks.  Llama-style
tokenizers isolate every digit so arithmetic can align digits across tokens.

This trains both on the same sample and reports fertility plus example encodes.

Usage::

    python scripts/tokenizer_digit_probe.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.bilingual import normalize
from kronllm.tokenizer import BPETokenizer, train_bpe_merges

ROOT = Path("data/bilingual")
OUT = ROOT / "tokenizer_probe"
SAMPLES = (
    "1234 + 5678 = 6912",
    "The result is 3.14159, not 3.14.",
    "x^2 + 2x - 15 = 0",
    "2024-10-26T18:40:35+03:00",
)


def load_sample(mb: float = 12.0) -> list[str]:
    """Read a mixed-language sample from the shards, deterministically."""
    rng = random.Random(11)
    texts: list[str] = []
    total = 0
    target = int(mb * 1e6)
    paths = sorted((ROOT / "shards").glob("*.jsonl"))
    rng.shuffle(paths)
    for path in paths:
        if total >= target:
            break
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)
                if doc.get("domain") not in {"math", "science", "encyclopedic",
                                             "literature", "ml", "logic"}:
                    continue
                texts.append(normalize(doc["text"]))
                total += len(texts[-1])
                if total >= target:
                    break
    return texts


def digit_isolating_pre_tokenizer():
    from tokenizers import Regex, pre_tokenizers

    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\d"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])


def train_variant(texts: list[str], path: Path, isolate_digits: bool) -> None:
    if not isolate_digits:
        train_bpe_merges(texts, vocab_size=8187, out_path=path)
        return
    from tokenizers import Tokenizer, decoders, models, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = digit_isolating_pre_tokenizer()
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8187, min_frequency=2,
                                 initial_alphabet=pre_tokenizers_alphabet(),
                                 special_tokens=[], show_progress=False)
    tok.train_from_iterator(list(texts), trainer=trainer)
    tok.save(str(path))


def pre_tokenizers_alphabet():
    from tokenizers import pre_tokenizers
    return pre_tokenizers.ByteLevel.alphabet()


def fertility(tok: BPETokenizer, texts: list[str]) -> float:
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_tokens = sum(len(tok.encode(t)) for t in texts)
    return n_bytes / max(1, n_tokens)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    texts = load_sample()
    print(f"[sample] {len(texts)} docs, "
          f"{sum(len(t.encode()) for t in texts)/1e6:.1f} MB", flush=True)

    for name, isolate in (("v1_gpt2", False), ("v2_digits", True)):
        path = OUT / f"{name}.json"
        train_variant(texts, path, isolate)
        tok = BPETokenizer(path, 8192, 1.0, 8191)
        print(f"\n=== {name} ===")
        print(f"  fertility {fertility(tok, texts[-500:]):.3f} bytes/token")
        for s in SAMPLES:
            enc = tok.encode(s)
            pieces = []
            for i in enc[:20]:
                try:
                    pieces.append(tok.decode([int(i)]))
                except Exception:
                    pieces.append("?")
            print(f"  {s!r} -> {len(enc)} tokens {pieces}")


if __name__ == "__main__":
    main()
