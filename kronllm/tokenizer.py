"""Tokenizers: the byte baseline and a byte-level BPE.

Both expose the same tiny interface::

    tok.vocab_size        # int
    tok.bytes_per_token   # float, used for bits-per-byte reporting
    tok.eot_id            # id reserved for <|endoftext|>
    tok.encode(text)      # str -> list[int]
    tok.decode(ids)       # list[int] -> str

The byte tokenizer is the TZ baseline (V = 256, one token per byte).  The BPE
tokenizer lifts ~3 bytes into one position, which is the principled version of
"three bytes at a time": instead of a fixed 3-byte chunk (a 16.7M vocabulary)
the merges are learned, so frequent trigrams such as ``the``/``ing``/``and``
become single ids while rare byte triples stay split.

``bits_per_byte`` is the metric that makes the two comparable::

    bpb = loss_nats_per_token / bytes_per_token / ln(2)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np

EOT_BYTE = 255
EOT_TEXT = "<|endoftext|>"


# ---------------------------------------------------------------------------
# byte baseline
# ---------------------------------------------------------------------------
class ByteTokenizer:
    name = "byte"
    vocab_size = 256
    bytes_per_token = 1.0
    eot_id = 255

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: Sequence[int]) -> str:
        clean = [int(i) & 0xFF for i in ids if int(i) != self.eot_id]
        return bytes(clean).decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return "ByteTokenizer(vocab=256)"


# ---------------------------------------------------------------------------
# byte-level BPE
# ---------------------------------------------------------------------------
class BPETokenizer:
    name = "bpe"

    def __init__(self, path: str | Path, vocab_size: int,
                 bytes_per_token: float, eot_id: int) -> None:
        from tokenizers import Tokenizer

        self.path = Path(path)
        self._tok = Tokenizer.from_file(str(self.path))
        self.vocab_size = int(vocab_size)
        self.bytes_per_token = float(bytes_per_token)
        self.eot_id = int(eot_id)
        if self._tok.get_vocab_size() > self.vocab_size:
            raise ValueError(
                f"tokenizer has {self._tok.get_vocab_size()} ids but "
                f"vocab_size={self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [e.ids for e in self._tok.encode_batch(list(texts))]

    def decode(self, ids: Sequence[int]) -> str:
        clean = [int(i) for i in ids if int(i) != self.eot_id]
        if not clean:
            return ""
        return self._tok.decode(clean, skip_special_tokens=False)

    def __repr__(self) -> str:
        return (f"BPETokenizer(vocab={self.vocab_size}, "
                f"{self.bytes_per_token:.2f} bytes/token)")


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def meta_path(data_dir: str | Path, name: str | None = None) -> Path:
    return Path(data_dir) / (str(name) if name else "tokenizer_meta.json")


def load_tokenizer(cfg: Mapping[str, Any]):
    """Build the tokenizer described by ``cfg['data']``."""
    dc = cfg["data"]
    kind = str(dc.get("tokenizer", "byte")).lower()
    if kind in ("byte", "bytes", "byte_level"):
        return ByteTokenizer()
    if kind != "bpe":
        raise ValueError(f"unknown tokenizer {kind!r}")
    path = meta_path(dc.get("data_dir", "data"), dc.get("meta_file"))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python scripts/prepare_bpe.py` first")
    meta = json.loads(path.read_text(encoding="utf-8"))
    tok_path = Path(str(meta["tokenizer"]))
    if not tok_path.exists():
        # The meta may carry the build machine's path (data/bilingual/...).  On
        # Kaggle the bins and the tokenizer sit together in /kaggle/input/<ds>,
        # so fall back to the file next to the meta.
        candidate = Path(dc.get("data_dir", "data")) / tok_path.name
        if candidate.exists():
            tok_path = candidate
    return BPETokenizer(tok_path, meta["vocab_size"],
                        meta["bytes_per_token"], meta["eot_id"])


def bits_per_byte(loss_nats: float, bytes_per_token: float) -> float:
    """Convert a per-token cross-entropy into bits per *byte* of text.

    This is the only fair way to compare the byte model (1 token = 1 byte)
    with a BPE model (1 token = ~3 bytes).
    """
    import math

    return float(loss_nats) / max(float(bytes_per_token), 1e-9) / math.log(2.0)


# ---------------------------------------------------------------------------
# training / corpus encoding
# ---------------------------------------------------------------------------
def train_bpe_merges(texts, vocab_size: int, out_path: str | Path,
                     min_frequency: int = 2,
                     special_tokens: Sequence[str] = ()) -> None:
    """Train a byte-level BPE whose ids are ``0..vocab_size-1``.

    ``vocab_size`` already includes the 256 byte ids and the special tokens.
    The end-of-story id is *not* part of the tokenizer: callers reserve
    ``vocab_size`` itself and insert it manually (see ``encode_corpus``).

    ``special_tokens`` are added after the merges, in the given order, so their
    ids are deterministic: ``vocab_size, vocab_size+1, ...``.  The chat
    template (``<|user|>``, ``<|assistant|>``) lives here so fine-tuning and
    inference share one encoding.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    # GPT-2 byte-level regex first (letters, numbers, punctuation as separate
    # chunks), then split every digit into its own token.  Without the Digits
    # step BPE merges "2026" into one token, which makes digit-level
    # arithmetic impossible: addition needs to align units, tens, hundreds.
    # Measured cost on the corpus sample: 0.9 % fertility (3.115 -> 3.088
    # bytes/token).
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        pre_tokenizers.Digits(individual_digits=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    tok.train_from_iterator(list(texts), trainer=trainer)
    if special_tokens:
        tok.add_special_tokens(list(special_tokens))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))


def stories_from_bytes(path: str | Path, limit: int | None = None) -> List[bytes]:
    """Split a byte-level corpus (``0xFF`` separated) into story blobs."""
    raw = Path(path).read_bytes()
    parts = raw.split(bytes([EOT_BYTE]))
    if limit:
        parts = parts[:limit]
    return [p for p in parts if p.strip()]


def encode_corpus(stories: Sequence[bytes], tok: BPETokenizer,
                  out_path: str | Path, eot_id: int,
                  chunk: int = 2000) -> dict:
    """Encode stories into a ``uint16`` memmap; returns byte/token stats.

    Writes each chunk straight to disk instead of accumulating every token
    array in memory.  At 2.5B tokens the old version held the whole corpus plus
    a second concatenation buffer (~10 GB); this one keeps peak memory at one
    chunk, which matters when the mix is 8 GB of text.
    """
    n_bytes = 0
    n_tokens = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        for start in range(0, len(stories), chunk):
            batch = stories[start:start + chunk]
            n_bytes += sum(len(s) for s in batch)
            texts = [s.decode("utf-8", errors="replace") for s in batch]
            for seq in tok.encode_batch(texts):
                arr = np.empty(len(seq) + 1, dtype=np.uint16)
                arr[:-1] = np.asarray(seq, dtype=np.uint16)
                arr[-1] = eot_id
                arr.tofile(fh)
                n_tokens += len(arr)
    return {"bytes": int(n_bytes), "tokens": int(n_tokens),
            "bytes_per_token": float(n_bytes) / max(1, n_tokens)}


def encode_docs(docs: Sequence[dict], tok: BPETokenizer,
                out_path: str | Path, eot_id: int,
                chunk: int = 2000, max_bytes: int = 0) -> dict:
    """Encode document dicts without materialising the whole corpus as bytes.

    ``encode_corpus`` expects a list of byte strings; at 8 GB of text that list
    is another 8 GB of RAM.  This variant reads ``doc['text']`` chunk by chunk,
    so peak memory stays at the document store plus one chunk.
    """
    n_bytes = 0
    n_tokens = 0
    used = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        for start in range(0, len(docs), chunk):
            batch = docs[start:start + chunk]
            texts = []
            for doc in batch:
                text = doc["text"]
                data = text.encode("utf-8")
                if max_bytes and used + len(data) > max_bytes:
                    break
                texts.append(text)
                used += len(data)
                n_bytes += len(data)
            if not texts:
                break
            for seq in tok.encode_batch(texts):
                arr = np.empty(len(seq) + 1, dtype=np.uint16)
                arr[:-1] = np.asarray(seq, dtype=np.uint16)
                arr[-1] = eot_id
                arr.tofile(fh)
                n_tokens += len(arr)
            if max_bytes and used >= max_bytes:
                break
    return {"bytes": int(n_bytes), "tokens": int(n_tokens),
            "bytes_per_token": float(n_bytes) / max(1, n_tokens)}
