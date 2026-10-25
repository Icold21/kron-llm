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
    return BPETokenizer(meta["tokenizer"], meta["vocab_size"],
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
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                 use_regex=True)
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

    Works in chunks and keeps the token ids as ``uint16`` numpy arrays, so peak
    memory stays around ``2 * n_tokens`` bytes (~120 MB for TinyStories-200K).
    """
    n_bytes = 0
    pieces: List[np.ndarray] = []
    for start in range(0, len(stories), chunk):
        batch = stories[start:start + chunk]
        n_bytes += sum(len(s) for s in batch)
        texts = [s.decode("utf-8", errors="replace") for s in batch]
        for seq in tok.encode_batch(texts):
            arr = np.empty(len(seq) + 1, dtype=np.uint16)
            arr[:-1] = np.asarray(seq, dtype=np.uint16)
            arr[-1] = eot_id
            pieces.append(arr)
    flat = np.concatenate(pieces) if pieces else np.zeros(0, np.uint16)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat.tofile(out_path)
    n_tokens = len(flat)
    return {"bytes": int(n_bytes), "tokens": int(n_tokens),
            "bytes_per_token": float(n_bytes) / max(1, n_tokens)}
