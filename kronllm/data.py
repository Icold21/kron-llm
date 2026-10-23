"""Byte-level data pipeline (TinyStories -> uint8 memmaps).

TinyStories is stored as UTF-8 bytes (no BPE): ``V = 256``.  Stories are
separated by ``0xFF`` which can never occur inside valid UTF-8, so it is a safe
end-of-text marker.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

EOT = 255


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
class ByteDataset(Dataset):
    """Random-access token blocks backed by a memmap.

    ``dtype=np.uint8`` is the byte-level corpus (V=256); ``np.uint16`` is the
    BPE corpus (V<=65535).  The class name is historical.
    """

    def __init__(self, path: str | Path, seq_len: int,
                 dtype: Any = np.uint8) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found -- run `python scripts/prepare_data.py` first")
        self.seq_len = int(seq_len)
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")
        if len(self.data) < self.seq_len + 1:
            raise ValueError(f"{self.path} is too small ({len(self.data)} tokens)")

    def __len__(self) -> int:
        return max(1, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = (i * self.seq_len) % (len(self.data) - self.seq_len - 1)
        chunk = np.asarray(self.data[start:start + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

    def get_batch(self, batch_size: int, device: torch.device | str = "cpu",
                  generator: Optional[torch.Generator] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(self.data)
        ix = torch.randint(0, n - self.seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(
            np.asarray(self.data[i:i + self.seq_len], dtype=np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            np.asarray(self.data[i + 1:i + 1 + self.seq_len], dtype=np.int64)) for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def iter_blocks(self, batch_size: int, device: torch.device | str = "cpu"
                    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic sequential pass (used for validation)."""
        n_blocks = len(self)
        for start in range(0, n_blocks, batch_size):
            idxs = range(start, min(start + batch_size, n_blocks))
            xs, ys = [], []
            for b in idxs:
                x, y = self[b]
                xs.append(x)
                ys.append(y)
            yield torch.stack(xs).to(device), torch.stack(ys).to(device)


def make_synthetic_bin(path: str | Path, n_bytes: int = 1 << 20,
                       seed: int = 0) -> Path:
    """Random bytes with a learnable first-order structure (smoke tests)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    alphabet = np.arange(32, 127, dtype=np.uint8)
    data = rng.choice(alphabet, size=n_bytes)
    # inject simple Markov structure so the model has something to learn
    for i in range(1, n_bytes):
        if rng.random() < 0.6:
            data[i] = min(255, data[i - 1] + 1)
    data.tofile(path)
    return path


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------
SEP = b"\xff"


def dataset_paths(cfg: Mapping[str, Any], data_dir: str | Path | None = None
                  ) -> dict:
    """Resolve corpus paths + dtype from the tokenizer choice.

    ``bpe``  -> ``data/{train,val}.bpe.bin`` as ``uint16``
    ``byte`` -> ``data/{train,val}.bin``     as ``uint8``
    """
    dc = cfg["data"]
    base = Path(data_dir or dc.get("data_dir", "data"))
    kind = str(dc.get("tokenizer", "byte")).lower()
    if kind in ("byte", "bytes", "byte_level"):
        return {"train": base / dc.get("train_bin", "train.bin"),
                "val": base / dc.get("val_bin", "val.bin"),
                "dtype": np.uint8, "tokenizer": "byte"}
    if kind == "bpe":
        return {"train": base / dc.get("train_bin", "train.bpe.bin"),
                "val": base / dc.get("val_bin", "val.bpe.bin"),
                "dtype": np.uint16, "tokenizer": "bpe"}
    raise ValueError(f"unknown tokenizer {kind!r}")


def open_dataset(cfg: Mapping[str, Any], split: str = "train",
                 data_dir: str | Path | None = None,
                 seq_len: int | None = None) -> "ByteDataset":
    paths = dataset_paths(cfg, data_dir)
    n = int(seq_len if seq_len is not None else cfg["data"]["seq_len"])
    return ByteDataset(paths[split], n, dtype=paths["dtype"])


def _encode_texts(texts, eot: int = EOT, budget: Optional[int] = None) -> bytes:
    sep = bytes([eot])
    out = bytearray()
    for text in texts:
        out += text.encode("utf-8", errors="ignore")
        out += sep
        if budget is not None and len(out) >= budget:
            break
    return bytes(out)


def prepare_tinystories(cfg: Mapping[str, Any]) -> dict:
    """Download/encode TinyStories into ``{data_dir}/train.bin`` and ``val.bin``."""
    from datasets import load_dataset

    dc = cfg["data"]
    data_dir = Path(dc.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    eot = int(dc.get("eot_id", EOT))
    max_stories = dc.get("max_stories", 200_000)
    max_val = dc.get("max_val_stories", 10_000)

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    meta: dict = {"dataset": dc["dataset"], "eot": eot}

    if train_path.exists() and val_path.exists() and not dc.get("force", False):
        meta["train_bytes"] = train_path.stat().st_size
        meta["val_bytes"] = val_path.stat().st_size
        meta["cached"] = True
        return meta

    ds = load_dataset(dc["dataset"])
    tr_key = "train" if "train" in ds else list(ds.keys())[0]
    va_key = "validation" if "validation" in ds else (
        "valid" if "valid" in ds else None)
    del ds

    n_tr = int(max_stories) if max_stories else None
    tr_split = f"{tr_key}[:{n_tr}]" if n_tr else tr_key
    print(f"[data] encoding train split {tr_split} ...")
    tr_ds = load_dataset(dc["dataset"], split=tr_split)
    tr_bytes = _encode_texts(tr_ds["text"], eot)
    train_path.write_bytes(tr_bytes)
    del tr_ds

    if va_key is not None:
        n_va = int(max_val) if max_val else None
        va_split = f"{va_key}[:{n_va}]" if n_va else va_key
        print(f"[data] encoding validation split {va_split} ...")
        va_ds = load_dataset(dc["dataset"], split=va_split)
        va_bytes = _encode_texts(va_ds["text"], eot)
        del va_ds
    else:
        n = len(tr_bytes)
        split_at = int(n * (1.0 - float(dc.get("val_split", 0.05))))
        va_bytes = tr_bytes[split_at:]
        tr_bytes = tr_bytes[:split_at]
        train_path.write_bytes(tr_bytes)
    val_path.write_bytes(va_bytes)

    meta["train_bytes"] = len(tr_bytes)
    meta["val_bytes"] = len(va_bytes)
    meta["train_stories"] = n_tr
    meta["cached"] = False
    return meta
