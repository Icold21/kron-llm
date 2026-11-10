"""A tokenizer meta written on Windows must still load on POSIX.

S10's first run died on this: ``tokenizer_meta.json`` shipped the build
machine's absolute path ``C:\\Users\\...\\data\\sft\\tokenizer.json``, which does
not exist in the container, so ``load_tokenizer`` falls back to "the file next
to the meta".  That fallback took ``Path(...).name``, and on POSIX a backslash is
not a separator -- the name came out as the whole string, the candidate could not
exist, and the crash surfaced as the tokenizer library's bare
``No such file or directory``.

The fix takes the last component under either separator, which is what these
tests pin down: a Windows path, a POSIX path, and the plain name all have to
resolve to the copy that sits beside the meta.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kronllm.tokenizer import BPETokenizer, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "tokenizer.json"

pytestmark = pytest.mark.skipif(not REAL.exists(),
                                reason="data/tokenizer.json is not in this checkout")


def _stage(tmp_path: Path, tokenizer_field: str) -> Path:
    """A data dir holding the tokenizer and a meta that points elsewhere."""
    shutil.copyfile(REAL, tmp_path / "tokenizer.json")
    meta = {
        "vocab_size": 8192,
        "eot_id": 8191,
        "bytes_per_token": 3.25,
        "train_tokens": 1,
        "val_tokens": 1,
        "tokenizer": tokenizer_field,
    }
    (tmp_path / "tokenizer_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def _load(data_dir: Path) -> BPETokenizer:
    cfg = {"data": {"tokenizer": "bpe", "data_dir": str(data_dir)}}
    tok = load_tokenizer(cfg)
    assert isinstance(tok, BPETokenizer)
    return tok


@pytest.mark.parametrize("field", [
    r"C:\Users\builder\kron_llm\data\sft\tokenizer.json",
    "/kaggle/input/datasets/icold21/kron-sft-corpus/tokenizer.json",
    "tokenizer.json",
])
def test_foreign_meta_path_falls_back_to_the_copy_beside_it(tmp_path, field):
    data_dir = _stage(tmp_path, field)
    tok = _load(data_dir)
    # the fallback has to land on the staged file, not on some other tokenizer
    assert Path(tok.path).resolve() == (data_dir / "tokenizer.json").resolve()


def test_a_real_path_wins_over_the_fallback(tmp_path):
    data_dir = _stage(tmp_path, "elsewhere.json")
    other = tmp_path / "other" / "tokenizer.json"
    other.parent.mkdir()
    shutil.copyfile(REAL, other)
    meta = json.loads((data_dir / "tokenizer_meta.json").read_text(encoding="utf-8"))
    meta["tokenizer"] = str(other)
    (data_dir / "tokenizer_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    tok = _load(data_dir)
    assert Path(tok.path).resolve() == other.resolve()
