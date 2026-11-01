"""The mix directory: a rebuild must not eat the chat pools.

``build_anneal_chat.py`` writes ``en_chat.jsonl.gz`` / ``ru_chat.jsonl.gz`` into
the same directory as the domain shards, and ``prepare_bilingual_bin.py
--interleave chat`` reads them from there.  Twice a rebuild swept the whole
directory and the decay bin came out without a single chat token, which is only
visible as a token count 6 % short of the target.
"""
from __future__ import annotations

from pathlib import Path

from kronllm.bilingual import sweep_stale_mix


def _touch(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).write_bytes(b"x")


def test_sweep_removes_domain_shards_and_keeps_chat(tmp_path):
    _touch(tmp_path, "en_math.jsonl.gz", "ru_cs.jsonl.gz",
           "en_chat.jsonl.gz", "ru_chat.jsonl.gz")
    removed = sweep_stale_mix(tmp_path)
    left = sorted(p.name for p in tmp_path.iterdir())

    assert sorted(removed) == ["en_math.jsonl.gz", "ru_cs.jsonl.gz"]
    assert left == ["en_chat.jsonl.gz", "ru_chat.jsonl.gz"]


def test_sweep_is_idempotent_and_creates_the_directory(tmp_path):
    target = tmp_path / "mix"
    assert sweep_stale_mix(target) == []
    assert target.is_dir()

    _touch(target, "en_science.jsonl.gz")
    assert sweep_stale_mix(target) == ["en_science.jsonl.gz"]
    assert sweep_stale_mix(target) == []


def test_a_chat_name_is_matched_on_the_file_name_not_the_stem():
    """``Path.stem`` strips one suffix, so the guard has to test the name."""
    name = "en_chat.jsonl.gz"
    assert Path(name).stem == "en_chat.jsonl"           # the trap, documented
    assert "_chat." in name
