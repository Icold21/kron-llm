"""Seen-set: the documents a step range actually consumed.

The point of these tests is that ``seen_from_index.consumed_docs`` and the
sampler agree.  A block contrived to pass under a wrong implementation is
useless, so the expectations are computed from the block spans directly, not
from the same permutation helper the tool uses.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from kronllm.tokenizer import BPETokenizer, doc_hash, encode_docs
from kronllm.seen import consumed_docs


@pytest.fixture()
def tok(tmp_path):
    from kronllm.tokenizer import train_bpe_merges

    path = tmp_path / "tok.json"
    texts = ["abcdefghij " * 40, "0123456789 " * 40]
    train_bpe_merges(texts, vocab_size=256, out_path=path, special_tokens=[])
    return BPETokenizer(path, vocab_size=256, bytes_per_token=4.0, eot_id=256)


@pytest.fixture()
def encoded(tmp_path, tok):
    docs = [{"text": f"document number {i} " * 40 + "tail"} for i in range(60)]
    out = tmp_path / "train.bpe.bin"
    idx = tmp_path / "train_index.npy"
    info = encode_docs(docs, tok, out, eot_id=tok.eot_id, index_path=idx)
    return {"path": out, "index": np.load(idx), "info": info, "docs": docs}


def test_index_spans_are_contiguous_and_hashes_distinct(encoded):
    index = encoded["index"]
    assert len(index) == encoded["info"]["documents"] == 60
    assert index["start"][0] == 0
    assert np.array_equal(index["start"][1:], index["start"][:-1] + index["n"][:-1])
    assert index["start"][-1] + index["n"][-1] == encoded["info"]["tokens"]
    assert len(set(index["h"].tolist())) == 60       # no two documents collapse


def test_doc_hash_matches_the_bytes_in_the_bin(encoded, tmp_path):
    index = encoded["index"]
    tokens = np.fromfile(encoded["path"], dtype=np.uint16)
    for row in index[:5]:
        span = tokens[row["start"]:row["start"] + row["n"]]
        assert doc_hash(span) == int(row["h"])


def test_full_epoch_touches_every_document(encoded):
    index = encoded["index"]
    n_tokens = encoded["info"]["tokens"]
    hashes = consumed_docs(index, n_tokens, seq_len=64, start_step=0,
                           end_step=10_000, batch=1)
    assert set(hashes.tolist()) == set(index["h"].tolist())


def test_partial_epoch_touches_exactly_the_drawn_blocks(encoded):
    index = encoded["index"]
    n_tokens = encoded["info"]["tokens"]
    seq_len, batch, steps = 64, 4, 3

    n_blocks = (n_tokens - 1) // seq_len
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_blocks, generator=g)
    drawn = perm[:steps * batch].numpy()

    # an independent expectation: a document is seen if a drawn block overlaps
    # its token span
    starts = index["start"]
    ends = index["start"] + index["n"]
    seen = set()
    for b in drawn:
        lo, hi = int(b) * seq_len, int(b) * seq_len + seq_len
        for i, (s, e) in enumerate(zip(starts, ends)):
            if s < hi and e > lo:
                seen.add(int(index["h"][i]))

    got = set(consumed_docs(index, n_tokens, seq_len=seq_len, start_step=0,
                            end_step=steps, batch=batch).tolist())
    assert got == seen
    assert len(got) < len(index)                  # a partial epoch reads less


def test_skip_lands_where_the_sampler_lands(encoded):
    index = encoded["index"]
    n_tokens = encoded["info"]["tokens"]
    seq_len, batch = 64, 2
    n_blocks = (n_tokens - 1) // seq_len
    steps_per_epoch = n_blocks // batch

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_blocks, generator=g)
    start_step = steps_per_epoch // 2
    drawn = perm[start_step * batch:(start_step + 1) * batch].numpy()

    starts = index["start"]
    ends = index["start"] + index["n"]
    seen = set()
    for b in drawn:
        lo, hi = int(b) * seq_len, int(b) * seq_len + seq_len
        for i, (s, e) in enumerate(zip(starts, ends)):
            if s < hi and e > lo:
                seen.add(int(index["h"][i]))

    got = set(consumed_docs(index, n_tokens, seq_len=seq_len,
                            start_step=start_step, end_step=start_step + 1,
                            batch=batch).tolist())
    assert got == seen


def test_min_share_narrows_the_set_and_a_full_epoch_still_reads_everything(
        encoded):
    index = encoded["index"]
    n_tokens = encoded["info"]["tokens"]

    any_touch = consumed_docs(index, n_tokens, seq_len=64, start_step=0,
                              end_step=40, batch=1)
    half = consumed_docs(index, n_tokens, seq_len=64, start_step=0,
                         end_step=40, batch=1, min_share=0.5)
    # "at least half read" can only ever be a subset of "touched": blocks are
    # drawn per token, so a long document is touched by many stray blocks
    assert set(half.tolist()) <= set(any_touch.tolist())

    full_any = consumed_docs(index, n_tokens, seq_len=64, start_step=0,
                             end_step=10_000, batch=1)
    full_half = consumed_docs(index, n_tokens, seq_len=64, start_step=0,
                              end_step=10_000, batch=1, min_share=0.5)
    assert set(full_half.tolist()) == set(index["h"].tolist())
    assert set(full_any.tolist()) == set(index["h"].tolist())


def test_encoding_skips_seen_documents_and_stops_at_the_target(tmp_path, tok):
    docs = [{"text": f"fresh document {i} " * 40} for i in range(40)]
    full = tmp_path / "full.bpe.bin"
    info = encode_docs(docs, tok, full, eot_id=tok.eot_id,
                       index_path=tmp_path / "full.npy")
    index = np.load(tmp_path / "full.npy")

    seen = set(int(h) for h in index["h"][:10].tolist())
    trimmed = tmp_path / "trimmed.bpe.bin"
    info2 = encode_docs(docs, tok, trimmed, eot_id=tok.eot_id,
                        skip_hashes=seen, index_path=tmp_path / "trimmed.npy")

    assert info2["skipped"] == 10
    assert info2["documents"] == info["documents"] - 10
    left = set(np.load(tmp_path / "trimmed.npy")["h"].tolist())
    assert not (left & seen)
    assert len(left) == 30
    # the bin on disk holds exactly the documents the index claims
    assert trimmed.stat().st_size == 2 * info2["tokens"]

    # and the target stops the encoder early instead of writing the whole mix
    short = tmp_path / "short.bpe.bin"
    half = encode_docs(docs, tok, short, eot_id=tok.eot_id,
                       target_tokens=info["tokens"] // 2,
                       index_path=tmp_path / "short.npy")
    assert half["tokens"] >= info["tokens"] // 2
    assert half["tokens"] < info["tokens"]
