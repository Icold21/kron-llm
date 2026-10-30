"""The prompt probe must measure the prompt tokens and touch no data.

The training curve breaks when the corpus switches on mid-run: documents begin
with <|system|> and an instruction the model has never seen.  The probe turns
that break into a series (loss on the prompt tokens) so the report can show the
model learning the tag instead of asking the reader to ignore a bump.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from kronllm.data import ByteDataset
from kronllm.testutils import make_cfg


class Tiny(torch.nn.Module):
    """A finite logit table: enough for the probe, no semantics needed."""

    def __init__(self, vocab: int = 8192):
        super().__init__()
        self.vocab = vocab
        self.seen = None

    def forward(self, x):
        logits = torch.zeros(*x.shape, self.vocab)
        logits.scatter_(-1, x.unsqueeze(-1), torch.full((*x.shape, 1), 5.0))
        self.seen = x
        return {"logits": logits}


def _dataset(tmp_path: Path, with_prompts: bool) -> ByteDataset:
    rs = np.random.default_rng(0)
    n = 400_000
    ids = rs.integers(0, 200, size=n, dtype=np.uint16)
    if with_prompts:
        # A document boundary with the system marker every 512 tokens.
        for start in range(0, n - 600, 48):
            ids[start:start + 8] = 8187
    bin_path = tmp_path / ("p.bin" if with_prompts else "b.bin")
    ids.tofile(bin_path)
    return ByteDataset(bin_path, seq_len=64, dtype=np.uint16)


def test_probe_consumes_no_data(tmp_path):
    from kronllm.train import prompt_probe

    ds = _dataset(tmp_path, with_prompts=True)
    ds.set_shard(0, 1, seed=11, skip=0, epoch=0)
    before = ds._pos
    out = prompt_probe(Tiny(), ds, 8, torch.device("cpu"), None)
    assert ds._pos == before, "the probe advanced the sampler"
    assert out, "probe returned nothing"


def test_probe_separates_prompt_tokens(tmp_path):
    from kronllm.train import prompt_probe

    ds = _dataset(tmp_path, with_prompts=True)
    ds.set_shard(0, 1, seed=11, skip=0, epoch=0)
    out = prompt_probe(Tiny(), ds, 16, torch.device("cpu"), None)
    assert out["share_prompt"] > 0, out
    assert "nats_prompt" in out and "nats_bare" in out, out
    # 8187 is rewarded as much as any other token by Tiny, so the two means are
    # close; what matters is that both series exist and are finite.
    assert torch.isfinite(torch.tensor(out["nats_prompt"]))
    assert torch.isfinite(torch.tensor(out["nats_bare"]))


def test_probe_without_prompts_reports_no_prompt_series(tmp_path):
    from kronllm.train import prompt_probe

    ds = _dataset(tmp_path, with_prompts=False)
    ds.set_shard(0, 1, seed=11, skip=0, epoch=0)
    out = prompt_probe(Tiny(), ds, 16, torch.device("cpu"), None)
    assert out.get("share_prompt", 0.0) == 0.0
    assert "nats_prompt" not in out
    assert "nats_bare" in out
