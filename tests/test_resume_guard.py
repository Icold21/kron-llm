"""Behavioural tests for the resume guards.

The first version of this file scanned `train.py` as text and passed or failed
on where a string happened to appear -- which is how a test lies.  These run the
trainer and look at what it wrote and whether it refused.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp: Path) -> Path:
    """A tiny CPU config with a 3-step budget."""
    from kronllm.config import deep_merge, DEFAULTS, save_config

    cfg = deep_merge(DEFAULTS, {
        "data": {"seq_len": 32, "vocab_size": 256},
        "model": {"d_model": 32, "n_layers": 2, "n_heads": 4,
                  "kron": {"n1": 4, "n2": 8}},
        "training": {"batch_size": 2, "grad_accum": 1, "amp": "none",
                     "device": "cpu", "warmup_steps": 1, "eval_every": 1000,
                     "save_every": 0, "max_minutes": 0},
    })
    path = tmp / "cfg.yaml"
    save_config(cfg, str(path))
    return path


def _train(cfg: Path, run: Path, extra: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "kronllm.cli", "train", "--config", str(cfg),
           "--run-dir", str(run), "--device", "cpu"] + extra
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=600)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("guard")
    cfg = _cfg(tmp)
    run = tmp / "run"
    proc = _train(cfg, run, ["--steps", "3"])
    assert proc.returncode == 0, proc.stderr[-2000:]
    return tmp, cfg, run


def test_final_checkpoint_keeps_origin_and_fingerprints(trained):
    """The published checkpoint is the final save; it must be complete.

    Session 3 published ``{'cfg', 'step', 'val_ppl'}`` alone -- no
    ``pass_origin``, no ``data_key`` -- so the corpus guard was inert on every
    session boundary and a decay pass recovered its origin from a fallback.
    """
    _, _, run = trained
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 3
    for key in ("pass_origin", "data_key", "tok_key"):
        assert key in ck, f"last.pt lost {key}"


def test_resume_refuses_a_changed_tokenizer(trained):
    """A different tokenizer means different strings for the same ids."""
    tmp, cfg, run = trained
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    ck["tok_key"] = "deadbeefdeadbeef"
    mutated = tmp / "mutated.pt"
    torch.save(ck, mutated)
    proc = _train(cfg, tmp / "run2", ["--resume", str(mutated), "--steps", "1"])
    assert proc.returncode != 0
    assert "tokenizer changed" in (proc.stderr + proc.stdout)


def test_corpus_change_needs_the_explicit_opt_in():
    """A rebuilt corpus must stop the run unless it was asked for."""
    from kronllm.config import experiment_config

    assert experiment_config("Q10")["training"]["allow_data_change"] is True
    assert experiment_config("E8")["training"].get("allow_data_change") is None
