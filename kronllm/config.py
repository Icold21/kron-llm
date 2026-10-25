"""Configuration handling for KronLLM.

Single source of truth for defaults is ``DEFAULTS`` below.  Experiments are
expressed as *overrides* (dotted keys) merged on top of the defaults, so that
every E-run is guaranteed to share the same seed / data / schedule unless it
explicitly says otherwise.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

DEFAULTS: Dict[str, Any] = {
    "project": "KronLLM-v2",
    "seed": 42,
    "data": {
        "dataset": "roneneldan/TinyStories",
        "tokenizer": "byte_level",
        "vocab_size": 256,
        "seq_len": 256,
        "val_split": 0.05,
        "data_dir": "data",
        "eot_id": 255,
        "num_proc": 8,
        "max_stories": 200000,
        "max_val_stories": 10000,
        "force": False,
    },
    "model": {
        "d_model": 128,
        "n_layers": 12,
        # Recursive weight sharing: use `n_unique_blocks` distinct blocks and
        # apply the stack `recurse` times.  Effective depth =
        # n_unique_blocks * recurse; parameters are counted once.
        "n_unique_blocks": None,
        "recurse": 1,
        "n_heads": 8,
        "n_kv_heads": None,          # None -> MHA
        "rope_theta": 10000.0,
        "norm": "rmsnorm",
        "norm_eps": 1e-6,
        "tie_embeddings": True,
        "ffn_silu": True,            # E9 is now the *linear* FFN ablation
        "ffn_gated": False,          # E11: SwiGLU-style gated FFN
        # Feature interaction in the FFN: "kron" is the linear composition,
        # "glu" is the bilinear (W_a x) * (W_b x) term (Q9/Q9k).
        "ffn_mix": "kron",
        "ffn_mix_rank": 16,
        "ffn_mix_kron": False,
        "use_kron_qkv": True,
        "use_kron_ffn": True,
        "dense_ffn_mult": 1.0,       # only used when use_kron_ffn == False
        # ALBERT-style factorised (tied) embedding: V x r then r x d.
        # 0 -> plain V x d table.  Keeps the embedding ~12% of the budget.
        "embed_rank": 0,
        # "auto" -> factorised when embed_rank > 0, dense otherwise.
        # "routed" -> top-k mixture of a dense (v_eff, d) bank (Q3).
        # "kron" -> the V x r table is a sum of Kronecker products
        #           (embed_kron); lookup and the tied head are unchanged.
        "embed_mode": "auto",
        "embed_kron": {
            # E = sum_k A_k (x) B_k,  V = v1*v2, r = r1*r2.
            # Cost: K*(v1*r1 + v2*r2) instead of V*r.
            "v1": 64, "v2": 64,
            "r1": 16, "r2": 8,
            "K": 32,
        },
        "routed": {
            "v_eff": 5000,
            "route_dim": 6,
            "top_k": 10,
            "temperature": 1.0,
        },
        # Intra-layer RMSNorm on q/k per head: removes logit drift and lets the
        # model use a higher lr at this scale.
        "qk_norm": False,
        # SDPA kernel: "auto" (best fused kernel), or "flash",
        # "mem_efficient", "math" to pin a backend for benchmarking.  Only the
        # attention kernel changes; the Kronecker maps are untouched.
        "attn_impl": "auto",
        # Fuse all composes of the model into a handful of batched GEMMs.
        # See kronllm/kron.py:KronBank for the launch-count rationale.
        "batched_prelude": True,
        "kron": {
            "L": 4,
            "K": 2,
            "K_win": 2,
            "n1": 8,
            "n2": 16,
            "stop_gradient": True,   # E8 flips this
            "composition": "product",  # E14 uses "sum": sI + sum_l alpha_l D_l
            "sum_alpha_init": 1.0,
            # E15/E16: input-dependent softmax weights over the L sum terms.
            # None (static alpha) | "dense" (d x L linear) | "kron" (Kron gate).
            "dynamic_gate": None,
            # E17: grouped composition -- sum inside `groups` blocks, product
            # across blocks.  G=1 is the sum, G=L is the product.
            "groups": 2,
            # E18: alternating composition -- one operation per micro-depth
            # step: sum, product, sum, product... starting from `sI`.
            "alternate_start": "sum",
            # E23/E24: elementwise nonlinearity between composition steps.
            "nonlinearity": "silu",
            "norm_between": False,
            "nonlinear_every": 1,
            "nonlinear_mode": "replace",
            "anchor": True,          # E4 turns the s*I anchor off
            # Per-role anchor, interpreted as the *effective* (post L-fold)
            # value: s = s_init ** (1/L).  See README "Deviations".
            "s_init": {"q": 1.0, "k": 1.0, "v": 1.0, "o": 1.0, "ffn": 1.0},
            "s_init_effective": True,
            "s_learnable": False,    # E3 turns this on
            # Orthogonal factor gain.  With normalize_D the gain is normalised
            # away in the forward pass, so it only matters for the ablation
            # table (E12).  "spec" is the literal TZ recipe 1/sqrt(d).
            "init_gain": "unit",
            # RMS-normalise D before it enters the anchor and the recurrence.
            # This is what makes the innovation *visible* to the gradient:
            # without it ||I|| ~ 1/d and the model is numerically the identity.
            "normalize_D": True,
            "innovation_scale": 0.1,  # tau init: RMS of every D_l
            "gate_init": 0.1,         # ReZero gamma init
            "gate": True,             # E12 keeps tau but drops the normaliser
            "theta_std": 0.1,
            # Depth-recurrence strength.  "learned" = sigmoid(theta) (default);
            # "fixed" pins it to sigma_value and freezes theta; "off" removes
            # the carry, leaving independent factors.  Ablations R1/R2.
            "sigma_mode": "learned",
            "sigma_value": 0.5,
            "rezero_eps": 0.0,
        },
    },
    "training": {
        "batch_size": 128,
        "seq_len": 256,
        "grad_accum": 1,
        "epochs": 3,
        "optimizer": "adamw",
        "betas": [0.9, 0.95],
        "lr": 3e-4,
        "lr_s": 3e-3,
        # theta is 4 scalars per module; its gradient is ~400x smaller than the
        # factor gradients because D is RMS-normalised (a scale-free mixing
        # coefficient). 10x the old value gives it a chance to move at all --
        # H3 ("fast/slow channels") is otherwise not even testable.
        "lr_theta": 1e-2,
        "warmup_steps": 500,
        "schedule": "cosine",
        "min_lr": 3e-5,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "amp": "none",   # fp16 autocast is slower than fp32 on the reference GPU
        "log_every": 100,
        "eval_every": 500,
        # Periodic checkpoint for crash / GPU-quota resilience.  ``last.pt``
        # is rewritten every ``save_every`` steps; ``best.pt`` follows the
        # validation curve.  `kronllm.cli train --resume runs/<run>/last.pt`
        # continues from the step counter stored in the file.
        "save_every": 1000,
        "eval_batches": 64,
        "early_stop_patience": 4,
        # Recursive weight sharing: sample r ~ U{1..recurse} per training step
        # so one model works at several depths (adaptive compute).
        "recurse_random": False,
        "num_workers": 0,
        "device": "cuda",
        "compile": False,
        "out_dir": "runs",
    },
    "metrics": {
        "primary": ["val_ppl", "val_loss"],
        "diagnostic": [
            "sigma_theta_mean",
            "sigma_theta_var",
            "D_norm",
            "D_rms",
            "tau_mean",
            "gate_mean",
            "h_norm_ratio",
            "sigma_min_W",
            "eff_rank_W",
            "eff_rank_D",
            "rank_excess",
            "attn_entropy",
            "grad_norm",
        ],
        "comparison": [
            "n_params",
            "n_params_matrix",
            "flops_per_token",
            "time_per_epoch",
            "peak_vram",
        ],
        "spectral_every": 2000,
    },
    "logging": {"tensorboard": True, "wandb": False, "wandb_project": "kronllm"},
}

# E-series from section 6.3 of the spec.  ``overrides`` are dotted paths.
# NOTE: SiLU on the FFN output is part of the *base* model now (it was E9 in
# the TZ), so E9 tests the opposite direction -- a purely linear FFN.  E11/E12
# are additions motivated by the diagnostics of the first training runs; see
# README "Deviations" for the full table.
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "E0": {
        "desc": "Dense baseline (MHA + dense FFN, ~0.97M params)",
        "overrides": {"model.use_kron_qkv": False, "model.use_kron_ffn": False,
                      # dense FFN hidden = 48 keeps the *control* under the
                      # project-wide 1M ceiling (it is 5x the Kron budget)
                      "model.dense_ffn_mult": 0.375},
    },
    "E1": {
        "desc": "Dense MHA + Kron FFN",
        "overrides": {"model.use_kron_qkv": False, "model.use_kron_ffn": True},
    },
    "E2": {
        "desc": "Full Kron (base)",
        "overrides": {},
    },
    "E3": {
        "desc": "E2 + learnable anchor s",
        "overrides": {"model.kron.s_learnable": True},
    },
    "E4": {
        "desc": "E2 without anchor, K_qkv=8 (pure innovation product)",
        "overrides": {
            "model.kron.anchor": False,
            "model.kron.K_qkv": 8,
            # without the identity carrier the product of L innovations has
            # norm ~ tau**L, so tau must be O(1) for the branch to be alive
            "model.kron.innovation_scale": 1.0,
        },
    },
    "E5": {
        "desc": "E2 + wider/deeper (d=192, N=16)",
        "overrides": {
            "model.d_model": 192,
            "model.n_layers": 16,
            "model.kron.n1": 12,
            "model.kron.n2": 16,
        },
    },
    "E6": {
        "desc": "E2 + L=6",
        "overrides": {"model.kron.L": 6},
    },
    "E7": {
        "desc": "E2 + K=4",
        "overrides": {"model.kron.K": 4},
    },
    "E8": {
        "desc": "E2 + full backprop (no stop-grad)",
        "overrides": {"model.kron.stop_gradient": False},
    },
    "E9": {
        "desc": "E2 with a linear FFN (SiLU ablation)",
        "overrides": {"model.ffn_silu": False},
    },
    "E10": {
        "desc": "E2 + GQA (n_kv=2)",
        "overrides": {"model.n_kv_heads": 2},
    },
    "E11": {
        "desc": "E2 + gated FFN: (W_a x) * silu(W_b x)",
        "overrides": {"model.ffn_gated": True},
    },
    "E12": {
        "desc": "E2 without D normalisation (raw innovation scale)",
        "overrides": {
            "model.kron.normalize_D": False,
            "model.kron.innovation_scale": 1.0,
        },
    },
    "E13": {
        "desc": "E2 + K=8 (budget-scaling point, ~651K params)",
        "overrides": {"model.kron.K": 8},
    },
    "E14": {
        "desc": "E2 + sum composition (sI + sum_l alpha_l D_l), full backprop",
        "overrides": {
            "model.kron.composition": "sum",
            "model.kron.stop_gradient": False,
        },
    },
    # E15/E16: the same sum, but the channel weights are generated from the
    # input by a softmax gate instead of being learned constants.  "dense" uses
    # a d x L linear layer; "kron" uses a factorised gate whose output is a
    # g1 x g2 Kronecker grid (g1*g2 == L), matching the project's style.
    "E15": {
        "desc": "E14 + input-dependent softmax gate (dense d x L)",
        "overrides": {
            "model.kron.composition": "sum",
            "model.kron.stop_gradient": False,
            "model.kron.dynamic_gate": "dense",
        },
    },
    "E16": {
        "desc": "E14 + input-dependent softmax gate (Kron g1 x g2)",
        "overrides": {
            "model.kron.composition": "sum",
            "model.kron.stop_gradient": False,
            "model.kron.dynamic_gate": "kron",
        },
    },
    # ------------------------------------------------------------------
    # Q-series: quality first.  Same Kron skeleton, but the budget is spent
    # where it pays off: a 3-bytes/token BPE vocabulary, more depth/width and
    # more Kronecker terms per matrix, with a factorised embedding so the
    # lookup table does not swallow the budget.
    # ------------------------------------------------------------------
    "Q1": {
        "desc": "BPE-1024, d=192/N=20, K=5, qk-norm (~926K params)",
        "overrides": {
            "data.tokenizer": "bpe",
            "data.vocab_size": 1024,
            "data.eot_id": 1023,
            "data.seq_len": 192,
            "model.d_model": 192,
            "model.n_layers": 20,
            "model.n_heads": 8,
            "model.embed_rank": 96,
            "model.qk_norm": True,
            "model.kron.n1": 12,
            "model.kron.n2": 16,
            "model.kron.K": 5,
            "training.seq_len": 192,
            # batch 96 is too large for the reference GPU at d=192/N=20;
            # 64 x 192 = 12.3k tokens per step.
            "training.batch_size": 64,
            "training.lr": 5e-4,
            "training.min_lr": 5e-5,
            "training.warmup_steps": 300,
            "training.eval_every": 500,
            "training.early_stop_patience": 99,
            # 10000 x 64 x 192 = 123M tokens ~ 2.1 epochs of the BPE corpus
            "training.max_steps": 10000,
        },
    },
    # ------------------------------------------------------------------
    # Controls: is the Kron parameterisation actually *worth* it at equal
    # parameters?  D1 is E0's dense recipe on the Q tokenizer, so the pair
    # D1 vs Q1 differs in exactly one thing: dense matrices vs Kron ones.
    # ------------------------------------------------------------------
    "D1": {
        "desc": "Dense+BPE control: E0 recipe, 1024-token BPE (~1.07M)",
        "overrides": {
            "data.tokenizer": "bpe",
            "data.vocab_size": 1024,
            "data.eot_id": 1023,
            "data.seq_len": 192,
            "model.use_kron_qkv": False,
            "model.use_kron_ffn": False,
            "model.dense_ffn_mult": 0.375,
            "model.qk_norm": True,
            "training.seq_len": 192,
            "training.batch_size": 64,
            "training.lr": 5e-4,
            "training.min_lr": 5e-5,
            "training.warmup_steps": 300,
            "training.eval_every": 500,
            "training.early_stop_patience": 99,
            # Exactly Q1's token budget: 4200 x 64 x 192 = 51.6M tokens.
            "training.max_steps": 4200,
        },
    },
    # Q1 was stopped at step ~4200 (val bpb 0.976) to spend the remaining time
    # on a 2.2x larger model; the design goal is quality, not the 1M budget.
    "Q2": {
        "desc": "BPE-1024, d=320/N=24, K=6, qk-norm (~2.08M params)",
        "overrides": {
            "data.tokenizer": "bpe",
            "data.vocab_size": 1024,
            "data.eot_id": 1023,
            "data.seq_len": 192,
            "model.d_model": 320,
            "model.n_layers": 24,
            "model.n_heads": 8,          # d_head = 40
            "model.embed_rank": 128,
            "model.qk_norm": True,
            "model.kron.n1": 16,
            "model.kron.n2": 20,         # n1 * n2 == d_model == 320
            "model.kron.K": 6,
            "training.seq_len": 192,
            # N*d*bs is held at Q1's level (24*320*32 == 20*192*64), so the
            # activation footprint stays the same as Q1's.
            "training.batch_size": 32,
            "training.lr": 4e-4,
            "training.min_lr": 4e-5,
            "training.warmup_steps": 300,
            "training.eval_every": 500,
            "training.early_stop_patience": 99,
            # 10000 x 32 x 192 = 61M tokens ~ 1 epoch of the BPE corpus
            "training.max_steps": 10000,
        },
    },
}

# Q3: dense routed embeddings on top of Q2.  Each token has a 6-d routing code;
# a linear router selects the top-10 rows of a 5000 x 320 bank and mixes them
# with softmax weights.  The effective V x d table is reused by the tied head,
# so the head stays parameter-free.  ~3.54M parameters.
EXPERIMENTS["Q3"] = {
    "desc": "Q2 + routed dense embeddings (v_eff=5000, top-k=10, ~3.54M params)",
    "overrides": {
        **EXPERIMENTS["Q2"]["overrides"],
        "model.embed_mode": "routed",
        "model.embed_rank": 0,
        "model.routed.v_eff": 5000,
        "model.routed.route_dim": 6,
        "model.routed.top_k": 10,
        "model.routed.temperature": 1.0,
        "training.lr": 3e-4,
        "training.min_lr": 3e-5,
    },
}

# Supervised fine-tuning on top of Q2.  The corpus is written by
# ``scripts/prepare_sft.py`` from Dolly-15k; only the model weights carry over,
# so the run gets a fresh optimizer and its own short cosine schedule.
EXPERIMENTS["S1"] = {
    "desc": "Q2 + supervised fine-tuning on Dolly-15k Q/A (~2.08M params)",
    "overrides": {
        **EXPERIMENTS["Q2"]["overrides"],
        "data.train_bin": "sft_train.bpe.bin",
        "data.val_bin": "sft_val.bpe.bin",
        "training.init_from": "runs/Q2/best.pt",
        "training.max_steps": 400,
        "training.lr": 1e-4,
        "training.min_lr": 1e-5,
        "training.warmup_steps": 40,
        "training.eval_every": 50,
        "training.early_stop_patience": 12,
    },
}

# Q4: test the E8 win at quality scale.  Q2 plus full backprop through the
# depth recurrence; same parameter count, tokenizer and activation footprint.
EXPERIMENTS["Q4"] = {
    "desc": "Q2 + full backprop (no stop-grad, ~2.08M params)",
    "overrides": {
        **EXPERIMENTS["Q2"]["overrides"],
        "model.kron.stop_gradient": False,
    },
}

# Q5: scale toward the 10M ceiling without increasing the activation
# footprint.  K=24 raises the matrix budget from 1.89M to 7.56M parameters
# while N*d*batch*seq stays at Q2's level, so it still targets a 6 GB card.
EXPERIMENTS["Q5"] = {
    "desc": "Q4 + K=24 (~7.75M params), same activation footprint",
    "overrides": {
        **EXPERIMENTS["Q4"]["overrides"],
        "model.kron.K": 24,
        # Same lr schedule as Q4: the first attempt used 3e-4/2k steps and is
        # not a clean K ablation.
        "training.max_steps": 4000,
        "training.eval_every": 500,
    },
}

# Q6: a wider tokenizer on top of Q4.  BPE-1024 gives ~3.07 bytes/token; a
# 4096-token vocabulary raises the compression rate and the generation
# bytes/s.  The embedding table is factorised (V x rank), so the extra
# vocabulary costs only ~393k parameters.  Artifacts are prepared with
# `python scripts/prepare_bpe.py --vocab 4096 --suffix 4096` and live next to
# the shipped BPE-1024 files.
EXPERIMENTS["Q6"] = {
    "desc": "Q4 + BPE-4096 tokenizer (~2.47M params)",
    "overrides": {
        **EXPERIMENTS["Q4"]["overrides"],
        "data.vocab_size": 4096,
        "data.eot_id": 4095,
        "data.train_bin": "train.bpe4096.bin",
        "data.val_bin": "val.bpe4096.bin",
        "data.meta_file": "tokenizer_meta4096.json",
        "model.loss_chunk": 1024,
        "training.max_steps": 4000,
    },
}

# Q7: Kronecker-structured embedding on top of Q6.  The V x r table becomes
# E = sum_k A_k (x) B_k with V = v1*v2 and r = r1*r2, so the embedding budget
# scales as K*(v1*r1 + v2*r2) instead of V*r: 49,152 parameters instead of
# 524,288 at the same rank ceiling (r1*r2 = r).  Lookup and the tied head are
# unchanged; only the table storage is compressed.
EXPERIMENTS["Q7"] = {
    "desc": "Q6 + kron embedding (49K vs 524K embed params)",
    "overrides": {
        **EXPERIMENTS["Q6"]["overrides"],
        "model.embed_mode": "kron",
        "model.embed_kron": {"v1": 64, "v2": 64, "r1": 16, "r2": 8, "K": 32},
    },
}

# Wikitext probes: the product-vs-sum question on a harder corpus than
# TinyStories, at the E2/E8 geometry (byte level, seq 256, batch 128).
EXPERIMENTS["E8w"] = {
    "desc": "E8 (full backprop) on WikiText-103 bytes",
    "overrides": {**EXPERIMENTS["E8"]["overrides"],
                  "data.data_dir": "data/wikitext"},
}
EXPERIMENTS["E14w"] = {
    "desc": "E14 (sum composition) on WikiText-103 bytes",
    "overrides": {**EXPERIMENTS["E14"]["overrides"],
                  "data.data_dir": "data/wikitext"},
}
EXPERIMENTS["E17w"] = {
    "desc": "E8 + grouped composition (G=2) on WikiText-103 bytes",
    "overrides": {**EXPERIMENTS["E8"]["overrides"],
                  "data.data_dir": "data/wikitext",
                  "model.kron.composition": "grouped",
                  "model.kron.groups": 2},
}
EXPERIMENTS["E18w"] = {
    "desc": "E8 + alternating sum/product composition on WikiText-103 bytes",
    "overrides": {**EXPERIMENTS["E8"]["overrides"],
                  "data.data_dir": "data/wikitext",
                  "model.kron.composition": "alternating",
                  "model.kron.alternate_start": "sum"},
}

# E19/E20: recursive weight sharing (Universal-Transformer style).  E19 keeps
# the effective depth at 12 with 6 unique blocks (half the parameters, same
# FLOPs); E20 keeps E2's parameter count and doubles the effective depth.
EXPERIMENTS["E19"] = {
    "desc": "E2 + weight sharing: 6 unique blocks x2 (12 effective, ~half params)",
    "overrides": {"model.n_unique_blocks": 6, "model.recurse": 2},
}
EXPERIMENTS["E20"] = {
    "desc": "E2 + weight sharing: 12 unique blocks x2 (24 effective, same params)",
    "overrides": {"model.n_unique_blocks": 12, "model.recurse": 2,
                  "training.batch_size": 64},
}
EXPERIMENTS["E21"] = {
    "desc": "E19 + randomized recursion r~U{1,2} + pass embedding",
    "overrides": {**EXPERIMENTS["E19"]["overrides"],
                  "training.recurse_random": True,
                  "model.pass_embedding": True},
}
EXPERIMENTS["E22"] = {
    "desc": "6 unique blocks x4 (24 effective, ~111K params)",
    "overrides": {"model.n_unique_blocks": 6, "model.recurse": 4,
                  "training.batch_size": 64},
}

# E23/E24: nonlinear composition at the E2 geometry.
# CLOSED (2026-10-24): every variant lost to the linear product
# (E25 1.519, E26 1.568, E26b 1.591 vs E8 1.479).  Kept for
# reproducibility only; do not schedule new runs.
EXPERIMENTS["E23"] = {
    "desc": "E2 + SiLU between composition steps",
    "overrides": {"model.kron.composition": "nonlinear"},
}
EXPERIMENTS["E24"] = {
    "desc": "E23 + RMSNorm between composition steps",
    "overrides": {"model.kron.composition": "nonlinear",
                  "model.kron.norm_between": True},
}
EXPERIMENTS["E25"] = {
    "desc": "nonlinear composition every 2 factors (2 fused GEMMs/token)",
    "overrides": {"model.kron.composition": "nonlinear",
                  "model.kron.nonlinear_every": 2},
}
EXPERIMENTS["E26"] = {
    "desc": "nonlinearity after the whole composition (1 GEMM, almost free)",
    "overrides": {"model.kron.composition": "nonlinear",
                  "model.kron.nonlinear_every": 4},
}
EXPERIMENTS["E26b"] = {
    "desc": "E26 but additive: keeps the linear product skeleton",
    "overrides": {"model.kron.composition": "nonlinear",
                  "model.kron.nonlinear_every": 4,
                  "model.kron.nonlinear_mode": "add"},
}
EXPERIMENTS["E25b"] = {
    "desc": "E25 (k=2) but additive: linear skeleton + 2 nonlinear branches",
    "overrides": {"model.kron.composition": "nonlinear",
                  "model.kron.nonlinear_every": 2,
                  "model.kron.nonlinear_mode": "add"},
}
# E27-E30: depth-recurrence ablation (2026-10-25).
# Motivated by an external review: with K_win=2 the state only crosses pairs of
# layers, sigma(theta) is one scalar per module, and no run isolated the carry
# itself.  All four share the E8 geometry and parameter count (theta stays in
# the state dict, so only 4 scalars per module are inert).
# E8r is the control: E8 re-run after the sigma_mode plumbing, so the four
# ablations are compared against a number produced by the same code path.
EXPERIMENTS["E8r"] = {
    "desc": "E8 re-run as the control for the E27-E30 sigma/K_win ablation",
    "overrides": {**EXPERIMENTS["E8"]["overrides"]},
}
EXPERIMENTS["E27"] = {
    "desc": "E8 without the depth carry: independent factors (sigma = 0)",
    "overrides": {**EXPERIMENTS["E8"]["overrides"],
                  "model.kron.sigma_mode": "off"},
}
EXPERIMENTS["E28"] = {
    "desc": "E8 with sigma frozen at 0.5 (the reviewer's ablation)",
    "overrides": {**EXPERIMENTS["E8"]["overrides"],
                  "model.kron.sigma_mode": "fixed",
                  "model.kron.sigma_value": 0.5},
}
EXPERIMENTS["E29"] = {
    "desc": "E8 with K_win=4: state crosses four layers instead of two",
    "overrides": {**EXPERIMENTS["E8"]["overrides"],
                  "model.kron.K_win": 4},
}
EXPERIMENTS["E30"] = {
    "desc": "E8 with K_win=0: state crosses the whole 12-layer stack",
    "overrides": {**EXPERIMENTS["E8"]["overrides"],
                  "model.kron.K_win": 0},
}
# Q8g: gated FFN at quality scale (Q6 + SwiGLU).
EXPERIMENTS["Q8g"] = {
    "desc": "Q6 + gated FFN (SwiGLU) (~2.85M params, batch 24)",
    "overrides": {**EXPERIMENTS["Q6"]["overrides"],
                  "model.ffn_gated": True,
                  "training.batch_size": 24},
}
# Q8w: parameter-matched control for Q8g.  20 layers x 6 matrices is the same
# block count and the same parameter count as Q6's 24 layers x 5 matrices, so
# the only difference from Q6 is the gated FFN, not extra capacity.
EXPERIMENTS["Q8w"] = {
    "desc": "Q8g with 20 layers: same params as Q6 (2.47M), gated FFN",
    "overrides": {**EXPERIMENTS["Q8g"]["overrides"],
                  "model.n_layers": 20},
}
# Q9/Q9k: multiplicative feature interaction.  The bilinear term is ~one Kron
# block in parameters (3*d*r = 15,360 vs 15,753), so both runs stay at ~2.47M.
EXPERIMENTS["Q9"] = {
    "desc": "Q6 with a bilinear (GLU) FFN instead of the linear Kron FFN (~2.46M)",
    "overrides": {**EXPERIMENTS["Q6"]["overrides"],
                  "model.ffn_mix": "glu",
                  "model.ffn_mix_rank": 16,
                  "training.batch_size": 24},
}
EXPERIMENTS["Q9k"] = {
    "desc": "Q9 + the Kron map as a non-interacting term (20 layers, ~2.47M)",
    "overrides": {**EXPERIMENTS["Q9"]["overrides"],
                  "model.ffn_mix_kron": True,
                  "model.n_layers": 20},
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def get_dotted(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def load_config(path: str | Path | None = None,
                overrides: Iterable[str] | Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Load YAML config on top of DEFAULTS, then apply overrides.

    ``overrides`` accepts either a mapping of dotted keys or an iterable of
    ``key=value`` strings (CLI style).
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})

    if overrides is None:
        return cfg

    if isinstance(overrides, Mapping):
        for key, value in overrides.items():
            set_dotted(cfg, key, value)
        return cfg

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, _, raw = item.partition("=")
        set_dotted(cfg, key.strip(), _parse_scalar(raw.strip()))
    return cfg


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def experiment_config(exp_id: str, base_path: str | Path | None = None,
                      extra: Iterable[str] | None = None) -> Dict[str, Any]:
    """Build the full config for one E-run."""
    if exp_id not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {exp_id!r}; known: {sorted(EXPERIMENTS)}")
    overrides: Dict[str, Any] = dict(EXPERIMENTS[exp_id]["overrides"])
    cfg = load_config(base_path, overrides)
    cfg["experiment"] = exp_id
    cfg["experiment_desc"] = EXPERIMENTS[exp_id]["desc"]
    if extra:
        for item in extra:
            key, _, raw = item.partition("=")
            set_dotted(cfg, key.strip(), _parse_scalar(raw.strip()))
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(json.loads(json.dumps(cfg, default=str)), fh,
                       sort_keys=False, allow_unicode=True)
