# Results

All numbers were measured on a single **GTX 1660 Ti (6 GB, TU116)** with
Python 3.11 and `torch 2.6.0+cu124`, unless stated otherwise. The corpus is
TinyStories (200 000 training stories, 10 000 validation stories). The metric
for cross-tokenizer comparison is bits per byte:

```
bpb = loss_nats_per_token / bytes_per_token / ln(2)
```

Raw artefacts live in `runs/<id>/` (`results.json`, `log.csv`, `best.pt`,
`analysis/`). They are not committed because of their size; every number below
is reproducible with the commands in §6 of the [README](../README.md).

## 1. Models

| model | tokenizer | params | matrix params | d / N | K / L | seq / batch | steps |
|---|---|---|---|---|---|---|---|
| E0 | byte | 969 856 | 933 888 | 128 / 12 | dense | 256 / 128 | 2 500 |
| E2 | byte | 190 108 | 154 140 | 128 / 12 | 2 / 4 | 256 / 128 | 2 500 |
| Q1 | BPE-1024 | 926 468 | 800 900 | 192 / 20 | 5 / 4 | 192 / 64 | 4 200 |
| Q2 | BPE-1024 | 2 079 992 | 1 890 360 | 320 / 24 | 6 / 4 | 192 / 32 | 10 000 |
| D1 | BPE-1024 | 1 068 544 | — (dense) | 128 / 12 | — | 192 / 64 | 2 500 |
| S1 | BPE-1024 | 2 079 992 | 1 890 360 | 320 / 24 | 6 / 4 | 192 / 32 | 400 |

## 2. Quality

| model | val loss | unit | val PPL | val bpb | note |
|---|---|---|---|---|---|
| E0 | 0.9789 | nats/byte | 2.661 | **1.412** | dense control, bytes |
| E2 | 1.1502 | nats/byte | 3.159 | **1.659** | Kron, bytes |
| Q1 | 2.0735 | nats/token | 7.953 | **0.976** | best checkpoint, step 4 000 |
| Q2 | <!-- Q2_LOSS --> | nats/token | <!-- Q2_PPL --> | **<!-- Q2_BPB -->** | 10 000 steps |
| D1 | <!-- D1_LOSS --> | nats/token | <!-- D1_PPL --> | **<!-- D1_BPB -->** | dense, same BPE |
| S1 | <!-- S1_LOSS --> | nats/token | <!-- S1_PPL --> | <!-- S1_BPB --> | fine-tuned on Dolly |

Validation curves for E2 and Q1 (`runs/<id>/log.csv`):

| step | E2 bpb | Q1 bpb | Q2 bpb |
|---|---|---|---|
| 500 | 1.768 | 1.525 | 1.482 |
| 1 000 | 1.659 | 1.266 | 1.245 |
| 1 500 | — | 1.149 | 1.132 |
| 2 000 | — | 1.083 | 1.068 |
| 2 500 | — | 1.044 | — |
| 3 000 | — | 1.016 | — |
| 3 500 | — | 0.994 | — |
| 4 000 | — | **0.976** | — |
| 6 000 | — | — | 0.902 |

The E2 run stops at 2 500 steps (0.45 epochs). Q1 was stopped at step 4 200 to
give the GPU to Q2; its best checkpoint is step 4 000. Q2 sees 61.4M tokens
(≈ 1.04 epochs of the BPE corpus) in 10 000 steps.

## 3. Cost

| model | ms/step | tokens/s | MFLOP/token | MFLOP/byte | peak VRAM |
|---|---|---|---|---|---|
| E0 | 361 | 91k | 3.51 | 3.51 | ~2.0 GB |
| E2 | 425 | 77k | 3.70 | 3.73 | 2.58 GB |
| Q1 | 540 | 22.8k | 11.2 | 3.66 | ~3.5 GB |
| Q2 | 810 | 7.6k | 36.2 | 11.9 | 4.22 GB |
| D1 | <!-- D1_MS --> | <!-- D1_TOKS --> | <!-- D1_FLOPS --> | <!-- D1_MPB --> | <!-- D1_VRAM --> |
| S1 | 810 | 7.6k | 36.2 | 11.9 | ~4.2 GB |

FLOPs are reported at `batch_size=64` (the convention used by
`flops_per_token`); tokens/s is the measured training throughput.

Generation speed (batch 1, `T=1.0`, `top_k=8`, no KV cache):

| model | tokens/s | bytes/s |
|---|---|---|
| E2 (byte) | 144 | 144 |
| Q1 (BPE) | 62 | 189 |
| Q2 (BPE) | <!-- Q2_GEN_TOKS --> | <!-- Q2_GEN_BYTES --> |

## 4. Diagnostics

Spectral diagnostics of E2 after 2 500 steps:

| metric | value | comment |
|---|---|---|
| `eff_rank(D)` | 91.2 / 128 | was 1.0 with the specification's `gain = 1/√d` |
| `eff_rank(P)` | 60.0 | the composition compresses rank |
| `eff_rank(P − s^L Id)` | 64.0 | the anchor dominates the spectrum |
| `cos(W_i, W_{i−1})` | 0.719 | layers differ (was 1.000) |
| `cos(P_i, P_{i−1})` | 0.083 | neighbouring compositions are nearly orthogonal |
| `τ` | 0.118 | grows with depth (0.105 → 0.17) |
| `γ` | 0.115 | grows with depth (0.10 → 0.185) |
| attention entropy | 1.65 | ln(256) = 5.55; the model uses context |
| `‖h_i‖/‖h_{i−1}‖` | 1.66 (max 6.6) | the residual does not explode |
| `Var_l σ(θ_l)` | 0.0039 | H3 is not confirmed (threshold 0.01) |
| Spearman(eff_rank D, depth) | 0.378 | H2 is not confirmed (threshold 0.7) |

Q1 and Q2 diagnostics: <!-- Q12_DIAG -->

## 5. Hypotheses

| hypothesis | criterion | E2 | verdict |
|---|---|---|---|
| H1: Kron is no worse than dense at fewer parameters | `PPL(E2) ≤ 1.2 · PPL(E0)` | 3.159 vs 2.661 (1.187) | confirmed |
| H2: expressiveness grows with depth | `Spearman > 0.7` | 0.378 | rejected |
| H3: channels split into fast and slow | `Var σ(θ) > 0.01` | 0.0039 | rejected |

H2 and H3 are rejected in the *formulation used here*, not in general: the
composition of matrices loses rank with depth, and `σ` multiplies an already
normalised state, so its gradient is ~250× smaller than the factors'. See
README §9 for the analysis and §14 for the proposed fix.

## 6. Quantisation

<!-- QUANT_TABLE -->

## 7. Supervised fine-tuning

<!-- SFT_SUMMARY -->
