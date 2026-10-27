# KronLLM — a compact, fully Kronecker-parameterised language model

KronLLM is a small decoder-only language model whose matrix layers are built
entirely from **Kronecker sums of rank-1 factors**. Every attention projection
and every FFN projection is a composition of `L` such factors instead of a
dense `d × d` weight, so a layer that would need 102,400 numbers needs 15,753.
The model is designed for a **6 GB consumer GPU**: the shipped configuration
trains and samples inside 4.1 GB of VRAM.

This repository is a research log as much as a model. It contains the
architecture, the training code, the measurements that killed several plausible
ideas, and the honest numbers behind each decision. Negative results are kept
because they are the reason the current design looks the way it does —
see [§8.4](#84-what-we-tried-and-rejected).

## At a glance

| | |
|---|---|
| Best model | **Q8g-long** — 2,851,280 parameters |
| Quality | **0.7542 bits/byte** (TinyStories val, 61.4M tokens seen) |
| Shipped artifact | [`artifacts/q8g_int8.pt`](artifacts/) — 3.29 MB, int8, +0.0003 bpb |
| Training cost | 3.3 h on a GTX 1660 Ti, 0.895 s/step, **4.09 GB** peak |
| Architecture | 24 layers, d=320, 8 heads, seq 192, BPE-4096, gated Kronecker FFN |
| Tests | 102 passed, 3 skipped (CPU only) |
| Hugging Face | [model](https://huggingface.co/Icold/kron-llm) · [corpus](https://huggingface.co/datasets/Icold/kron-corpus) |
| Corpus repo | [github.com/Icold21/kron-corpus](https://github.com/Icold21/kron-corpus) |

**Read the budget, not just the number.** TinyStories is a *development bench*,
never the target corpus. Every comparison below is made at equal tokens,
because step counts alone are misleading: Q6 trained with `batch 32` and Q8g
with `batch 24`, so 10,000 steps of one is not 10,000 steps of the other.

### The next run: 10M parameters on the bilingual corpus

The current checkpoint is a development bench. The real target is a **10M
model trained on a 1.6B-token bilingual corpus** and shipped as an int8 artifact
under 10 MB, so it fits a 6 GB GPU and runs on CPU. The reference
configuration is 9,938,886 parameters (d=512, 25 layers, d×d = 16×32
Kronecker factors, K=10, 8K bilingual BPE). Purpose, success criteria, corpus
contract and compute plan are in [`docs/goal_10m.md`](docs/goal_10m.md); the
corpus itself is published as
[**Icold21/kron-corpus**](https://github.com/Icold21/kron-corpus). Training is
scheduled for a Kaggle T4, with the 1660 Ti as the restartable fallback.

| model | parameters | tokens | bits/byte | note |
|---|---:|---:|---:|---|
| Q2 | 2,079,992 | 61.4M | 0.8391 | previous shipped model |
| Q6 | 2,473,208 | 61.4M | 0.7644 | BPE-4096 tokenizer |
| **Q8g-long** | **2,851,280** | **61.4M** | **0.7542** | **+ gated FFN** |

Same token budget, same corpus, same seed. The difference is the tokenizer and
the gated FFN, both of which are isolated in [§8.2](#82-tokenizer-and-budget-two-effects-that-are-easy-to-confuse)
and [§8.3](#83-gated-ffn-the-one-architectural-win).

## Contents

- [1. What "Kronecker" means here](#1-what-kronecker-means-here)
- [2. Depth recurrence](#2-depth-recurrence)
- [3. The scale defect and its fix](#3-the-scale-defect-and-its-fix)
- [4. Parameter budget](#4-parameter-budget)
- [5. Data](#5-data)
- [6. Running](#6-running)
- [7. Metrics](#7-metrics)
- [8. Results](#8-results)
- [9. Hypotheses: verdicts](#9-hypotheses-verdicts)
- [10. Tests](#10-tests)
- [11. Efficiency](#11-efficiency)
- [12. Repository layout](#12-repository-layout)
- [13. Limitations](#13-limitations)
- [14. The bilingual corpus](#14-the-bilingual-corpus)
- [15. License and citation](#15-license-and-citation)

---

## 1. What "Kronecker" means here

A dense projection `y = W x` stores `d_out × d_in` numbers. KronLLM instead
writes the projection as a **Kronecker sum** of `K` rank-1 factor pairs:

```
W = s·I  +  c · Σ_k  α_k · (a_k ⊗ b_k)
```

where `a_k ∈ R^{n1_out}` and `b_k ∈ R^{n2_out}` are the row/column factors and
`d_out = n1_out · n2_out`. The identity term is the **anchor**: it guarantees
that the layer starts near the identity map, and the factors learn a low-rank
correction on top of it. Per module the factors cost

```
L · K · (n1_out² + n2_out²)  +  2L  +  1
```

instead of `d_out · d_in`. For the shipped model (`L=4, K=6, n1=16, n2=20,
d=320`) that is **15,753 parameters per module** instead of 102,400 — 6.5×
smaller, and the saving grows with `d`.

A composition step is therefore one `d × d` matmul that *never materialises a
dense weight*: the factors are composed into a matrix on the fly (or, in
training, batched across all modules by `KronBank` and applied as a single
GEMM).

## 2. Depth recurrence

A module is not a single factor but a **recursion over depth**:

```
D_0 = D_init
D_l = s·D_{l-1} + c · (a_l ⊗ b_l) · γ_l        l = 1 … L
P   = D_L
y   = γ · P x
```

Two things matter here.

1. `D_l` carries a **depth state** across the `L` steps, so the composition is
   not a sum of independent terms but a repeated `s·D + innovation` update.
   The state also flows between transformer layers (`stop_gradient` controls
   whether the gradient crosses the boundary).
2. Because all `L` steps are linear, the whole stack collapses into a single
   matrix `P` that is computed once per layer and applied as one GEMM. This is
   the property that makes the architecture cheap — and it is also the property
   that later experiments tried (and failed) to break; see
   [§8.4.2](#842-nonlinearity-inside-the-composition).

**Measured, not assumed (E8r/E27–E30, 190K params, equal tokens).** The carry
is real but small, and it is short-range:

| variant | bpb | cost |
|---|---:|---:|
| learned σ, `K_win=2` | **1.5282** | — |
| no carry (σ=0) | 1.5455 | +1.1 % |
| σ frozen at 0.5 | 1.5409 | +0.8 % |
| `K_win=4` | 1.5496 | +1.4 % |
| `K_win=0` (whole stack) | 1.5712 | +2.8 % |

The state helps when it crosses a pair of layers and hurts when it travels
further, so this is not Universal-Transformers-style depth recurrence; making
it one is worse. Keep the cheapest form. Details in `docs/results.md` §15.

### Parameters of one module

| component | formula | count (`d=320`) |
|---|---|---:|
| factors `a_lk`, `b_lk` | `L·K·(n1² + n2²)` = `4·6·656` | 15,744 |
| `m_l` (memory) | `L` | 4 |
| `σ_l` (scale of `D_l`) | `L` | 4 |
| `γ` (ReZero) | 1 | 1 |
| **total** | | **15,753** |

Six modules per layer (`q, k, v, o, ffn, ffn_gate`) × 24 layers = 144 modules →
**2,268,432**; embedding 4,096 × 128 = 524,288 (factorised, 582,848 with the
up-projection); RMSNorm and the rest bring the total to **2,851,280**.

## 3. The scale defect and its fix

This section is the most important one in the repository: without these
changes the architecture from the specification **does not train at all**.
Every item is a measurement, not a matter of taste.

### 3.1 Root defect: `gain = 1/√d` makes the model the identity

With factors initialised at `gain = 1/√d = 0.088`, and the anchor dominant:

```
‖D_l‖_RMS ≈ 1e-3   →   W_l = s·I + D_l ≈ s·I
P = W_4 W_3 W_2 W_1 ≈ s^L · I = 0.1 · I
```

The whole Kronecker layer degenerates into multiplication by a constant.
Measured on a real 300-step TinyStories run:

| metric | value | meaning |
|---|---|---|
| `∂L/∂D` norm | **0.000** | the recurrence receives no gradient |
| `cos(W_i, W_{i−1})` | **1.000** | all layers are the same matrix |
| `eff_rank(W)` | **128** (all layers) | full algebraic rank comes from the anchor, not the factors; effective rank is far lower |
| val PPL | **20.06** | ≈ a unigram model |
| samples | "letter soup" | — |

### 3.2 The naive fix also breaks

With `gain = 1` (spectral norm of a factor ≈ 1) the hidden state grows across
layers as `11 → 226 → 1858 → 6830`, the attention contribution drops to
0.2–3 %, and `std(attn logits) = 0.0098`: the model again ignores context and
only rescales the residual.

### 3.3 What was actually done

| # | specification | implemented | reason (measurement) |
|---|---|---|---|
| 1 | `gain = 1/√d` | `D` is RMS-normalised; `gain` no longer acts in the forward pass | see 3.1: otherwise `P ≈ s^L·I` and the gradient is zero |
| 2 | one `s_init = 0.1` for QKV | per-role `s_init = 1.0`, applied as `s = s_init^{1/L}` | with `s = 0.1` and `D ≈ 1e-3` each layer halves the signal |
| 3 | — | learned `σ_l` scaling of each `D_l`, init 0.1 | sets `RMS(D)` directly; `eff_rank(D)` went 1.0 → **91.3/128** |
| 4 | — | ReZero gate `γ` on the branch output, init 0.1 | the branch starts as a perturbation; without it the residual grows 600× |
| 5 | `s` shared by QKV | its own `s` per role, optionally learnable | per-role scales differ by an order of magnitude |
| 6 | SiLU = experiment E9 | SiLU after the FFN is the default | the more useful baseline |
| 7 | `amp: fp16` | `amp: none` (fp32) | fp16 autocast was **2.4× slower** on the reference GPU |
| 8 | one composition per layer | `KronBank` fuses all 144 compositions | 439 vs 505 ms/step, identical output (test `T16`) |

Items 1–4 are one mechanism: **the scale must be normalised, not set by the
initialisation.** After the fix `eff_rank(D) = 91`, `cos(W_i, W_{i−1}) = 0.72`
instead of `1.000`, and the model trains.

## 4. Parameter budget

```bash
python -m kronllm.cli budget        # exits 1 if a configuration exceeds 10M
```

| model | d | layers | n1/n2 | K | L | modules | parameters |
|---|---:|---:|---|---:|---:|---:|---:|
| E-series | 128 | 12 | 8/16 | 4 | 4 | 60 | 190,108 |
| Q1 | 192 | 20 | 12/16 | 5 | 4 | 100 | 926,153 |
| Q2 / Q4 | 320 | 24 | 16/20 | 6 | 4 | 120 | 2,079,992 |
| Q6 | 320 | 24 | 16/20 | 6 | 4 | 120 | 2,473,208 |
| **Q8g** | 320 | 24 | 16/20 | 6 | 4 | 144 | **2,851,280** |

The soft ceiling is 10M parameters. Q8g uses 2.85M, i.e. 28 % of the ceiling:
the limit is not the parameter count but the 6 GB of VRAM, the 61M tokens per
run, and the throughput of a mobile GPU.

## 5. Data

All results in this README are on **TinyStories-200K**, byte-level or BPE-4096,
and this is deliberate: it is a *development bench* that separates architecture
effects from data effects, and it fits in a few minutes of GPU time.

TinyStories is **not** the target corpus. The final model is trained on a
bilingual English/Russian corpus (60/40 by bytes, science-heavy); the corpus is
published separately as
**[Icold21/kron-corpus](https://github.com/Icold21/kron-corpus)**, and its
measured composition is summarised in [§14](#14-the-bilingual-corpus).

Every conclusion below that mentions "quality" should be read as "quality on
the development bench".

## 6. Running

```bash
# environment (from the repo root)
pip install -r requirements.txt

# ...or install the package itself; it ships the `kronllm` CLI:
pip install "git+https://github.com/Icold21/kron-llm.git"        # needs git
pip install https://github.com/Icold21/kron-llm/archive/refs/heads/main.zip
pip install -e ".[dev]"                                          # editable, from a clone

# data: downloads TinyStories, writes a byte memmap
python scripts/prepare_data.py

# BPE-4096 tokenizer + re-tokenised corpus (~4 bytes/token)
python scripts/prepare_bpe.py --vocab 4096

# training (Q8g = the shipped configuration)
python -m kronllm.cli train --exp Q8g --steps 13333 --run-dir runs/Q8g_long

# evaluation, generation, analysis
python -m kronllm.cli eval --run runs/Q8g_long
python -m kronllm.cli generate --run runs/Q8g_long \
    --prompt "Once upon a time" --temperature 1.0 --top-k 8 --samples 2
python -m kronllm.cli analyze --run runs/Q8g_long
python -m kronllm.cli quantize --run runs/Q8g_long --bits 8
```

One trap worth naming: `pip install https://github.com/Icold21/kron-llm` does
**not** work — that URL is the repository's web page, so pip downloads HTML and
fails with `cannot detect archive format`. Use one of the three forms above;
nothing else in the recipe changes.

### Hugging Face interface

```python
from kronllm.hf import KronLM

model = KronLM.from_run("runs/Q8g_long")
print(model.generate("Once upon a time", max_new_tokens=100, temperature=1.0, top_k=8))
```

### Dashboard

`python scripts/dashboard.py` starts a local Streamlit page with the training
curves, the effective-rank spectra and side-by-side samples.

## 7. Metrics

| metric | why it is used |
|---|---|
| **bits/byte** | the only cross-tokenizer comparison; bpb = loss / ln 2 |
| PPL | convenient for humans, but tokenizer-dependent — never compare across vocabularies |
| `eff_rank(P)` | Roy–Vetterli effective rank of the composed matrix; exact rank saturates at `d`, so the effective rank is what distinguishes compositions |
| `cos_prev_max` | maximum cosine between consecutive layer matrices; → 1.0 means depth collapse (Q5) |
| `Var σ(θ)` | spread of the per-layer scale; a collapse indicator |

## 8. Results

### 8.1 The shipped model

**Q8g-long**: 24 layers, `d=320`, 8 heads, seq 192, BPE-4096, gated Kronecker
FFN. 61.4M tokens (13,333 steps × 24 × 192), 0.895 s/step, **4.09 GB** peak.

```
val loss 2.0693   PPL 7.919   bpb 0.7542   eff_rank(P) 144.8
```

The curve was still falling at the last step (0.7553 → 0.7542), so 61.4M tokens
is a budget choice, not a measured ceiling.

Samples (`T=1.0`, `top_k=8`):

> Once upon a time there was a little girl named Lily. She was three years old
> and loved to explore. One day she went to the beach and saw a big, wide waves.
> She was excited and wanted to go on a boat to explore the water.

> Once upon a time there was a little girl named Lily. She had a big box full of
> toys and she loved to play with her toys. One day, she found a big, red book
> under a tree. She picked it up and looked at it. It was so bright and beautiful!

Artifacts:

| file | size | quality |
|---|---:|---|
| `artifacts/q8g_int8.pt` | 3.29 MB | +0.0003 bpb vs fp32 (3.55× smaller) |
| `artifacts/q2_int8.pt` | 2.34 MB | previous model, kept for comparison |
| `runs/Q8g_long/best.pt` | 11.1 MB | fp32 reference checkpoint |

int8 quantisation is per-channel symmetric and leaves the Kronecker factors
intact; the composed matrix is rebuilt in fp32 at load time, which is why the
delta is 3e-4 rather than the usual 1e-2.

### 8.2 Tokenizer and budget: two effects that are easy to confuse

BPE-1024 compresses TinyStories to 3.07 bytes/token; BPE-4096 reaches
**3.96 bytes/token** on the same corpus. At a fixed parameter count that alone
is worth ~8 % bpb, because each token carries more text:

| model | tokenizer | parameters | steps | tokens | bpb |
|---|---|---:|---:|---:|---:|
| Q2 | BPE-1024 | 2,079,992 | 4,000 | 24.6M | 0.962 |
| Q4 | BPE-1024 | 2,079,992 | 4,000 | 24.6M | 0.9248 |
| Q6 | BPE-4096 | 2,473,208 | 4,000 | 24.6M | 0.8489 |
| Q6 | BPE-4096 | 2,473,208 | 10,000 | 61.4M | 0.7644 |
| **Q8g** | **BPE-4096** | **2,851,280** | **13,333** | **61.4M** | **0.7542** |

Q4 vs Q2 isolates the training change (full backprop instead of truncated):
**−3.9 %** at equal budget. Q6 vs Q4 isolates the tokenizer: **−8.2 %**, and
the only extra parameters are 393K in the factorised embedding.

To stay inside 6 GB with a 4,096-token vocabulary, the loss is computed in
checkpointed row chunks (`model.loss_chunk`), so the `24 × 192 × 4096` logits
tensor is never materialised. Peak VRAM is 4.09 GB.

### 8.3 Gated FFN: the one architectural win

`ffn_gated=True` adds a second Kronecker module and multiplies it by the first
through SiLU: `f = (W_a x) · SiLU(W_b x)`. Two controls isolate the effect:

| run | layout | parameters | tokens | bpb |
|---|---|---:|---:|---:|
| Q6 | 24 layers × 5 modules | 2,473,208 | 24.6M | 0.8489 |
| **Q8w** | **20 layers × 6 modules** | **2,470,328** | **24.6M** | **0.8359** (−1.5 %) |
| Q8g | 24 layers × 6 modules | 2,851,280 | 24.6M | 0.8318 (−2.0 %) |

Q8w is the honest control: same parameters (0.1 % apart), same number of matrix
modules per token (120), same tokens. The only difference from Q6 is how they
are arranged — 20 layers with a gate versus 24 layers without. The gate wins
**1.5 %**, and that is architecture, not capacity. The remaining 0.5 % of Q8g
is the extra 378K parameters.

So the design rule is: **a gated FFN beats four extra layers at equal budget.**
Everything else in the model is unchanged from Q2.

### 8.4 What we tried and rejected

Each entry lists the hypothesis, the measurement, and why the idea was dropped.
The weights are not kept — only the numbers, because the numbers are the result.

#### 8.4.1 Composition: sum, grouped, alternating

*Hypothesis:* a fully additive composition (`P = sI + Σ α_l D_l`) has a much
higher effective rank than the product, so it should be more expressive.

| composition | mult. depth | bpb (TinyStories) | eff_rank(P) |
|---|---:|---:|---:|
| product (E8) | 3 | **1.479** | 60.1 |
| sum (E14) | 0 | 1.5508 | 97.7 |
| dynamic gate (E15) | 0 | 1.5091 | 96.4 |
| Kron gate (E16) | 0 | 1.5504 | 97.4 |

The sum reaches rank 97.7 and still loses 4.9 % to the product. Rank was never
the bottleneck. Grouped and alternating compositions on WikiText-103 initially
looked 1.3 % better than the product (2.1561 / 2.1578 vs 2.1848), but **a second
seed reversed the sign**: product 2.1559 vs grouped 2.1624. The seed-to-seed
spread (~0.03 bpb) is larger than the effect, so the "win" was noise.

*Verdict:* product stays. The rank ordering is stable across seeds
(product 60.5 < grouped 79.2), so it remains a useful diagnostic — just not a
predictor of quality.

#### 8.4.2 Nonlinearity inside the composition

*Hypothesis:* the layer is linear by construction; putting an activation
between composition steps turns it into a small deep network inside a single
matrix, which should be a large gain per parameter.

| variant | nonlinearities per composition | bpb | cost |
|---|---:|---:|---:|
| product (E8) | 0 | **1.4789** | 3.67 MFLOP/token |
| E26, k=4 | 1 | 1.5675 | 3.67 |
| E25, k=2 | 2 | 1.5189 | 5.63 |
| E26b, additive k=4 | 1 | 1.5906 | 3.67 |

The nonlinearity cannot be fused away: `L` factors no longer collapse into one
matrix, so the layer costs `ceil(L/k)` GEMMs instead of one. Even at `k = L`,
where the FLOP count is unchanged, wall-clock time rose from 0.405 to 0.655
s/step (+62 %) — the loss of fusion, not the activation.

The interesting part is the ordering: *more* nonlinearity helped (k=2 beat k=4)
but every variant still lost to the purely linear product, and the additive
formulation — which keeps the linear skeleton — was the worst of all. The extra
branch doubles the output scale, and the model spends capacity compensating
instead of learning. `eff_rank(P)` rose to 62.3 while bpb got worse, the third
independent sign that rank is not the constraint.

*Verdict:* closed. The layer is linear, and at this scale that is a feature.

#### 8.4.3 Bilinear feature interaction (GLU)

*Hypothesis:* cross-feature products `(W_a x) ⊙ (W_b x)` give interactions that
a linear composition cannot express, at ~one module's parameter cost
(`3·d·r = 15,360` at `r=16` versus 15,753).

| run | FFN | layers | parameters | bpb |
|---|---|---:|---:|---:|
| Q6 | linear Kron | 24 | 2,473,208 | **0.8489** |
| Q9 | GLU only | 24 | 2,463,776 | 0.8740 |
| Q9k | GLU + Kron | 20 | 2,462,468 | 0.8673 |

Both lost (2.9 % and 2.2 %), and `cos_prev_max` rose to **0.939** in Q9 —
the layers became more similar, not more expressive. Adding the linear
Kronecker term back helped a little, but not enough.

A note on methodology: the first Q9 run was invalid because the generic
`std=0.02` initialisation left the bilinear branch 300× smaller than a Kron
module (output std 0.005 vs 1.76), i.e. switched off. The branch is now
initialised to unit activation and a regression test guards it. The numbers
above are from the corrected run — a false negative would have been worse than
no experiment.

*Verdict:* closed at this scale.

#### 8.4.4 Rank: `K = 24`

*Hypothesis:* more factors per module (`K=24` instead of 6) buy expressiveness
for a few extra parameters.

Q5 (`K=24`, 7.75M parameters) scored **0.9530** against Q4's **0.9248** at the
same budget, and `cos_prev_max` reached **0.99999**: every layer learned the
same matrix. The extra factors made the innovation dominate the anchor until
depth differences vanished.

*Verdict:* closed. `K=6` is the useful range at this width.

#### 8.4.5 Kronecker embedding

*Hypothesis:* the embedding is 524K of the 2.47M parameters; a Kronecker
embedding would shrink it.

Q7 scored **0.9533** against Q6's **0.8489** — 12 % worse. The factorised
`V × rank → rank × d` form is already cheap (393K for the wider vocabulary),
and the Kronecker structure costs too much in the input representation.

*Verdict:* closed; the factorised embedding stays.

#### 8.4.6 Recursive weight sharing

*Hypothesis:* reuse `n_unique` blocks for `r` passes and get depth for free.

| run | unique | passes | effective depth | parameters | bpb |
|---|---:|---:|---:|---:|---:|
| E8 | 12 | 1 | 12 | 190,108 | **1.479** |
| E19 | 6 | 2 | 12 | 111,502 | 1.717 |
| E21 | 6 | 1–2 random | 6–12 | 111,758 | 1.734 @ r=1 · 1.718 @ r=2 |
| E20 | 12 | 2 | 24 | 190,108 | **1.427** |
| E22 | 6 | 4 | 24 | 111,502 | 1.521 |

Three findings, all negative for the "free depth" hope:

1. **Sharing costs quality.** E19 at 59 % of the parameters and the same
   effective depth is 16 % worse than E8.
2. **Depth helps, but at 2× compute.** E20 gains 3.5 % over E8 with the same
   parameters and twice the FLOPs per token.
3. **Randomised training makes recursion usable.** E21 (r ~ U{1,2}, plus a
   per-pass embedding) matches E19 at r=2 and stays coherent at r=1 (1.734),
   where E19 collapses to 2.554. One model, two usable depths.

*Verdict:* not a default. Recursion is a deployment/compute trade-off
(adaptive compute), not a quality upgrade.

#### 8.4.7 Input-dependent channel gates

E15 (dense dynamic gate) reached **1.5091** and E16 (Kronecker gate) **1.5504**
against E8's **1.4789**. Both cost 2.4–2.6× the step time, because the gate
depends on the input and the composition can no longer be precomputed.

*Verdict:* closed.

#### 8.4.8 What did not survive contact with the data, in one line each

- **Sum composition**: higher rank, worse quality (1.5508 vs 1.479).
- **Hybrid compositions**: a 1.3 % win on one seed, reversed on the second.
- **Dynamic gates**: 3× slower, 2 % worse.
- **Nonlinear composition**: 2.7–7.5 % worse in every form.
- **Bilinear GLU FFN**: 2.2–2.9 % worse, with layer collapse.
- **`K=24`**: 3 % worse and complete depth collapse.
- **Kronecker embedding**: 12 % worse.
- **Weight sharing**: 16 % worse at equal effective depth.
- **KV cache**: exact (2.8e-7) but no speed-up on this GPU.
- **Factorised `P` apply**: exact, but 0.20–0.31× the speed of one dense GEMM.

The pattern is consistent: **what wins is the stuff that keeps the layer
linear and fusable**; what loses is anything that breaks the single-GEMM
composition, regardless of how much rank or nonlinearity it promises.

## 9. Hypotheses: verdicts

| # | hypothesis | verdict |
|---|---|---|
| H1 | normalising the branch scale is what makes the Kron layer trainable | **confirmed** (PPL 20.06 → 12.3 after the fix) |
| H2 | the depth recurrence adds useful state beyond independent factors | **untested** — no run removes the carry (E27/E28 pending); E8 vs E14 (1.479 vs 1.551, −4.9 %) tests the composition, not the recurrence |
| H3 | a wider tokenizer is worth more than extra parameters | **confirmed** (BPE-4096: −8.2 % at equal params) |
| H4 | effective rank is monotone in additive depth | **confirmed empirically** (60/60 modules; survives seeds) — but rank does not predict bpb |
| H5 | hybrid compositions beat the product | **rejected** (seed noise) |
| H6 | randomised recursion gives one model with usable adaptive compute | **confirmed** (r=1 1.734 vs 2.554 untrained) |
| H7 | extra depth at equal parameters is a clear win | **partially**: +3.5 %, but 2× compute |

H4 is deliberately called a hypothesis, not a theorem: effective rank is not
monotone under matrix products, and counterexamples exist for exact rank. The
observed ordering (product < alternating < grouped < sum) is empirical.

## 10. Tests

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests
# 102 passed, 3 skipped
```

The suite runs on CPU in ~20 s and covers: the Kronecker forward against a
naive reference, gradient flow through the recurrence, `KronBank` prelude
equivalence, the KV cache (exact to 2.8e-7), quantisation round-trips, the HF
wrapper, and regression guards for the failures described above (dead
composition scale, dead GLU branch, depth collapse).

## 11. Efficiency

Measured on a GTX 1660 Ti (6 GB, sm75; no FlashAttention, no Triton).

| model | batch × seq | s/step | µs/token | peak VRAM |
|---|---|---|---:|---:|
| Q6 | 32 × 192 | 0.847 | 138 | 4.28 GB |
| Q8g | 24 × 192 | 0.895 | 194 | 4.09 GB |

The gated FFN adds a sixth module per layer (+20 % matmuls) and the smaller
batch loses some GPU efficiency, so cost per token rises ~40 % while peak
memory falls, because 24 layers with batch 24 fit better than 32.

Two serving optimisations are implemented and exact but not useful here:

- **KV cache** — matches the no-cache path to 2.8e-7, but the 1660 Ti is
  compute-bound at these sizes, so decoding does not speed up.
- **Factorised apply** (`y = x·Pᵀ` computed factor by factor) — mathematically
  identical, but only 0.20–0.31× the speed of one dense GEMM. Composing once
  and multiplying once is the fast path.

`torch.compile` is unavailable in this environment (no Triton on Windows), so
all speed-ups are structural, not compiler-driven.

## 12. Repository layout

```
kronllm/            library: kron.py (composition), model.py, train.py,
                    config.py (experiments), data.py, tokenizer.py,
                    bilingual.py (corpus pipeline), quantization.py,
                    hf.py, cli.py
scripts/            data prep (TinyStories, BPE, WikiText, SFT,
                    build_bilingual.py, prepare_bilingual_bin.py),
                    dashboard
tests/              102 tests, CPU-only
docs/               architecture.md, results.md (full tables),
                    reproducing.md (one command per claim),
                    data_plan.md (bilingual corpus), roadmap.md
artifacts/          q8g_int8.pt (shipped), q2_int8.pt (previous),
                    q10_tokenizer.json (8K bilingual BPE for Q10 inference)
runs/               per-experiment logs and metrics; only the two best
                    checkpoints are kept (Q8g_long, Q6_10k)
notebooks/          walkthrough of the architecture and the measurements
current_progress.md rolling log of the experiment queue
```

## 13. Limitations

- **Development bench only.** Every quality number is TinyStories, English.
  Nothing here proves the same ordering on the bilingual corpus; the corpus
  pilot is built but no model has been trained on it yet.
- **One GPU, one seed.** Most experiments are single-seed; the one seed control
  performed (hybrid compositions) reversed a published-looking result, which is
  a warning about the rest.
- **No FlashAttention, no Triton.** Throughput is well below what the same
  model would reach with a fused kernel.
- **512-token context at most.** RoPE is implemented, but training used 192–256
  tokens; long-context behaviour is untested.
- **No instruction tuning in the shipped model.** S1 was an SFT probe on
  Dolly-15k, not a product.
- **int8 only.** int4 is implemented but the delta at this size is not yet
  worth the extra complexity.

## 14. The bilingual corpus

The corpus now lives in its own repository:
**[Icold21/kron-corpus](https://github.com/Icold21/kron-corpus)** — a bilingual
English/Russian corpus with a science accent, published with the full pipeline,
the per-shard manifest and the measured composition.

| stage | result |
|---|---|
| fetched text | 7.65 GB · 807k documents (+2.9 GB of Russian bulk imported later) |
| after dedup | 787k documents · 7.12 GB (20,022 dropped, 2.5 %) |
| mixed | **8.62 GB · 968,334 documents** (identity upsampled ×8) |
| measured size | **2,250,807,719 train tokens** (8K BPE with digit isolation, 3.69 bytes/token) |
| by language | EN 46.2 % / RU 53.8 % of bytes · **EN 61.9 % / RU 38.1 % of tokens** |
| domains | encyclopedic 30.3 %, literature 21.2 %, math 19.1 %, science 11.1 %, code 7.5 %, ML 5.4 %, logic 3.2 %, CS/engineering 2.3 %, identity 0.19 % |
| dedup | MinHash (Jaccard 0.8) · held-out 2 % per language, disjoint · title-level dedup for the bulk imports |
| tokenizer | 8K byte-level BPE, bilingual, digits split before ByteLevel so every digit is one token; fertility EN 2.92, RU 4.96 bytes/token |
| chat template | `<\|system\|>` 8187 · `<\|user\|>` 8188 · `<\|assistant\|>` 8189 · `<\|eot\|>` 8190 · `<\|endoftext\|>` 8191 |
| identity slice | 800 plain + 850 pastiche per language, no numeric facts |

**The ratio policy is the main measured surprise.** Bytes per token is not
comparable across scripts: Russian costs about five bytes per token against
three for English. The quota-based mix therefore gave Russian only **15.7 % of
tokens** even though it had 29 % of the bytes. The fix was not a quota tweak but
more Russian data: the full Russian Wikipedia (2.0 GB of new articles) and the
full Russian Wikisource (1.1 GB of public-domain literature), with titles
already present in the curated shards skipped. That moved Russian to **38.1 % of
tokens** — the original plan wanted 45–47 %, and closing the rest would need
another ~1 GB of Russian prose that does not exist under a reusable licence.

Known gaps are reported rather than hidden, and the corpus README repeats them:
logic lands at ~3 % against a 10 % quota (only formal proofs scale), ML at
~5 % against 12 % (most prose is paywalled), code at 7.5 % against 15 %
(Russian code is absent by nature), and humour is a rounding error. The mixer
emits these as shortfalls; nothing is padded with mislabelled text.

Next steps:

1. train **Q10** (9,938,886 params, d=512, 25 layers, K=10, seq 512) on the
   corpus on one Kaggle T4 with fp16 and batch 32 (DDP is unstable in the
   Kaggle image; see `docs/goal_10m.md`), one epoch = 137,377 steps of 16,384
   tokens = 2.251B, checkpoint every 500 steps;
2. package the result, then compare against public small-model baselines
   before the SFT stage;
3. decide the language rebalance for SFT (Russian is 38 % of tokens now).

**Reproducibility details that matter for the run.** The epoch sampler is
sharded: all ranks build the same permutation from a shared seed and take
`rank::world`, so one epoch is one full pass instead of the ~63 % a random
sampler covers. DDP writes logs and checkpoints only on rank 0; `last.pt`
carries the optimizer and the step counter, so a killed Kaggle session resumes
exactly where it stopped.

The long-term plan, including the SFT families and the comparison phase, is in
[docs/roadmap.md](docs/roadmap.md).

## 15. License and citation

MIT — see [LICENSE](LICENSE).

```bibtex
@misc{kronllm2026,
  title  = {KronLLM: a compact fully Kronecker-parameterised language model},
  author = {Ivan Kholodilo},
  year   = {2026},
  url    = {https://github.com/Icold21/kron-llm}
}
```
