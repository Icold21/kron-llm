# KronLLM — a compact, fully Kronecker-parameterised language model

Every weight matrix in a decoder-only transformer is replaced by a Kronecker
composition with a depth recurrence. The result is a family of small language
models — **190K to 2.08M parameters**, **0.87–8.55 MB** on disk — that train on a
single consumer GPU and generate on CPU. On TinyStories, the 926K model reaches
**0.976 bits per byte**; the 2.08M model goes further.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen.svg)](tests)
[![Parameters](https://img.shields.io/badge/parameters-190K%E2%80%932.08M-orange.svg)](#at-a-glance)

> **Naming.** The repository is `kron_llm`, the Python package is `kronllm`, and
> the model is **KronLLM**. *Kron* stands for **Kronecker**.

## At a glance

| property | value |
|---|---|
| Models | 190,108 · 926,468 · 2,079,992 · 2,473,208 parameters |
| Matrix parameterisation | Kronecker products, 4 channels × 2–8 summands; product or sum |
| Tokenizer | byte-level BPE, 1,024–4,096 tokens, 3.07–3.96 bytes/token |
| Context | 192 tokens |
| Validation quality | **0.976 bpb** (Q1, 926K) · **0.839** (Q2, 2.08M) · **0.849** (Q6, 2.47M, 4k steps) |
| Checkpoint size | 8.55 MB fp32 → 2.45 MB int8 → 1.42 MB int4 (Q2/S1) |
| Training footprint | Fits a single **6 GB** consumer GPU (Q2 4.22 GB, Q6 4.28 GB peak) |
| Inference | CPU or one consumer GPU; no custom kernels |
| License | MIT |

**Why compactness matters.** A dense transformer spends most of its parameters
on `d × d` matrices. KronLLM parameterises each matrix as a sum of Kronecker
products: at equal depth and width, the matrix budget of the dense control drops
**6.1×** (933,888 → 154,140 parameters) while the layer remains full-rank. The
whole 2.08M model is an 8.55 MB file, small enough to inspect, fine-tune and
quantise on ordinary hardware. The largest run trains inside **6 GB of VRAM**
(4.22 GB peak), so a single consumer card is enough; that is a capability, not
the organising goal.

**What is in this repository.** The full pipeline: data preparation, BPE
tokenizer, training, evaluation, spectral diagnostics, sampling, Dolly Q/A
fine-tuning, int8/int4 post-training quantisation, an experimental routed dense
embedding (Q3), a Kronecker-structured embedding (Q7), a Hugging Face
`PreTrainedModel` wrapper, a 28-cell acceptance notebook and 71 tests.

**Quick start** (details in §6):

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/prepare_bpe.py --vocab 1024
python -m kronllm.cli train --exp Q2
```

Q2 trains in about 3 hours on one consumer GPU; the 190K byte-level model
`E2` reaches a usable checkpoint in about 18 minutes.

## Contents

1. [What Kronecker means here](#1-what-kronecker-means-here)
2. [Depth recurrence](#2-depth-recurrence)
3. [Deviations from the specification](#3-deviations-from-the-specification--and-why)
4. [Parameter budget](#4-parameter-budget)
5. [Data](#5-data-why-tinystories-200k-and-not-more)
6. [Running](#6-running)
7. [Metrics](#7-metrics-how-to-read-them)
8. [Results](#8-results)
9. [Hypotheses H1–H3](#9-hypotheses-h1h3-verdicts)
10. [Tests](#10-tests)
11. [Compactness and efficiency](#11-compactness-and-efficiency)
12. [Repository layout](#12-repository-layout)
13. [Limitations](#13-limitations)
14. [What to do next](#14-what-to-do-next)
15. [License and citation](#15-license-and-citation)

---

## 1. What Kronecker means here

Every linear layer `d → d` (and `d → d_kv` for GQA) is not a `128 × 128` matrix
but `L` channels, each a sum of `K` Kronecker products:

```
I_l = Σ_{k=1..K} A_lk ⊗ B_lk        A_lk ∈ R^{8×8},  B_lk ∈ R^{16×16}
(A ⊗ B) x = A · X · Bᵀ,             X = reshape(x, 8, 16)
```

Such a matrix `I_l` holds `K·(n1² + n2²) = 640` parameters instead of
`d² = 16,384` — 26× fewer — while not being a low-rank approximation:
`rank(A ⊗ B) = rank(A)·rank(B)`, which reaches the full `d = 128` for full-rank
factors.

## 2. Depth recurrence

Every layer and every role has **its own** factors `A_lk, B_lk` and its own
`τ_l, θ_l`; otherwise all layers would compute the same function. Kronecker here
is not one matrix for the whole network but a **parameterisation of each
matrix**. Layer diversity is amplified by the state `D` carried across blocks:

```
D_l^{(i)} = τ_l · rms_norm_ℝ^{d×d} ( Î_l + σ(θ_l) · D̂_l^{(i-1)} )
W_l^{(i)} = s_l · Id + D_l^{(i)}
P^{(i)}   = W_L^{(i)} ⋯ W_1^{(i)}                 (a single d×d matrix)
y         = γ^{(i)} · P^{(i)} x                   (ReZero gate, γ_init = 0.1)
```

* `Î_l`, `D̂` are RMS normalisations over the whole matrix (not per row); E12
  (`normalize_D: false`) removes the normalisation, leaving `τ` as the only
  scale.
* `σ(θ_l)` is a learned memory speed per channel, `θ_init = 0`, `σ = 0.5`.
* `stop_gradient` on `D̂` is on by default; E8 turns it off.
* `K_win = 2`: the state resets every 2 layers (recurrence window).
* `s = s_init^{1/L}` is the effective anchor value (see §3, item 2).

`P` is computed batched for all 60 modules at once (`KronBank`) and applied as an
ordinary `x @ Pᵀ`: 5 matrix multiplications per layer instead of 5 × 4 small
matvecs.

### Parameters of one module

| component | formula | count |
|---|---|---|
| factors `A_lk, B_lk` | `L·K·(n1² + n2²)` = 4·2·320 | 2,560 |
| `θ_l` (memory) | `L` | 4 |
| `τ_l` (scale of D) | `L` | 4 |
| `γ` (ReZero) | 1 | 1 |
| **total** | | **2,569** |

60 modules (12 layers × 5 roles) → **154,140**; embedding 256 × 128 = 32,768;
RMSNorm 25 × 128 = 3,200; **190,108 in total**.

Q1 (§8.3) uses the same formulas with other numbers: `n1=12, n2=16, K=5, L=4,
d=192`. One module costs `L·K·(n1²+n2²) + 2L + 1 = 4·5·400 + 9 = 8,009`
parameters; 100 modules (20 layers × 5 roles) → **800,900**. Q2 (§8.4):
`n1=16, n2=20, K=6, L=4, d=320` → `4·6·656 + 9 = 15,753` per module; 120
modules → **1,890,360**.

---

## 3. Deviations from the specification — and why

This is the most important section: without these changes the architecture from
the specification **does not train at all**. Every item below is a measurement,
not a matter of taste.

### 3.1 Root defect: `gain = 1/√d` makes the model the identity

With factors initialised as `gain = 1/√d = 0.088`:

```
‖I‖_RMS ≈ 1e-3   →   W_l = s·Id + D_l ≈ s·Id
P = W_4 W_3 W_2 W_1 ≈ s^L · Id = 0.1 · Id
```

The whole Kronecker layer degenerates into multiplication by a constant.
Measured on a real 300-step TinyStories run:

| metric | value | meaning |
|---|---|---|
| `‖∂L/∂θ‖` | **0.000** | the recurrence receives no gradient |
| `cos(W_i, W_{i-1})` | **1.000** | all layers are the same matrix |
| `eff_rank(W)` | **128** (for all) | full rank comes from the anchor, not the innovation |
| val PPL | **20.06** | ≈ a unigram model |
| samples | "letter soup" | — |

### 3.2 The naive fix also breaks

With `gain ≈ 1` (spectral norm of a factor ≈ 1) the hidden state grows across
layers as `11 → 226 → 1858 → 6830`, the attention contribution drops to 0.2–3 %,
and `std(attn logits) = 0.0098` — the model again ignores context and only
rescales the residual.

### 3.3 What was actually done

| # | specification v2.0 | implemented | reason (measurement) |
|---|---|---|---|
| 1 | `gain = 1/√d` | `D` is RMS-normalised; `gain` is irrelevant in the forward pass (kept as an ablation knob `init_gain`, E12) | see 3.1: otherwise `P ≈ s^L·Id` and `∂L/∂θ = 0` |
| 2 | one `s_init = 0.1` for QKV | per-role `s_init = {q,k,v,o,ffn} = 1.0`, treated as an effective value: `s = s_init^{1/L}` | with `s = 0.1` and `D ≈ 1e-3` the matrix is `W = 0.56·Id`, i.e. a layer halves the signal; with `s = 1` the innovation and the anchor are of the same order |
| 3 | — | `τ_l` (learned scale of each `D_l`), init 0.1 | sets `RMS(D)` directly; `eff_rank(D)` grew from 1.0 to **91.3/128** |
| 4 | — | ReZero gate `γ` on the branch output, init 0.1 | the branch starts as a small perturbation of the residual; without it the residual grows 600× (3.2) |
| 5 | `s` shared by QKV | its own `s` per role; `s_learnable` switches `Parameter` vs `buffer` | in E2 `s` is a buffer, in E3 a `Parameter` |
| 6 | SiLU = experiment E9 | SiLU at the FFN output is the default; E9 became the linear-FFN ablation | SiLU is standard, the comparison is more useful as a baseline |
| 7 | E0–E10 | added **E11** (gated FFN `(W_a x) * silu(W_b x)`), **E12** (no normalisation of D) and **E13** (`K=8`, ~651K) | E12 prices the normalisation, E11 an alternative to SiLU, E13 a point on the budget→quality curve |
| 8 | `s_init=0.1` + always anchor | E4: `anchor=False` and `τ=1.0` | without the anchor `‖P‖ ≈ τ^L`; with τ=0.1 the branch is dead (1e-4) |
| 9 | `amp: fp16` | `amp: none` (fp32) | fp16 autocast was 2.4× slower than fp32 on the reference GPU, so fp32 is the default |
| 10 | E0 — plain dense | dense FFN hidden=48 → E0 = 969,856 (not 1.22M) | the hard ≤ 1M limit is respected by the control too |
| 11 | — | `KronBank`: all 60 compositions computed batched | 439 vs 505 ms/step; prelude = 4.9 ms; mathematically identical (test T16) |
| 12 | seq/batch from the spec | `batch_size=128, seq_len=256` | chosen for throughput; see §11 |
| 13 | CUDA graphs / `torch.compile` | disabled | capture fails on `baddbmm` in this environment |

Items 1–5 are really one mechanism: **the scale must be normalised, not set by
initialisation**. After the fixes `eff_rank(D) = 91`, `cos(W_i, W_{i-1}) = 0.72`
instead of `1.000`, and PPL is what the results table shows.

---

## 4. Parameter budget

```bash
python -m kronllm.cli budget        # E-series table, exits 1 if > 1M
python -m kronllm.cli budget --all    # also Q1-Q3, D1, S1 (above 1M by design)
```

All values are ≤ 1,000,000 (the maximum is E0 at 97.0 % of the limit).

| E | what | total | matrix | vs E0 |
|---|---|---|---|---|
| E0 | dense baseline | 969,856 | 933,888 | 1.00× |
| E1 | dense MHA + Kron FFN | 853,228 | 817,260 | 0.88× |
| **E2** | **full Kron (base)** | **190,108** | **154,140** | **0.17×** |
| E3 | + learnable `s` | 190,168 | 154,200 | 0.17× |
| E4 | no anchor, `K_qkv=8` | 282,268 | 246,300 | 0.26× |
| E5 | `d=192, N=16` | 312,208 | 256,720 | 0.27× |
| E6 | `L=6` | 267,148 | 231,180 | 0.25× |
| E7 | `K=4` | 343,708 | 307,740 | 0.33× |
| E8 | no stop-gradient | 190,108 | 154,140 | 0.17× |
| E9 | linear FFN (no SiLU) | 190,108 | 154,140 | 0.17× |
| E10 | GQA (n_kv=2) | 141,580 | 105,612 | 0.11× |
| E11 | gated FFN | 220,936 | 184,968 | 0.20× |
| E12 | no normalisation of D | 190,108 | 154,140 | 0.17× |
| E13 | `K=8` (budget ~651K) | 650,908 | 614,940 | 0.66× |
| E14 | sum composition, full backprop | 190,348 | 154,380 | 0.17× |

---

## 5. Data: why TinyStories-200K and not more

`data/train.bin` = **180,723,995** bytes = 180.7M tokens at the byte level
(1 byte = 1 token). That is **950 tokens per parameter** for the 190K model.

* The Chinchilla optimum for 190K parameters is ≈ 3.8M tokens. We are already
  **47×** past it.
* The full TinyStories (2.1M stories) would give ~1.9B tokens = 10,000
  tokens/parameter. For a 190K model that buys nothing: **the bottleneck is
  capacity, not data**.

So 200K stories (10 % of the corpus) is a deliberate choice, not a limitation.
TinyStories is the **development bench**: it is what makes the ablations cheap
and comparable, and it stays as a regression check for the frozen architecture.
It is **not** the production corpus and will not be part of the final
pretraining mix; the planned bilingual English–Russian corpus is described in
[docs/roadmap.md](docs/roadmap.md). To prepare a different amount:

```bash
python scripts/prepare_data.py --max-stories 500000     # more stories
python scripts/prepare_data.py --max-stories 20000      # faster for experiments
```

The dataset is set in `kronllm/config.py:data.dataset` (any HF dataset with a
`text` field); `max_stories` caps the number of stories.

**If you want better text within 1M parameters, grow the model, not the corpus.**
The Kronecker parameterisation scales with `n1² + n2²`, not with `d²`, so there
are two cheap directions: `K=8` (E13: 651K parameters with almost no step-time
cost) and more width (`d=256, N=24, n1=n2=16` ≈ 570K but 4× the FLOPs).

**For Q1/Q2 (§8.3–8.4) the arithmetic is different:** 59.2M BPE tokens for 926K
parameters is **64 tokens per parameter**, three times the Chinchilla optimum
(20). Q1 was stopped at 4,200 steps (51.6M tokens = 56 tokens/parameter), while
Q2 sees 61.4M tokens with 2.08M parameters — **29 tokens/parameter**, close to
optimal. The next lever is again data: the full corpus (2.1M stories) would give
~620M tokens = 300 tokens/parameter for Q2.

---

## 6. Running

```bash
# environment
pip install -r requirements.txt          # torch 2.6.0+cu124, gradio, matplotlib, pandas

# data (downloads TinyStories from the HF hub and writes a memmap)
python scripts/prepare_data.py

# BPE tokenizer with 1,024 tokens (~3 bytes/token) + corpus re-tokenisation
python scripts/prepare_bpe.py --vocab 1024

# training
python -m kronllm.cli train --exp E2 --steps 2500
python -m kronllm.cli train --exp Q2                     # 2.08M model, 10k steps
python -m kronllm.cli train --exp Q1                     # 926K model (4.2k steps here)
python -m kronllm.cli train --exp D1 --run-dir runs/D1   # dense control on BPE
python -m kronllm.cli train --exp E12 --override model.kron.K=4 --run-dir runs/E12_k4

# evaluation / generation / analysis
python -m kronllm.cli eval --run runs/Q2
python -m kronllm.cli generate --run runs/Q2 --prompt "Once upon a time" --samples 3 \
    --temperature 1.0 --top-k 8
python scripts/analyze.py --run runs/Q2
```

Supervised fine-tuning and quantisation (see §8.6–8.7):

```bash
python scripts/prepare_sft.py                        # Dolly-15k -> data/sft_*.bpe.bin
python -m kronllm.cli train --exp S1                 # Q2 + SFT, 400 steps
python -m kronllm.cli quantize --run runs/S1         # writes model_int8.pt / model_int4.pt
python -m kronllm.cli generate --run runs/S1 --checkpoint model_int8.pt \
    --prompt "### Question:\nWhat is the capital of France?\n### Answer:\n"
```

Notebook with tests, plots and generation:

```bash
jupyter lab notebooks/kronllm_tests.ipynb
```

The invariants run inside the notebook, so it can be executed top-to-bottom as
an acceptance test: parameter budget, `kron_apply == dense`, bank == per-module
path, causality, bits-per-byte curves, diagnostics and samples.

### Hugging Face interface

`kronllm/hf.py` wraps the model in a standard `PreTrainedModel` and registers it
with the `AutoClass` registry:

```python
import torch
from kronllm.hf import KronForCausalLM
from kronllm.tokenizer import load_tokenizer

model = KronForCausalLM.from_kron_checkpoint("artifacts/q2_fp32.pt")
tok = load_tokenizer(model.config.kron_config)
ids = torch.tensor([tok.encode("Once upon a time")])
out = model.generate(ids, attention_mask=torch.ones_like(ids),
                     max_new_tokens=64, do_sample=True,
                     temperature=1.0, top_k=8)
print(tok.decode(out[0].tolist()))
```

The wrapper supports `save_pretrained` / `from_pretrained` / `push_to_hub`; the
nested KronLLM config is stored under `kron_config` in `config.json`, and after
`import kronllm.hf`, `AutoModelForCausalLM.from_pretrained(...)` resolves to
`KronForCausalLM`. `from_kron_checkpoint` also accepts the int8/int4 artefacts.
Install the optional dependency with `pip install "kronllm[hf]"`.

Limitations: there is no KV cache (every token is a full forward), only all-ones
attention masks are exact (no padded batches), and tokenisation remains the
custom BPE from `kronllm.tokenizer`, not a `PreTrainedTokenizer`.

The whole E series:

```bash
python scripts/run_experiments.py --all --steps 2500
python scripts/run_experiments.py --exp E2 E4 --steps 2500
```

Results land in `runs/<E>/results.json`, `runs/<E>/diagnostics.json`,
`results/summary.csv` and images in `runs/<E>/analysis/`.

### Dashboard

```bash
python scripts/dashboard.py --port 7860
```

Four tabs: **Training** (start/stop runs, live log and loss curve, data
preparation, run the whole series), **Generation** (samples from any
checkpoint, A/B two runs), **Metrics** (metric table, hypotheses H1–H3,
heatmaps, spectra, rank-vs-depth, τ/γ), **Experiments** (comparison table and
plots).

---

## 7. Metrics: how to read them

| metric | meaning | healthy value |
|---|---|---|
| `eff_rank_D` | effective rank of the innovation `D_l` (singular values > 1 % of max) | 60–110 at `d=128` (was 1.0) |
| `eff_rank_W` | rank of `W = s·Id + D`; with the anchor it is always ≈ d, so it is not informative | ≈ d, ignore |
| `rank_excess` | `eff_rank(W) − eff_rank(D)` — how much rank the anchor contributes | > 0 |
| `eff_rank_P` | rank of the composition `P = W_L ⋯ W_1` | ≈ 60–90 |
| `eff_rank_sum` | rank of `sI + ΣD_l`; on product runs it reuses the trained `D_l`, on E14 it is the trained composition | > `eff_rank_P` means multiplication compresses |
| `eff_rank_excess` | rank of `P − s^L·Id` for product runs, `P − s·Id` for E14 | the larger the better |
| `cos_prev_max` | cosine between `W_l` of neighbouring layers | 1.0 = identical layers; 0.7 = distinct |
| `tau_mean`, `D_rms` | scale of D; `D_rms` equals τ by construction | 0.05–0.5 |
| `D_rms / s`, `σ_max(D) / s` | innovation scale relative to the anchor | ≪ 1 = sum-like; ≈ 1 = product-like |
| `gate_mean` | ReZero γ | 0.05–1.0 |
| `sigma_theta_var` | `Var_l σ(θ_l)` — hypothesis H3 | > 0.01 = channels specialised |
| `attn_entropy` | attention entropy | ln(256) = 5.55 is the maximum; 2–3 means the model uses context |
| `h_norm_ratio` | `‖h_i‖ / ‖h_{i−1}‖` | ≈ 1; > 5 means the residual explodes |

`theta_grad_norm` in `results.json` is always 0: diagnostics run after
`optimizer.zero_grad()`. Per-group gradients are logged to `log.csv` instead
(`grad_embed`, `grad_factors`, `grad_theta`, `grad_tau`, `grad_gate`).

---

## 8. Results

### 8.1 Base model E2 (2,500 steps, 0.45 epochs)

| | |
|---|---|
| parameters | **190,108** (154,140 in matrices, 81 %) |
| val loss / PPL | **1.1502 / 3.159** |
| time | 1,107 s (0.443 s/step), 77k tok/s |
| FLOPs | 3.70 MFLOP/token forward (E0: 3.51M — only +5 %) |

Diagnostics after training:

| metric | value | comment |
|---|---|---|
| `eff_rank(D)` | **91.2 / 128** | was 1.0 with `gain = 1/√d` |
| `eff_rank(P)` | 60.0 | the composition **compresses** rank |
| `eff_rank(sI + ΣD_l)` | **101.0** | same trained `D_l`, sum diagnostic |
| sum / product | **1.70** | multiplication removes ~40 % of the rank |
| `eff_rank(P − s^L·Id)` | **64.0** | > `eff_rank(P)`: the anchor dominates the spectrum |
| `cos(W_i, W_{i−1})` | 0.719 | was 1.000 — layers differ |
| `cos(P_i, P_{i−1})` | 0.083 | neighbouring compositions are nearly orthogonal |
| `D_rms / s` | 0.118 | `s = 1.0` per factor; the RMS innovation is 12 % of the anchor |
| `‖D‖_F / s` | 15.2 | high-rank `D`: 128 × RMS, so this is not a scale-free number |
| `σ_max(D) / s` | 3.60 | the spectral norm is 3.6 × the anchor |
| `τ` (D RMS) | 0.118 | learned, grows with depth (0.105 → 0.17) |
| `γ` (ReZero) | 0.115 | grows with depth (0.10 → 0.185) |
| `attn_entropy` | 1.65 | ln(256) = 5.55 — the model looks at context |
| `‖h_i‖ / ‖h_{i−1}‖` | 1.66 (max 6.6) | the residual does not explode |
| `Var_l σ(θ_l)` | 0.0039 | H3 not confirmed (threshold 0.01) |

The RMS ratio is small, but `σ_max(D)/s` is 3.3–3.6: the innovation is not an
infinitesimal perturbation. The rank loss is a property of the product and of
the correlated channels, not just of a small RMS. The sum row uses the trained
`D_l` from the product model; it is a diagnostic, not a trained sum-composition
result. It shows that multiplication itself removes ~40 % of the rank. The
trained sum ablation is E14 (§8.8).

Two mechanisms are visible directly in the plots (`runs/E2/analysis/`):

* **`rank_vs_depth.png`** — a sawtooth with period 2: layers where the
  recurrence is reset (`K_win=2`) have `eff_rank(D) ≈ 89`, layers with memory
  ≈ 92. There is no monotonic growth (`Spearman = 0.38`), and
  `eff_rank(P − s^L·Id)` even **decreases** with depth (69 → 61). This is the
  H2 verdict.
* **`scales.png`** — `τ` and `γ` grow towards the last layers (FFN: 0.105 →
  0.17 and 0.10 → 0.185). The model compensates for the missing rank growth
  with the amplitude of the branch: deeper layers change the residual more.

**Where the model fails** (`worst_tokens.json`, top-20 by loss): the hardest
positions are all the first letter of a word after a space (`…their big V`,
`…his old V`, `…day, the V`), loss 10.4–10.8 nats. That is the structural price
of a byte-level vocabulary: the first letter of an English word is almost
unpredictable from context, and the model has to keep a distribution over 26+
options. No architecture fixes this — only BPE/word-level tokenisation does.

### 8.2 The E0–E14 series: why it was stopped

The plan was 14 runs, E0–E13. After E2 it became clear that they differ by 1–3
hyperparameters and change PPL by ±5–10 %: that answers "which of nearly
identical configurations is slightly better", not "how to get good text". The
time budget went into the quality branch Q: first **Q1** (BPE + 926K, §8.3),
then **Q2** — the same architecture with 2.08M parameters (§8.4). E8 (full
backprop) and E14 (sum composition) were added later as targeted ablations of
the two assumptions behind the product: stop-gradient and multiplication.

| run | parameters | matrix | val loss | val PPL (token) | **val bpb** | steps |
|---|---|---|---|---|---|---|
| **E0** (dense, bytes) | 969,856 | 933,888 | 0.9789 | 2.661 | **1.412** | 2,500 |
| **E2** (Kron, bytes) | 190,108 | 154,140 | 1.1502 | 3.159 | **1.659** | 2,500 |
| **E8** (full backprop, bytes) | 190,108 | 154,140 | 1.0251 | 2.79 | **1.479** | 4,000 |
| **E14** (sum, bytes) | 190,348 | 154,380 | 1.0750 | 2.93 | **1.551** | 4,000 |
| **Q1** (Kron, BPE) | 926,468 | 800,900 | 2.0735 | 7.953 | **0.976** | 4,200 |
| **Q2** (Kron, BPE) | 2,079,992 | 1,890,360 | 1.7833 | 5.95 | **0.839** | 10,000 |
| **Q4** (full backprop, BPE) | 2,079,992 | 1,890,360 | 1.9653 | 7.137 | **0.9248** | 4,000 |
| **Q6** (BPE-4096, full backprop) | 2,473,208 | 1,890,360 | 2.3292 | 10.27 | **0.8489** | 4,000 |

`val loss`/`PPL` in this table are in different units (nats/byte for
E0/E2/E8/E14, nats/token for Q1/Q2/Q4/Q6), so only the last column can be
compared across rows.

**Full backprop (E8).** Removing the stop-gradient from the depth recurrence
improves quality: at the matched 2,500-step point E8 reaches **1.586 bpb**
versus E2's **1.659**, and its best checkpoint at step 4,000 reaches **1.479**.
The run was stopped at step 4,400 because the default 3-epoch schedule would
have run ~16k steps; 4,000 is the matched comparison point. The spectral
picture does not change: at step 4,000 `eff_rank(P) = 60.27` versus
`eff_rank(sI + ΣD_l) = 101.23` (ratio 1.70), `Var_l σ(θ_l) = 0.00248`, and
`cos(W_i, W_{i−1}) = 0.761`. Full backprop buys quality but does **not** remove
the product's rank compression; H3 is still not confirmed.

**Full backprop at quality scale (Q4).** The E8 win survives at 2M: Q4 reaches
**0.9248 bpb** at step 4,000 versus Q2's **0.962** at the same step, at
identical cost (2,079,992 parameters, 36.2 MFLOP/token, 4.31 GB peak VRAM).
The stop-gradient stays off in all following runs.

**Wider tokenizer (Q6).** BPE-4096 raises the compression from 3.07 to
**3.96 bytes/token**. Q6 reaches **0.8489 bpb** at step 4,000 versus Q4's
0.9248 — **8.2 % better** — and nearly matches Q2's 10,000-step 0.839 in a
quarter of the steps. The gain is the tokenizer, not new capacity: the matrix
layers are identical to Q4, and the only extra parameters are 393K in the
factorised embedding (2,473,208 total). To stay inside 6 GB, the loss is
computed in checkpointed row chunks (`model.loss_chunk`), so the
`32 × 192 × 4096` logits tensor is never materialised. Peak VRAM 4.28 GB.

**Kronecker embedding (Q7) is a negative result.** Replacing the `V × r` table
with `Σ_k A_k ⊗ B_k` cuts the embedding from 565K to 90K parameters at the same
rank ceiling, but at step 4,000 it reaches **0.9533 bpb** versus Q6's 0.8489 —
**12 % worse**. The dense factorised table stays the default. The code is kept
(test-covered) for vocabularies where the dense table is unaffordable, but it
needs a better initialisation or training scheme before it can be used.

`python scripts/run_experiments.py --rebuild-summary` builds
`results/summary.csv` from `results/E*.json`; single runs are started as
`python -m kronllm.cli train --exp E7 --run-dir runs/E7`.

### 8.3 The quality branch Q1: "three bytes at a time", but learned

E2 is trained on bytes for 0.45 epochs: it genuinely learns spelling, but a word
costs 4–5 predictions and the most expensive one is the first letter after a
space (§8.1). The "three bytes at a time" idea addresses exactly that. It is
implemented not as a fixed chunking (a `256³ = 16.7M` vocabulary is impossible
within 1M parameters) but as a **byte-level BPE with 1,024 tokens** trained on
the same corpus:

| | train | val |
|---|---|---|
| bytes (without separators) | 180,723,995 | 8,310,677 |
| tokens | 59,237,168 | 2,707,417 |
| **bytes/token** | **3.05** | **3.07** |

One prediction therefore covers ~3 bytes, and the boundaries are chosen by the
merges: `the`, `ing`, `and`, ` once` become single ids, rare triples stay split.
`EOT = 1023` is reserved and never produced by the merges. The corpus is the
same (same files, same validation split), so the comparison metric is
**bits per byte**:

$$\mathrm{bpb} = \frac{\mathcal{L}_{\text{nats/token}}}{\text{bytes/token}\cdot\ln 2}.$$

**Q1's budget is 926,468 parameters, and it is balanced:**

| block | parameters | share | what |
|---|---|---|---|
| Kronecker matrices | 800,900 | 86.4 % | 20 layers × 5 roles × (4 channels × 5 summands × 400 + 9) |
| embedding | 116,736 | 12.6 % | factorised tied: `1024×96 + 96×192` instead of `1024×192` |
| norms | 8,832 | 1.0 % | pre-norm RMSNorm + QK-norm |

The factorised embedding (ALBERT-style) saves 80K parameters, and the head is
**free**: `logits = h Uᵀ Eᵀ`, i.e. the same weights read in the opposite
direction.

**Configuration:** `d=192 · N=20 · H=8 (d_head=24) · K=5 · L=4 · n1=12 · n2=16 ·
seq_len=192 · batch=64 · lr=5e-4`. The plan was 10,000 steps (123M tokens ≈ 2.1
epochs), but the run was stopped at **4,200 steps** (51.6M tokens ≈ 0.87
epochs): the remaining time went to Q2 — a model that spends three times more
compute per byte of text and therefore has a higher quality ceiling.

**Intra-layer normalisation.** Besides the usual pre-norm, **QK-norm** is added
— an RMSNorm on `q` and `k` per head before RoPE. It keeps the attention
operands on the same scale regardless of how the Kronecker branch has grown,
removes logit drift and allows `lr=5e-4` without divergence. It costs
`2·d_head` parameters per layer (0.04 % of the budget).

**FLOP cost.** 11.2 MFLOP/token at 3.05 bytes/token = **3.66 MFLOP/byte**
against 3.73 for E2 — that is 4.9× more parameters at the same compute per byte
of text.

**Q1 result** (best checkpoint, step 4,000, val): loss 2.0735 nats/token →
**PPL 7.95**, **bpb 0.976**. That is **41 %** fewer bits per byte than E2
(1.659) and **31 %** fewer than the dense control E0 (1.412) at almost the same
parameter count (926K vs 970K). The curve was still falling at step 4,200
(~0.005 bpb per 500 steps), so Q1 is not a converged model but one stopped by
the time budget.

**Honest caveat.** Q1 differs from E2 in three things at once (tokenisation,
capacity, number of steps), so the bpb gain cannot be attributed to any single
one of them. Separating the contributions needs dedicated runs; the point here
is that the final quality per byte is reached within the same FLOP budget and
under the 1M parameter limit.

### 8.4 Q2: quality at 2.08M parameters

Once quality became more important than the parameter count, the 1M limit was
lifted. But "just make the model bigger" is not the goal: Q2 adds parameters
where they buy quality **without inflating the training footprint**. Activation
memory grows as `N · d · batch · seq`, so Q2 widens and deepens the model while
halving the batch, keeping `N · d · batch` identical to Q1:

| | Q1 | Q2 |
|---|---|---|
| `d` / `N` | 192 / 20 | **320 / 24** |
| `K` / `L` | 5 / 4 | **6 / 4** |
| `n1 × n2` | 12 × 16 | **16 × 20** |
| parameters | 926,468 | **2,079,992** |
| `N · d · batch` | 20·192·64 = 245,760 | 24·320·32 = **245,760** |
| MFLOP/token | 11.2 | **36.2** |
| MFLOP/byte | 3.66 | **11.9** |
| fp32 checkpoint | 3.90 MB | **8.55 MB** |

The activation footprint matches to the last unit, so Q2 trains with the same
memory budget as Q1 while carrying 2.2× the parameters. The price is **3.2×
compute per byte**: a step takes 1.06 s instead of 0.54 s with half the batch
size.

Q2's budget (2,079,992 parameters):

| block | parameters | share |
|---|---|---|
| Kronecker matrices | 1,890,360 | 90.9 % |
| embedding (tied, `r=128`) | 172,032 | 8.3 % |
| norms (pre-norm + QK-norm) | 17,600 | 0.85 % |

**Configuration:** `d=320 · N=24 · H=8 (d_head=40) · K=6 · L=4 · n1=16 · n2=20 ·
r=128 · seq_len=192 · batch=32 · lr=4e-4 · 10,000 steps` = 61.4M tokens ≈ 1.04
epochs of the corpus.

**Q2 result** (best checkpoint, step 10,000): loss **1.7833** nats/token →
**PPL 5.95**, **bpb 0.839**. That is **14.0 %** fewer bits per byte than Q1
(0.976) at 2.2× the parameters, and **40.6 %** fewer than the dense byte-level
control E0 (1.412). The validation curve was still improving at the end
(~0.006 bpb per 500 steps), so Q2 is compute-limited, not converged.

Samples at `T=1.0`, `top_k=8` (prompt in bold):

> **Once upon a time** — Once upon a time, there was a little boy named Timmy.
> Timmy loved to play with his toys and eat candy. One day, Timmy's mommy told
> him that he was going to have fun with his friend.
>
> Timmy's mom said, "Sure, let's play a game. Let's go outside and play!"
>
> **The little girl found a** — The little girl found a beautiful toy. She was
> so happy that she could play with it. She was very excited to give it a new
> toy.
>
> The little girl was very happy and had so much fun playing with the toy all
> day. She played with her friends for a little while, but soon the toy was too
> late.

The samples are coherent at the sentence level and keep a consistent cast for
a paragraph or two; they lose the thread after that, which is expected at 2.08M
parameters and ~1 epoch of TinyStories.

**Honest caveat.** Q2 differs from Q1 in parameters (2.2×), in tokens seen
(1.2×) and in compute per byte (3.2×), so it is not a controlled "more
parameters at equal compute" experiment but a choice in favour of quality at a
fixed time budget. The clean comparison "same architecture, equal compute per
byte" is Q1 against E2 in §8.3; the clean comparison "same tokens, same
tokenizer, Kronecker vs dense" is D1 against Q1 in §8.5.

#### Routed-embedding variant (Q3, experimental)

The repository also implements a routed dense embedding. Each token has a
6-dimensional routing code; a linear router produces 5,000 logits, the top 10
are softmaxed, and the corresponding rows of a `5000 x 320` bank are mixed. The
effective `V x d` table is reused by the tied head, so the head adds no
parameters. Q3 has **3,544,104** parameters.

A from-scratch 500-step pilot reached **2.08 bpb at step 400** (val loss 4.427,
PPL 83.7), versus **1.48 bpb at step 500** for Q2, and peaked at **5.56 GB** of
VRAM at batch 32. The router does not learn quickly without help. A longer run
(`Q3long`, 5,000-step budget, batch 24, `lr=4e-4`, warmup 300) reached
**1.277 bpb at step 2,000**, against Q2's **1.068** at 2,000 and **1.245** at
1,000. Route usage improved smoothly, not as a jump: usage PPL 497 -> 656,
unique rows 817 -> 842, unique top-1 179 -> 355, effective table rank
245 -> 220. The run was stopped at 2,000; Q3warm remains an initialisation
test, not evidence of from-scratch convergence.

A warm start from Q2's effective table (`scripts/init_routed.py`) is different:
it starts at Q2's loss and reaches **0.8236 bpb** after 500 fine-tuning steps
(val loss 1.7502, PPL 5.76, peak 3.51 GB at batch 24). That is **1.85 % better**
than Q2, but with **70 % more parameters** and a 14.4 MB checkpoint. The router
stays close to its initialization: all 1,024 tokens keep their own bank row as
top-1, the effective-table cosine is 0.99997, and 4,397 / 5,000 rows are used.
The gain therefore comes from fine-tuning a near-copy of Q2's table, not from a
newly discovered sparse code. Q2 remains the compact published model; Q3warm is
reproducible (see [`docs/results.md`](docs/results.md#8-routed-embedding-q3)) but
is not shipped. The design and the exact mixture math are in
[`docs/architecture.md`](docs/architecture.md#51-routed-dense-embedding-q3-experimental).

### 8.5 D1: the price of Kronecker, measured on the same tokenizer

Q1 vs E2 changes tokenisation, capacity and step count at once, so neither of
them prices the Kronecker parameterisation itself. D1 closes that gap: it is the
dense E0 recipe (`use_kron_qkv=False`, `use_kron_ffn=False`,
`dense_ffn_mult=0.375`, QK-norm on) trained on the **same BPE tokenizer, the
same corpus and the same number of steps as Q1**.

| | D1 (dense) | Q1 (Kronecker) |
|---|---|---|
| parameters | 1,068,544 | 926,468 |
| `d` / `N` | 128 / 12 | 192 / 20 |
| val loss (nats/token) | 2.0763 | 2.0735 |
| val bpb | 0.9770 | **0.976** |
| steps | 4,200 | 4,200 |
| tokens seen | 51.6M | 51.6M |

**D1 result:** val loss **2.0763** nats/token, PPL **7.97**, bpb **0.9770**.
Q1 is **0.9757 bpb**, so Q1 is 0.13 % better with **15.3 % fewer parameters**
and **3.4× fewer FLOPs per token**. The Kronecker parameterisation is therefore
not a large quality win at 1M parameters, but it is a real parameter and compute
win at parity. D1 is 3× faster in wall-clock time (0.178 s vs 0.54 s per step)
because dense GEMMs are cheaper; the comparison is parameter- and token-matched,
not wall-clock-matched.

**Caveat.** D1 matches Q1 in tokenizer, sequence length, batch size and token
budget, so the comparison isolates the parameterisation. It does not match it in
wall-clock time: a dense step is cheaper, so D1 costs less time for the same
number of tokens.

### 8.6 S1: supervised fine-tuning on Dolly-15k

A 2M-parameter model cannot become a general assistant. What it can do is learn
the Q/A **format** and the closed-vocabulary questions that dominate a small
fine-tuning set. S1 is the honest version of that experiment:

* Data: [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
  15,011 human-written instruction/response pairs. The Q/A-style categories
  (`closed_qa`, `open_qa`, `general_qa`) are kept: 7,706 rows, of which 4,882
  fit the 192-token context (1,711 of them only after dropping the context
  paragraph; the remaining 2,824 are still too long and are dropped).
* Template (also used at inference time):

  ```
  ### Question:
  {instruction}
  ### Answer:
  {response}<eot>
  ```

* Training: the Q2 weights are loaded (`training.init_from`), the optimizer and
  schedule are fresh (400 steps, lr 1e-4, warmup 40, batch 32), and the
  validation split is 5 % of the pairs.

```bash
python scripts/prepare_sft.py     # writes data/sft_train.bpe.bin / sft_val.bpe.bin
python -m kronllm.cli train --exp S1
python -m kronllm.cli generate --run runs/S1 \
    --prompt "### Question:\nWhat is the capital of France?\n### Answer:"
```

**S1 result** (400 steps): val loss **2.3593** nats/token, PPL **10.58**, bpb
**1.110** on the Dolly validation split. That bpb is **not comparable** to the
TinyStories numbers above: Dolly Q/A is a different, much smaller distribution,
and the score covers the full templated sequence.

The model reliably learns the template:

> **### Question:** What is the capital of Germany?
> **### Answer:** Germany is the first capital of Germany.

It does not acquire reliable facts. At 2.08M parameters and 400 steps, S1 is a
format demonstration, not an assistant. It is kept because it is the smallest
end-to-end Q/A artifact in the repo, and because the failure mode is
informative: the Kronecker backbone can fit the interface, but factual recall
needs more capacity, more data, or retrieval.

### 8.7 int8 / int4 quantisation

KronLLM has no `nn.Linear` layer, so `torch.ao.quantization` cannot see any of
its matrices. `kronllm/quantization.py` therefore quantises the **stored
tensors** — factors, embeddings, tied head — with symmetric per-channel scales,
and dequantises them when the checkpoint is loaded. The forward pass is
unchanged; the win is file size, which is what matters for deployment.

```bash
python -m kronllm.cli quantize --run runs/S1 --bits 8 4
```

The command writes `model_int8.pt` / `model_int4.pt`, reports the compression
ratio and measures the bits-per-byte change on the validation split:

| artifact | size | compression | val bpb (Dolly) | Δ bpb |
|---|---:|---:|---:|---:|
| fp32 | 8.55 MB | 1.00× | 1.1102 | — |
| int8 | 2.45 MB | **3.49×** | 1.1113 | **+0.0011** |
| int4 | 1.42 MB | **6.03×** | 1.4984 | +0.3882 |

int8 is the recommended deployment format: it is near-lossless and 3.5× smaller.
int4 is 6× smaller but visibly degrades this 2M model; it is included for
completeness, not as the default. Symmetric per-channel scales have no way to
preserve the small factors at 4 bits. A larger model, group-wise scales, or
mixed-precision factors would be needed to make int4 competitive.

Q2 quantises similarly on TinyStories: int8 is **2.45 MB** with **+0.0004 bpb**
(32-batch evaluation), and int4 is **1.42 MB** with **+0.2193 bpb**. Only the Q2
int8 artifact is shipped; the Q2 int4 file is not competitive.

### 8.8 Sum composition (E14)

E14 keeps E2's geometry and full backprop but replaces the L-fold product with
an additive composition:

```
P = sI + sum_l alpha_l D_l
```

`alpha_l` is learnable per channel and initialised to 1.0. There is no
exponentiation in this path: `s_init` is used directly, unlike the product
where each factor receives `s_init ** (1/L)` and the composed anchor is
`s ** L`. E14 has **190,348** parameters (154,380 in matrices; +240 alpha) and
**11.5 MFLOP/token** at batch 64, versus 3.70 for E8, because the sum
materialises `P` on every forward.

E14 reaches **1.551 bpb** at step 4,000 versus E8's **1.479**: the sum is
**4.9 % worse** while using **3.1×** the FLOPs per token. The rank diagnostic
does exactly what it was designed to do - `eff_rank(P)` rises from 60.3 to
**97.66** - but quality does not follow. The alpha coefficients stay near 1.0
(mean 0.993, std 0.044, range 0.915-1.206), so the channels do not specialise,
and `Var_l σ(θ_l) = 0.00767` remains below the 0.01 H3 threshold.

The conclusion is negative and useful: **rank compression is not the quality
bottleneck at this scale**. The product remains the default; the sum costs
3.1× the FLOPs and does not buy quality. E14 is an ablation, not a shipped
model.

---

## 9. Hypotheses H1–H3: verdicts

| hypothesis | criterion | E2 | verdict |
|---|---|---|---|
| **H1** the Kronecker model is no worse than dense with fewer parameters | `PPL(E2) ≤ 1.2 · PPL(E0)` | 3.159 vs **2.661** (ratio 1.187) | ✅ **confirmed**: 5.1× fewer parameters for 19 % PPL |
| **H2** expressiveness grows with depth | `Spearman(eff_rank(D), layer) > 0.7` | **0.378** | ❌ **not confirmed** |
| **H3** channels split into "fast" and "slow" | `Var_l σ(θ_l) > 0.01` | **0.0039** | ❌ **not confirmed** |

At 2.08M parameters the pattern is unchanged: Q2 has
`Spearman(eff_rank(D), depth) = 0.265` and `Var_l σ(θ_l) = 0.0087`, so H2 and
H3 remain unconfirmed. E8, which removes the stop-gradient, has
`Spearman = 0.238` and `Var_l σ(θ_l) = 0.00248`: full backprop does not change
the verdict either. H1 is still supported by Q2's bpb; D1 supplies the
same-tokenizer dense control.

**H2 is not confirmed, and that is a meaningful result.** The rank of `D` saws
between 89 and 92 with the `K_win=2` period and has no trend, while the rank of
the "non-anchor" part of the composition `P − s^L·Id` actually **decreases**
with depth (Spearman −0.895, 69 → 61). The reason is clear: `P = ∏(Id + D_i)` is
a product of matrices, and a product compresses the spectrum (the product of
singular values), so depth does not accumulate rank but loses it. The RMS
normalisation of `D` (without which the model does not train, §3) also removes
the scale accumulation that could hide this effect.

What the model does instead of growing rank: it **grows the amplitude of the
branch** — `τ` and `γ` increase monotonically towards the last layers (§8.1,
`scales.png`). Depth is used, but through scale rather than through the rank of
the innovation.

**H3 is not confirmed even with `lr_theta = 1e-2`** (10× the original). `σ(θ_l)`
stays in the narrow band 0.40–0.53 around the initialisation value 0.5. The
reason is measurable: after the RMS normalisation `σ` is a dimensionless mixing
coefficient, and `‖∂L/∂θ‖ ≈ 0.02` versus `‖∂L/∂factors‖ ≈ 5`, i.e. ~250×
smaller. Even with a 10× higher learning rate, θ moves by ~0.02 in 2,500 steps,
which is not enough for the channels to diverge.

This is a limitation of the **formulation** (σ multiplies an already normalised
`D`), not of the experiment: for H3 to be testable, `σ` must enter the
innovation before normalisation, e.g. `D = τ·rms_norm(σ·I + prev)`. That
modification changes the geometry of the branch and needs a separate run; it was
not part of the current series.

---

## 10. Tests

`python -m pytest tests -q` → **66 passed, 2 skipped** on CPU (T13/T14 and the
integration runs need CUDA).

| test | what it checks |
|---|---|
| T1 | `kron_apply(A,B,x) == (A⊗B)x == A·X·Bᵀ` (3 sizes + batched) |
| T2 | with `D=0` the layer equals `γ·s^L·x` (the anchor) |
| T3 / T3b | `materialize` yields exactly `s·Id + A⊗B`; matrix mode == sequential |
| **T3c / T3d** | **sum composition** equals `sI + Σ α_l D_l`; the anchor is `s`, not `s^L` |
| T4 | `D_l = I_l + σ·D_{l−1}` (unnormalised); normalised `D` has RMS = τ; `K_win` reset |
| T5 | stop-gradient: `∂L/∂θ` does not flow through `D_{l−1}`; full backprop and `α_l` do |
| T6 | RoPE preserves the norm and depends on position |
| T7 | causal attention (against a manual mask) |
| T8 | weight tying |
| T9 | **budget ≤ 1,000,000** and the matrix parameter share |
| T10 | no NaN/Inf in logits or gradients |
| T11 | overfit a single batch (loss < 0.05) |
| T12 | overfit 100 batches of a Markov task (loss < 0.5) |
| T13 | CUDA memory smoke test at batch=32 |
| T14 | CUDA step-time smoke test at batch=32 |
| T15 | determinism: two runs produce identical losses |
| **T16** | **`KronBank` == the per-module path**: forward and every gradient |
| **T16b / T16c** | **`KronBank` == the sum path**: forward and every gradient |
| **T17** | the ReZero gate is active, differentiable and small at init |
| **T18** | BPE: `decode(encode(x)) == x`, `max(id) < EOT`, 2.5–4 bytes/token |
| **T19** | factorised tied embedding: `V×r + r×d` parameters, `logits == h·U·Eᵀ`, both tensors receive gradients |
| **T20** | QK-norm: per-head RMS = 1, costs exactly `2·d_head·N` parameters, finite forward |
| **T21** | Q1 budget ≤ 1M **and balance**: 80–90 % matrices, 8–16 % embedding |
| **T22** | `bits_per_byte` and corpus selection (`uint8` vs `uint16`) per tokenizer |
| **T23** | the eval-time prelude cache: does not change the output, resets on `train()` |
| **T24** | `CSVLogger` keeps its history when restarted in the same directory |
| **T25** | Q2 budget: 2,079,992 parameters, activation footprint equal to Q1's |
| **T26** | int8/int4 quantisation: half-step error bound, nibble packing, round-trip, smaller checkpoint |
| **T27** | HF wrapper: config round-trip, tied head, `save_pretrained`/`from_pretrained` through `AutoModelForCausalLM`, `generate`, quantized checkpoint |
| **T28** | routed embedding: the effective table equals the softmax-weighted top-k bank mixture |
| **T29** | routed embedding: bank/router gradients, tied effective head, adjustable top-k and cache invalidation |

T13/T14 require CUDA and are skipped without a GPU.

## 11. Compactness and efficiency

| model | parameters | fp32 checkpoint | MFLOP/token | MFLOP/byte |
|---|---|---|---|---|
| E2 (byte) | 190,108 | 0.87 MB | 3.70 | 3.73 |
| E14 (byte, sum) | 190,348 | 0.87 MB | 11.5 | 11.5 |
| Q1 (BPE) | 926,468 | 3.90 MB | 11.2 | 3.66 |
| Q2 (BPE) | 2,079,992 | 8.55 MB | 36.2 | 11.9 |
| Q6 (BPE-4096) | 2,473,208 | 10.13 MB | 37.0 | 9.35 |

The 2.08M model is an **8.55 MB** fp32 file; int8 and int4 bring it to
**2.45 MB** and **1.42 MB** (§8.7). Generation needs no custom kernels: the
Hugging Face wrapper runs Q2 at about **50 tokens/s on CPU** (32 tokens in
0.6 s, batch 1, no KV cache).

Training throughput on the reference GPU (one consumer card, fp32; absolute
times will vary):

| configuration | ms/step | tok/s |
|---|---|---|
| E2, bs=128, seq=256 | 425 | 77k |
| E8, bs=128, seq=256 | 414 | 79k |
| E14, bs=128, seq=256 | 416 | 79k |
| E0 dense, bs=128, seq=256 | 361 | 91k |
| E2 without `KronBank` (per module) | 505 | 65k |
| E2, seq=128 | 187 | 87k |
| E2, 4 heads (d_head=32) | 339 | 96k |
| **Q1, bs=64, seq=192** | **540** | **22.8k** |
| **Q2, bs=32, seq=192** | **1,055** | **5.8k** |
| **Q4, bs=32, seq=192** | **820** | **7.5k** |
| **Q6, bs=32, seq=192** | **869** | **7.1k** |

Q2 is 3.2× more expensive per byte than Q1 — the price of 2.08M parameters at
Q1's activation footprint. In bytes per second Q2 is 3× slower than Q1, but one
training step sees half as many tokens, so Q2's 10k steps are 61M tokens against
Q1's 51.6M.

Generation (batch 1, `T=1.0`, `top_k=8`, no KV cache):

| model | tok/s | bytes/s |
|---|---|---|
| E2 (byte) | 144 | 144 |
| **Q1 (BPE)** | 62 | **189** |
| Q2 (BPE) | 33.7 | 103 |

Q1 has 4.9× the parameters of E2 but is faster in bytes: one token is 3.05
bytes. The Kronecker prelude is cached between steps in eval mode
(`_prelude_cache`); without it the prelude alone would cost ~13 ms/token.

Why the implementation is shaped this way:

* **Kernel-launch overhead.** On Windows/WDDM every launch costs ~45 µs, so
  `KronBank` computes everything batched: ~8 calls per network (`bmm` for the
  innovations + `L−1` `baddbmm` for the composition, one group per signature)
  instead of ~240 small GEMMs. That is worth 15 %.
* **Attention dominates.** With `d_head=16` SDPA is inefficient; 4 heads
  (`d_head=32`) give +20 %, but that deviates from the specification.
* CUDA graphs are not used: capture fails on `baddbmm` in this environment.

---

## 12. Repository layout

```
kronllm/
  kron.py          KronCompose (innovation, recurrence, τ, γ, product/sum) + KronBank
  model.py         KronBlock, KronTransformer, the P-path forward,
                   factorised/routed tied embedding, QK-norm, prelude cache
  tokenizer.py     ByteTokenizer + byte-level BPE, bits_per_byte, corpus encoding
  quantization.py  int8/int4 per-channel post-training quantisation
  hf.py            Hugging Face PreTrainedModel / AutoClass wrapper
  config.py        DEFAULTS + EXPERIMENTS E0-E14, Q1-Q7, D1, S1
  train.py         training loop, param groups, diagnostics, CSV log with bpb
  diagnostics.py   spectral_report (eff_rank D/W/P, cos, τ, γ), σ(θ),
                   routed_report (Q3 usage/rank), gradients
  data.py          memmap corpus (uint8 bytes / uint16 tokens)
  generate.py      sampling through the tokenizer
  plots.py         curves and comparison plots
  cli.py           train / eval / generate / analyze / quantize / budget
  testutils.py     a small config for tests
scripts/
  prepare_data.py      byte-level corpus preparation
  prepare_bpe.py       BPE training and corpus re-tokenisation
  prepare_sft.py       Dolly-15k -> SFT memmaps
  run_experiments.py   the E series + summary.csv
  analyze.py           spectra, heatmaps, rank-vs-depth, τ/γ, samples
  init_routed.py       warm-start the Q3 routed embedding from Q2's table
  dashboard.py         Gradio dashboard
tests/
  test_kron.py         T1–T10, T16/T16b/T16c, T17
  test_integration.py  T11–T15 + generation
  test_quality.py      T18-T25 (tokenizer, budgets, QK-norm, caching, logging)
  test_quantization.py T26 (int8/int4)
  test_hf.py           T27 (Hugging Face wrapper)
  test_routed.py       T28-T29 (routed embedding math, gradients, top-k)
notebooks/
  kronllm_tests.ipynb  tests, plots and generation (T=1.0, top_k=5/8/10)
docs/
  architecture.md      the Kronecker parameterisation and the depth recurrence
  results.md           all measured numbers in one place
artifacts/
  q2_fp32.pt           final Q2 weights
  s1_fp32.pt           Q2 fine-tuned on Dolly-15k
  s1_int8.pt           int8 quantised S1
  s1_int4.pt           int4 quantised S1
```

---

## 13. Limitations

1. **Text quality** is bounded by capacity and data, not by the Kronecker
   parameterisation: E2 (190K, byte level, 0.45 epochs) produces coherent
   fragments and breaks on long-range coherence; Q1 and Q2 remove much of that
   limitation but remain 1–2M parameter models trained on ~10 % of TinyStories.
2. **H2/H3 are not confirmed** — see §9. That is a result, not a bug.
3. **The product composition is rank-compressing, but rank is not the
   bottleneck.** At E2, `eff_rank(P) = 60 < eff_rank(D) = 91`; Q1/Q2 show the
   same pattern, and E8 (full backprop) leaves it at 60.3. E14 trains the
   additive composition: `eff_rank(P)` rises to 97.66, but bpb worsens from
   E8's 1.479 to 1.551 while FLOPs/token grow 3.1×. The product remains the
   default. The orthogonal-anchor fix is not implemented, and the dense-2M and
   E1 controls are not run. Treat the product as an empirical choice, not a
   proven advantage.
4. **No KV cache** at generation time: every new token is a full forward. The
   Kronecker prelude is cached in eval mode (a 3× speed-up), but attention is
   still recomputed from scratch: ~16 ms/token for Q1.
5. **One GPU, fp32.** Neither DDP nor gradient checkpointing was tested.
6. `sigma_theta_report` computes SVDs on CPU tensors for stability — about 2 s
   for 20 layers, and it only runs during evaluation.
7. **S1 is not an assistant.** With 2.08M parameters the fine-tuned model learns
   the question/answer format and a handful of frequent patterns; it hallucinates
   on everything else. It is a demonstration of the pipeline, not a product.

---

## 14. What to do next

| step | why | expected effect |
|---|---|---|
| **Architecture freeze at ~10M** | Q5 (`K=24`) lost 3 % and collapsed layer diversity, so the budget must go to width/depth/context, not more summands | a `B1` config that passes the Phase-0 gate in [docs/roadmap.md](docs/roadmap.md) |
| **Bilingual tokenizer + corpus** | TinyStories is only the development bench; the product corpus is EN–RU | 8K–16K BPE, measured fertility per language, contamination report |
| **Q6 long run (10,000 steps)** | Q6 is at 0.849 after 4,000; Q2 is 0.839 after 10,000, so the shipped comparison is not like-for-like yet | a clean Q6-vs-Q2 curve and a possible new shipped model |
| **Orthogonal anchor** (`O_l` Householder + `D_l`) | keeps `W_l` full-rank and stable while allowing larger `D_l` | tests H2 without the near-identity compromise; higher risk |
| **E1 / dense-2M controls** | isolate attention vs FFN and parameter count | the missing controls behind the current claims |
| **`σ` before normalisation**: `D = τ·rms_norm(σ·Î + prev)` | today `σ` multiplies an already normalised `D`, so its gradient is ~250× smaller than the factors' and H3 is untestable | `σ` starts controlling the scale of memory → `Var σ(θ)` grows by orders of magnitude |
| **KV cache + `torch.compile`** | generation is currently O(T²) per token | ×10–20 sampling speed at no quality cost |
| **Distillation from the 10M base** | a smaller student at a similar bpb | a compact artefact for CPU and edge deployment |
| **Vector gate `γ` per layer** (`γ` is a scalar today) | `γ` learned to 0.05–0.19, i.e. the model already switches layers off | a soft per-layer depth selector |
| **Separate the Q1/Q2 contributions** (tokenizer / capacity / steps) | §8.3–8.4 compare several changes at once | a clean answer to "what exactly produced the bpb" |

E14 ruled out a sum-composition scaling run (4.9 % worse at 3.1× the FLOPs).
Q4 confirmed full backprop (0.9248 vs 0.962 bpb at step 4,000) and Q6 confirmed
the tokenizer (0.8489 at 4,000 steps, 8.2 % better than Q4). Q5 and Q7 are
negative: `K=24` lost 3 % and collapsed layer diversity (`cos_prev_max =
0.99999`), and the Kronecker embedding lost 12 %. The horizon is therefore:
freeze the architecture under 10M, move to the bilingual corpus, and keep
TinyStories only as a sanity bench — see [docs/roadmap.md](docs/roadmap.md).

### Long-term horizon

The plan beyond these ablations — freeze the architecture, build a curated
English–Russian corpus and a shared tokenizer, pretrain at ~10M parameters
inside 6 GB, then structured Q/A SFT and optional distillation — is in
[docs/roadmap.md](docs/roadmap.md); the bilingual mix, source licences and
corpus pipeline are fixed in [docs/data_plan.md](docs/data_plan.md). Each phase
has an explicit gate and kill criteria, so a failed phase sends the project
back instead of silently accumulating compute.

---

## 15. License and citation

MIT — see [LICENSE](LICENSE). If you use this code or the measurements, please
cite the repository:

```bibtex
@software{kronllm2026,
  title  = {KronLLM: a compact, fully Kronecker-parameterised language model},
  author = {KronLLM contributors},
  year   = {2026},
  url    = {https://github.com/Icold21/kron_llm}
}
```
