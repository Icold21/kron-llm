# KronLLM — a fully Kronecker-parameterised language model: 190K → 2.08M parameters on a 6 GB GPU

A decoder-only transformer in which **every weight matrix** (Q, K, V, O, FFN) is
replaced by a Kronecker composition with a depth recurrence. Trained and sampled
on a single GTX 1660 Ti (6 GB, TU116, no tensor cores).

```
Q2: d_model=320 · N=24 · H=8 (d_head=40) · vocab=1024 BPE · seq_len=192
    A (16×16) ⊗ B (20×20) · L=4 channels · K=6 summands
```

| model | total params | matrix params | val bpb (TinyStories) | step |
|---|---|---|---|---|
| **KronLLM Q2** (BPE-1024, 10k steps) | **2 079 992** | **1 890 360** | <!-- Q2_BPB --> | 0.81 s · bs=32 · seq=192 |
| **KronLLM Q1** (BPE-1024, 4.2k steps) | **926 468** | **800 900** | **0.976** | 0.54 s · bs=64 · seq=192 |
| **KronLLM E2** (byte, 2.5k steps) | **190 108** | **154 140** | 1.659 | 425 ms · bs=128 · seq=256 |
| E0 (dense control, 2.5k steps) | 969 856 | 933 888 | 1.412 | 361 ms · bs=128 · seq=256 |

E2 compresses the matrix budget of E0 **6.4×** at the same depth and width.
E0/E2/Q1 fit the original 1M parameter limit; **Q2 deliberately exceeds it**:
the real constraint is the 6 GB of VRAM, not the parameter count (memory is set
by `N·d·batch`, and Q2 occupies the same ~4.2 GB as Q1). See §8.3–8.4.

Also in this repository:

* **D1** — a dense control on the *same* BPE tokenizer and corpus, so that the
  price of the Kronecker parameterisation can be measured in bits per byte
  instead of parameter counts (§8.5).
* **S1** — Q2 fine-tuned on Dolly-15k Q/A pairs as a supervised fine-tuning
  stage (§8.6), and **int8/int4** post-training quantisation that turns the
  2 MB checkpoint into 0.5–1 MB artefacts (§8.7).

---

## 1. What exactly is Kronecker here

Every linear layer `d → d` (and `d → d_kv` for GQA) is not a `128×128` matrix but
`L` "channels", each a sum of `K` Kronecker products:

```
I_l   = Σ_{k=1..K}  A_lk ⊗ B_lk            A_lk ∈ R^{8×8},  B_lk ∈ R^{16×16}
(A ⊗ B) x  =  A · X · Bᵀ,   X = reshape(x, 8, 16)
```

Such a matrix `I_l` holds `n1² + n2² = 64 + 256 = 320` parameters instead of
`d² = 16 384` — 51× fewer — while still being not a low-rank approximation:
`rank(A⊗B) = rank(A)·rank(B)`, which reaches the full `d = 128` for full-rank
factors.

## 2. Depth recurrence

Every layer and every role has **its own** factors `A_lk, B_lk` and its own
`τ_l, θ_l` (otherwise all layers would compute the same function). Kronecker
here is not "one matrix for the whole network" but a **parameterisation of each
matrix**. Layer diversity is further amplified by the state `D` carried across
blocks:

```
D_l^{(i)} = τ_l · rms_norm_ℝ^{d×d} ( Î_l + σ(θ_l) · D̂_l^{(i-1)} )
W_l^{(i)} = s_l · Id + D_l^{(i)}
P^{(i)}   = W_L^{(i)} ⋯ W_1^{(i)}                 (a single d×d matrix)
y         = γ^{(i)} · P^{(i)} x                   (ReZero gate, γ_init = 0.1)
```

* `Î_l`, `D̂` are RMS normalisations over the whole matrix (not per row); E12
  (`normalize_D: false`) removes the normalisation, leaving `τ` as the only
  scale;
* `σ(θ_l)` is a learned "memory speed" per channel, `θ_init = 0`, `σ = 0.5`;
* `stop_gradient` on `D̂` is on by default (as required); E8 turns it off;
* `K_win = 2`: the state resets every 2 layers (recurrence window);
* `s = s_init^{1/L}` — the *effective* anchor value (see deviation #2).

`P` is computed batched for all 60 modules at once (`KronBank`, §12) and applied
as an ordinary `x @ Pᵀ` — 5 matrix multiplications per layer instead of
5 × 4 small matvecs.

### Parameters of one module

| component | formula | count |
|---|---|---|
| factors `A_lk, B_lk` | `L·K·(n1² + n2²)` = 4·2·320 | 2 560 |
| `θ_l` (memory) | `L` | 4 |
| `τ_l` (scale of D) | `L` | 4 |
| `γ` (ReZero) | 1 | 1 |
| **total** | | **2 569** |

60 modules (12 layers × 5 roles) → **154 140**; embedding 256×128 = 32 768;
RMSNorm 25×128 = 3 200; **190 108 in total**.

Q1 (§8.3) uses the same formulas with other numbers: `n1=12, n2=16, K=5, L=4,
d=192`. One module costs `L·K·(n1²+n2²) + 2L + 1 = 4·5·400 + 9 = 8 009`
parameters, 100 modules (20 layers × 5 roles) → **800 900**.
Q2 (§8.4): `n1=16, n2=20, K=6, L=4, d=320` → `4·6·656 + 9 = 15 753` per module,
120 modules → **1 890 360**.

---

## 3. Deviations from the original specification — and why

This is the most important section: without it the architecture from the
specification **does not train at all**. Every item below is a measurement made
on this machine, not a matter of taste.

### 3.1 Root defect: `gain = 1/√d` makes the model the identity

With factors initialised as `gain = 1/√d = 0.088`:

```
‖I‖_RMS ≈ 1e-3   →   W_l = s·Id + D_l ≈ s·Id
P = W_4W_3W_2W_1 ≈ s^L · Id = 0.1 · Id
```

The whole "Kronecker" layer degenerates into multiplication by a constant.
Measured on a real 300-step TinyStories run:

| metric | value | meaning |
|---|---|---|
| `‖∂L/∂θ‖` | **0.000** | the recurrence receives no gradient |
| `cos(W_i, W_{i-1})` | **1.000** | all layers are the same matrix |
| `eff_rank(W)` | **128** (for all) | "full rank" comes from the anchor, not the innovation |
| val PPL | **20.06** | ≈ a unigram model |
| samples | "letter soup" | — |

### 3.2 The naive fix also breaks

With `gain ≈ 1` (spectral norm of a factor ≈ 1) the hidden state grows across
layers as `11 → 226 → 1858 → 6830`, the attention contribution drops to
0.2–3 %, and `std(attn logits) = 0.0098` — the model again ignores context and
only rescales the residual.

### 3.3 What was actually done

| # | specification v2.0 | implemented | reason (measurement) |
|---|---|---|---|
| 1 | `gain = 1/√d` | `D` is RMS-normalised; `gain` is irrelevant in the forward pass (kept as an ablation knob `init_gain`, E12) | see 3.1: otherwise `P ≈ s^L·Id` and `∂L/∂θ = 0` |
| 2 | one `s_init = 0.1` for QKV | per-role `s_init = {q,k,v,o,ffn} = 1.0`, treated as an **effective** value: `s = s_init^{1/L}` | with `s = 0.1` and `D ≈ 1e-3` the matrix is `W = 0.56·Id`, i.e. a layer halves the signal; with `s = 1` the innovation and the anchor are of the same order |
| 3 | — | `τ_l` (learned scale of each `D_l`), init 0.1 | sets `RMS(D)` **directly**; `eff_rank(D)` grew from 1.0 to **91.3/128** |
| 4 | — | ReZero gate `γ` on the branch output, init 0.1 | the branch starts as a small perturbation of the residual; without it the residual grows 600× (3.2) |
| 5 | `s` shared by QKV | its own `s` per role; `s_learnable` honestly switches `Parameter` vs `buffer` | in E2 `s` is a buffer, in E3 a `Parameter` |
| 6 | SiLU = experiment E9 | SiLU at the FFN output is the **default**; E9 became the "linear FFN" ablation | SiLU is standard, the comparison is more useful as a baseline |
| 7 | E0–E10 | added **E11** (gated FFN `(W_a x)*silu(W_b x)`), **E12** (no normalisation of D) and **E13** (`K=8`, ~651K) | E12 prices the normalisation, E11 an alternative to SiLU, E13 a point on the budget→quality curve |
| 8 | `s_init=0.1` + always anchor | E4: `anchor=False` **and** `τ=1.0` | without the anchor `‖P‖ ≈ τ^L`; with τ=0.1 the branch is dead (1e-4) |
| 9 | `amp: fp16` | `amp: none` (fp32) | TU116 has **no tensor cores**: fp16 = 1004 ms/step vs 425 ms fp32 |
| 10 | E0 — plain dense | dense FFN hidden=48 → E0 = 969 856 (not 1.22M) | the hard "≤ 1M" limit is respected by the control too |
| 11 | — | `KronBank`: all 60 compositions computed batched | 439 vs 505 ms/step; prelude = 4.9 ms; mathematically identical (test T16) |
| 12 | seq/batch from the spec | `batch_size=128, seq_len=256` | 77k tok/s is the throughput maximum of this card; VRAM 2.5 GB |
| 13 | CUDA graphs / `torch.compile` | disabled | capture fails on `baddbmm` in this environment |

Items 1–5 are really one mechanism: **the scale must be normalised, not set by
initialisation**. After the fixes `eff_rank(D) = 91`, `cos(W_i, W_{i-1}) = 0.72`
instead of `1.000`, and PPL is what the results table shows.

---

## 4. Parameter budget

```bash
python -m kronllm.cli budget        # prints the table below, exits 1 if > 1M
```

All values ≤ 1 000 000 (the maximum is E0 at 97.0 % of the limit).

| E | what | total | matrix | vs E0 |
|---|---|---|---|---|
| E0 | dense baseline | 969 856 | 933 888 | 1.00× |
| E1 | dense MHA + Kron FFN | 853 228 | 817 260 | 0.88× |
| **E2** | **full Kron (base)** | **190 108** | **154 140** | **0.17×** |
| E3 | + learnable `s` | 190 168 | 154 200 | 0.17× |
| E4 | no anchor, `K_qkv=8` | 282 268 | 246 300 | 0.26× |
| E5 | `d=192, N=16` | 312 208 | 256 720 | 0.27× |
| E6 | `L=6` | 267 148 | 231 180 | 0.25× |
| E7 | `K=4` | 343 708 | 307 740 | 0.33× |
| E8 | no stop-gradient | 190 108 | 154 140 | 0.17× |
| E9 | linear FFN (no SiLU) | 190 108 | 154 140 | 0.17× |
| E10 | GQA (n_kv=2) | 141 580 | 105 612 | 0.11× |
| E11 | gated FFN | 220 936 | 184 968 | 0.20× |
| E12 | no normalisation of D | 190 108 | 154 140 | 0.17× |
| E13 | `K=8` (budget ~651K) | 650 908 | 614 940 | 0.66× |

---

## 5. Data: why TinyStories-200K and not "more"

`data/train.bin` = **180 723 995** bytes = 180.7M tokens (byte level: 1 byte =
1 token). That is **950 tokens per parameter** for the 190K model.

* The Chinchilla optimum for 190K parameters is ≈ 3.8M tokens. We are already
  **47×** past it.
* The full TinyStories (2.1M stories) would give ~1.9B tokens = 10 000 tok/param.
  For a 190K model that buys nothing: **the bottleneck is capacity, not data**.

So 200K stories (10 % of the corpus) is a deliberate choice, not a limitation.
To prepare a different amount:

```bash
python scripts/prepare_data.py --max-stories 500000     # more stories
python scripts/prepare_data.py --max-stories 20000      # faster for experiments
```

The dataset is set in `kronllm/config.py:data.dataset` (any HF dataset with a
`text` field); `max_stories` caps the number of stories.

**If you want better text within 1M parameters, grow the model, not the corpus.**
The Kronecker parameterisation scales with `n1²+n2²`, not with `d²`, so there
are two cheap directions: `K=8` (E13: 651K parameters with almost no step-time
cost) and more width (`d=256, N=24, n1=n2=16` ≈ 570K but 4× the FLOPs).

**For Q1/Q2 (§8.3–8.4) the arithmetic is different:** 59.2M BPE tokens for 926K
parameters is **64 tokens per parameter**, three times the Chinchilla optimum
(20). Q1 was stopped at 4 200 steps (51.6M tokens = 56 tok/param), while Q2 sees
61.4M tokens with 2.08M parameters — **29 tok/param**, i.e. almost optimal. The
next lever is again data: the full corpus (2.1M stories) would give ~620M tokens
= 300 tok/param for Q2.

---

## 6. Running

```bash
# environment
pip install -r requirements.txt          # torch 2.6.0+cu124, gradio, matplotlib, pandas

# data (downloads TinyStories from the HF hub and writes a memmap)
python scripts/prepare_data.py

# BPE tokenizer with 1024 tokens (~3 bytes/token) + corpus re-tokenisation
python scripts/prepare_bpe.py --vocab 1024

# training
python -m kronllm.cli train --exp E2 --steps 2500
python -m kronllm.cli train --exp Q2                 # final 6 GB model, 10k steps
python -m kronllm.cli train --exp Q1                 # the 926K version (4.2k steps here)
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

The whole E series:

```bash
python scripts/run_experiments.py --all --steps 2500      # ~4 hours on a 1660 Ti
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
| `eff_rank_D` | effective rank of the **innovation** `D_l` (singular values > 1 % of max) | 60–110 at `d=128` (was 1.0) |
| `eff_rank_W` | rank of `W = s·Id + D`. With the anchor it is **always ≈ d** — not informative | ≈ d, ignore |
| `rank_excess` | `eff_rank(W) − eff_rank(D)` — how much rank the anchor contributes | > 0 |
| `eff_rank_P` | rank of the composition `P = W_L⋯W_1` | ≈ 60–90 |
| `eff_rank_excess` | rank of `P − s^L·Id` — the "non-anchor" part of the composition | the larger the better |
| `cos_prev_max` | cosine between `W_l` of neighbouring layers | 1.0 = identical layers; 0.7 = distinct |
| `tau_mean`, `D_rms` | scale of D; `D_rms` equals τ by construction | 0.05–0.5 |
| `gate_mean` | ReZero γ | 0.05–1.0 |
| `sigma_theta_var` | `Var_l σ(θ_l)` — hypothesis H3 | > 0.01 = channels specialised |
| `attn_entropy` | attention entropy | ln(256)=5.55 is the maximum; 2–3 means the model uses context |
| `h_norm_ratio` | `‖h_i‖/‖h_{i-1}‖` | ≈ 1; > 5 means the residual explodes |

`theta_grad_norm` in `results.json` is **always 0**: diagnostics run after
`optimizer.zero_grad()`. Per-group gradients are logged to `log.csv` instead
(`grad_embed`, `grad_factors`, `grad_theta`, `grad_tau`, `grad_gate`).

---

## 8. Results

### 8.1 Base model E2 (2 500 steps, 0.45 epochs)

| | |
|---|---|
| parameters | **190 108** (154 140 in matrices, 81 %) |
| val loss / PPL | **1.1502 / 3.159** |
| time | 1107 s (0.443 s/step), VRAM 2.58 GB, 77k tok/s |
| FLOPs | 3.70 MFLOP/token forward (E0: 3.51M — only +5 %) |

Diagnostics after training:

| metric | value | comment |
|---|---|---|
| `eff_rank(D)` | **91.2 / 128** | was 1.0 with `gain=1/√d` |
| `eff_rank(P)` | 60.0 | the composition **compresses** rank |
| `eff_rank(P − s^L·Id)` | **64.0** | > `eff_rank(P)`: the anchor dominates the spectrum |
| `cos(W_i, W_{i−1})` | 0.719 | was 1.000 — layers differ |
| `cos(P_i, P_{i−1})` | 0.083 | compositions of neighbouring layers are almost orthogonal |
| `τ` (D RMS) | 0.118 | learned, grows with depth (0.105 → 0.17) |
| `γ` (ReZero) | 0.115 | grows with depth (0.10 → 0.185) |
| `attn_entropy` | 1.65 | ln(256) = 5.55 — the model looks at context |
| `‖h_i‖/‖h_{i−1}‖` | 1.66 (max 6.6) | the residual does not explode |
| `Var_l σ(θ_l)` | 0.0039 | H3 not confirmed (threshold 0.01) |

Two mechanisms are visible directly in the plots (`runs/E2/analysis/`):

* **`rank_vs_depth.png`** — a sawtooth with period 2: layers where the
  recurrence is reset (`K_win=2`) have `eff_rank(D) ≈ 89`, layers with memory
  ≈ 92. There is no monotonic growth (`Spearman = 0.38`), and
  `eff_rank(P − s^L Id)` even **decreases** with depth (69 → 61). This is the
  H2 verdict.
* **`scales.png`** — `τ` and `γ` grow towards the last layers (FFN: 0.105 →
  0.17 and 0.10 → 0.185). The model compensates for the missing rank growth
  with the **amplitude of the branch**: deeper layers change the residual more.

**Where the model fails** (`worst_tokens.json`, top-20 by loss): the hardest
positions are all **the first letter of a word after a space** (`…their big V`,
`…his old V`, `…day, the V`), loss 10.4–10.8 nats. That is the structural price
of a byte-level vocabulary: the first letter of an English word is almost
unpredictable from context, and the model has to keep a distribution over 26+
options. No architecture fixes this — only BPE/word-level tokenisation does.

### 8.2 The E0–E13 series: why it was stopped

The plan was 14 runs, E0–E13. After E2 it became clear that they differ by 1–3
hyperparameters and change PPL by ±5–10 %: that answers "which of nearly
identical configurations is slightly better", not "how to get good text". The
time budget went into the quality branch Q: first **Q1** (BPE + 926K, §8.3),
then **Q2** — the same architecture with 2.08M parameters under the 6 GB VRAM
ceiling (§8.4).

| run | parameters | matrix | val loss | val PPL (token) | **val bpb** | steps |
|---|---|---|---|---|---|---|
| **E0** (dense, bytes) | 969 856 | 933 888 | 0.9789 | 2.661 | **1.412** | 2 500 |
| **E2** (Kron, bytes) | 190 108 | 154 140 | 1.1502 | 3.159 | **1.659** | 2 500 |
| **Q1** (Kron, BPE) | 926 468 | 800 900 | 2.0735 | 7.953 | **0.976** | 4 200 |
| **Q2** (Kron, BPE) | 2 079 992 | 1 890 360 | <!-- Q2_LOSS --> | <!-- Q2_PPL --> | **<!-- Q2_BPB -->** | 10 000 |

`val loss`/`PPL` in this table are in **different units** (nats/byte for E0/E2,
nats/token for Q1/Q2), so only the last column can be compared across rows.

`python scripts/run_experiments.py --rebuild-summary` builds
`results/summary.csv` from `results/E*.json`; single runs are started as
`python -m kronllm.cli train --exp E7 --run-dir runs/E7`.

### 8.3 The quality branch Q1: "three bytes at a time", but learned

E2 is trained on bytes for 0.45 epochs: it genuinely learns spelling, but a word
costs 4–5 predictions and the most expensive one is the first letter after a
space (§8.1). The "three bytes at a time" idea addresses exactly that. It is
implemented not as a fixed chunking (a `256³ = 16.7M` vocabulary is impossible
within 1M parameters) but as a **byte-level BPE with 1024 tokens** trained on
the same corpus:

| | train | val |
|---|---|---|
| bytes (without separators) | 180 523 995 | 8 300 677 |
| tokens | 59 237 168 | 2 707 417 |
| **bytes/token** | **3.05** | **3.07** |

One prediction therefore covers ~3 bytes, and the boundaries are chosen by the
merges: `the`, `ing`, `and`, ` once` become single ids, rare triples stay split.
`EOT = 1023` is reserved and never produced by the merges. The corpus is the
same (same files, same validation split), so the comparison metric is
**bits per byte**:

$$\mathrm{bpb} = \frac{\mathcal{L}_{\text{nats/token}}}{\text{bytes/token}\cdot\ln 2}.$$

**Q1's budget is 926 468 parameters, and it is balanced:**

| block | parameters | share | what |
|---|---|---|---|
| Kronecker matrices | 800 900 | 86.4 % | 20 layers × 5 roles × (4 channels × 5 summands × 400 + 9) |
| embedding | 116 736 | 12.6 % | factorised tied: `1024×96 + 96×192` instead of `1024×192` |
| norms | 8 832 | 1.0 % | pre-norm RMSNorm + QK-norm |

The factorised embedding (ALBERT-style) saves 80K parameters, and the head is
**free**: $\text{logits} = h\,U^\top E^\top$, i.e. the same weights read in the
opposite direction.

**Configuration:** `d=192 · N=20 · H=8 (d_head=24) · K=5 · L=4 · n1=12 · n2=16 ·
seq_len=192 · batch=64 · lr=5e-4`. The plan was 10 000 steps (123M tokens ≈ 2.1
epochs), but the run was stopped at **4 200 steps** (51.6M tokens ≈ 0.87
epochs): the remaining ~55 minutes of GPU time went to Q2 — a model that spends
three times more compute per byte of text and therefore has a higher quality
ceiling.

**Intra-layer normalisation.** Besides the usual pre-norm, **QK-norm** is added
— an RMSNorm on `q` and `k` per head before RoPE. It keeps the attention
operands on the same scale regardless of how the Kronecker branch has grown,
removes logit drift and allows `lr=5e-4` without divergence. It costs
`2·d_head` parameters per layer (0.04 % of the budget).

**FLOP cost.** 11.2 MFLOP/token at 3.05 bytes/token = **3.66 MFLOP/byte**
against 3.73 for E2 — that is 4.9× more parameters at the same compute per byte
of text.

**Q1 result** (best checkpoint, step 4 000, val): loss 2.0735 nats/token →
**PPL 7.95**, **bpb 0.976**. That is **41 %** fewer bits per byte than E2
(1.659) and **31 %** fewer than the dense control E0 (1.412) at almost the same
parameter count (926K vs 970K). The curve was still falling at step 4 200
(~0.005 bpb per 500 steps), so Q1 is not a converged model but one stopped by
the time budget.

**Honest caveat.** Q1 differs from E2 in three things at once (tokenisation,
capacity, number of steps), so the bpb gain cannot be attributed to any single
one of them. Separating the contributions needs dedicated runs; the point here
is that the final quality per byte is reached within the same FLOP budget and
under the 1M parameter limit.

### 8.4 Q2: the final model — the limit is 6 GB, not 1M

Once quality became more important than the parameter count, the 1M limit was
lifted. But "just make the model bigger" does not work: on 6 GB the constraint
is not the weight memory (2M parameters with Adam is ~32 MB) but the
**activations**, and they grow as `N · d · batch · seq`.

Q2 is therefore built to **keep Q1's activation footprint** and spend the freed
parameters on width and depth:

| | Q1 | Q2 |
|---|---|---|
| `d` / `N` | 192 / 20 | **320 / 24** |
| `K` / `L` | 5 / 4 | **6 / 4** |
| `n1 × n2` | 12 × 16 | **16 × 20** |
| parameters | 926 468 | **2 079 992** |
| `N · d · batch` | 20·192·64 = 245 760 | 24·320·32 = **245 760** |
| MFLOP/token | 11.2 | **36.2** |
| MFLOP/byte | 3.66 | **11.9** |
| peak VRAM | ~3.5 GB | **4.22 GB** |

The activation footprint matches to the last unit, so Q2 fits the same 6 GB —
measured peak 4.22 GB (including eval). The price is **3.2× compute per byte**:
a step takes 0.81 s instead of 0.54 s with half the batch size.

Q2's budget (2 079 992 parameters):

| block | parameters | share |
|---|---|---|
| Kronecker matrices | 1 890 360 | 90.9 % |
| embedding (tied, `r=128`) | 172 032 | 8.3 % |
| norms (pre-norm + QK-norm) | 17 600 | 0.85 % |

**Configuration:** `d=320 · N=24 · H=8 (d_head=40) · K=6 · L=4 · n1=16 · n2=20 ·
r=128 · seq_len=192 · batch=32 · lr=4e-4 · 10 000 steps` = 61.4M tokens ≈ 1.04
epochs of the corpus.

<!-- Q2_RESULTS -->

**Honest caveat.** Q2 differs from Q1 both in parameters (2.2×) and in the
number of tokens seen (0.5×), so it is not a controlled "more parameters at
equal compute" experiment but a choice in favour of quality at a fixed GPU time.
The clean comparison "same architecture, equal compute per byte" is Q1 against
E2 in §8.3.

### 8.5 D1: the price of Kronecker, measured on the same tokenizer

Q1 vs E2 changes tokenisation, capacity and step count at once, so neither of
them prices the Kronecker parameterisation itself. D1 closes that gap: it is
the dense E0 recipe (`use_kron_qkv=False`, `use_kron_ffn=False`,
`dense_ffn_mult=0.375`, QK-norm on) trained on the **same BPE tokenizer, the
same corpus and the same number of steps as Q1**.

| | D1 (dense) | Q1 (Kronecker) |
|---|---|---|
| parameters | 1 068 544 | 926 468 |
| `d` / `N` | 128 / 12 | 192 / 20 |
| val loss (nats/token) | <!-- D1_LOSS --> | 2.0735 |
| val bpb | <!-- D1_BPB --> | **0.976** |
| steps | 2 500 | 4 200 |

<!-- D1_RESULTS -->

**Caveat.** D1 matches Q1 in tokenizer and step count but not in step count per
byte of text (dense steps are cheaper), so the comparison is "same data, same
tokens per step" rather than "same wall-clock time".

### 8.6 S1: supervised fine-tuning on Dolly-15k

A 2M-parameter model cannot become a general assistant. What it *can* do is
learn the Q/A **format** and the closed-vocabulary questions that dominate a
small fine-tuning set. S1 is the honest version of that experiment:

* Data: [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
  15 011 human-written instruction/response pairs. The Q/A-style categories
  (`closed_qa`, `open_qa`, `general_qa`) are kept: 7 706 rows, of which 4 882
  fit the 192-token context (2 824 answer/context pairs are longer and are
  dropped; 1 711 examples are shortened by removing the context paragraph).
* Template (also used at inference time):

  ```
  ### Question:
  {instruction}
  ### Answer:
  {response}<eot>
  ```

* Training: the Q2 weights are loaded (`training.init_from`), the optimizer and
  schedule are fresh (400 steps, lr 1e-4, warmup 40, batch 32), the validation
  split is 5 % of the pairs.

```bash
python scripts/prepare_sft.py     # writes data/sft_train.bpe.bin / sft_val.bpe.bin
python -m kronllm.cli train --exp S1
python -m kronllm.cli generate --run runs/S1 \
    --prompt "### Question:\nWhat is the capital of France?\n### Answer:"
```

<!-- S1_RESULTS -->

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

<!-- QUANT_RESULTS -->

---

## 9. Hypotheses H1–H3: verdicts

| hypothesis | criterion | E2 | verdict |
|---|---|---|---|
| **H1** the Kronecker model is no worse than dense with fewer parameters | `PPL(E2) ≤ 1.2·PPL(E0)` | 3.159 vs **2.661** (ratio 1.187) | ✅ **confirmed**: 5.1× fewer parameters for 19 % PPL |
| **H2** expressiveness grows with depth | `Spearman(eff_rank(D), layer) > 0.7` | **0.378** | ❌ **not confirmed** |
| **H3** channels split into "fast" and "slow" | `Var_l σ(θ_l) > 0.01` | **0.0039** | ❌ **not confirmed** |

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
smaller. Even with a 10× higher learning rate, θ moves by ~0.02 in 2 500 steps,
which is not enough for the channels to diverge.

This is a limitation of the **formulation** (σ multiplies an already normalised
`D`), not of the experiment: for H3 to be testable, `σ` must enter the
innovation *before* normalisation, e.g. `D = τ·rms_norm(σ·I + prev)`. That
modification changes the geometry of the branch and needs a separate run; it was
not part of the current series.

---

## 10. Tests

`python -m pytest tests -q` → 52 tests (T13/T14 and the integration runs need
CUDA).

| test | what it checks |
|---|---|
| T1 | `kron_apply(A,B,x) == (A⊗B)x == A·X·Bᵀ` (3 sizes + batched) |
| T2 | with `D=0` the layer equals `γ·s^L·x` (the anchor) |
| T3 / T3b | `materialize` yields exactly `s·Id + A⊗B`; matrix mode == sequential |
| T4 | `D_l = I_l + σ·D_{l-1}` (unnormalised); normalised `D` has RMS = τ; `K_win` reset |
| T5 | stop-gradient: `∂L/∂θ` does not flow through `D_{l-1}`; and optionally does |
| T6 | RoPE preserves the norm and depends on position |
| T7 | causal attention (against a manual mask) |
| T8 | weight tying |
| T9 | **budget ≤ 1 000 000** and the matrix parameter share |
| T10 | no NaN/Inf in logits or gradients |
| T11 | overfit a single batch (loss < 0.05) |
| T12 | overfit 100 batches of a Markov task (loss < 0.5) |
| T13 | VRAM < 5.5 GB at batch=32 |
| T14 | step < 160 ms at batch=32 |
| T15 | determinism: two runs produce identical losses |
| **T16** | **`KronBank` == the per-module path**: forward and every gradient |
| **T17** | the ReZero gate is active, differentiable and small at init |
| **T18** | BPE: `decode(encode(x)) == x`, `max(id) < EOT`, 2.5–4 bytes/token |
| **T19** | factorised tied embedding: `V×r + r×d` parameters, `logits == h·U·Eᵀ`, both tensors receive gradients |
| **T20** | QK-norm: per-head RMS = 1, costs exactly `2·d_head·N` parameters, finite forward |
| **T21** | Q1 budget ≤ 1M **and balance**: 80–90 % matrices, 8–16 % embedding |
| **T22** | `bits_per_byte` and corpus selection (`uint8` vs `uint16`) per tokenizer |
| **T23** | the eval-time prelude cache: does not change the output, resets on `train()` |
| **T24** | `CSVLogger` keeps its history when restarted in the same directory |
| **T25** | Q2 budget: 2 079 992 parameters, activation footprint equal to Q1's |
| **T26** | int8/int4 quantisation: half-step error bound, nibble packing, round-trip, smaller checkpoint |

T13/T14 require CUDA and are skipped without a GPU.

---

## 11. Performance on a GTX 1660 Ti (TU116)

| configuration | ms/step | tok/s |
|---|---|---|
| E2 fp32, bs=128, seq=256 | **425** | 77k |
| E2 fp16 autocast | 1004 | 33k |
| E2 without `KronBank` (per module) | 505 | 65k |
| E0 dense fp32 | 361 | 91k |
| E2, seq=128 | 187 | 87k |
| E2, 4 heads (d_head=32) | 339 | 96k |
| **Q1 fp32, bs=64, seq=192** | **540** | **22.8k** |
| **Q2 fp32, bs=32, seq=192** | **810** | **7.6k** |

Q2 is 3.2× more expensive per byte of text than Q1 — that is the price of 2.08M
parameters at the same activation memory. In bytes per second Q2 is 3× slower
than Q1, but one training step sees half as many tokens, so Q2's 10k steps are
61M tokens against Q1's 51.6M.

Generation (batch 1, `T=1.0`, `top_k=8`, no KV cache):

| model | tok/s | bytes/s |
|---|---|---|
| E2 (byte) | 144 | 144 |
| **Q1 (BPE)** | 62 | **189** |

Q1 has 4.9× the parameters but is **faster in bytes**: one token is 3.05 bytes.
The Kronecker prelude is cached between steps in eval mode (`_prelude_cache`);
without it the prelude alone would cost ~13 ms/token.

Why:

* **No tensor cores.** fp16 on TU116 runs on ordinary CUDA cores and adds casts;
  fp32 is therefore 2.4× faster. `amp` defaults to `none`.
* **WDDM.** Every kernel launch costs ~45 µs, so `KronBank` computes everything
  batched: ~8 calls per network (`bmm` for the innovations + `L−1` `baddbmm`
  for the composition, one group per signature) instead of ~240 small GEMMs.
  That is worth 15 %.
* **Attention dominates.** With `d_head=16` SDPA is inefficient; 4 heads
  (`d_head=32`) give +20 %, but that deviates from the spec.
* CUDA graphs are not used: capture fails on `baddbmm` in this environment.

---

## 12. Repository layout

```
kronllm/
  kron.py          KronCompose (innovation, recurrence, τ, γ) + KronBank
  model.py         KronBlock, KronTransformer, the P-path forward,
                   factorised tied embedding, QK-norm, prelude cache
  tokenizer.py     ByteTokenizer + byte-level BPE, bits_per_byte, corpus encoding
  quantization.py  int8/int4 per-channel post-training quantisation
  config.py        DEFAULTS + EXPERIMENTS E0–E13, Q1, Q2, D1, S1
  train.py         training loop, param groups, diagnostics, CSV log with bpb
  diagnostics.py   spectral_report (eff_rank D/W/P, cos, τ, γ), σ(θ), gradients
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
  dashboard.py         Gradio dashboard
tests/
  test_kron.py         T1–T10, T16, T17
  test_integration.py  T11–T15 + generation
  test_quality.py      T18–T25 (tokenizer, budgets, QK-norm, caching, logging)
  test_quantization.py T26 (int8/int4)
notebooks/
  kronllm_tests.ipynb  tests, plots and generation (T=1.0, top_k=5/8/10)
docs/
  architecture.md      the Kronecker parameterisation and the depth recurrence
  results.md           all measured numbers in one place
```

---

## 13. Limitations

1. **Text quality** is bounded by capacity and data, not by the Kronecker
   parameterisation: E2 (190K, byte level, 0.45 epochs) produces coherent
   fragments and breaks on long-range coherence; Q1 and Q2 remove much of that
   limitation but remain 1–2M parameter models trained on ~10 % of TinyStories.
2. **H2/H3 are not confirmed** — see §9. That is a result, not a bug.
3. **No KV cache** at generation time: every new token is a full forward. The
   Kronecker prelude is cached in eval mode (a 3× speed-up), but attention is
   still recomputed from scratch: ~16 ms/token for Q1.
4. **One GPU, fp32.** Neither DDP nor gradient checkpointing was tested. At
   `d=192/N=20` a batch of 96 does not fit in 6 GB — the working batch is 64;
   for Q2 (`d=320/N=24`) it is 32 (peak 4.22 GB).
5. `sigma_theta_report` computes SVDs on CPU tensors for stability — about 2 s
   for 20 layers, and it only runs during evaluation.
6. **S1 is not an assistant.** With 2.08M parameters the fine-tuned model learns
   the question/answer format and a handful of frequent patterns; it hallucinates
   on everything else. It is a demonstration of the pipeline, not a product.

---

## 14. What to do next (driven by results, not by the plan)

| step | why | expected effect |
|---|---|---|
| **`σ` before normalisation**: `D = τ·rms_norm(σ·Î + prev)` | today `σ` multiplies an already normalised `D`, so its gradient is ~250× smaller than the factors' and H3 is untestable | `σ` starts controlling the *scale* of memory → `Var σ(θ)` grows by orders of magnitude |
| **More data** (200K → 2.1M TinyStories stories) | Q2 sees 61M tokens ≈ 1 epoch, while the full corpus is ~620M tokens | the main remaining quality lever after Q2: expected −0.04…−0.08 bpb |
| **KV cache + `torch.compile`** | generation is currently O(T²) per token | ×10–20 sampling speed at no quality cost |
| **Distillation from Q2 into 500–700K** | 2.08M → ~600K at a similar bpb | a compact artefact for weak hardware |
| **Vector gate `γ` per layer** (`γ` is a scalar today) | `γ` learned to 0.05–0.19, i.e. the model already switches layers off | a soft per-layer depth selector |
| **Separate the Q1/Q2 contributions** (tokenizer / capacity / steps) | §8.3–8.4 compare several changes at once | a clean answer to "what exactly produced the bpb" |

---

## 15. License and citation

MIT — see [LICENSE](LICENSE). If you use this code or the measurements, please
cite the repository:

```bibtex
@software{kronllm2026,
  title  = {KronLLM: a fully Kronecker-parameterised language model on a 6 GB GPU},
  author = {KronLLM contributors},
  year   = {2026},
  url    = {https://github.com/your-name/kronllm}
}
```
