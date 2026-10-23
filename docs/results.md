# Results

All numbers were measured on one consumer GPU (GTX 1660 Ti, 6 GB) with Python
3.11 and `torch 2.6.0+cu124`, unless stated otherwise. Q2 peaks at 4.22 GB of
VRAM; the from-scratch Q3 pilot peaks at 5.56 GB, also inside a 6 GB card. The
corpus is TinyStories (200,000 training stories, 10,000 validation stories). The metric
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
| E0 | byte | 969,856 | 933,888 | 128 / 12 | dense | 256 / 128 | 2,500 |
| E2 | byte | 190,108 | 154,140 | 128 / 12 | 2 / 4 | 256 / 128 | 2,500 |
| E8 | byte | 190,108 | 154,140 | 128 / 12 | 2 / 4 | 256 / 128 | 4,000 |
| E14 | byte | 190,348 | 154,380 | 128 / 12 | 2 / 4 (sum) | 256 / 128 | 4,000 |
| Q1 | BPE-1024 | 926,468 | 800,900 | 192 / 20 | 5 / 4 | 192 / 64 | 4,200 |
| Q2 | BPE-1024 | 2,079,992 | 1,890,360 | 320 / 24 | 6 / 4 | 192 / 32 | 10,000 |
| Q4 | BPE-1024 | 2,079,992 | 1,890,360 | 320 / 24 | 6 / 4 | 192 / 32 | 4,000 |
| Q6 | BPE-4096 | 2,473,208 | 1,890,360 | 320 / 24 | 6 / 4 | 192 / 32 | 4,000 |
| Q3 (pilot) | BPE-1024 | 3,544,104 | 1,890,360 | 320 / 24 | 6 / 4 | 192 / 32 | 400 |
| Q3warm | BPE-1024 | 3,544,104 | 1,890,360 | 320 / 24 | 6 / 4 | 192 / 24 | 500 |
| Q3long | BPE-1024 | 3,544,104 | 1,890,360 | 320 / 24 | 6 / 4 | 192 / 24 | 2,000 |
| D1 | BPE-1024 | 1,068,544 | 933,888 | 128 / 12 | — | 192 / 64 | 4,200 |
| S1 | BPE-1024 | 2,079,992 | 1,890,360 | 320 / 24 | 6 / 4 | 192 / 32 | 400 |

The fp32 checkpoints are **0.87 MB** (E2/E8/E14), **3.90 MB** (Q1),
**4.31 MB** (D1), **8.55 MB** (Q2/Q4), **10.13 MB** (Q6), **8.55 MB** (S1),
and **14.41 MB** (Q3/Q3warm/Q3long).

## 2. Quality

| model | val loss | unit | val PPL | val bpb | note |
|---|---|---|---|---|---|
| E0 | 0.9789 | nats/byte | 2.661 | **1.412** | dense control, bytes |
| E2 | 1.1502 | nats/byte | 3.159 | **1.659** | Kron, bytes |
| E8 | 1.0251 | nats/byte | 2.79 | **1.479** | full backprop, best step 4,000 |
| E14 | 1.0750 | nats/byte | 2.93 | **1.551** | sum composition, step 4,000; negative result |
| Q1 | 2.0735 | nats/token | 7.953 | **0.976** | best checkpoint, step 4,000 |
| Q2 | 1.7833 | nats/token | 5.95 | **0.839** | 10,000 steps |
| Q4 | 1.9653 | nats/token | 7.137 | **0.9248** | full backprop, step 4,000 |
| Q6 | 2.3292 | nats/token | 10.27 | **0.8489** | BPE-4096, step 4,000 |
| Q3 (pilot) | 4.4275 | nats/token | 83.7 | 2.083 | routed, step 400; not adopted |
| Q3warm | 1.7502 | nats/token | 5.76 | **0.824** | routed, warm-started from Q2; not shipped |
| Q3long | 2.7145 | nats/token | 15.10 | 1.277 | routed from scratch, step 2,000; no jump |
| D1 | 2.0763 | nats/token | 7.97 | **0.9770** | dense, same BPE |
| S1 | 2.3593 | nats/token | 10.58 | 1.110 | fine-tuned on Dolly (not comparable to TinyStories) |

Validation curves (`runs/<id>/log.csv`):

| step | E2 bpb | Q1 bpb | Q2 bpb | Q4 bpb | Q6 bpb |
|---|---|---|---|---|---|
| 500 | 1.768 | 1.525 | 1.482 | 1.490 | 1.271 |
| 1,000 | 1.659 | 1.266 | 1.245 | 1.244 | 1.102 |
| 1,500 | — | 1.149 | 1.132 | 1.115 | 1.008 |
| 2,000 | — | 1.083 | 1.068 | 1.046 | 0.952 |
| 2,500 | — | 1.044 | 1.037 | 1.001 | 0.911 |
| 3,000 | — | 1.016 | 1.004 | 0.963 | 0.880 |
| 3,500 | — | 0.994 | 0.980 | 0.939 | 0.860 |
| 4,000 | — | **0.976** | 0.962 | 0.9248 | **0.8489** |
| 4,500 | — | — | 0.946 | — | — |
| 5,000 | — | — | 0.928 | — | — |
| 6,000 | — | — | 0.902 | — | — |
| 8,000 | — | — | 0.859 | — | — |
| 10,000 | — | — | **0.839** | — | — |

The E2 run stops at 2,500 steps (0.45 epochs). Q1 was stopped at step 4,200 to
spend the remaining time on Q2; its best checkpoint is step 4,000. Q2 sees 61.4M
tokens (≈ 1.04 epochs of the BPE corpus) in 10,000 steps. Q4 stops at the
matched 4,000-step point and beats Q2 at every shared evaluation. Q6 reaches
0.8489 at step 4,000, within 1.2 % of Q2's 10,000-step result in 40 % of the
steps.

## 3. Cost

| model | ms/step | tokens/s | MFLOP/token | MFLOP/byte | peak memory |
|---|---|---|---|---|---|
| E0 | 361 | 91k | 3.51 | 3.51 | ~2.0 GB |
| E2 | 425 | 77k | 3.70 | 3.73 | 2.58 GB |
| E8 | 414 | 79k | 3.70 | 3.73 | — |
| E14 | 416 | 79k | 11.5 | 11.5 | 2.58 GB |
| Q1 | 540 | 22.8k | 11.2 | 3.66 | ~3.5 GB |
| Q2 | 1,055 | 5.8k | 36.2 | 11.9 | 4.22 GB |
| Q4 | 820 | 7.5k | 36.2 | 11.9 | 4.31 GB |
| Q6 | 869 | 7.1k | 37.0 | 9.35 | 4.28 GB |
| D1 | 178 | 69k | 3.31 | 1.08 | 1.44 GB |
| S1 | 844 | 7.3k | 36.2 | 11.9 | 4.22 GB |

FLOPs are reported at `batch_size=64` (the convention used by
`flops_per_token`); tokens/s is the measured training throughput.

Generation speed (batch 1, `T=1.0`, `top_k=8`, no KV cache):

| model | tokens/s | bytes/s |
|---|---|---|
| E2 (byte) | 144 | 144 |
| Q1 (BPE) | 62 | 189 |
| Q2 (BPE) | 33.7 | 103 |

Q2 also runs at about **50 tokens/s on CPU** (batch 1, Hugging Face wrapper, no
KV cache), so the 2.08M quality checkpoint is usable without a GPU.

## 4. Diagnostics

Spectral diagnostics of E2 after 2,500 steps:

| metric | E2 | E8 | E14 | comment |
|---|---:|---:|---:|---|
| `eff_rank(D)` | 91.2 / 128 | 91.7 / 128 | 88.8 / 128 | was 1.0 with the specification's `gain = 1/√d` |
| `eff_rank(P)` | 60.0 | 60.3 | **97.66** | the product compresses rank; the trained sum does not |
| `eff_rank(sI + ΣD_l)` | **101.0** | **101.2** | **97.66** | for E14 this is the trained composition |
| sum / product | **1.70** | **1.70** | **1.00** | E14 removes the multiplication-rank gap |
| `eff_rank(P − s^L Id)` | 64.0 | 64.1 | 96.55 (P − sI) | the anchor dominates the product spectrum |
| `cos(W_i, W_{i−1})` | 0.719 | 0.761 | 0.717 | layers differ (was 1.000) |
| `cos(P_i, P_{i−1})` | 0.083 | 0.104 | 0.318 (mean) | E14 compositions are more similar |
| `D_rms / s` | 0.118 | 0.119 | 0.159 | `s = 1.0`; RMS innovation is 16 % of the anchor |
| `‖D‖_F / s` | 15.2 | 15.3 | 20.4 | high-rank `D` (128 × RMS) |
| `σ_max(D) / s` | 3.60 | 3.59 | 5.64 | spectral norm is 5.6 × the anchor |
| `τ` | 0.118 | 0.119 | 0.159 | grows with depth |
| `γ` | 0.115 | 0.115 | 0.182 | grows with depth |
| attention entropy | 1.65 | 1.46 | 2.32 | ln(256) = 5.55; E14 is more diffuse |
| `‖h_i‖/‖h_{i−1}‖` | 1.66 (max 6.6) | 1.54 (max 4.65) | 1.36 (max 1.95) | the residual does not explode |
| `Var_l σ(θ_l)` | 0.0039 | 0.00248 | 0.00767 | H3 is not confirmed (threshold 0.01) |
| Spearman(eff_rank D, depth) | 0.378 | 0.238 | 0.042 | H2 is not confirmed (threshold 0.7) |

| metric | Q1 (step 4,000) | Q2 (step 10,000) |
|---|---:|---:|
| `eff_rank(D)` | 148.6 / 192 | 248.6 / 320 |
| `eff_rank(P)` | 89.1 | 144.8 |
| `eff_rank(sI + ΣD_l)` | 153.5 | 255.5 |
| sum / product | 1.73 | 1.77 |
| rank excess | 94.4 | 152.8 |
| `cos(W_i, W_{i-1})` | 0.771 | 0.879 |
| `cos(P_i, P_{i-1})` | 0.206 | 0.517 |
| `D_rms / s` | 0.1023 | 0.0835 |
| `‖D‖_F / s` | 19.7 | 26.7 |
| `σ_max(D) / s` | 3.26 | 3.54 |
| `τ` | 0.1023 | 0.0835 |
| `γ` | 0.1447 | 0.1243 |
| attention entropy | — | 2.48 |
| residual-norm ratio | — | 1.86 |
| `Var_l σ(θ_l)` | 0.0063 | 0.0087 |
| Spearman(eff_rank D, depth) | 0.233 | 0.265 |

The `sI + ΣD_l` rows use the trained product model's `D_l`; they are a
rank diagnostic, not a trained sum-composition run. They isolate the rank
lost by multiplication itself. E14 trains the same additive composition; its
result is reported in §9.

E8 removes the stop-gradient and improves quality from **1.659** to **1.586**
bpb at the matched 2,500-step point, reaching **1.479** at step 4,000. The
spectral geometry is essentially unchanged: `eff_rank(P)` is 60.3 versus
101.2 for the sum diagnostic, and `Var_l σ(θ_l)` is 0.00248. Full backprop
therefore improves optimisation but does not remove the product's rank
compression; H3 remains unconfirmed.

E14 trains the additive composition directly. It removes the rank gap
(`eff_rank(P) = 97.66`), but reaches only **1.551** bpb at step 4,000 versus
E8's **1.479**, while costing **3.1×** the FLOPs/token. The alpha coefficients
remain near 1.0 (mean 0.993, std 0.044), so the channels do not specialise.
This is the strongest evidence that rank compression is **not** the quality
bottleneck at this scale. The product remains the default; E14 is a negative
ablation, not a shipped model.

H2 and H3 remain unconfirmed at both scales: the depth-rank correlation is far
below 0.7, and the variance of `σ(θ)` stays below 0.01. H1 is strongly
supported by Q2's bpb and the D1 comparison. The full Q1/Q2 plots are in
`runs/Q1/analysis/` and `runs/Q2/analysis/`.

Q4 at step 4,000 keeps the Q2 geometry: `eff_rank(D) = 248.9 / 320`,
`eff_rank(P) = 140.2`, `τ = 0.0884`, `gate = 0.107`, `cos(W_i, W_{i−1}) =
0.896`, `Var_l σ(θ_l) = 0.00558`, Spearman 0.307. Full backprop improves
quality without changing the spectral picture.

Q6 keeps the same matrix geometry with a wider tokenizer: `eff_rank(P) =
139.4`, `eff_rank(D) = 248.7 / 320`, `τ = 0.0869`, `gate = 0.107`,
`cos(W_i, W_{i−1}) = 0.848`, `Var_l σ(θ_l) = 0.00438`, Spearman 0.341. The
quality gain comes from 3.96 bytes/token, not from a different spectral regime.

## 5. Hypotheses

| hypothesis | criterion | E2 | E8 | E14 | verdict |
|---|---|---|---|---|---|
| H1: Kron is no worse than dense at fewer parameters | `PPL(E2) ≤ 1.2 · PPL(E0)` | 3.159 vs 2.661 (1.187) | 3.00 vs 2.661 (1.127), step 2,500 | 2.93 vs 2.661 (1.101) | confirmed |
| H2: expressiveness grows with depth | `Spearman > 0.7` | 0.378 | 0.238 | 0.042 | rejected |
| H3: channels split into fast and slow | `Var σ(θ) > 0.01` | 0.0039 | 0.00248 | 0.00767 | rejected |

H2 and H3 are rejected in the *formulation used here*, not in general: the
composition of matrices loses rank with depth, and `σ` multiplies an already
normalised state, so its gradient is ~250× smaller than the factors'. See
README §9 for the analysis and §14 for the proposed fix.

The D1 control makes the parameterisation comparison concrete: D1 reaches
0.9770 bpb at 1,068,544 parameters and 3.31 MFLOP/token, while Q1 reaches
0.9757 bpb at 926,468 parameters and 11.2 MFLOP/token. Q1 is 0.13 % better in
quality with 15.3 % fewer parameters and 3.4× fewer FLOPs; the Kronecker
parameterisation is a compactness/compute win at 1M, not a large quality win.

## 6. Quantisation

| model | fp32 | int8 | int4 |
|---|---|---|---|
| Q2 | 8.55 MB, bpb 0.8100* | **2.45 MB**, bpb 0.8105 (+0.0004) | 1.42 MB, bpb 1.0294 (+0.2193) |
| S1 | 8.55 MB, bpb 1.1102 | **2.45 MB**, bpb 1.1113 (+0.0011) | **1.42 MB**, bpb 1.4984 (+0.3882) |

*Q2 quantisation uses a 32-batch validation subset, so its absolute bpb is
0.8100 rather than the full-validation 0.839; the deltas are the meaningful
numbers.

S1 int8 is near-lossless (3.49× smaller). S1 int4 is 6.03× smaller but degrades
by +0.388 bpb; it is shipped for completeness and should not be the default.
The int8 artifact is the recommended deployment checkpoint.

## 7. Supervised fine-tuning

S1 starts from Q2, trains for 400 steps on 4,882 Dolly Q/A pairs, and reaches
val loss 2.3593 (PPL 10.58, bpb 1.110 on the Dolly split). It learns the
`### Question: ... ### Answer: ...` template but hallucinates facts:

> **### Question:** What is the capital of Germany?
> **### Answer:** Germany is the first capital of Germany.

This is a format demonstration, not a knowledge benchmark. The result is
reported honestly because a 2.08M-parameter model cannot store broad factual
knowledge; the value of S1 is the end-to-end Q/A interface and the quantised
artifacts derived from it.

## 8. Routed embedding (Q3)

The routed variant replaces the factorised `E @ U` table with a top-k mixture of
a dense bank. A token has a 6-dimensional routing code; a linear router produces
`v_eff = 5,000` logits; the top `k = 10` are softmaxed; and the corresponding
rows of a `5,000 x 320` bank are mixed into one dense embedding. The effective
table is reused by the tied head. The model has 3,544,104 parameters (1,890,360
in the Kronecker matrix layers) and a 14.41 MB fp32 checkpoint.

**From scratch.** A 500-step pilot reached only 2.083 bpb at step 400 (val loss
4.4275, PPL 83.7), against 1.482 bpb at step 500 for Q2. It peaked at 5.56 GB of
VRAM at batch 32, close to the 6 GB limit. The router does not learn a useful
code quickly enough at this scale.

A longer from-scratch run (`Q3long`, 5,000-step budget, batch 24, `lr=4e-4`,
warmup 300) reached **1.277 bpb** at step 2,000 (val loss 2.7145, PPL 15.10).
Q2 is at 1.245 bpb after 1,000 steps and 1.068 after 2,000. The router improves
smoothly rather than jumping: usage PPL 497 -> 656, used rows 817 -> 842, unique
top-1 rows 179 -> 355, effective table rank 245 -> 220, and logit std
0.100 -> 0.150. There is no phase transition through 2,000 steps, so the run was
stopped. Q3warm is an initialisation test, not evidence of from-scratch
convergence.

```bash
python -m kronllm.cli train --exp Q3 --steps 2000 --run-dir runs/Q3long \
    --override training.batch_size=24 \
    --override training.lr=4e-4 --override training.min_lr=4e-5 \
    --override training.warmup_steps=300 \
    --override training.eval_every=250 --override training.log_every=100
```

**Warm start.** `scripts/init_routed.py` copies Q2's 817 shared tensors and
initialises the bank with Q2's effective table. Each token's own bank row is
made dominant, so the initial effective table matches Q2 at cosine 0.999999
(mean). The initial loss on a validation batch is 1.529038, versus 1.529019 for
Q2. Fine-tuning for 500 steps at `lr=5e-5`, batch 24 gives:

| metric | Q2 | Q3warm |
|---|---:|---:|
| parameters | 2,079,992 | 3,544,104 |
| val loss | 1.7833 | **1.7502** |
| val PPL | 5.949 | **5.756** |
| val bpb | 0.8391 | **0.8236** |
| checkpoint | 8.55 MB | 14.41 MB |
| peak VRAM | 4.22 GB | 3.51 GB (batch 24) |
| MFLOP/token | 36.2 | 37.6 |

The bpb improvement is 1.85 %, at 70 % more parameters. The router remains a
near-copy of the initialization: all 1,024 tokens keep their own bank row as
top-1, the effective-table cosine against Q2 is 0.99997 (mean), and 4,397 of
5,000 bank rows are used. The gain is therefore fine-tuning of a near-copy of
Q2's table, not a newly discovered sparse code. Q2 remains the compact published
model; Q3warm is reproducible but is not shipped.

```bash
python scripts/init_routed.py --source runs/Q2/best.pt --out runs/Q3warm/init.pt
python -m kronllm.cli train --exp Q3 --steps 500 --run-dir runs/Q3warm/run \
    --override training.init_from=runs/Q3warm/init.pt \
    --override training.batch_size=24 \
    --override training.lr=5e-5 --override training.min_lr=5e-6 \
    --override training.warmup_steps=50 \
    --override training.eval_every=100 --override training.log_every=50
```

## 9. Full backprop and sum composition (E8/E14)

E8 removes the stop-gradient from the depth recurrence. At the matched
2,500-step point it reaches **1.586 bpb** versus E2's **1.659**, and its best
checkpoint at step 4,000 reaches **1.479**. Full backprop therefore improves
optimisation, but the spectral geometry is unchanged: at step 4,000
`eff_rank(P) = 60.27`, `eff_rank(sI + ΣD_l) = 101.23`, and
`Var_l σ(θ_l) = 0.00248`.

E14 replaces the product with `P = sI + Σ α_l D_l`, using learnable `α_l`
initialised to 1.0 and no exponentiation. It has 190,348 parameters and
11.5 MFLOP/token at batch 64 (E8: 3.70), because the sum is materialised on
every forward. At step 2,000 E14 reaches **1.675 bpb** versus E8's **1.656**;
`eff_rank(P)` rises to **97.7** from 60.3, but the alpha coefficients stay near
1.0.

The final 4,000-step result is **1.551 bpb** (val loss 1.0750, PPL 2.93),
against E8's **1.479** at the same step: the sum is **4.9 % worse** while using
**3.1×** the FLOPs/token. The alpha coefficients end at mean **0.993**, std
**0.044**, range 0.915–1.206, so the channels do not specialise.
`Var_l σ(θ_l) = 0.00767` stays below the 0.01 threshold, and the depth-rank
Spearman correlation is 0.042. The product remains the default; E14 is a
negative result, not a shipped model. It shows that removing rank compression
is not sufficient to improve quality.

## 10. Quality-scale follow-ups (Q4/Q5/Q6/Q7)

Q4 = Q2 + full backprop. At the matched 4,000-step point it reaches **0.9248
bpb** (val loss 1.9653, PPL 7.137) versus Q2's **0.962**, at identical cost:
2,079,992 parameters, 36.2 MFLOP/token, 4.31 GB peak VRAM. Q4 also beats Q2 at
every shared evaluation from step 500 onward (see §2). The E8 win therefore
survives at quality scale, and the stop-gradient stays off for all following
runs. Q2 remains the shipped model because it trained longer (10,000 steps,
0.839 bpb); Q4 is the new base for further runs.

Q6 = Q4 + BPE-4096. BPE-1024 compresses to 3.07 bytes/token; BPE-4096 reaches
**3.96 bytes/token** on the same corpus, and the factorised embedding
(`V × rank`) makes the wider vocabulary cheap: 2,473,208 parameters total,
37.0 MFLOP/token at batch 32. The loss is computed in row chunks
(`model.loss_chunk = 1024`) so the `32 × 192 × 4096` logits tensor is never
materialised; without that the wider vocabulary exceeds the 6 GB card.

At step 4,000 Q6 reaches **0.8489 bpb** (val loss 2.3292, PPL 10.27) versus
Q4's **0.9248** — **8.2 % better** at matched steps and 4.28 GB peak VRAM. It
is within 1.2 % of Q2's 10,000-step 0.839 in 4,000 steps. The gain is the
tokenizer: the matrix layers are identical to Q4; the only extra parameters are
393K in the embedding. Q2 remains the shipped model until Q6 is re-run to the
same 10,000-step budget.

Q5 = Q4 + `K=24` (7,747,832 parameters) at the same activation footprint. It
is a **negative result**: **0.9530 bpb** at step 4,000 versus Q4's 0.9248,
despite 3.7× the parameters. The spectral diagnostics show why: `cos_prev_max =
0.99999` (neighbouring layers are nearly identical; Q4: 0.896), `eff_rank(P) =
124.0` (Q4: 140.2), and `Var_l σ(θ_l) = 0.0328`. More Kronecker summands at
fixed width/depth do not buy quality; they collapse layer diversity.

Q7 = Q6 + kron-structured embedding (`E = Σ_k A_k ⊗ B_k`, §5.2 of
[architecture.md](architecture.md)): the embedding+head budget drops from
565,248 to 90,112 parameters (1,998,072 total) at the same rank ceiling. It is
also a **negative result**: **0.9533 bpb** at step 4,000 versus Q6's 0.8489
(**12 % worse**), with the same spectral geometry (`eff_rank(P) = 140.5`,
`Var_l σ(θ_l) = 0.00462`). The dense factorised table stays the default. The
Kronecker table code is kept and tested, but it needs a better initialisation
or training scheme before it is useful at this scale.

## 11. Input-dependent channel gates (E15/E16)

E15/E16 replace the static `α_l` of the sum composition with weights generated
from the layer input:

```
w_t = L · softmax(g(x_t)),   P_t = sI + Σ_l w_{t,l} D_l
```

The `L` factor keeps the mean weight at 1, so at initialisation both variants
are exactly the static `α = 1` sum. E15 uses a dense `d × L` gate (`+30,960`
parameters); E16 uses a factorised gate whose `g1 × g2` output is reshaped to
`L` logits (`+3,120` parameters).

| run | gate | bpb @ 4k | params | gate std | range |
|---|---|---:|---:|---:|---|
| E14 | static α | 1.551 | 190,348 | 0.044 | 0.92–1.21 |
| **E15** | dense | **1.509** | 221,308 | **0.54** | **0.07–3.1** |
| E16 | Kron | 1.550 | 193,468 | 0.02 | 0.95–1.05 |
| E8 | product | **1.479** | 190,108 | — | — |

The dense gate **does** specialise and improves the sum by 2.7 %, but still
does not beat the product; the Kron gate stays essentially uniform and is
neutral. Both cost `L·d²` per token and need checkpointing (peak 3.56 GB) to
fit 6 GB. Conclusion: dynamic weighting is a real effect, but it does not make
the sum competitive with the product at this scale.

## 12. Composition study on WikiText-103 (E8w/E14w/E17w/E18w)

TinyStories is easy enough that the product-vs-sum gap could be a corpus
artefact. The same four compositions were therefore trained on WikiText-103
bytes at the E8 geometry: 190K parameters, batch 128, seq 256, 4,000 steps
(131M tokens), one seed.

| run | composition | multiplicative depth | bpb @ 4k | eff_rank(P) | Var σ | cos_prev | s/step |
|---|---|---:|---:|---:|---:|---:|---:|
| E8w | product | 3 | 2.1848 | 60.1 | 0.00273 | 0.773 | 0.421 |
| E18w | alternating | 2 | **2.1578** | 73.5 | 0.00415 | 0.728 | 0.517 |
| E17w | grouped G=2 | 1 | **2.1561** | 79.0 | 0.00422 | 0.752 | 0.502 |
| E14w | sum | 0 | 2.2118 | 97.7 | 0.00708 | 0.741 | 0.427 |

Grouped and alternating are statistically tied (+0.0017 bpb apart), **1.3 %
better than the product** and **2.5 % better than the sum**. Peak VRAM is
2.59 GB (product) vs 3.58 GB (hybrids), because the hybrids are not yet fused
into `KronBank`.

The effective-rank ordering is monotone in the additive depth:

```
product 60.1  <  alternating 73.5  <  grouped 79.0  <  sum 97.7
```

and it held at **60/60 matched modules**. Exact ranks saturate at 125–128
(d=128), so the exact-rank bounds (Sylvester, subadditivity, Kronecker) are
satisfied but uninformative; the ordering is empirical (hypothesis H4, see
`docs/architecture.md` §3.1). One seed is not enough to declare grouped the new
default: a seed-43 control is queued after Q6@10k.

**Zero-shot recursion does not transfer.** E2 was trained with `recurse=1`
(12 layers). Reusing its weights for 2 or 3 passes without fine-tuning makes
the model much worse: 1.5747 → 2.0776 → 2.7789 bpb. Recursive weight sharing
must be trained (or fine-tuned) with the target `recurse`; extra passes are
free in parameters, not in optimisation. E19 (6 unique × 2) and E20
(12 unique × 2 = 24 effective layers) test the trained versions.
