# Architecture

KronLLM is a decoder-only transformer in which every weight matrix is a
Kronecker composition with a depth recurrence. This document describes the
mathematics, the parameter accounting and the implementation invariants. For
the empirical results see [results.md](results.md); for the motivation behind
the deviations from the original specification see section 3 of the
[README](../README.md).

## 1. Kronecker product

For matrices `A ∈ R^{n₁×n₁}` and `B ∈ R^{n₂×n₂}` the Kronecker product is the
`n₁n₂ × n₁n₂` block matrix

```
A ⊗ B = [[A₁₁B, …, A₁ₙB],
         [  ⋮  ,  ⋱ ,  ⋮  ],
         [Aₙ₁B, …, AₙₙB]]
```

It acts on a vector reshaped as an `n₁ × n₂` matrix by multiplying from both
sides:

```
(A ⊗ B) vec(X) = vec(A · X · Bᵀ),     X = reshape(x, n₁, n₂).
```

The implementation uses exactly this identity (`kron_apply`), so a factorised
layer never materialises the `d × d` matrix in the forward pass.

## 2. Layer parameterisation

Every linear map `d → d` (and `d → d_kv` for grouped-query attention) is a sum
of `L` channels, each of which is a sum of `K` Kronecker products:

```
I_l = Σ_{k=1..K} A_lk ⊗ B_lk,     A_lk ∈ R^{n₁×n₁},  B_lk ∈ R^{n₂×n₂}
W_l = s · Id + I_l
```

with `n₁ · n₂ = d`. The anchor `s · Id` guarantees full rank and a stable
gradient path at initialisation; the innovation `I_l` carries the learned
structure.

`rank(A ⊗ B) = rank(A) · rank(B)`, so with full-rank factors one summand is
already full-rank — this is not a low-rank approximation. The saving is in
parameters, not in rank: one channel costs `K(n₁² + n₂²)` instead of `d²`.

### Parameter count of one module

| component | count |
|---|---|
| factors `A_lk, B_lk` | `L · K · (n₁² + n₂²)` |
| memory speeds `θ_l` | `L` |
| scales `τ_l` | `L` |
| ReZero gate `γ` | `1` |

For Q2 (`L=4, K=6, n₁=16, n₂=20`): `4·6·(256+400) + 8 + 1 = 15,753` parameters
per module. A block has five roles (q, k, v, o, ffn), and Q2 has 24 blocks:
`120 · 15,753 = 1,890,360` matrix parameters.

### Recursive weight sharing (`n_unique_blocks`, `recurse`)

The transformer stack can reuse a smaller set of blocks:

```
effective_depth = n_unique_blocks * recurse
parameters       = the unique blocks only (counted once)
```

`KronTransformer.set_recurse(r)` rebuilds the execution schedule for inference
without adding parameters. The blocks are applied in order with the effective
layer index, so `K_win` resets and the depth-recurrence state `D` keep
evolving across passes.

**The recursion must be trained.** E2, trained with `recurse=1`, degrades
badly when the same weights are applied more times at inference:

| passes | effective layers | val bpb (E2, no fine-tuning) |
|---:|---:|---:|
| 1 | 12 | **1.5747** |
| 2 | 24 | 2.0776 (+32 %) |
| 3 | 36 | 2.7789 (+76 %) |

The shared block learns a one-step transformation; composing it shifts the
residual and attention statistics. Training with the target `recurse`, or
fine-tuning after changing it, is required. Randomized recursion (sampling
`1..R` per step) is the standard way to get one model that works at several
depths; it is not implemented yet. E19 (6 unique × 2 = 12 effective, ~half the
parameters) and E20 (12 × 2 = 24 effective) test the trained versions.

## 3. Depth recurrence

Every layer owns its factors, so the composition over channels is not the only
source of depth. The innovation is additionally carried across blocks:

```
D_l^{(i)} = τ_l · rms_norm( Î_l + σ(θ_l) · D̂_l^{(i-1)} )
W_l^{(i)} = s · Id + D_l^{(i)}
P^{(i)}   = W_L^{(i)} ⋯ W_1^{(i)}
y         = γ^{(i)} · P^{(i)} x
```

* `rms_norm` is taken over the whole `d × d` matrix, not per row. Without it
  the product of matrices collapses to `s^L · Id` and the recurrence receives no
  gradient (see README §3.1).
* `σ(θ) = sigmoid(θ)` with `θ_init = 0` is the learned mixing rate of the
  previous depth state; `stop_gradient` (on by default) detaches `D̂`.
* `K_win` resets the state every `K_win` blocks, bounding the effective depth of
  the recurrence.
* `γ` is a ReZero gate initialised at 0.1, so the branch starts as a small
  perturbation of the residual stream.

The composition `P` is computed once per block and applied as a single
`x @ Pᵀ`; see §6.

### Sum-composition ablation (E14)

The product is an empirical choice, not a requirement of the Kronecker
parameterisation. E14 keeps the same innovations and depth recurrence but
replaces the L-fold product with an additive composition:

```
P_sum^(i) = s * Id + sum_l alpha_l * D_l^(i)
```

`alpha_l` is a learnable per-channel coefficient initialised to 1.0. There is
no exponentiation in this path: `s_init` is used directly as the coefficient of
the identity, unlike the product path where each factor receives
`s_init ** (1/L)` and the composed anchor is `s ** L`. The sum is therefore a
direct test of whether the product, rather than the recurrence or the budget,
causes the measured rank compression. `KronBank` evaluates the sum with one
batched weighted reduction followed by a single GEMM.

### 3.1 Rank relations: proven bounds and the effective-rank ordering

Two hybrid compositions sit between the product and the sum:

- **grouped** (`G` blocks): `S_g = s^(1/G) I + sum_{l in g} alpha_l D_l`,
  `P = S_G ... S_1`; `G=1` is the sum, `G=L` is the product.
- **alternating** (start `sum`): start at `P = sI`, then for each depth step
  apply `P <- P + alpha_l D_l` (sum) or `P <- P (I + alpha_l D_l)` (product),
  alternating the operation.

**Proven bounds** (standard linear algebra):

| statement | source |
|---|---|
| `rank(AB) <= min(rank A, rank B)` | Sylvester's rank inequality |
| `rank(A + B) <= rank A + rank B` | rank subadditivity |
| `rank(A ⊗ B) = rank(A) rank(B)` | Kronecker product rank |

Applied to the compositions:

```
sum:         rank(P) <= 1 + sum_l rank(D_l)
grouped:     rank(P) <= min_g rank(S_g) <= 1 + sum_l rank(D_l)
product:     rank(P) <= min_l rank(sI + D_l)
alternating: rank(P) <= min over the factors in the multiplicative chain
```

**Exact ranks saturate.** On all 60 modules of E8w/E14w/E17w the exact rank of
`P` is 125–128 (d=128) and `max(rank(P) - bound) = 0`; exact rank therefore
cannot distinguish the methods. The informative quantity is the **effective
rank** (Roy & Vetterli, 2007): `exp(H(sigma / sum sigma))`.

Measured on WikiText-103 at 4,000 steps (60 modules each):

| composition | mean effective rank | range |
|---|---:|---:|
| product, depth 3 (E8w) | **60.1** | 45.4–78.5 |
| alternating, depth 2 (E18w) | **73.5** | 61.0–85.0 |
| grouped `G=2`, depth 1 (E17w) | **79.0** | 67.6–90.8 |
| sum, depth 0 (E14w) | **97.7** | 90.4–103.0 |

The non-strict ordering

```
eff_rank(product) <= eff_rank(alternating) <= eff_rank(grouped) <= eff_rank(sum)
```

held at **60 / 60 matched modules**, and is monotone in the number of additive
(vs multiplicative) steps. This is consistent with Sylvester's bound, but it is
**not a theorem**: effective rank is not monotone under matrix products, and
counterexamples exist for exact rank. We record it as hypothesis **H4** and keep
it empirical (one corpus; the rank ordering survived a second seed, while the
bpb advantage of the hybrids did not). The sum-product construction is
motivated by SPNs, where sum and product nodes alternate by design (Poon &
Domingos, 2012).

## 4. Attention

The block is pre-norm RMSNorm → attention → residual, then pre-norm RMSNorm →
FFN → residual.

* **RoPE** is applied to `q` and `k` per head before the dot product.
* **QK-norm** (Q1 and Q2) is an RMSNorm on `q` and `k` per head before RoPE.
  It keeps attention operands on one scale independently of how the Kronecker
  branch has grown, which removes logit drift and allows `lr ≈ 4·10⁻⁴` without
  divergence. Cost: `2 · d_head` parameters per layer.
* **GQA** is available as experiment E10 (`n_kv = 2`); the default is full MHA.

## 5. Factorised tied embedding

Instead of a `V × d` table the model stores `E ∈ R^{V×r}` and
`U ∈ R^{r×d}` with `r < d`. The logits are computed as

```
logits = h · Uᵀ · Eᵀ
```

so the head is tied to the embedding and costs no extra parameters. For Q2
(`V=1024, r=128, d=320`) this is `172,032` parameters instead of `327,680`.

### 5.1 Routed dense embedding (Q3, experimental)

The factorised table has an `r`-dimensional bottleneck. Q3 replaces it with a
top-k mixture of a dense bank:

```
r_v   = route[v]                       # (route_dim,)
logits = r_v W_route                   # (v_eff,)
w      = softmax(topk(logits, k))      # (k,)
E_eff[v] = sum_i w_i * bank[i]         # (d,)
```

`route` is `(V, route_dim)`, `W_route` is `(route_dim, v_eff)` and `bank` is
`(v_eff, d)`. The effective table `E_eff` is dense in width but never stored:
it is rebuilt from the selected rows on each forward pass (and cached in
`eval()`). The same `E_eff` is used as the tied output head,
`logits = h @ E_eff.T`, so the head adds no parameters.

For Q3 (`V=1024, v_eff=5000, route_dim=6, k=10, d=320`) the embedding costs
`6,144 + 30,000 + 1,600,000 = 1,636,144` parameters and the model totals
`3,544,104`. Gradients reach the selected bank rows through the mixture weights
and the router through the selected logits; the top-k indices themselves are
discrete. `model.set_route_top_k(k)` changes `k` at generation time and
invalidates the effective-table cache.

### 5.2 Kronecker-structured embedding (`embed_mode="kron"`, Q7)

The factorised `V × r` table grows linearly with the vocabulary. In kron mode
the table is a sum of Kronecker products:

```
E = sum_k A_k (x) B_k,    V = v1*v2,  r = r1*r2
A_k: (v1, r1),  B_k: (v2, r2)
```

The parameter cost is `K·(v1·r1 + v2·r2)` instead of `V·r`, and the rank
ceiling is unchanged: `rank(A ⊗ B) = rank(A)·rank(B) ≤ r1·r2 = r`, so `K`
summands can reach the same rank `min(V, r)` as the dense factorisation.
Lookup and the tied head are untouched (`logits = h·Uᵀ·Eᵀ`); only the storage
of `E` is compressed. The table is materialised fresh on every forward
(`K·V·r` MACs, amortised over the batch; never cached, because a cached tensor
would carry a stale training graph).

For Q7 (`V=4096, r=128, v1=v2=64, r1=16, r2=8, K=32`) the factors cost
`32·(1024+512) = 49,152` parameters versus `524,288` for the dense factor --
a 10.7× reduction, with the model dropping from 2,473,208 to 1,998,072
parameters. The ratio improves as the vocabulary grows.

**Q7 result: negative.** At step 4,000 the composed table reaches **0.9533
bpb** versus the dense factor's **0.8489** (12 % worse) at the same rank
ceiling, with the same spectral geometry. The factors are a harder
optimisation problem than a free table, and the dense factorised table stays
the default. The module remains available (and test-covered) for vocabularies
where the dense table is unaffordable, but it needs a better initialisation
(for example, warm-starting the factors to approximate a trained dense table)
before it is competitive.

## 6. Batched prelude (`KronBank`)

On WDDM every kernel launch costs ~45 µs, so computing 120 small compositions
one by one is wasteful. `KronBank` groups the modules by signature
`(role, n₁, n₂, K, L, composition)` and computes the innovations with batched
`bmm` calls and the composition with `L−1` `baddbmm` calls (product) or one
weighted reduction (sum). The result is mathematically
identical to the per-module path (test T16 checks forward and all gradients) and
saves ~15 % of step time.

The prelude depends only on the weights, so in `eval()` mode it is cached in
`_prelude_cache` and invalidated when the model returns to `train()` or the
cache is cleared. This is what makes generation practical: without it the
prelude alone would cost ~13 ms per token.

### 6.1 Serving: KV cache and factorized application

**KV cache.** `KronBlock.forward` accepts `past_kv=(k, v)` (already
RoPE-rotated, before GQA expansion) and returns the updated cache; RoPE uses
absolute positions, and chunked prefill gets an explicit causal mask. Logits
match a full forward to `2.8e-7`. On the reference GPU it is **not faster**:
Q2 goes from 32.4 to 32.9 tok/s and E2 from 81.9 to 79.7, because per-token
cost is dominated by 24 layers of small kernel launches, not by attention
(`O(T²)` is tiny at `T≤192`). It becomes useful at long context or with a
fused kernel.

**Factorized application.** `KronCompose.apply_innovation` computes
`(A ⊗ B)x = A X Bᵀ` without materialising `d×d`:

```
cost per summand  = n1_out·n1_in·n2_in + n2_out·n2_in·n1_out
dense cost        = d_in·d_out
```

With square factors this is `O(d^1.5)` per summand; the crossover against the
dense map is `K · 2 · d^1.5 < d²`. Measured on the reference GPU, PyTorch's
dense GEMM still wins everywhere we can afford to train: 0.21× at
`d=320, K=6`, 0.31× at `d=320, K=1`, 0.20× at `d=1024, K=6`. The reasons are
kernel-launch overhead, the `K`-summand loop, and the recurrence state `prev`
being dense. A real `O(d^1.5)` win needs a fused kernel **and** a factorized
recurrence state.

`torch.compile` is not available on this machine: the Windows PyTorch 2.6
build ships without Triton, so Inductor CUDA cannot run.

## 7. FLOPs

Per token, forward:

* QKV and output projections: `4 · d²` per layer;
* attention: `2 · seq_len · d` per layer;
* FFN: `2 · d · d_ff` per layer;
* plus the Kronecker prelude, amortised over the sequence.

`KronTransformer.flops_per_token(seq_len, batch_size=64)` returns the measured
formula. The reported values are 3.70M (E2), 11.2M (Q1) and 36.2M (Q2) per
token. Since the tokenizers differ, the honest unit is FLOPs per **byte**:
3.73 (E2), 3.66 (Q1) and 11.9 (Q2) MFLOP/byte.

The sum path is not FLOP-free. It materialises `sI + Σ α_l D_l` on every forward,
so its prelude costs `L·d²` per token, whereas the product is amortised over the
sequence. At the E2 geometry, E8 (product) reports 3.70 MFLOP/token and E14
(sum) reports 11.5 MFLOP/token at batch 64; the sum is an expressiveness
ablation, not a cheaper model.

## 8. Implementation invariants

The following properties are enforced by the test suite:

| id | property |
|---|---|
| T1 | `kron_apply(A, B, x)` equals `(A ⊗ B) x` and `A · X · Bᵀ` |
| T2 | with `D = 0` a layer equals `γ · s^L · x` (the anchor) |
| T3 | `kron_materialize` equals `torch.kron`; matrix mode equals sequential |
| T3c | sum composition equals `sI + Σ α_l D_l`; the anchor is `s`, not `s^L` |
| T3d | sum composition with `D = 0` is the identity times `s` |
| T4 | the recurrence matches `D_l = I_l + σ·D_{l-1}`; normalised `D` has RMS `τ` |
| T5 | stop-gradient blocks `∂L/∂θ` through `D_{l-1}`; full backprop and `α_l` are differentiable |
| T6 | RoPE preserves the norm and depends on position |
| T7 | attention is causal |
| T8 | the head is tied to the embedding |
| T16 | `KronBank` equals the per-module path in forward and gradients |
| T16b/T16c | `KronBank` equals the per-module sum path in forward and gradients |
| T17 | the ReZero gate is active and differentiable |
| T19 | the factorised tied embedding has `V·r + r·d` parameters |
| T20 | QK-norm has unit per-head RMS and costs `2·d_head·N` |
| T23 | the prelude cache does not change the output |
| T28 | the routed table equals the softmax-weighted top-k bank mixture |
| T29 | routed gradients reach the bank and router; top-k is adjustable |

## 9. Design constraints

* **Compactness first.** Activation memory grows as `N · d · batch · seq`. Q1 and
  Q2 were designed to have exactly the same activation footprint
  (`20·192·64 = 24·320·32 = 245,760`), so Q2's extra parameters add almost no
  training memory. Q2 peaks at 4.22 GB, so the family fits a 6 GB consumer GPU.
* **One GPU, fp32.** fp16 autocast is 2.4× slower than fp32 on the reference
  GPU, so `amp` defaults to `none`. No DDP, no gradient checkpointing, no CUDA
  graphs (capture fails on `baddbmm` in this environment).

## 10. The Q10 configuration (the run that matters)

Q10 is the first model in this repo trained at token budget, not at toy scale, so
its numbers are the ones the papers and the article quote. Everything below is
measured from the built model (`KronTransformer(experiment_config("Q10"))`).

### Geometry

| knob | value |
|---|---|
| `d_model` | 512 |
| layers / heads | 25 / 8 (`d_head = 64`, full MHA) |
| context | 512 |
| vocabulary | 8,192 (BPE, digit-split) |
| embedding | factorised `V×256` + `256×512`, tied head |
| Kronecker | `L = 4` channels, `K = 10` terms, `n₁ × n₂ = 16 × 32` |
| depth recurrence | `K_win = 2`, product composition, `σ` learned, `stop_gradient = False` |
| anchor | `s = 1.0`, frozen (not learnable) |
| FFN | gated Kronecker: `silu(W_f h) ⊙ W_g h`, both `d → d` |
| norms | RMSNorm, `qk_norm` on every layer |
| initialisation | `τ = 0.1`, `γ = 0.1` (ReZero), `θ ~ N(0, 0.1)`, unit-gain factors |

### Parameters (measured, not estimated)

| component | parameters | share |
|---|---:|---:|
| 150 Kronecker modules (6 per block × 25) | 7,681,350 | 77.3 % |
| embedding `E` (8192 × 256) | 2,097,152 | 21.1 % |
| embedding up-projection `U` (256 × 512) | 131,072 | 1.3 % |
| RMSNorms (2 × 512 per block + final) | 26,112 | 0.3 % |
| QK-norm (2 × 64 per block) | 3,200 | 0.03 % |
| **total** | **9,938,886** | **100 %** |

One module costs

```
L·K·(n₁² + n₂²) + 2L + 1 = 4·10·(256 + 1024) + 8 + 1 = 51,209
```

against 262,144 for a dense `512 × 512` map: **5.12× fewer parameters at full
rank**, because `rank(A ⊗ B) = rank(A)·rank(B)` and the `s·Id` anchor keeps the
composed map away from zero. The six roles per block are `q, k, v, o, f, f_g`;
the only dense matrices left are the tied embedding pair.

For comparison, a dense transformer with the same width and depth would spend
`150 × 262,144 = 39.3M` parameters on matrices alone, i.e. ~4× the whole Q10
model.

### Why the budget looks thin and the data looks fat

9.94M parameters against 2.25B training tokens is **226 tokens per parameter**
— roughly 40× past compute-optimal for a model this size. That is deliberate:
the Kronecker parameterisation is only interesting if it holds up when the
*data* is the constraint, so the run is set up to be data-bound from the first
step. The decay pass (Q10A) then spends the last 30 % of the schedule at
`LR 3e-4 → 1e-6` on curated data.

### Arithmetic cost: the prelude is not free

`flops_per_token(seq_len)` at the Q10 geometry:

| seq len | forward MFLOP/token |
|---:|---:|
| 128 | 1,348.6 |
| 256 | 726.0 |
| 512 | **424.5** |
| 1024 | 293.4 |

The composition `P = (sI + D_4) ⋯ (sI + D_1)` costs `L·d³` MACs per module per
forward: `4 × 512³ ≈ 537M` MACs, i.e. `1.07 GFLOP`, times 6 modules and 25
blocks = **161 GFLOP per forward at `seq_len = 512`** — 315 MFLOP per token.
The body (attention, FFN, and the single `h @ Pᵀ` application per module) is
`~105 MFLOP` per token plus `2·T·d` for attention. So at 512 tokens the
Kronecker prelude is **about three quarters of the forward FLOPs**, and that
share falls as `1/T`: at 1024 tokens the measured cost is 293 MFLOP/token, of
which 157 is prelude.

This is the honest trade of the whole approach: parameters are bought with
arithmetic, not with rank. The saving matters when the model must fit in a
memory budget (a T4, a phone, a 4 GB checkpoint); it does not make training
cheaper per token. A `torch.compile`/Triton fused prelude and a factorised
recurrence state are the obvious next steps, and neither exists on this setup
(Windows PyTorch ships without Triton).

### Training setup

| knob | value |
|---|---|
| optimiser | AdamW, `β` defaults, `ε` default |
| LR | 3e-4 constant through the epoch (WSD), `θ` at 1e-2, `τ`/`γ` at LR |
| weight decay | 0.1 on matrices, 0 on `θ` (learned mixing rates) |
| grad clip | 1.0 |
| precision | fp16 autocast, single T4 |
| batch | 32 × 512 = 16,384 tokens/step |
| epoch | 137,377 steps = 2.2508B tokens |
| measured | 2.53 s/step, ~6,500 tok/s, ~95.6 GPU-h per epoch |
| sessions | 500 minutes, resume from `last.pt`, `save_every = 500` |

### What Q10 deliberately does **not** use

* **GQA** (`n_kv_heads = 8`, i.e. plain MHA) — at `d_head = 64` the KV cache is
  not the bottleneck.
* **Recursion / weight sharing** (`recurse = 1`, 25 unique blocks) — E2 showed
  untrained recursion degrades, and the run has no budget for the fine-tune.
* **Mixture-of-experts or routed embedding** — Q3/Q7 both lost to the dense
  factorised table.
* **Dropout, gradient checkpointing, DDP** — a 10M model on 500-minute sessions
  does not need them; DDP is blocked by the 32 GB host cgroup, not by the code.
* **The sum/grouped/alternating compositions** — E14/E17/E18 measured lower
  effective rank and no better bpb than the product; Q10 is the product path.

### What `val_bpb` measures (and why it was optimistic)

The training-time `val_bpb` comes from `evaluate()` with `eval_batches = 64`, a
*sequential* pass over the first 512 blocks of the val bin.  Two things were
wrong with it, and only one of them was about the data:

1. **Denominator (the big one).** `bits_per_byte` divides by the corpus-wide
   3.694 bytes per token, but that span is English Python at **2.558** bytes per
   token.  Every logged number is therefore low by exactly `3.694 / 2.558 =
   1.444`.
2. **Coverage.** The bin is written bucket by bucket and `en/code` sorts first,
   so 262,144 tokens of 46.2M were measured — 0.6 % of the held-out set, all
   from one domain.

Rescaled by 1.444, the step 11,000 number (0.884) is **1.276**; the measured
value at step 11,723 is **1.265** on the code span and **1.248** on a strided
sample of the whole held-out set (EN 1.518, RU 0.990).  The mixed number is
*lower* than the code span because Russian carries ~5 bytes per token against
2.9 for Latin text, so the byte-weighted average is pulled down — bpb is not
comparable across languages, only across models on the same text.

GPT-2 (124M) on the same strided blocks: **1.554** (EN 1.306, RU 1.792), and
**1.365** on the code span.  So at 8.5 % of the epoch a 10M bilingual model
already wins overall, loses on English by 0.21 bpb, and wins on code by 0.10.

`training.eval_uniform` now adds a strided pass over the whole bin next to the
legacy number, and `scripts/eval_reference.py` scores any checkpoint — or GPT-2
— on identical blocks.  The legacy column is kept so the existing curve stays
continuous; it is not the number to quote.

## References

- Rank inequalities (Sylvester, subadditivity): *Rank (linear algebra)*,
  Wikipedia, <https://en.wikipedia.org/wiki/Rank_(linear_algebra)>.
- Kronecker rank `rank(A ⊗ B) = rank(A) rank(B)`: *Kronecker product*,
  Wikipedia, <https://en.wikipedia.org/wiki/Kronecker_product>.
- Effective rank: O. Roy and M. Vetterli, "The Effective Rank: A Measure of
  Effective Dimensionality", EUSIPCO 2007,
  <https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf>.
- Sum-product networks: H. Poon and P. Domingos, "Sum-Product Networks: A New
  Deep Architecture", 2012, <https://arxiv.org/abs/1202.3732>.
