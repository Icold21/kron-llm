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
parameters. The ratio improves as the vocabulary grows. Whether the composed
table stores the *same information* as the dense factor is exactly what Q7
tests at equal steps against Q6.

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
