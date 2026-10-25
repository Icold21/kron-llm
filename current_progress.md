# Current progress

Rolling log of the most recent results, newest first. At most five entries are
kept; pruning is triggered only by overflow. This file is committed and pushed
only at important milestones, not after every run, and entries are written in
detail.

## In progress

- No GPU run is active. The architecture is frozen at **Q8g** (gated Kronecker
  FFN, 2.85M parameters); Q8g-long is the new best checkpoint and the new
  shipped candidate.
- Next: total build of the bilingual English/Russian corpus (Phase 1), then
  re-tokenisation, then a 10M-scale run on the real corpus. TinyStories remains
  a sanity bench only.

## Results

## 2026-10-25T05:10+03:00 — Q8g-long 0.7542 bpb: new best; bilinear interaction dead

Chain: Q9 → Q9k → Q8g-long, all at 61.4M tokens where applicable.

- **Q8g-long** (24 layers, gated Kronecker FFN, 2,851,280 params, 13,333 steps
  × batch 24 × 192): **0.7542 bpb**, loss 2.0693, PPL 7.919, 0.895 s/step,
  **4.09 GB** peak, `eff_rank(P)` 144.8, `cos_prev_max` 0.834
  - vs **Q6@10k 0.7644** at the identical 61.44M-token budget: **−1.3 %**
  - the curve was still falling at the last step (0.7553 → 0.7542), so the
    budget is a choice, not a measured ceiling
  - int8 quantisation: **+0.0003 bpb**, 3.29 MB vs 11.1 MB fp32 (3.55×);
    moved to `artifacts/q8g_int8.pt`
- **Q9** (bilinear GLU FFN, no Kron term, 2,463,776 params, 24 layers):
  **0.8740** vs Q6 **0.8489** at equal tokens — **+2.9 %**, and
  `cos_prev_max` jumped to **0.939**: the layers became more similar
- **Q9k** (GLU + the Kron map as a non-interacting term, 20 layers,
  2,462,468 params): **0.8673** — **+2.2 %**. Adding the linear term back helps
  a little but does not close the gap
- **Methodology note:** the first Q9 run was invalid — the generic `std=0.02`
  init left the bilinear branch 300× smaller than a Kron module (output std
  0.005 vs 1.76), i.e. switched off. Fixed to unit-activation init, guarded by
  `test_glu_branch_is_alive_at_init`; the numbers above are from the corrected
  run. A false negative would have been worse than no experiment
- **Verdict:** multiplicative feature interaction is closed at this scale. The
  pattern across all experiments is consistent: what wins keeps the layer
  linear and fusable into one GEMM
- checkpoints of Q9/Q9k removed; `runs/Q8g_long/best.pt` and
  `runs/Q6_10k/best.pt` are the only ones kept on disk

## 2026-10-24T23:50+03:00 — Q8w: gated FFN wins 1.5 % at equal parameters

- **Q8g** (24 layers × 6 modules, 2,851,280 params): **0.8318** at 24.58M tokens
- **Q8w** (20 layers × 6 modules, 2,470,328 params, same 120 modules/token):
  **0.8359** — vs Q6's **0.8489** at identical parameters (±0.1 %) and tokens:
  **−1.5 %**
- decomposition of the gain: **1.5 % is architecture** (equal parameters, equal
  tokens, equal module count), **0.5 % is the extra 378K parameters** of Q8g
- the design rule that follows: *a gated FFN beats four extra layers at equal
  budget*
- Q8w is also cheaper in memory (3.44 GB vs 4.28) because 20 layers with batch
  24 fit better than 24 layers with batch 32

## 2026-10-24T22:42+03:00 — nonlinearity inside the composition: closed

| variant | bpb | vs E8 (1.4789) |
|---|---:|---:|
| E26, k=4 (one activation) | 1.5675 | +6.0 % |
| E25, k=2 (two activations) | 1.5189 | +2.7 % |
| E26b, additive k=4 (linear skeleton kept) | 1.5906 | +7.5 % |

- more nonlinearity helped (k=2 beat k=4), but **every variant lost to the
  linear product**, and the additive form was the worst
- wall-clock cost at equal FLOPs: **0.655 vs 0.405 s/step** (+62 %) — the loss
  of the single-GEMM fusion, not the activation itself
- `eff_rank(P)` rose to 62.3 while bpb got worse: third independent sign that
  rank is not the bottleneck
- E23 (k=1, 9.56 MFLOP, 2.6× compute) and E24 (k=1 + RMSNorm) were dropped
  rather than run: the trend is monotone and points away from the hypothesis
- `E23–E26b` annotated as CLOSED in `config.py`; code kept for reproducibility
  only

## 2026-10-24T06:50+03:00 — recursion: depth helps, sharing costs, randomization works

- **E20** (12 unique ×2 = 24 effective, 190K params, 8k steps × batch 64 =
  131M tokens): **1.4271** vs E8 **1.479** at the same params/tokens — extra
  depth buys **3.5 %** for 2× compute
- **E22** (6 unique ×4 = 24 effective, 111K params, same tokens): **1.5214** —
  11 % better than E19 but still worse than E8: depth does not close the
  capacity gap
- **E21** (randomized recursion r∈{1,2}, pass embedding): r=2 **1.7180** ≈ E19
  fixed r=2 **1.7172**; r=1 **1.7344** vs E19 at r=1 **2.5541**. One model now
  works at both depths
- **H6 confirmed** (randomized recursion gives usable adaptive compute);
  **H5 rejected** (sharing at equal effective depth costs 16 %);
  **H7 partially confirmed**
- recursion stays a deployment/compute option, not a default

## 2026-10-24T03:50+03:00 — seed control: the hybrid advantage does not reproduce

- WikiText-103, product vs grouped, 4,000 steps:

  | seed | product (E8w) | grouped (E17w) |
  |---|---:|---:|
  | 42 | 2.1848 | **2.1561** |
  | 43 | **2.1559** | 2.1624 |

- seed-to-seed spread is ~**0.03 bpb** (product 2.1848 → 2.1559), larger than
  the 0.029 "win" on seed 42; on seed 43 grouped is 0.007 worse
- **verdict:** the quality difference is seed noise; the product remains the
  default. The *rank* ordering is stable across seeds (product 60.5 < grouped
  79.2), so H4 survives as a diagnostic but does not predict bpb
- this result is the reason every later comparison in this file is quoted with
  an explicit token budget and, where possible, a parameter-matched control
