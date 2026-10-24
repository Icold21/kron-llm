# Current progress

Rolling log of the most recent results, newest first. At most five entries are
kept. Pruning is triggered only by overflow: with five or fewer entries nothing
is removed, regardless of age. When a new result would make six, entries younger
than one hour are kept temporarily; the oldest is removed once it is at least
one hour old. This file is committed and pushed only at important milestones,
not after every run; entries are written in detail.

## In progress

- **E19** (6 unique blocks ×2 = 12 effective) is running; E21 (randomized
  recursion), E20 (12×2) and E22 (6×4) are queued automatically.

## Results

## 2026-10-24T03:50+03:00 — seed control: hybrid advantage does not reproduce

- WikiText product vs grouped, two seeds at 4,000 steps:

  | seed | product (E8w) | grouped (E17w) |
  |---|---:|---:|
  | 42 | 2.1848 | **2.1561** |
  | 43 | **2.1559** | 2.1624 |

- seed-to-seed spread is ~**0.03 bpb** (product 2.1848 → 2.1559), larger than
  the 0.029 “win” seen on seed 42; on seed 43 grouped is 0.007 worse
- **verdict: H5 (hybrid beats product) is not confirmed.** Product remains the
  default; grouped/alternating stay as tested code and a rank experiment, not
  as a quality claim
- the *rank* ordering is stable across seeds: product 60.5 < grouped 79.2
  (seed 43), matching 60.1 < 79.0 (seed 42); H4 (rank vs depth) survives, the
  quality claim does not
- E19 (6 unique ×2) started right after; E21/E20/E22 queued

## 2026-10-24T02:50+03:00 — Q6 at 10k: 0.7644 bpb, new best

- val loss **2.0972**, PPL **8.143**, bpb **0.7644** (2,473,208 params, BPE-4096)
- Q2 at the same budget: **0.839**; Q6 is **8.9 % better**, and Q6@4k was
  0.8489 — the curve was still improving at 10k
- 0.847 s/step, ~2.4 h, peak **4.28 GB**; `eff_rank(P)` **144.3**,
  `Var sigma(theta)` **0.0108**, `cos_prev_max` **0.839**
- Q6 is the new best checkpoint; Q2 remains the packaged int8 artifact until
  Q6 is quantised and added to `artifacts/`
- seed control of the hybrid compositions started right after this run

## 2026-10-24T00:35+03:00 — WikiText composition study: hybrids beat product

- WikiText-103 bytes, 190K params, 4,000 steps, 131M tokens, one seed:
  E8w product **2.1848** · E18w alternating **2.1578** · E17w grouped G=2
  **2.1561** · E14w sum **2.2118** bpb
- grouped and alternating tie (0.002 bpb apart): **−1.3 % vs product**,
  **−2.5 % vs sum**; hybrids cost 0.50–0.52 s/step and 3.58 GB because they
  bypass `KronBank` (product: 0.421 s, 2.59 GB)
- effective rank is monotone in additive depth: product **60.1** < alternating
  **73.5** < grouped **79.0** < sum **97.7**; the non-strict ordering held at
  **60/60 matched modules**
- exact ranks saturate at 125–128 (d=128), so Sylvester/subadditivity/Kronecker
  bounds hold but are uninformative; H4 (effective-rank ordering) is empirical,
  not a theorem — documented in `docs/architecture.md` §3.1 with references
- one seed only: seed-43 control for E8w/E17w queued after Q6@10k; Q6@10k is
  running now (10,000 steps, ~2.4 h)

## 2026-10-23T22:10+03:00 — E15/E16 dynamic gates + serving measurements

- **E15 (dense softmax gate):** bpb **1.5091** vs E14 **1.551** (+2.7 %),
  221,308 params; the gate specialises (std **0.54**, range **0.07–3.1**) —
  but it is still 2 % behind the product E8 **1.479**
- **E16 (Kron gate, 52 params/module):** bpb **1.5504** — neutral; the gate
  stays uniform (std **0.02**) and is too weak to specialise
- both cost `L·d²` per token and need checkpointing; peak VRAM **3.56 GB**
- **KV cache:** implemented and bit-exact (logits match to **2.8e-7**), but no
  speedup on this GPU: Q2 32.4 → 32.9 tok/s, E2 81.9 → 79.7 — generation is
  launch-bound, not attention-bound
- **factorized innovation apply:** exact, but **0.20–0.31×** the speed of a
  dense GEMM from d=320 to d=2048; the `O(d^1.5)` win needs a fused kernel and
  a factorized recurrence state
- `torch.compile` unavailable: Windows PyTorch 2.6 has no Triton
- E15/E16 checkpoints cleaned; conclusions in `docs/results.md` §11 and
  `docs/architecture.md` §6.1

## 2026-10-23T19:15+03:00 — Q7 kron embedding: negative

- val loss **2.6155**, PPL **13.67**, bpb **0.9533** (1,998,072 params)
- Q6 at the same step: bpb **0.8489**; the compressed table is **12 % worse**
  despite an identical rank ceiling
- embedding+head budget 565,248 → **90,112** params; the matrix layers are
  unchanged (`eff_rank(P)` **140.5**, `Var sigma(theta)` **0.00462**)
- 0.930 s/step, peak **4.27 GB**; the dense factorised table stays the
  default, and the kron table is kept only as tested code for large vocabularies
- both Q5 and Q7 are now cleaned from `runs/`; conclusions live in
  `docs/results.md` and `docs/roadmap.md`
