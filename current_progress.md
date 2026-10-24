# Current progress

Rolling log of the most recent results, newest first. At most five entries are
kept. Pruning is triggered only by overflow: with five or fewer entries nothing
is removed, regardless of age. When a new result would make six, entries younger
than one hour are kept temporarily; the oldest is removed once it is at least
one hour old. This file is committed and pushed only at important milestones,
not after every run; entries are written in detail.

## In progress

- No GPU run is active. Next: bilingual corpus pilot (Phase 1) and packaging
  Q6@10k; architecture freeze follows the corpus.

## Results

## 2026-10-24T06:50+03:00 — recursion: depth helps, sharing costs, randomization works

- **E20** (12 unique ×2 = 24 effective, 190K params, 8k steps × batch64 =
  131M tokens): bpb **1.4271** vs E8 **1.479** at the same params/tokens —
  extra depth buys **3.5 %** for 2× compute
- **E22** (6 unique ×4 = 24 effective, 111K params, same tokens): **1.5214** —
  11 % better than E19 (1.7172) but still worse than E8: depth does not close
  the capacity gap
- **E21** (randomized recursion r∈{1,2}, pass embedding): r=2 **1.7180** ≈ E19
  fixed r=2 **1.7172**; r=1 **1.7344** vs E19 at r=1 **2.5541**. One model now
  works at both depths, and the r=1 mode costs only ~1 % bpb
- **H6 confirmed** (randomized recursion gives usable adaptive compute);
  **H5 rejected** (sharing at the same effective depth costs 16 %);
  **H7 partially confirmed** (extra depth helps, but 2× compute, and half-param
  sharing stays behind)
- E19/E21/E22 checkpoints removed; conclusions in `docs/results.md` §12 and
  `docs/architecture.md` §2

## 2026-10-24T04:20+03:00 — E19 recursive sharing at half params: negative

- **1.7172 bpb** (loss 1.1902, PPL 3.288), 111,502 params, effective depth 12
  (6 unique blocks ×2), 0.407 s/step, 2.58 GB
- E8 (12 unique blocks, 190,108 params) reaches **1.479**: sharing the same
  depth with 59 % of the parameters costs **+16 % bpb**. The recursion does not
  replace unique layers at this scale
- `eff_rank(P)` **59.2** (close to product 60), `Var sigma(theta)` 0.00408,
  `cos_prev_max` 0.718 — no collapse, just less capacity/diversity
- verdict: weight sharing is a parameter-saving tool, not a free quality win;
  E21/E20/E22 test randomized recursion and extra depth
- E19 checkpoint removed (not a shipped candidate)

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

## 2026-10-24T00:35+03:00 — WikiText composition study (seed 42)

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
