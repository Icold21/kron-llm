# Current progress

Rolling log of the most recent results, newest first. At most five entries are
kept. Pruning is triggered only by overflow: with five or fewer entries nothing
is removed, regardless of age. When a new result would make six, entries younger
than one hour are kept temporarily; the oldest is removed once it is at least
one hour old. This file is committed and pushed only at important milestones,
not after every run; entries are written in detail.

## In progress

- **Product-vs-sum on a harder corpus:** the next probe is the E8/E14 recipes on
  WikiText-103 (matched params/steps), to test whether TinyStories is masking
  the sum's extra rank.
- **Q6 to 10,000 steps:** the current best bpb at 4,000 is 0.8489 vs the
  shipped Q2 0.839 at 10,000; a like-for-like run is needed before shipping.
- **Bilingual corpus (Phase 1):** mix and pipeline fixed in `docs/data_plan.md`
  (60/40 bytes, math/science/logic/literature/encyclopedic); pilot next.

## Results

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

## 2026-10-23T16:55+03:00 — Q6 BPE-4096: tokenizer is the win

- val loss **2.3292**, PPL **10.27**, bpb **0.8489** (2,473,208 params)
- Q4 at the matched step 4,000: bpb **0.9248**; Q6 is **8.2 % better** and
  within 1.2 % of Q2's 10,000-step 0.839 in 40 % of the steps
- tokenizer compression **3.07 → 3.96 bytes/token**; matrix layers are
  identical to Q4, the only extra parameters are 393K in the factorised
  embedding — this is a tokenizer gain, not new architectural capacity
- the first attempt OOMed (the `32×192×4096` logits tensor is ~100 MB fp32);
  the loss now runs in checkpointed row chunks (`model.loss_chunk=1024`)
- 0.869 s/step, 58 min, peak VRAM **4.28 GB**; diagnostics `eff_rank(P)`
  **139.4**, `Var sigma(theta)` **0.00438**, `cos_prev_max` **0.848**
- Q2 remains the shipped checkpoint until Q6 is trained to 10,000 steps

## 2026-10-23T10:30+03:00 — Q4 full backprop at 2M: win

- val loss **1.9653**, PPL **7.137**, bpb **0.9248** (2,079,992 params)
- Q2 at the matched step 4,000: bpb **0.962**; full backprop is **3.9 %
  better** at the same cost (36.2 MFLOP/token, 4.31 GB peak VRAM)
- `eff_rank(P)` **140.2**, `Var sigma(theta)` **0.00558**; the stop-gradient
  stays off for all following runs
- 0.820 s/step, 54.7 min; Q4 is the new 2M reference

## 2026-10-23T09:32+03:00 — E14 sum composition: final

- val loss **1.0750**, PPL **2.93**, bpb **1.551** (190,348 params)
- E8 product at the same step: bpb **1.479**; the sum is **4.9 % worse** while
  using **3.1×** the FLOPs/token (11.5 vs 3.70 MFLOP)
- `eff_rank(P)` **97.66** vs E8's **60.27**: the sum removes rank compression,
  but that does not translate into quality
- `alpha` mean **0.993**, std **0.044**, range 0.915–1.206: no channel
  specialisation; `Var sigma(theta)` **0.00767** is still below 0.01
- 0.428 s/step, 28.6 min, peak VRAM **2.58 GB**; the product remains the
  default and E14 is a negative result, not a shipped model
