# Current progress

Rolling log of the most recent results, newest first. At most five entries are
kept. Pruning is triggered only by overflow: with five or fewer entries nothing
is removed, regardless of age. When a new result would make six, entries younger
than one hour are kept temporarily; the oldest is removed once it is at least
one hour old. This file is committed and pushed only at important milestones,
not after every run; entries are written in detail.

## In progress

- **Architecture freeze (gate A):** Q5 and Q7 are closed as negative, so the
  10M budget must go to width/depth/context, not more Kronecker summands or a
  compressed embedding table. Next: allocation runs at matched activation
  footprint, then `seq_len=256`.
- **Bilingual data + tokenizer (roadmap Phase 1):** 8K BPE on mixed EN/RU with
  fertility and contamination reports. TinyStories remains only the bench.

## Results

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

## 2026-10-23T09:03+03:00 — E8 stopped; best at step 4,000

- best val loss **1.0251**, PPL **2.79**, bpb **1.479** (190,108 params)
- at step 2,500: bpb **1.586** vs E2 **1.659**; full backprop helps quality
- diagnostics at 4,000: `eff_rank(P)` **60.27**, `eff_rank(sum)` **101.23**,
  ratio **1.70**; `Var sigma(theta)` **0.00248**, `cos_prev_max` **0.761**
- full backprop does **not** remove the product's rank compression; H3 is not
  confirmed
