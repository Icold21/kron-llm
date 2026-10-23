# Current progress

Rolling log of the most recent results, newest first. At most five entries are
kept. Pruning is triggered only by overflow: with five or fewer entries nothing
is removed, regardless of age. When a new result would make six, entries younger
than one hour are kept temporarily; the oldest is removed once it is at least
one hour old. This file is committed and pushed after every result.

## In progress

- **E8** (E2 + full backprop, no stop-gradient): stopped at step 4,400; best
  checkpoint is step 4,000 (bpb 1.479). The default 3-epoch schedule would have
  run ~16k steps, so the ablation was cut short to keep it cheap.
- **E14** (E2 + sum composition + full backprop): code implemented in
  `KronCompose`/`KronBank`; 66 tests pass, 2 skipped on CPU. Starting now for
  4,000 steps, matching E8's best checkpoint.

## Results

## 2026-10-23T09:03+03:00 — E8 stopped; best at step 4,000

- best val loss **1.0251**, PPL **2.79**, bpb **1.479** (190,108 params)
- at step 2,500: bpb **1.586** vs E2 **1.659**; full backprop helps quality
- diagnostics at 4,000: `eff_rank(P)` **60.27**, `eff_rank(sum)` **101.23**,
  ratio **1.70**; `Var sigma(theta)` **0.00248**, `cos_prev_max` **0.761**
- full backprop does **not** remove the product's rank compression; H3 is not
  confirmed

## 2026-10-23T08:52+03:00 — E8 at step 3,000

- val loss **1.0742**, PPL **2.93**, bpb **1.550**
- 190,108 params, ~0.42 s/step; full backprop enabled
- still running to the end of the 3-epoch schedule

## 2026-10-23T08:46+03:00 — E8 live at step 2,000

- val loss **1.1476**, PPL **3.15**, bpb **1.656**
- 190,108 params, 0.414 s/step; run continues to 2,500
- full backprop is enabled (`model.kron.stop_gradient=False`)
