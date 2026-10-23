# Current progress

Rolling log of the most recent results, newest first. At most five entries are
kept. An entry is never removed before it is one hour old; if more than five
results arrive within an hour, all are kept temporarily and the oldest is
pruned once it is older than one hour. This file is committed and pushed after
every result.

## In progress

- **E8** (E2 + full backprop, no stop-gradient): 190,108 params; at step 2,200
  of 2,500. Latest validation at step 2,000: bpb 1.656.
- **E14** (E2 + sum composition + full backprop): code implemented in
  `KronCompose`/`KronBank`; CPU forward and bank-parity checks pass. Full test
  suite and training run are pending until E8 finishes.

## Results

## 2026-10-23T08:46+03:00 — E8 live at step 2,000

- val loss **1.1476**, PPL **3.15**, bpb **1.656**
- 190,108 params, 0.414 s/step; run continues to 2,500
- full backprop is enabled (`model.kron.stop_gradient=False`)
