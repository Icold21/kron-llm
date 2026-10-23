# Current progress

Rolling log of the most recent results, newest first. At most five entries are
kept. Pruning is triggered only by overflow: with five or fewer entries nothing
is removed, regardless of age. When a new result would make six, entries younger
than one hour are kept temporarily; the oldest is removed once it is at least
one hour old. This file is committed and pushed after every result.

## In progress

- **E8** (E2 + full backprop, no stop-gradient): 190,108 params; at step 3,000.
  Latest validation at step 3,000: bpb 1.550. The run continues to the end of
  its 3-epoch schedule.
- **E14** (E2 + sum composition + full backprop): code implemented in
  `KronCompose`/`KronBank`; 66 tests pass, 2 skipped on CPU. Training will start
  after E8 finishes.

## Results

## 2026-10-23T08:52+03:00 — E8 at step 3,000

- val loss **1.0742**, PPL **2.93**, bpb **1.550**
- 190,108 params, ~0.42 s/step; full backprop enabled
- still running to the end of the 3-epoch schedule

## 2026-10-23T08:46+03:00 — E8 live at step 2,000

- val loss **1.1476**, PPL **3.15**, bpb **1.656**
- 190,108 params, 0.414 s/step; run continues to 2,500
- full backprop is enabled (`model.kron.stop_gradient=False`)
