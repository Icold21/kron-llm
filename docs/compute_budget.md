# GPU budget per stage

Measured speed: **2.552 s/step** (median of the last ten steps of session 5, one Kaggle T4, batch 32, seq 512), **16,384 tokens per step**. The corpus size is read from the epoch bin's `tokenizer_meta.json`, so the table below is arithmetic on real numbers, not an estimate.

| stage | steps | tokens | GPU hours |
|---|---|---|---|
| Q10 pretraining, full epoch | 169,884 | 2.78B | 120.4 |
| done so far (step 36,549) | 36,549 | 0.60B | 25.9 |
| left in the epoch | 133,335 | 2.18B | 94.5 |
| decay Q10A (the anneal bin) | 49,384 | 0.81B | 35.0 |
| SFT S10 (4,000 steps) | 4,000 | 0.07B | 2.8 |

**132 GPU hours remain** -- about 12 sessions of 11 h, which is 4.4 weeks at the free Kaggle quota of ~30 h/week.

One session (`MINUTES = 660`, 11 h) is ~15,517 steps (0.25B tokens). It is 11 h and not 12 because the platform caps a session at 720 min and the 5.7 GB bin copy takes ~10 min.

## What the stages are

- **Epoch**: 2,783,383,150 tokens = 169,884 steps, one pass over the rebuilt corpus (EN 59 / RU 41 by tokens).
- **Decay Q10A**: 49,384 steps (29.1 % of the epoch -- the anneal bin divided by the step size, so it is read exactly once), the same domain profile selected by quality, plus 5 % chat, LR 3e-4 -> 1e-6.
- **SFT S10**: 4,000 steps = 65.5M tokens, 18 % of the SFT reservoir; a full epoch there would memorise.

## What is not counted

- reference evals (`eval_reference.py`, GPT-2 on the same blocks) -- minutes each;
- corpus builds and dataset uploads -- CPU and network, no GPU;
- local runs on the 1660 Ti -- they do not touch the Kaggle quota.
