# GPU budget per stage

Measured speed: **2.552 s/step** (median of the last ten steps of session 5, one Kaggle T4, batch 32, seq 512), **16,384 tokens per step**. The corpus size is read from the epoch bin's `tokenizer_meta.json`, so the table below is arithmetic on real numbers, not an estimate.

| stage | steps | tokens | GPU hours |
|---|---|---|---|
| Q10 pretraining, full epoch | 172,273 | 2.82B | 122.1 |
| done so far (step 36,549) | 42,700 | 0.70B | 30.3 |
| left in the epoch | 129,573 | 2.12B | 91.9 |
| decay Q10A (the anneal bin) | 51,294 | 0.84B | 36.4 |
| SFT S10 (4,000 steps) | 4,000 | 0.07B | 2.8 |

**131 GPU hours remain** -- about 12 sessions of 11 h, which is 4.4 weeks at the free Kaggle quota of ~30 h/week.

One session (`MINUTES = 660`, 11 h) is ~15,517 steps (0.25B tokens). It is 11 h and not 12 because the platform caps a session at 720 min and the 5.7 GB bin copy takes ~10 min.

## What the stages are

- **Epoch**: 2,822,521,746 tokens = 172,273 steps, one pass over the rebuilt corpus (EN 59 / RU 41 by tokens).
- **Decay Q10A**: 51,294 steps (29.8 % of the epoch -- the anneal bin divided by the step size, so it is read exactly once), the same domain profile selected by quality, plus 5 % chat, LR 3e-4 -> 1e-6.
- **SFT S10**: 4,000 steps = 65.5M tokens, 18 % of the SFT reservoir; a full epoch there would memorise.

## What is not counted

- reference evals (`eval_reference.py`, GPT-2 on the same blocks) -- minutes each;
- corpus builds and dataset uploads -- CPU and network, no GPU;
- local runs on the 1660 Ti -- they do not touch the Kaggle quota.
