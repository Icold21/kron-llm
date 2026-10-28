# Decoding settings, measured

The recipe the reports quote (T = 1.0, top_k = 8, repetition penalty 1.1) started
as a judgement call.  This document replaces that judgement with measurements on
the step-35,115 checkpoint, and records what the numbers do **not** show.

Reproduce with:

```
python scripts/tune_decoding.py --seeds 2 --out data/decoding_sweep.json      # 20 configs
python scripts/tune_decoding.py --seeds 2 --grid "20,21,...,27" \
    --out data/decoding_sweep_b.json                                          # top_p stage
python scripts/tune_decoding.py --seeds 8 --grid "11,12,10,1" \
    --out data/decoding_sweep_c.json                                          # 8 seeds
python scripts/decoding_stats.py --sweep data/decoding_sweep_c.json \
    --reference 1.0/8/0.0/1.1 --pick 1.0/8/0.0/1.05 0.8/8/0.0/1.1 1.0/8/0.0/1.2
python scripts/decoding_mass.py                                               # kept mass
python scripts/decoding_bind.py                                               # when top_p binds
python scripts/decoding_report.py --sweep data/decoding_sweep.json \
    data/decoding_sweep_b.json                                                # ranked table
python scripts/decoding_examples.py --pick 1.0/8/0.0/1.1 0.8/8/0.0/1.1        # the texts
```

## How it is scored

Every configuration runs the same 7 prompts with the same seeds, so comparisons
are paired.  Four things are measured:

| metric | what it is | why it is not self-report |
|---|---|---|
| judge bpb (EN) | GPT-2 scores bits/byte of the continuation | a different model, different tokenizer, never saw these weights |
| judge bpb (RU) | ruGPT3-small, same computation | ditto |
| loops | share of samples where a character 3-gram repeats 3+ times | degeneration is the failure mode of small models, so it is counted |
| distinct-3 | unique character 3-grams over all of them | catches a setting that wins the judge by collapsing into safe phrases |
| task checks | code parses, `17 + 25 =` starts with 42, identity mentions the model | the behaviours a reader will try first |

The two judge columns are never pooled: English runs at ~2.9 bytes per token and
Cyrillic at ~5, so one average would be dominated by the Russian column.

## Result: truncation is doing real work

The checkpoint is at 26 % of one epoch, and its distribution is flat.  Measured
per decoding step (`scripts/decoding_mass.py`, 288 steps):

| | mean |
|---|---:|
| mass of the top-1 token | **0.34** |
| mass of the top-8 | **0.61** |
| mass of the top-40 | 0.81 |
| tokens needed to reach mass 0.9 | **~135** |
| tokens needed to reach mass 0.95 | ~258 |

So `top_k = 8` discards 39 % of the model's own probability mass at every step,
and the text still reads better than the untruncated distribution (EN judge 1.65
versus 2.24, paired difference +0.59 ± 0.14 — four standard errors).  This is
worth stating plainly in any write-up: at 26 % of the epoch, part of the visible
quality is the decoder, not the model.

## Result: what actually matters

Eight seeds, paired against the reference recipe (T = 1.0, top_k = 8, rp 1.1):

| config | EN bpb | Δ EN | RU bpb | Δ RU | verdict |
|---|---:|---:|---:|---:|---|
| T=1.0, k=8, rp=1.1 (reference) | 1.739 | — | 0.989 | — | — |
| T=1.0, k=8, rp=1.05 | 1.652 | −0.087 ± 0.077 | 0.937 | −0.053 ± 0.051 | noise |
| T=0.8, k=8, rp=1.1 | 1.594 | −0.145 ± 0.081 | 0.935 | −0.054 ± 0.058 | borderline (1.8σ) |
| T=1.0, k=8, rp=1.2 | 1.973 | **+0.233 ± 0.088** | 1.008 | +0.019 ± 0.035 | **worse on EN** |

Only one effect clears two standard errors: **repetition penalty 1.2 is worse for
English**.  The rest is inside the noise, and the sign of the T = 0.8 difference
even flipped between the two-seed and the eight-seed runs, which is what noise
looks like.  The honest statement is therefore:

* **rp 1.05–1.1 is the band.**  1.1 stays the default because 1.05 is not
  measurably better and 1.1 matches what the reports already quote.
* **rp 1.2 costs English**, so the earlier "не выше 1.2" rule was right but the
  border is already crossed at 1.2 itself.
* **T 0.8–1.0 cannot be separated** by this experiment on prose.  It can on code
  (below), which is why the code sample uses 0.8.

## Result: top_p is not an alternative to top_k

`top_p` truncates by mass, `top_k` by count.  On a flat distribution they are not
interchangeable — `top_p = 0.9` keeps ~135 tokens here, seventeen times wider than
`top_k = 8`:

| truncation | EN bpb | RU bpb | loops |
|---|---:|---:|---:|
| top_k = 8 | **1.652** | **0.909** | 0.79 |
| top_k = 40 | 2.123 | 1.146 | 0.86 |
| top_p = 0.90 | 2.206 | 1.141 | 0.64 |
| top_p = 0.95 | 2.355 | 1.451 | 0.50 |
| none | 2.239 | 1.480 | 0.57 |

`top_p = 0.9` and no truncation at all land in the same place, which is the whole
point: at this checkpoint a probability threshold is not a truncation.

## Result: combining the two

Combining is not a bad idea in general — it bounds both the count and the mass —
but whether it does anything depends on which of the two binds first.  Given
`mass@8 = 0.61` on average, `top_p = 0.9` should almost never bite under a
`top_k = 8` cap.  It does bite, occasionally, and only on the confident steps
(`scripts/decoding_bind.py`, 288 steps):

| | share of steps where `top_p` keeps fewer than 8 tokens |
|---|---:|
| top_p = 0.5 | 65.3 % |
| top_p = 0.8 | 25.3 % |
| **top_p = 0.9** | **17.0 %** |
| top_p = 0.95 | 11.8 % |
| top_p = 0.99 | 5.6 % |

The binding steps are concentrated in Russian (12–15 of 48 steps per prompt)
rather than English (5–10), because Russian text has more steps where one token
dominates.

Combining `top_k = 8` with `top_p = 0.9` therefore tightens only those confident
steps, and on eight seeds that is a small but real gain (paired against plain
`top_k = 8`, rp 1.1):

| config | EN bpb | Δ EN | RU bpb | Δ RU |
|---|---:|---:|---:|---:|
| top_k=8 (reference) | 1.739 | — | 0.989 | — |
| **top_k=8, top_p=0.90** | **1.630** | **−0.109 ± 0.052 (2.1σ)** | **0.906** | −0.083 ± 0.063 |
| top_k=8, top_p=0.95 | 1.763 | +0.024 ± 0.070 | 0.979 | −0.010 ± 0.050 |

`top_p = 0.9` clears two standard errors on English and moves Russian the same
way without reaching significance; `top_p = 0.95` does nothing, which is what the
17 % / 12 % binding rates predict.  So the combination is not decoration after
all — but the useful threshold is the one that bites on a sixth of the steps, not
the comfortable one that bites on a tenth.

The margin is thin (2.1σ, one checkpoint, eight seeds) and should be re-measured
when the distribution sharpens at the end of the epoch.  It is adopted anyway
because it cannot hurt: the two settings are identical on 83 % of steps, and on
the remaining 17 % the narrower one won.


## The recipe

| purpose | settings | why |
|---|---|---|
| measurement (default) | T = 1.0, top_k = 8, top_p = 0.9, rp = 1.1 | nothing in the grid beats it; `top_p = 0.9` adds a measured −0.11 bpb on English |
| code | T = 0.8, top_k = 8, rp = 1.1 | lower entropy keeps indentation and `def` lines; the judge cannot see this, the samples can |
| demo only, must be labelled | T = 0.8, top_k = 40, rp = 1.2 | smoothest reading; do not quote it as the model's quality |

## What this does not show

* **Two to eight seeds, two prompts per language.**  The standard error on a
  single configuration is 0.05–0.10 bpb, so differences below ~0.15 are not
  resolvable here.  A claim that T = 0.9 beats T = 1.0 would need a bigger
  experiment than this one.
* **Judges are 125M models.**  They reward text that looks like ordinary prose,
  which is not the same as text that is true, or structured, or useful.  The task
  checks are there to keep that from being the only criterion.
* **One checkpoint.**  Everything here is measured at 26 % of one epoch.  A
  finished model has a sharper distribution, so the optimal truncation will move
  — probably towards wider `top_k` or towards `top_p` becoming meaningful.  The
  scripts are cheap to re-run at the epoch end; the numbers should be.
* **`identity` and arithmetic do not discriminate yet.**  No configuration got a
  single identity sample to mention the model, and none got `17 + 25 =` to start
  with 42, so those columns are recorded but they are not evidence about decoding.
