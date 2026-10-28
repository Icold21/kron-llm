# What can be improved for free

The question was whether the model can be made better **without training and
without Kaggle quota**.  Five levers were measured at step 35,115 (25.6 % of one
epoch).  One of them works and is worth adopting; three do nothing; one is a
trap that looks like it works because of how it is scored.

Reproduce with:

```
python scripts/soup_q10.py --blocks 1024 --save-best data/kaggle_out/q10/soup.pt
python scripts/free_gains.py --part all --seeds 3 --out data/free_gains.json
python scripts/free_gains.py --ckpt data/kaggle_out/q10/last.pt --part samplers \
    --only "recipe+top_p0.9" --seeds 20 --out data/free_gains_gen_last.json
python scripts/free_gains.py --ckpt data/kaggle_out/q10/soup.pt --part samplers \
    --only "recipe+top_p0.9" --seeds 20 --out data/free_gains_gen_soup.json
python scripts/compare_generation.py data/free_gains_gen_last.json \
    data/free_gains_gen_soup.json --label-a last --label-b soup
```

## 1. Weight averaging (a "model soup") — works, −1.9 % bpb

`best.pt` (step 35,000) and `last.pt` (step 35,115) are 115 steps apart in the
same constant-LR phase, which by the usual rule of thumb should be too close to
matter.  It matters anyway.  Bits per byte on 1,024 strided blocks of the val bin
(0.5 M tokens, the same blocks `scripts/eval_reference.py` uses):

| weights | bpb | EN | RU |
|---|---:|---:|---:|
| `best.pt` (α = 0) | 1.1736 | 1.4289 | 0.9288 |
| 75 / 25 | 1.1563 | 1.4085 | 0.9145 |
| **50 / 50** | **1.1511** | **1.4024** | **0.9102** |
| 25 / 75 | 1.1575 | 1.4100 | 0.9154 |
| `last.pt` (α = 1) | 1.1760 | 1.4322 | 0.9304 |

α is the weight of `last.pt`.  The curve is smooth and its minimum sits at the
middle, which is the signature of averaging out SGD noise rather than of one
lucky point: a single interpolated model beats **both** parents by 0.022–0.025
bpb, and the two parents are 0.002 bpb apart from each other.  Both languages
move together (EN −1.9 %, RU −2.0 %), so this is not a language-specific effect.

The weights really are different — mean relative L2 distance between the two
checkpoints is 2.8 % for the transformer blocks, 5.4 % for the embeddings, 0.1 %
for the final norm.  115 steps are enough to decorrelate the noise, not the
signal.

**Caveat, stated plainly:** the val bin is the pretraining val split, so this is
in-domain bpb.  The independent check is generation: 7 prompts × 20 seeds with
the recipe fixed, scored by the same two judges, paired per (prompt, seed)
(`scripts/compare_generation.py`):

| metric | `last.pt` | soup | paired Δ (soup − last) | verdict |
|---|---:|---:|---:|---|
| judge bpb (EN+RU prose, n=80) | 1.295 | 1.211 | **−0.083 ± 0.038** | **2.2σ, real** |
| loops (n=140) | 0.871 | 0.929 | +0.057 ± 0.032 | 1.8σ, slightly worse |
| distinct-3 (n=140) | 0.763 | 0.754 | −0.009 ± 0.013 | 0.7σ, no change |
| code parses (n=140) | 0.200 | 0.200 | 0.000 ± 0.020 | no change |
| continuation length, chars | — | — | −6.9 ± 3.6 | 1.9σ, shorter |

So the soup buys about **0.08 bpb of judged prose quality** and pays for it with
**more repetition** — the two effects are the same order of significance, and
neither is large.  That is a fair summary of a 2 % weight average: better
average behaviour, slightly more self-repetition.  It is worth adopting because
the bpb gain is solid and costs nothing, not because it is a transformation.

Why this is the only lever that moved the number: bpb is a property of the
weights, and averaging is the only free operation that changes the weights.

### The soup cannot be extended right now

A soup wants *distant* checkpoints (different LR phases, different data orders),
not two neighbours.  The distant ones are not reachable:

* `icold21/kron-q10-ckpt` returns **HTTP 401** to the public API, so its file
  list cannot even be enumerated without a token.
* the Kaggle kernel is re-pushed under the same slug every session, so each push
  supersedes the previous output; only the latest `last.pt`/`best.pt` survive.
* `train.py` only writes `last.pt` and `best.pt` into the kernel output.

So 35,000 + 35,115 is the whole menu today.  Two things follow:

1. **Publish `soup.pt`, not `last.pt`,** as the checkpoint that demos and the
   README point at (weights-only, same 40 MB).
2. **Soup across phases at the epoch end.**  `epoch1.pt` (LR 3e-4) and
   `annealed.pt` (LR 1e-6, after Q10A) are genuinely distant points; averaging
   those is where the effect should be larger, and it costs one script run.

The cheaper structural version of the same idea is an **EMA of the weights inside
the trainer** (a shadow copy updated every step, decay ~0.999, no extra compute,
no extra memory worth counting at 10 M parameters).  It gives the same noise
averaging continuously instead of once, and it would have to be added to
`train.py` before the Q10A launch to be usable there.

## 2. Sampler families the earlier sweep did not try

`docs/decoding.md` settled temperature, `top_k`, `top_p` and the repetition
penalty.  It never tried `min_p` (a floor relative to the top token), locally
typical sampling, or an n-gram ban.  All three are now in
`KronTransformer.generate`; 7 prompts × 3 seeds, judges as in `decoding.md`:

| sampler | EN bpb | RU bpb | loops | distinct-3 | code parses |
|---|---:|---:|---:|---:|---:|
| greedy | 0.462 | 0.492 | **1.00** | **0.241** | 1.00 |
| greedy + no-repeat-3 | 1.728 | 0.926 | 0.86 | 0.741 | 0.00 |
| T=1, k=8, rp=1.1 | 1.593 | **0.894** | 0.86 | 0.811 | 0.33 |
| **T=1, k=8, top_p=0.9, rp=1.1** | **1.538** | 0.932 | 0.81 | 0.774 | 0.33 |
| min_p = 0.02 | 1.957 | 1.004 | 0.86 | 0.786 | 0.00 |
| min_p = 0.05 | 1.956 | 1.046 | 0.81 | 0.754 | 0.00 |
| min_p = 0.10 | 1.691 | 0.970 | 0.95 | 0.640 | 0.33 |
| typical = 0.5 | 2.240 | 1.147 | 0.71 | **0.858** | 0.33 |
| typical = 0.9 | 2.231 | 1.285 | **0.67** | 0.837 | 0.00 |
| k=8, rp=1.1, no-repeat-3 | 1.715 | 1.027 | 0.81 | 0.834 | 0.33 |

Two things to read out of this table:

* **Greedy is the trap.**  Its judge bpb (0.46) is the *lowest in the table* and
  it is the worst text in the table: `loops = 1.00`, `distinct-3 = 0.24` — every
  greedy sample loops, and the judge loves it because `" 1000000000000000"` is
  perfectly predictable.  Any claim of the form "greedy decodes better because
  GPT-2 gives it a lower bpb" is measuring the loop.  This is why the probes
  carry `loops` and `distinct-3` next to the judge.
* **Nothing beats the existing recipe.**  `min_p` and typical sampling lose on
  the judge by 0.15–0.7 bpb; typical sampling does cut loops (0.67 vs 0.81) and
  raises diversity, at a price the prose judge refuses to pay.  The n-gram ban
  costs 0.12–0.18 bpb of English and does not reduce loops here, because the
  loops this model produces are token-level, not exact 3-gram repeats.

So `docs/decoding.md`'s recipe stands, now against a wider field.  The one
addition worth keeping in the toolbox is `temperature=0` + `no_repeat_ngram_size`
for structured output, where loops are the failure mode and prose quality is not
the criterion.

## 3. Prompt priming — does not work

A base model has no instruction tuning, so the prefix is part of the program.
Three prefixes were prepended to the same prompts, recipe fixed:

| primer | EN bpb | RU bpb | loops | distinct-3 |
|---|---:|---:|---:|---:|
| none | **1.538** | 0.932 | 0.81 | 0.774 |
| style ("Ниже — отрывок ясной русской прозы.") | 1.568 | **0.906** | 0.95 | 0.796 |
| few-shot ("Вопрос: … Ответ: …") | 1.982 | 1.030 | 0.76 | 0.710 |

A style primer moves Russian the right way by 0.026 bpb and English the wrong way
by 0.030 — that is a wash at this sample size.  Few-shot priming makes both
languages clearly worse (−0.44 EN), which is expected at 26 % of an epoch: the
model cannot use an instruction pattern it has not been trained on, and the extra
context only gives it more ways to continue.  **Do not add primers to the demos
in the hope of a free gain.**  After SFT (`S10`) this experiment has to be re-run;
there the answer should change, and that is the point of SFT.

## 4. Best-of-N reranking — real, but mostly not for the reason it looks

Sample N continuations, keep the one the model itself finds most likely.  No
verifier, no extra model, N× inference only.  Mean GPT-2 bpb of the 4 prose
prompts (2 EN, 2 RU):

| N | pick by model likelihood | random pick | first non-looping |
|---|---:|---:|---:|
| 1 | 1.123 | 1.123 | 1.123 |
| 4 | 1.053 | 1.286 | 1.178 |
| 8 | **0.929** | 1.441 | 1.178 |

Read the three columns carefully, because two of them are honest and one is a
tautology:

* **random pick** gets *worse* with N (1.12 → 1.44) — that is not a bug, it is
  the baseline: with more draws you are more likely to draw a bad one.
* **first non-looping** is the honest part of the gain: 1.44 → 1.18.  Rejecting
  samples that repeat themselves is a real improvement in text quality and needs
  no likelihood at all.
* **pick by model likelihood** (0.93) is partly circular: the score *is* "how
  likely is this text to a 125 M LM", so choosing the most likely of eight
  optimises the metric by construction.  It is still useful — it is the standard
  best-of-N recipe and it does select the samples a human would call cleaner —
  but the number 0.93 must not be quoted as "2× better text".

What would make best-of-N genuinely valuable is a **verifier that is not a
likelihood**: does the code parse, does the arithmetic check out.  For code this
works today (a third of samples parse, so best-of-8 almost always has one that
parses).  For arithmetic it does not — see below.

## 5. Self-consistency on arithmetic — 0 % and no way around it

50 two-digit sums, 8 samples each, majority vote over the first number in the
continuation:

| | accuracy |
|---|---:|
| first sample | 2 % (1/50) |
| majority of 8 | **0 %** |
| any of 8 correct | 4 % |

The model does not compute: `17 + 25 =` continues as `11000 # 91110`, `10 = 10.59
= 40`, `0.15 + 34 = .4090 * 89 = (1.7`.  Majority voting cannot recover an answer
that is never sampled, and voting makes it *worse* than taking one sample,
because the wrong answers agree with each other more often than the right one
appears.  Any "free" arithmetic gain at this checkpoint is an illusion; this is
what SFT and the decay pass are for.

## Summary

| lever | cost | measured effect | adopt? |
|---|---|---|---|
| weight averaging of two nearby checkpoints | one script run | **−0.025 bpb (−1.9 %), both languages**; judged prose −0.083 ± 0.038 bpb at 2.2σ, loops +0.057 ± 0.032 | **yes — publish `soup.pt`** |
| EMA of weights during training | none (shadow copy) | same mechanism, continuous | yes, for Q10A |
| min_p / typical / n-gram ban | none | worse than the current recipe | no |
| prompt priming | none | wash on style, worse on few-shot | no |
| best-of-N by likelihood | N× inference | real for loop rejection, partly circular as scored | only with a non-likelihood verifier |
| self-consistency on arithmetic | N× inference | 0 % | no |

## What this does not show

* **One pair of checkpoints.**  The soup result is one interpolation of two
  neighbours.  It should be re-measured at the epoch end on `epoch1` + `annealed`,
  where the expected gain is larger and the right α is unknown.
* **In-domain bpb.**  The val bin is the pretraining distribution.  A 1.9 % gain
  there does not automatically transfer to held-out SFT data, code, or anything
  the model has not seen; the paired generation probe in §1 is the only
  out-of-metric check, and it is 140 samples.
* **Generation differences are small and paired.**  The 20-seed run gives
  2.2σ on judged prose and 1.8σ on loops, in opposite directions; the other
  columns do not move.  The soup is a small win, not a different model.
* **Nothing here creates knowledge.**  Every lever either averages existing
  weights or selects among existing samples.  Arithmetic stays at 0 %, identity
  stays unmentioned, and the model is still at 25.6 % of one epoch.
