# The 8k vocabulary: what it costs and what it would buy

Decision (2026-10-29): **keep the 8192-token BPE.** No retraining, no
transplant before the epoch ends. This note records the measurement behind that
decision, because the question ("is 8k enough, or will SFT fall apart?") comes
back every time the model is compared with something that has 32k or 50k.

Everything below is measured by `scripts/vocab_probe.py` on the real corpus:
two BPEs (8192 and 16384) trained on the *same* 164.7 MB sample with the *same*
recipe (byte-level, digits isolated, five reserved ids), then scored on held-out
rows of every (domain, language) bucket. Raw numbers in
`data/vocab_probe_digits.json`.

## 1. The vocabulary is bilingual, not English with Cyrillic bolted on

Decoding all 8,187 learned pieces of `data/bilingual/tokenizer.json`:

| kind | pieces | average size |
|---|---:|---:|
| pure Cyrillic | 4,036 | 8.8 bytes (~4.4 letters) |
| Latin, length > 1 | 3,564 | -- |
| bytes, digits, punctuation | ~580 | 1 byte |
| specials | 5 | -- |

Half the merge table is Russian. That is why Russian runs at 4.96 bytes/token
instead of the 1.7 a monolingual-English tokenizer manages, and why the model
needs no vocabulary work to *speak* Russian at all.

## 2. Size is not the variable -- balance is

Same text, two tokenizers, bytes per token (higher is better):

| text | our 8k | GPT-2 (50k) | ratio |
|---|---:|---:|---:|
| Russian maths | **3.69** | 1.69 | 2.2x better |
| Russian chat | **5.01** | 1.72 | 2.9x better |
| English maths | 2.46 | 3.35 | 1.36x worse |
| English chat | 3.06 | 4.36 | 1.42x worse |

A vocabulary six times larger, trained on English, is 2.2-2.9x *worse* on
Russian than ours. The 60/40 contract is a byte contract; paying 1.4x on
English to avoid paying 2.5x on Russian is the correct side of that trade.

## 3. What 16k would buy

| bucket | 8k | 16k | gain | records fitting 512 tokens |
|---|---:|---:|---:|---:|
| math_en | 2.817 | 2.961 | +5.1 % | 0.77 -> 0.84 |
| math_ru | 4.351 | 4.800 | +10.3 % | 0.79 -> 0.86 |
| code_en | 3.369 | 3.603 | +6.9 % | **0.30 -> 0.45** |
| code_ru | 4.316 | 4.623 | +7.1 % | 0.90 -> 0.96 |
| qa_en | 3.632 | 4.032 | +11.0 % | 0.44 -> 0.51 |
| qa_ru | 4.985 | 5.746 | +15.3 % | 0.69 -> 0.72 |
| sites_en | 3.468 | 3.813 | +9.9 % | 0.58 -> 0.64 |
| sites_ru | 5.127 | 5.656 | +10.3 % | 1.00 -> 1.00 |

Weighted by the bytes each bucket holds in the corpus: **+8.8 %**. That is the
whole prize -- the same 65.5M-token SFT budget would cover 9 % more text, and
the 512-token filter would keep 4-9 percentage points more records.

## 4. What 16k would cost

- parameters **9.94M -> 12.04M (+21 %)**, and every one of those 2.1M is an
  embedding row at rank 256. The trunk stays at 7.71M: the model gets bigger,
  not smarter.
- the logits path goes from 2.10M to 4.19M MACs per token (the factored
  embedding is `(h . U) . E^T`, so cost scales with the row count). Whole
  forward pass +21 %, **+11 % per byte of text** -- 16k is more expensive per
  byte, not less.
- a full retrain is 137k steps ~ 33 h plus 9.6 h of decay ~ **42 GPU-hours**,
  about a week and a half of the 30 h/week quota, and the 35,115 steps already
  spent (8.3 h) are thrown away.

Nine percent of text for eleven percent more compute, twenty-one percent more
parameters and a lost trajectory is not a trade worth making at 10M.

## 5. The real risk to SFT is not the vocabulary

A scan of `data/bilingual/train.bpe.bin` (2.25B tokens) for the chat tokens:

| token | occurrences |
|---|---:|
| `<|system|>` | 0 |
| `<|user|>` | 5 |
| `<|assistant|>` | 6 |
| `<|eot|>` | 0 |
| `<|endoftext|>` | 948,975 |

The four chat tokens are **cold**: their embeddings are untrained and the model
has never seen the format. That, not the merge table, is what would make the
first few hundred SFT steps useless. Fixed on 2026-10-29 by putting 2.5 % chat
into the decay pass (`scripts/build_anneal_chat.py`, shares in
`scripts/build_anneal_mix.py`), so Q10A hands SFT a model that already knows
what `<|user|>` means.

The remaining risks are the 512-token window (the filter drops 23 % of maths and
70 % of code records), 4,000 steps at LR 3e-4 (forgetting), and the fact that
the base is at 25.6 % of an epoch. None of them is fixed by a bigger table.

## 6. If a 16k vocabulary is ever wanted

It does not have to be a retrain. Because the vocabulary keys are byte strings,
old pieces map to new ones exactly, and a new merge is by construction the
concatenation of its two children, so its row can be initialised as the **sum of
the two child rows** -- a near-exact initialisation rather than a random one.
Then 2-3k steps at low LR adapt it: 30-60 minutes of GPU instead of 42
hours. This is the fallback, not the plan.
