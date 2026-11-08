# SFT corpus: contract, sources, filters

The supervised stage needs a different corpus from pretraining: short
question/answer pairs in a fixed chat template, weighted towards mathematics,
and bilingual in the same 60/40 English/Russian **byte** ratio the pretraining
mix uses (`docs/data_plan.md` §1).  This document records what goes in, what is
filtered out, and where the gaps are.  The generated numbers live in
`data/report_sft.md`; the code is `scripts/sft_registry.py` and
`scripts/build_sft.py`.

## 1. Contract

| | value | why |
|---|---|---|
| language split | 60 % EN / 40 % RU **by bytes** | same ratio as pretraining, so SFT does not silently re-balance the model |
| math | 38.5 % of bytes | the stage's purpose: structured task→solution pairs |
| code | 20 % of bytes | code is what carries the mathematics: following a statement and writing a loop is closer to algebra than a word problem is, and the structure of the answer stays visible |
| open Q/A | 28 % | instruction following, definitions, and — since 2026-10-30 — dialogue: the chat sources live here |
| forum Q/A (Stack Exchange, mail.ru, ELI5, Reddit) | 10 % | real questions with real answers, including technical threads |
| general instructions | 3 % | Dolly-style tasks that are neither math nor forum |
| identity | 0.5 % | 100–300 examples, guarded (see `docs/roadmap.md` §6) |

The template is the one the notebook and the sampler already use:

```
<|system|>You are KronLLM, a compact bilingual model.<|eot|>
<|user|>…<|eot|><|assistant|>…<|eot|>
```

`<|eot|>` ends a turn, token `8191` ends an example.  `sft_index.npy` stores
`(start, prompt_tokens, total_tokens, file)` per example, so a trainer *can* mask
the loss on the prompt without re-tokenising.  The current trainer does not
consume that index yet — it trains on the whole rendered text, which still
teaches the answer but spends capacity on predicting the prompt.  Wiring the
mask in is a follow-up, and the index exists so it is a data-loader change, not
a rebuild.

Examples longer than **512 tokens** (the Q10 context) are dropped at packing
time, not truncated: a window cut through the middle of an answer teaches
nothing, and the model cannot be given a longer context without retraining its
position encoding.  How much is dropped is recorded per part in `meta.json`
(`dropped_long`).

### 1.1 What the build actually achieved

The mix is built by filling each (domain, language) cell up to its byte budget
from whatever the sources can supply.  Where a cell cannot fill its budget, the
other cells keep theirs and the realised mix drifts.  The full build on
2026-10-30 produced **517,239,848 tokens** (506.9 M train / 10.4 M val),
3,111,653 examples, and this is the drift:

| | realised (bytes) | contract | note |
|---|---:|---:|---|
| English | **60.5 %** | 60 % | the ratio now holds |
| Russian | **39.5 %** | 40 % | |
| math | 46.4 % | 38.5 % | the only large domain that fills its budget, so its share rises |
| code | 16.9 % | 20 % | 234.3 MB; `code_en` stops at 166.3/194.9 |
| open Q/A | 24.5 % | 28 % | chat now fills most of it (see below) |
| forum Q/A | 11.7 % | 10 % | over budget, both cells full |
| general | 0.4 % | 3 % | Dolly is the only licensed source; 23 MB short |
| identity | 0.1 % | 0.5 % | 800 examples per language by design |

Three things changed on 2026-10-30 and they are the reason the numbers moved:

* **Dialogue.**  `qa` is 28 % of the budget and the chat sources fill most of
  it: SlimOrca 68k rows, OpenHermes 77k (multi-turn), oasst1-en 18k, counsel
  2.1k, UltraChat 3k.  The Russian cell has the first genuine dialogue it has
  ever had aside from Saiga: 3,069 oasst1-ru pairs.
* **Cross-source deduplication.**  Each shard was deduplicated against itself at
  download time but not against its neighbours, and the code sources overlap
  heavily: 99.9 % of `evol_codealpaca` was already present in
  `opencodeinstruct`/`magicoder`/`codefeedback` (measured with
  `scripts/dup_probe.py`).  `write_parts` now drops exact duplicates across the
  whole corpus, which is why `code_en` stops at 166.3 MB: 43 MB of what used to
  fill that cell was the same records a second time.
* **The stream is shuffled.**  Examples used to be written in cell order — all
  of the code, then all of the maths, then 2.9 M instruction pairs with the
  dialogue in one contiguous stretch.  They are now emitted in a deterministic
  global order (seed 20261029) and `sft_index.npy` records where each one
  landed, so a batch is a mix of domains regardless of how the trainer samples.

Two constraints still cause the drift:

* **Code is capped by length and by the per-source ceiling.**  The 512-token
  filter keeps 176k of the code rows, and each source may contribute at most
  1.8× its even share of the cell.  Raising the code share needs **answer
  chunking** (split a long solution into ≤512-token windows), not more sources.
* **The same filter costs most of UltraChat.**  Multi-turn threads are long:
  of 86k records only 3,067 fit 512 tokens.  Splitting a thread into per-turn
  examples is the natural fix and is the next change to this builder; it matters
  because that is the only source of deep multi-turn dialogue in the corpus.

The language ratio holds at 60.5/39.5 because `russian_instructions_2` carries
`cap_share 0.70` in `qa_ru` (it is the only Russian source with real volume,
and its licence turned out to be MIT), so the Russian cell fills instead of
leaving the corpus English-heavy.


## 2. Sources

Everything comes from the Hub through a mirror with a declared licence.  Nothing
is scraped: a raw crawl of `otvet.mail.ru` or Reddit would violate the licence
policy in `docs/data_plan.md` §2, so those sites enter through
`Den4ikAI/mailruQA-*`, `Pavithree/eli5` and `zhoudoe23/reddit-top-qa-dataset`.

| group | sources |
|---|---|
| math, EN | OpenMathInstruct-2 (1M slice), MetaMathQA, Orca-Math, MathInstruct, GSM8K, MATH (algebra/geometry/number theory) |
| math, RU | translated Orca-Math (`d0rj/…-ru`), translated MathInstruct, Russian math dialogues, ru-alpaca-math |
| math, synthetic | 16 templated arithmetic families in both languages: exact answers, computed by Python, with a written-out derivation. Three of them are **column drills** — addition with carries, subtraction with borrows, one-digit multiplication with partial products — written digit by digit, because a one-line "47 + 38 = 85" teaches the format but not the algorithm |
| code, EN | OpenCodeReasoning (competitive programming: statement + reasoning + solution), Magicoder-OSS-Instruct, self-oss-instruct-sc2, CodeAlpaca, python-instructions-18k |
| code, RU | English code tasks whose **question** was machine-translated to Russian; the code stays English, as it does in the wild |
| Q/A, EN | OpenOrca, OpenHermes-2.5, HotpotQA |
| Q/A, RU | saiga_scored (`opus_score ≥ 6`), russian_instructions_2 |
| forum, EN | Stack Exchange dump (best answer per thread, score ≥ 2), ELI5, Reddit top Q&A |
| forum, RU | mail.ru QA (two mirrors, short answers by design) |
| general | Dolly-15k |
| identity | generated locally from `data/identity` (CC0) |

### 2.1 Arithmetic drills write the algorithm, not just the answer

The pretraining corpus contains mathematics **prose** and almost no worked
examples: measured with `scripts/count_arithmetic_drills.py`, the mix holds
**14.5** `a op b = c` pairs per million characters, and only 3.7 in `en/math` —
because textbooks carry notation and proofs, not input-output pairs. That is why
exact arithmetic sits at 0/60 before SFT (`eval_samples.py`, step 35,115).

The synthetic arithmetic families close that gap, and since 2026-11-01 three of
them per language are **column drills**: the assistant writes the computation
digit by digit,

```
units: 7 + 4 = 11 -> write 1, carry 1
tens: 0 + 6 + 1 = 7 -> write 7
hundreds: 6 + 7 = 13 -> write 3, carry 1
thousands: 1 -> write 1
```

and the same in Russian (`пишем`, `запоминаем`, `занимаем`, `заём`). The rows are
produced *from* the digits rather than typed by hand, so a row cannot contradict
the answer, and `tests/test_arith_columns.py` parses the rendered rows back into
a number and compares it with Python. Every drill is retried until it actually
needs a carry or a borrow — a drill without one teaches nothing. They take 39 %
of the arithmetic examples (55-58 % of that source's bytes) and stay bounded by
`cap_share=0.40` of the math cell.

### 2.2 Russian code questions by translation

Code is written in English everywhere, so a bilingual code slice does not need a
Russian solution — it needs a Russian question.  `scripts/translate_code_questions.py`
takes English code tasks, splits every question into prose and non-prose
(fenced blocks, inline spans, URLs, formulas), sends only the prose through
MarianMT (`Helsinki-NLP/opus-mt-en-ru`) and puts the code back verbatim.  The
answer is copied unchanged.

Two things this pipeline learned the hard way: placeholders for code do not
survive the translator (a bracket gets re-spaced and the round-trip check fails
on 80 % of questions), so the question is *segmented* rather than masked; and
the acceptance gate has to look at the Cyrillic share of the translated prose
only, because a question that is mostly code legitimately has little Russian in
it.

The third thing it learned is worse.  Translation is batched, and the translated
prose pieces are scattered back into the questions by index.  The first version
restarted that index at zero for every question, so a question could receive the
*first* question's prose while keeping its own answer: 874 of 11,234 pairs were
such hybrids.  They are fluent, they pass every aggregate counter, and they are
only visible when the pairs are read.  `scripts/verify_translated_pairs.py` now
does exactly that — it maps each Russian question back to the question that
produced it and checks that the answer carried alongside is that question's
answer.  It found the 874; the fix was to use absolute offsets in the flattened
batch; the shard is rebuilt from the translation cache, so re-running the audit
costs no GPU.

Getting that check right took two rounds, and both false alarms are worth
knowing about.  The same English question sits in several code datasets with
different answers (the alpaca-style sets share whole rows), so comparing against
one answer per question flagged 2,474 healthy pairs; and two questions that
differ only in trailing whitespace translate to the same Russian sentence, so
keying on the hash flagged the second of each pair.  The index is therefore
`russian question -> every answer of every question that translated to it`: a
hybrid still fails, because its answer belongs to a question whose Russian text
is a different sentence.  The current shard is 33,405 records with 0 mismatches.

The cache is keyed by the hash of the English question, and the shard is always
rebuilt from it in candidate order.  A rerun therefore never re-translates what
it already has, and the file is a function of (raw shards, cache) rather than of
whichever run happened to write it.

The upstream dataset stays English — if a Russian answer is ever needed, it comes
from the model, not from a translation of somebody else's solution.

Two source-level details matter for quality: `saiga_scored` labels a role `bot`,
not `assistant`, and its rows carry model-generated scores — anything below 6 or
flagged by its own regex filter is dropped.  mail.ru answers are short by
nature; using the forum default of 120 characters rejected 98 % of them, so
those two sources carry their own 32-character floor.

## 3. Filters

Applied per row, in this order:

1. **length** — question 12–4,000 characters, answer 20–8,000 (forum floor 120,
   mail.ru 32).
2. **refusal / link-only** — answers that start with "I'm sorry", "As an AI",
   "не могу помочь" or are nothing but URLs are dropped; they teach the model
   that an answer is an apology.
3. **language** — a Cyrillic-share test (≥ 25 % for Russian, ≤ 2 % for English).
   No language-ID model is bundled; the share test is enough because the sources
   are already per language and only mislabelled rows leak.
4. **loops** — an answer whose 5-gram variety is below 70 % is dropped.  This is
   the failure mode of small models and it is counted, not eyeballed.
5. **exact duplicates** — 64-bit BLAKE2 over the whole thread.
6. **cap** — each source is limited to its share of the cell budget, so the
   first dataset in the list cannot swallow the cell.

### 3.1 The 512-token context, not the catalogue, is the binding constraint

Q10 reads 512 tokens.  An example longer than that cannot be learned as one unit
— the window would cut the answer in half — so it is dropped at packing time and
does not count towards its cell's budget.  Sampled at 4,000 rows per cell:

| cell | > 512 tokens | mean tokens | p95 |
|---|---:|---:|---:|
| math_en | 49.3 % | 579 | 1,178 |
| math_ru | 11.3 % | 321 | 624 |
| qa_en | 38.1 % | 484 | 1,094 |
| qa_ru | 57.6 % | 699 | 1,969 |
| qa_sites_en | 41.5 % | 542 | 1,238 |
| qa_sites_ru | 0.0 % | 60 | 80 |
| general_en | 11.8 % | 259 | 742 |
| code_ru | 0.0 % | 60 | 81 |
| identity | 0.0 % | 110 | 144 |

The consequence is that the reservoir is smaller than the sources suggest: a
500M-token target is filled to roughly **350M tokens**, and the shortfall is not
evenly spread.  English math loses half its rows, Russian forum Q/A loses none
(answers there are one or two sentences by construction), and English code loses
more than half because the good code instruction sets are long
chain-of-thought examples.  Padding the context is not an option without
retraining the position encoding, so the honest response is to report the
reservoir as it is and to prefer *short* sources when adding more.

That is also the argument for translating code **questions**: a translated
question keeps its original answer, so it costs one short paragraph of Russian
and lands entirely inside the context, while an English chain-of-thought example
mostly does not.

## 4. Known gaps

- **English code** used to be the largest gap.  It is now filled from
  OpenCodeInstruct, Magicoder-OSS-Instruct, Magicoder-Evol, self-oss-instruct-sc2,
  CodeAlpaca and python-instructions-18k, plus OpenCodeReasoning for competitive
  programming.  The cell still stops at about half its budget: the remaining
  shortfall is the 512-token filter (§3.1), not a missing licence.
- **Russian general instructions**: Dolly is English-only; the Russian general
  budget is absorbed into Russian math.
- **`squad_v2` and `NelvoraAI/russian-math-dialogues`** cannot be read with
  `datasets` 3.5 (`List` / `Json` feature types); both contribute nothing.
  `Vikhrmodels/russian_math` is gated.
- **`ai-bond/ru-alpaca-math`** answers are bare numbers below the 20-character
  floor: 191 rows out of 9,231.
- **The trainer does not mask the prompt yet.**  `sft_index.npy` carries the
  boundary; the loss is still computed on the whole rendered example.
- **Translation throughput limits the Russian code slice.**  MarianMT runs at
  ~2.5 questions/s on the 1660 Ti, so a full night of translation is ~60–70k
  questions; that is the practical ceiling without a bigger GPU.

## 5. Reproducing

```bash
python scripts/gen_arithmetic_sft.py --per-lang 700000 --out E:/kron_sft/raw
python scripts/fetch_sft_sources.py --raw E:/kron_sft/raw      # per-source timeout
python scripts/translate_code_questions.py --per-source 9000   # RU code questions
python scripts/verify_translated_pairs.py                     # must print PASS
python scripts/build_sft.py mix --raw E:/kron_sft/raw --target-tokens 500000000
python scripts/build_sft.py report --raw E:/kron_sft/raw
```

`fetch` is resumable (`<source>.done.json` marks a finished shard), `mix` is
deterministic given the raw shards, and the per-source timeouts exist because a
Hub CDN read can stall for minutes without erroring (MetaMathQA did, then
succeeded on retry).  `translate` is resumable through its cache;
`--rebuild-only` rewrites the shard from the cache without a GPU.

## 6. How much SFT the model actually sees

The reservoir is 502,728,415 train tokens; a run takes a slice of it.  The
arithmetic that decided the slice:

| steps | tokens | share of the reservoir | tokens per parameter | GPU hours |
|---:|---:|---:|---:|---:|
| 4,000 (S10) | 65.5M | 13.0 % | 6.6 | 2.8 |
| 8,000 (S10 + S10b) | 131M | 26.1 % | 13.2 | 5.6 |
| 12,000 (S10 + S10b) | 197M | 39.1 % | 19.8 | 8.5 |
| 16,000 | 262M | 52.1 % | 26.4 | 11.3 |

S10 alone is a **verdict probe**, not a recipe: 6.6 tokens per parameter is far
below what a 10M model can absorb, and the point of the first 4,000 steps is to
find out whether fine-tuning moves anything at all.  The decision rule:

* format learned **and** exact-match arithmetic above zero -> the recipe works;
  S10b continues from S10's weights for 8,000 more steps (lr 1e-5 -> 1e-6, a
  fresh cosine), taking the total to 197M tokens.
* format learned and arithmetic still zero -> a capacity wall.  More SFT tokens
  would buy memorisation of the examples, not the skill, so the next GPU hours
  go to a bigger model instead.

`kaggle/s10b` (generated by `scripts/make_s10b_kernel.py`, code v24) is the
second pass; `scripts/after_s10.py` does the whole hand-off in one command --
download S10's output, publish `kron-s10-ckpt`, wait for the version to go
live, push S10b -- because publishing and pushing by hand is how a stale mount
happens.

Context length is 512 everywhere, and that is a real constraint on this corpus:
a chat example has to fit in 512 tokens or it is dropped, which is why UltraChat
loses ~95 % of its threads at the filter.  Measured cost of 1024 at equal
tokens/step is nil (1.94 s/step at 1024x1 against 1.99 s/step at 512x2, same
4.39 GB): the Kronecker prelude dominates and it is paid once per step, not per
token.  But the model has never seen 1024 -- RoPE was trained at 512 -- so
doubling the window needs a short continued-pretraining adaptation first, and
that is a separate stage, not a flag.
