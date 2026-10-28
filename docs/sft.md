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
| math | 45 % of bytes | the stage's purpose: structured task→solution pairs |
| open Q/A | 30 % | instruction following, translation, definitions |
| forum Q/A (Stack Exchange, mail.ru, ELI5, Reddit) | 15 % | real questions with real answers, including code and technical threads |
| code | 6 % | docstring→snippet pairs; kept small because pretraining already had a code slice |
| general instructions | 3.5 % | Dolly-style tasks that are neither math nor forum |
| identity | 0.5 % | 100–300 examples, guarded (see `docs/roadmap.md` §6) |

The template is the one the notebook and the sampler already use:

```
<|system|>You are KronLLM, a compact bilingual model.<|eot|>
<|user|>…<|eot|><|assistant|>…<|eot|>
```

`<|eot|>` ends a turn, token `8191` ends an example.  `sft_index.npy` stores
`(start, prompt_tokens, total_tokens)` per example, so the trainer can mask the
loss on the prompt without re-tokenising — the roadmap asks for that masking and
the flat bin alone does not carry it.

## 2. Sources

Everything comes from the Hub through a mirror with a declared licence.  Nothing
is scraped: a raw crawl of `otvet.mail.ru` or Reddit would violate the licence
policy in `docs/data_plan.md` §2, so those sites enter through
`Den4ikAI/mailruQA-*`, `Pavithree/eli5` and `zhoudoe23/reddit-top-qa-dataset`.

| group | sources |
|---|---|
| math, EN | OpenMathInstruct-2 (1M slice), MetaMathQA, Orca-Math, MathInstruct, GSM8K, MATH (algebra/geometry/number theory) |
| math, RU | translated Orca-Math (`d0rj/…-ru`), translated MathInstruct, Russian math dialogues, ru-alpaca-math |
| math, synthetic | 13 templated arithmetic families in both languages: exact answers, computed by Python, with a written-out derivation |
| Q/A, EN | OpenOrca, OpenHermes-2.5, SQuAD v2, HotpotQA |
| Q/A, RU | saiga_scored (`opus_score ≥ 6`), russian_instructions_2 |
| forum, EN | Stack Exchange dump (best answer per thread, score ≥ 2), ELI5, Reddit top Q&A |
| forum, RU | mail.ru QA (two mirrors, short answers by design) |
| code | Russian code Q/A (question → snippet) |
| general | Dolly-15k |
| identity | generated locally from `data/identity` (CC0) |

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

## 4. Known gaps

- **English code**: no permissively licensed EN code SFT source is registered,
  so the 6 % code budget for English has nowhere to go.  Its bytes are moved
  into the English math cell and reported as an absorbed cell rather than
  silently dropped.
- **Russian general instructions**: Dolly is English-only; the Russian general
  budget is likewise absorbed into Russian math.
- **`NelvoraAI/russian-math-dialogues`** cannot be read with `datasets` 3.5
  (`Feature type 'Json' not found`); it is small (11k rows) and currently
  contributes nothing.
- **`Vikhrmodels/russian_math`** is gated; skipped.
- **No Russian OpenMathInstruct equivalent** exists at the same scale, which is
  why synthetic arithmetic carries a large share of the Russian math cell — and
  why the Russian math slice is the least "natural" part of the corpus.

## 5. Reproducing

```bash
python scripts/gen_arithmetic_sft.py --per-lang 700000 --out E:/kron_sft/raw
python scripts/fetch_sft_sources.py --raw E:/kron_sft/raw      # per-source timeout
python scripts/build_sft.py mix --raw E:/kron_sft/raw --target-tokens 500000000
python scripts/build_sft.py report --raw E:/kron_sft/raw
```

`fetch` is resumable (`<source>.done.json` marks a finished shard), `mix` is
deterministic given the raw shards, and the per-source timeouts exist because a
Hub CDN read can stall for minutes without erroring (MetaMathQA did, then
succeeded on retry).
