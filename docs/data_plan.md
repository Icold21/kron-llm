# Bilingual corpus plan (English–Russian)

Status: Phase 1 of [roadmap.md](roadmap.md), started 2026-10-23. This document
fixes the mix, the source policy and the pipeline; the source list is a living
registry and every shard must carry its licence.

## 1. Mix

**60 % English / 40 % Russian by bytes** is the working ratio. It is a
deliberate compromise, not a neutrality claim:

- English has more high-quality math/science material, so equal bytes would
  underuse the better English sources;
- Russian tokenizer fertility is lower (≈3.0–3.5 bytes/token vs ≈4.0 for
  English), so 40 % of bytes is roughly **45–47 % of tokens** — the model still
  sees a substantial Russian share;
- the ratio is a **starting point**: per-language and per-domain bpb decide
  whether it moves. If Russian bpb lags by more than ~0.05, raise the Russian
  share rather than accept a bilingual model that is English with a Russian
  accent.

Domain mix inside each language (target, by bytes):

| domain | share | why |
|---|---:|---|
| mathematics | 25 % | dense reasoning, formulas, proofs |
| science / scientific articles | 20 % | formal prose, terminology, long context |
| code | 15 % | precise syntax, structured reasoning, transfer to math |
| logic | 10 % | proofs, definitions, structured argument |
| literature | 15 % | long-range coherence, style, vocabulary |
| general encyclopedic | 15 % | world knowledge and coverage |

The domain mix is **not** a replacement for general text: a corpus made only of
proofs and classics produces a model that writes like a textbook. The 15 %
encyclopedic slice is the anchor; if generation samples look narrow, raise it.
Code is capped at 15 % because its token distribution (indentation, operators,
identifiers) differs sharply from natural language; it is there to sharpen
structure, not to turn the model into a code model.

## 2. Sources and licences

Only public-domain or permissively licensed material. Every shard records
`source`, `licence`, `url`, `sha256`, `bytes`, `language`, `domain`.

| domain | English | Russian | licence notes |
|---|---|---|---|
| mathematics | OpenStax, ProofWiki, public-domain textbooks, Wikibooks/Wikiversity mathematics shelves | public-domain textbooks, open lecture notes, ru Wikibooks/Wikiversity | check each item; prefer CC-BY / CC-BY-SA / public domain |
| science | arXiv (CC-BY subset), PubMed Central OA, peS2o (S2ORC open-access, ODC-By), Wikibooks/Wikiversity chemistry/physics/biology textbooks, Wikipedia chemistry categories, public-domain chemistry manuals (Gutenberg) | CyberLeninka (item-level check), open university repositories, ru Wikibooks/Wikiversity, ru Wikipedia chemistry | arXiv licences vary per paper; only CC-BY or explicit permission; peS2o is ODC-By and is *papers*, not textbooks |
| practical / DIY | Wikibooks Cookbook, Do It Yourself, Electronics, Engineering; Gutenberg handicraft, woodwork, cookery | ru Wikibooks Кулинария / Электроника / Ремонт, ru Wikisource | public domain or CC-BY-SA; Gutenberg manuals are 1850–1930 and are *procedure* prose, not current practice |
| logic | Lean/Coq docs, public-domain logic texts | public-domain logic textbooks | documentation under permissive licences |
| code | permissively licensed repositories (MIT/Apache/BSD), Python docs and exercises | same, plus Russian-language programming tutorials with a clear licence | licence detection per file; no copyleft-only or unlicensed code |
| literature | Project Gutenberg, Wikisource | public-domain classics (Tolstoy, Dostoevsky, Chekhov), Wikisource | author died >70 years / US public domain |
| encyclopedic | Wikipedia dumps (CC-BY-SA) | Wikipedia RU dumps (CC-BY-SA) | attribution required |
| instructions (SFT only) | Dolly-15k, open instruct sets | Saiga / open Russian instruct sets | never used for pretraining |

**Science expansion, 2026-10-30.**  A keyword scan of the science pools found
chemistry at 7.4 % of `en/science` (47 MB of 634 MB) and 11.5 % of `ru/science`
(37 MB of 318 MB) — enough to be present, not enough to answer a chemistry
question.  The plan adds three kinds of text, all inside the existing `science`
bucket so the quota mixer needs no new domain:

* **textbooks** — `scripts/fetch_textbooks.py` crawls `Category:Subject:<name>`
  on Wikibooks (the plain `Category:Chemistry` there is empty; books live under
  `Category:Subject:Chemistry`, 60 books, and `Category:Book:<name>`, e.g. 116
  chapters for *General Chemistry*) and ordinary categories on Wikiversity;
* **reference** — Wikipedia chemistry categories at scale, threaded, because a
  depth-2 category walk is thousands of requests;
* **procedure** — Gutenberg `chemistry` / `engineering` / `cookery` filters
  (`--corpus-domain science`): soldering, carpentry, preserving, laboratory
  manuals.  Old, but textbook-shaped: definition, reaction, worked example.

Journals are covered by arXiv and peS2o rather than by scraping publisher sites:
both are open-access-only and both carry a licence we can attribute.  New shards
are appended to `mix/` with `scripts/add_bulk_to_mix.py`, which dedups by title
against the target bucket — no full MinHash rebuild for a few hundred MB.

**Status of the fetch (2026-10-30): blocked.**  The link died mid-crawl: HTTPS to
Wikimedia, Hugging Face, Gutenberg, GitHub and Kaggle times out ~90 % of the time
(1/10 sequential requests to `en.wikipedia.org/w/api.php` succeeded).  The only
shard that landed is `wikipedia_ru_science_chem.jsonl` (47 docs, 0.4 MB) out of
310 titles — not worth adding.  The fetchers, the filters, the `--corpus-domain`
flag and the rebalanced `SHARES` are written and tested; the crawl itself runs
when the link returns, and the decay bin is rebuilt after it.

Explicitly rejected: pirated book collections, sites with unclear terms,
scraped textbooks without a licence, and any source that cannot be attributed.
A shard with an unknown licence does not enter the mix.

## 3. Pipeline

1. **Registry** — one JSONL row per shard with the fields above; raw files live
   outside git.
2. **Language ID** — fastText `lid.176` (or equivalent); keep items with
   confidence ≥ 0.9 and the expected language; drop mixed-language pages.
3. **Normalisation** — Unicode NFKC, whitespace cleanup, remove navigation,
   boilerplate, references-only pages and markup residues. Code **keeps its
   indentation and line structure**; aggressive whitespace collapsing would
   destroy the signal it is meant to teach.
4. **Quality filter** — minimum length, punctuation/letter ratio, digit and
   symbol ratio, repeated n-gram ratio, compression ratio; optional perplexity
   filter with a small reference model.
5. **Deduplication** — exact hash + MinHash LSH near-duplicate removal within
   and across sources; keep the highest-quality copy.
6. **Decontamination** — remove overlap with the held-out sets and any Q/A
   evaluation data before training.
7. **Mixing** — 60/40 by bytes, domain quotas above, deterministic shuffle.
8. **Tokenizer** — byte-level BPE trained on a stratified sample; 8K default,
   16K only if the embedding budget allows; report bytes/token per language.
9. **Held-out** — 10M tokens per language plus a small per-domain slice, fixed
   before any training run.

## 4. Sizes

**The 100 MB pilot is not the pretraining corpus.** It exists to validate the
pipeline and the tokenizer cheaply; training starts only after the full build.

| stage | size | purpose |
|---|---|---|
| pilot | ~250 MB per language (≈60–80M tokens) | filters, dedup, licences, tokenizer fertility |
| full corpus | **0.5–2B tokens** (≈2–8 GB raw text at 3.7–4.0 bytes/token) | actual pretraining |
| held-out | 10M tokens per language + 1M per domain | evaluation, fixed before training |
| minimum per domain-language | enough for ≥20M tokens after filtering | avoid a starved domain |

Reference points: a 10M-parameter model is compute-optimal at ≈200M tokens
(20 tokens/parameter), but small models usually keep improving well past that,
so the run is checkpointed and the bpb-vs-tokens curve decides where to stop.
At ~7k tokens/s, 1B tokens is ≈40 h of GPU time and 2B is ≈80 h.

## 5. Evaluation of the corpus itself

- bytes/token per language and per domain after tokenizer training;
- token share per language (must stay near the 45–47 % Russian target);
- duplicate and contamination rates;
- per-domain bpb on the held-out slices — a domain that is much worse than the
  average points to a data problem, not an architecture problem;
- code validity: after decoding a held-out code slice, measure the fraction of
  samples that parse (Python `ast.parse`) and the character-level round-trip
  after tokenization.

## 6. Pilot build (2026-10-25)

First end-to-end run of the pipeline. Sources: English and Russian Wikipedia,
Russian Wikisource and the CPython standard library. Every shard is registered
with its licence in `data/bilingual/manifest.jsonl`.

| stage | result |
|---|---|
| shards fetched | 12 · 420.2 MB · 35,919 documents |
| by language | EN 231.4 MB / 25,715 docs · RU 188.8 MB / 10,204 docs |
| licences | CC BY-SA 4.0 (407.7 MB) · PSF-2.0 (12.6 MB) |
| exact + near dedup | 34,912 kept (394.0 MB) · 1,007 dropped (2.8 %) |
| mixed corpus | **273.4 MB** · RU share **39.3 %** |
| held-out | 196 docs (1 MB per language, disjoint from training) |
| tokenizer | 8K BPE · 80.3M train tokens · 1.52M val tokens · 3.34 bytes/token |

Achieved domain mix against the targets in §1 (by bytes, both languages):

| domain | target | achieved |
|---|---:|---:|
| mathematics | 25 % | 22.3 % |
| science | 20 % | 25.4 % |
| code | 15 % | 4.2 % |
| logic | 10 % | 9.8 % |
| literature | 15 % | 19.2 % |
| encyclopedic | 15 % | 19.1 % |

Shortfalls are reported rather than hidden: `en/math` 33.5 of 52.2 MB,
`en/code` 11.4 of 31.4 MB, `en/logic` 16.6 of 20.9 MB, `ru/math` 27.4 of
34.8 MB, `ru/logic` 10.3 of 13.9 MB, `ru/code` empty. **Code is the largest
gap**: the Python standard library was the only permissively licensed code
source in the pilot, and no Russian-language code corpus with a clear licence
was found. The next build needs MIT/Apache repositories with per-repository
licence records to reach 15 %.

### 6.1 Finding: bytes per token is not comparable across scripts

| language | bytes/token | characters/token |
|---|---:|---:|
| English | 2.79 | ≈2.8 |
| Russian | 5.24 | ≈2.6 |

Both languages receive roughly the same number of *characters* per token, but
Cyrillic costs two UTF-8 bytes per character. Consequence: at the planner's
60/40 **byte** split the corpus is 39.3 % Russian by bytes and only **33 % of
tokens** are Russian.

The 45-47 % token share predicted in §1 would need Russian at ~55-58 % of
bytes. The pilot deliberately honoured the stated 60/40 byte policy; the next
stage must decide whether the target is byte share or token share and record
one of them as the contract. `data/bilingual/tokenizer_meta.json` stores both.

## 7. Implementation

The pipeline is implemented and does not depend on anything outside the
standard library:

| stage | where |
|---|---|
| MediaWiki cleaning, filters, MinHash dedup, mixing | `kronllm/bilingual.py` |
| domain system prompts for pretraining | `kronllm/domain_prompts.py` (see `docs/prompts.md`) |
| fetch (Wikipedia, Wikisource, Wikibooks, Wikiversity, Gutenberg, CPython stdlib) | `scripts/build_bilingual.py` |
| threaded textbook / category crawl | `scripts/fetch_textbooks.py` |
| append a bulk shard to an existing mix bucket | `scripts/add_bulk_to_mix.py` |
| tokenisation into uint16 memmaps | `scripts/prepare_bilingual_bin.py` |
| tests | `tests/test_bilingual.py`, `tests/test_domain_prompts.py` |

```bash
python scripts/build_bilingual.py fetch-wiki --lang ru --domain math --target-mb 40
python scripts/build_bilingual.py fetch-wikisource --lang ru --domain literature
python scripts/build_bilingual.py fetch-gutenberg --lang en --domain literature
python scripts/build_bilingual.py fetch-gutenberg-catalog --lang en \
    --domain chemistry --corpus-domain science
python scripts/build_bilingual.py fetch-code --target-mb 12
python scripts/fetch_textbooks.py --site wikibooks --lang en --domain science \
    --category "Chemistry,Physics,Biology" --suffix _textbooks
python scripts/add_bulk_to_mix.py --shard data/bilingual/shards/wikibooks_en_science_textbooks.jsonl \
    --lang en --domain science
python scripts/build_bilingual.py build --target-mb 250
python scripts/prepare_bilingual_bin.py --vocab 8192 --phase anneal --interleave chat
```

Every fetcher writes one JSONL shard and one manifest row with `source`,
`licence`, `url`, `sha256`, `bytes`, `language`, `domain`. `record_shard`
refuses an unknown licence, so a shard without provenance cannot enter the
mix.

## 8. Immediate steps

- [x] freeze the shard manifest schema and create `data/bilingual/manifest.jsonl`;
- [x] implement normalisation, quality filters, deduplication, mixing and the
      held-out split;
- [ ] grow the pilot toward 250 MB per language and close the code gap
      (MIT/Apache repositories with per-repository licence records);
- [ ] compare 8K and 16K tokenizers on the pilot;
- [ ] only then start the full 0.5-2B-token download and build.

The pilot is deliberately small: it is cheaper to discover a licence or
quality problem on 500 MB than on 8 GB.
