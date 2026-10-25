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
| mathematics | OpenStax, ProofWiki, public-domain textbooks | public-domain textbooks, open lecture notes | check each item; prefer CC-BY / CC-BY-SA / public domain |
| science | arXiv (CC-BY subset), PubMed Central OA | CyberLeninka (item-level check), open university repositories | arXiv licences vary per paper; only CC-BY or explicit permission |
| logic | Lean/Coq docs, public-domain logic texts | public-domain logic textbooks | documentation under permissive licences |
| code | permissively licensed repositories (MIT/Apache/BSD), Python docs and exercises | same, plus Russian-language programming tutorials with a clear licence | licence detection per file; no copyleft-only or unlicensed code |
| literature | Project Gutenberg, Wikisource | public-domain classics (Tolstoy, Dostoevsky, Chekhov), Wikisource | author died >70 years / US public domain |
| encyclopedic | Wikipedia dumps (CC-BY-SA) | Wikipedia RU dumps (CC-BY-SA) | attribution required |
| instructions (SFT only) | Dolly-15k, open instruct sets | Saiga / open Russian instruct sets | never used for pretraining |

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

## 6. Implementation

The pipeline is implemented and does not depend on anything outside the
standard library:

| stage | where |
|---|---|
| MediaWiki cleaning, filters, MinHash dedup, mixing | `kronllm/bilingual.py` |
| fetch (Wikipedia, Wikisource, Gutenberg, CPython stdlib) | `scripts/build_bilingual.py` |
| tokenisation into uint16 memmaps | `scripts/prepare_bilingual_bin.py` |
| tests | `tests/test_bilingual.py` (9 tests) |

```bash
python scripts/build_bilingual.py fetch-wiki --lang ru --domain math --target-mb 40
python scripts/build_bilingual.py fetch-wikisource --lang ru --domain literature
python scripts/build_bilingual.py fetch-gutenberg --lang en --domain literature
python scripts/build_bilingual.py fetch-code --target-mb 12
python scripts/build_bilingual.py build --target-mb 250
python scripts/prepare_bilingual_bin.py --vocab 8192
```

Every fetcher writes one JSONL shard and one manifest row with `source`,
`licence`, `url`, `sha256`, `bytes`, `language`, `domain`. `record_shard`
refuses an unknown licence, so a shard without provenance cannot enter the
mix.

## 7. Immediate steps

- [x] freeze the shard manifest schema and create `data/bilingual/manifest.jsonl`;
- [x] implement normalisation, quality filters, deduplication, mixing and the
      held-out split;
- [ ] reach the 250 MB-per-language pilot target (the first build is smaller);
- [ ] compare 8K and 16K tokenizers on the pilot;
- [ ] only then start the full 0.5-2B-token download and build.

The pilot is deliberately small: it is cheaper to discover a licence or
quality problem on 500 MB than on 8 GB.
