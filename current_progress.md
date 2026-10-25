# Current progress

Rolling log of the most recent results, newest first. At most five entries are
kept; pruning is triggered only by overflow. This file is committed and pushed
only at important milestones, not after every run, and entries are written in
detail.

## In progress

- Identity regeneration (800 plain + 850 pastiche per language) is finishing;
  the night chain then builds the corpus, tokenizes 8K with the reserved chat
  ids and uploads the bins to Kaggle.
- Architecture is frozen; **Q10** (d=512, 25 layers, K=10, 9,938,886 params,
  seq 512, 82k steps = 2.015B tokens) is the pretraining run, on two T4s via
  `torchrun --nproc_per_node=2`.
- After pretraining: Habr article #1, then SFT/Q&A with 10–20 % pretrain
  replay, then article #2. Distillation is shelved until an unaligned teacher
  exists.

## Results

## 2026-10-26T02:30+03:00 — 2B corpus assembled; identity, repos and Q10 ready

- **Corpus, fetched**: 7.3 GB of text, ~795k documents, ~2.15B tokens at
  3.34 bytes/token; EN 4.9 GB / RU 2.4 GB. Published as its own repository
  (`kron-corpus`) and an HF dataset.
- **Domains**: math 26 %, science 22 % (physics/astronomy/cosmology curated),
  encyclopedic 16 %, code 14 %, literature 11 %, ML 5 %, logic 3 % (Lean
  mathlib), CS/engineering 3 %, humour <1 %, identity <0.1 %.
- **Selection**: provenance per shard (source, licence, URL, SHA-256), primary
  sources over scrapes, MinHash 0.8 dedup, 2 % disjoint held-out. Shortfalls
  reported, not padded: logic 3 % vs 10 %, ML 5 % vs 12 %, RU code absent.
- **Identity slice**: 800 plain + 850 pastiche per language (3300 docs,
  2.5 MB). Facts deliberately carry **no numbers and no framing** (name,
  compact/efficient, Kronecker origin), guarded by `SPECS`, `FRAMING` and
  `MISSPELL` filters. Earlier versions with "ten million" and "6 GB" were
  discarded: a hard-coded size breaks the moment the model is scaled.
- **Easter eggs**: 81 episode transcripts plus 11 season-9 plot summaries
  (S9 transcripts are not published anywhere yet).
- **Chat template reserved in the 8K tokenizer**: system 8187, user 8188,
  assistant 8189, `eot` 8190 (turn end), `endoftext` 8191 (document end) — SFT
  never needs to resize the embedding.
- **Q10**: 9,938,886 params, 82k steps × 24,576 tokens (batch 24 × seq 512 ×
  2 ranks) = 2.015B tokens = one full epoch.
- **DDP** verified on two CPU ranks: rank-0 logging and checkpoints, periodic
  `last.pt` every 500 steps with optimizer state, `--resume` tested; Windows
  rendezvous-store bug worked around.
- **Sharded epoch sampler**: one shared permutation per epoch, ranks take
  `rank::world`, so an epoch is a full pass (the old random sampler covered
  ~63 %). Tests cover coverage, disjointness, determinism and reshuffling.
- **Repos**: `kron-llm` (renamed), `kron-corpus` (new, public); HF model
  `Icold/kron-llm`, dataset `Icold/kron-corpus`; Kaggle code dataset
  `icold21/kron-llm-code`.

## 2026-10-25T08:30+03:00 — bilingual corpus pilot built end to end

- **12 shards, 420.2 MB, 35,919 documents**: EN Wikipedia 231.4 MB (25,715
  docs), RU Wikipedia 188.8 MB (10,204), RU Wikisource 25.4 MB, CPython
  standard library 12.6 MB. Licences: CC BY-SA 4.0 (407.7 MB), PSF-2.0
  (12.6 MB); every shard is registered in `data/bilingual/manifest.jsonl`
- **dedup** (SHA-256 + MinHash, Jaccard 0.8): 34,912 kept (394.0 MB), 1,007
  dropped (2.8 %) — lower than expected because category crawls overlap little
- **mixed corpus 273.4 MB, RU share 39.3 %** at a 60/40 byte policy, with
  per-domain quotas and a 196-document held-out split disjoint from training
- **8K BPE**: 80.3M train tokens, 1.52M val, 3.34 bytes/token. Round-trip is
  exact on Russian, English and LaTeX
- **Finding (changes the ratio policy):** bytes per token is not comparable
  across scripts. Both languages get ≈2.8 characters/token, but Cyrillic costs
  two UTF-8 bytes per character, so 60/40 by *bytes* gives only **33 % of
  tokens** to Russian, not the 45-47 % the plan predicted. The next stage must
  pick byte share or token share as the contract
- **Gaps, reported not hidden:** code 4.2 % vs a 15 % target (only the stdlib
  is permissively licensed in the pilot; no RU code corpus found), `en/math`
  33.5 of 52.2 MB, `ru/code` empty
- pipeline lives in `kronllm/bilingual.py` + `scripts/build_bilingual.py` +
  `scripts/prepare_bilingual_bin.py`, guarded by 12 tests (114 total pass)

## 2026-10-25T05:10+03:00 — Q8g-long 0.7542 bpb: new best; bilinear interaction dead

Chain: Q9 → Q9k → Q8g-long, all at 61.4M tokens where applicable.

- **Q8g-long** (24 layers, gated Kronecker FFN, 2,851,280 params, 13,333 steps
  × batch 24 × 192): **0.7542 bpb**, loss 2.0693, PPL 7.919, 0.895 s/step,
  **4.09 GB** peak, `eff_rank(P)` 144.8, `cos_prev_max` 0.834
  - vs **Q6@10k 0.7644** at the identical 61.44M-token budget: **−1.3 %**
  - the curve was still falling at the last step (0.7553 → 0.7542), so the
    budget is a choice, not a measured ceiling
  - int8 quantisation: **+0.0003 bpb**, 3.29 MB vs 11.1 MB fp32 (3.55×);
    moved to `artifacts/q8g_int8.pt`
- **Q9** (bilinear GLU FFN, no Kron term, 2,463,776 params, 24 layers):
  **0.8740** vs Q6 **0.8489** at equal tokens — **+2.9 %**, and
  `cos_prev_max` jumped to **0.939**: the layers became more similar
- **Q9k** (GLU + the Kron map as a non-interacting term, 20 layers,
  2,462,468 params): **0.8673** — **+2.2 %**. Adding the linear term back helps
  a little but does not close the gap
- **Methodology note:** the first Q9 run was invalid — the generic `std=0.02`
  init left the bilinear branch 300× smaller than a Kron module (output std
  0.005 vs 1.76), i.e. switched off. Fixed to unit-activation init, guarded by
  `test_glu_branch_is_alive_at_init`; the numbers above are from the corrected
  run. A false negative would have been worse than no experiment
- **Verdict:** multiplicative feature interaction is closed at this scale. The
  pattern across all experiments is consistent: what wins keeps the layer
  linear and fusable into one GEMM
- checkpoints of Q9/Q9k removed; `runs/Q8g_long/best.pt` and
  `runs/Q6_10k/best.pt` are the only ones kept on disk

## 2026-10-24T23:50+03:00 — Q8w: gated FFN wins 1.5 % at equal parameters

- **Q8g** (24 layers × 6 modules, 2,851,280 params): **0.8318** at 24.58M tokens
- **Q8w** (20 layers × 6 modules, 2,470,328 params, same 120 modules/token):
  **0.8359** — vs Q6's **0.8489** at identical parameters (±0.1 %) and tokens:
  **−1.5 %**
- decomposition of the gain: **1.5 % is architecture** (equal parameters, equal
  tokens, equal module count), **0.5 % is the extra 378K parameters** of Q8g
- the design rule that follows: *a gated FFN beats four extra layers at equal
  budget*
- Q8w is also cheaper in memory (3.44 GB vs 4.28) because 20 layers with batch
  24 fit better than 24 layers with batch 32

## 2026-10-24T22:42+03:00 — nonlinearity inside the composition: closed

| variant | bpb | vs E8 (1.4789) |
|---|---:|---:|
| E26, k=4 (one activation) | 1.5675 | +6.0 % |
| E25, k=2 (two activations) | 1.5189 | +2.7 % |
| E26b, additive k=4 (linear skeleton kept) | 1.5906 | +7.5 % |

- more nonlinearity helped (k=2 beat k=4), but **every variant lost to the
  linear product**, and the additive form was the worst
- wall-clock cost at equal FLOPs: **0.655 vs 0.405 s/step** (+62 %) — the loss
  of the single-GEMM fusion, not the activation itself
- `eff_rank(P)` rose to 62.3 while bpb got worse: third independent sign that
  rank is not the bottleneck
- E23 (k=1, 9.56 MFLOP, 2.6× compute) and E24 (k=1 + RMSNorm) were dropped
  rather than run: the trend is monotone and points away from the hypothesis
- `E23–E26b` annotated as CLOSED in `config.py`; code kept for reproducibility
  only
