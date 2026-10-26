# Current progress

Rolling log of the most recent results, newest first. At most five entries are
kept; pruning is triggered only by overflow. This file is committed and pushed
only at important milestones, not after every run, and entries are written in
detail.

## In progress

- The corpus is final: **8.61 GB, 945,234 documents, 2,183,929,941 train
  tokens**, Russian at **37.9 % of tokens**. Bins are on Kaggle as
  `icold21/kron-corpus-2b` (4.37 GB train, 89.6 MB val).
- **Q10** (d=512, 25 layers, K=10, 9,938,886 params, seq 512) is running on two
  Kaggle T4s via `torchrun`; one epoch is **88,864 steps** (24,576 tokens per
  step), `max_steps=0` derives it from the bin. Checkpoints every 500 steps,
  500-minute sessions with resume.
- After pretraining: Habr article #1, then SFT/Q&A with a KL anchor to the
  frozen pretrain policy plus 5–10 % replay, then article #2. Distillation is
  shelved until an unaligned teacher exists.

## Results

## 2026-10-26T15:10+03:00 — corpus final: 2.184B tokens, Russian 37.9 %; Q10 running

- **Size**: 8.61 GB of mixed text, 945,234 documents, **2,183,929,941 train
  tokens** at 3.80 bytes/token (val 44.8M). EN 46.2 % / RU 53.8 % of bytes,
  which is **EN 62.1 % / RU 37.9 % of tokens**.
- **Russian top-up**: the quota mix gave Russian 15.7 % of tokens because
  Russian costs 5.12 bytes/token against 3.00 for English. The fix was data:
  the full Russian Wikipedia (**2.0 GB** of new articles after skipping 18,337
  titles already in curated shards) and the full Russian Wikisource (**1.1 GB**
  of public-domain literature, 100,261 texts). Russian moved from 15.7 % to
  37.9 % of tokens; closing the remaining gap to 45 % would need ~1 GB of
  Russian prose that does not exist under a reusable licence.
- **English shortfalls**: +47 MB theoretical physics, +63 MB engineering
  (2,505 duplicate titles skipped), +100 MB IETF RFCs (2,696 documents) and
  +130 MB Cosmopedia for ML (30,118 synthetic educational texts, labelled as
  such in the manifest).
- **Domains**: encyclopedic 30.3 %, literature 21.2 %, math 19.1 %, science
  11.1 %, code 7.5 %, ML 5.4 %, logic 3.2 %, CS/engineering 2.3 %, humour
  <0.1 %, identity 0.09 %. The encyclopedic overshoot is the price of the
  Russian requirement; the report states it instead of hiding it.
- **Identity v7**: 3,300 documents, **0 defects** under the generator's own
  gate. Both pastiches regenerated at temperature 0.7, plain slices repaired
  (616 kept / 184 regenerated in EN, 596 / 204 in RU). The gate rejects digits,
  assistant phrasing, dialogue and stage directions, markdown, mixed scripts,
  wrong mathematician names, fragments, and 5-gram near-duplicates; it requires
  the three facts (name, KronLLM, Kronecker origin).
- **Pipeline bugs fixed on the way**: `str.splitlines()` split JSON on U+2028
  inside wiki text; bulk appends lacked `sha256` for the train/val split; the
  BPE sample used the first 10 MB of each language (i.e. the code shard) and now
  takes an equal share per domain; `encode_docs` streams text so an 8.6 GB mix
  does not need three copies in RAM; `--max-train-mb` defaulted to the 400 MB
  pilot cap and silently truncated the corpus 13×.
- **Kaggle**: the dataset mount moved to `/kaggle/input/datasets/<owner>/<slug>`
  and two DDP ranks reading bins straight from that mount were OOM-killed
  (SIGKILL, no traceback). The kernel now copies the bins to
  `/kaggle/working/data` first; the run survives.
- **Q10**: 9,938,886 params, 88,864 steps of 24,576 tokens = 2.184B = one epoch,
  `max_steps=0` and `epochs=1` derive the count from the bin. Batch 24 × seq 512
  per GPU, fp16, checkpoint every 500 steps, 500-minute session budget.

- **Easter eggs**: 81 episode transcripts plus 11 season-9 plot summaries
  (S9 transcripts are not published anywhere yet).
- **Chat template reserved in the 8K tokenizer**: system 8187, user 8188,
  assistant 8189, `eot` 8190 (turn end), `endoftext` 8191 (document end) — SFT
  never needs to resize the embedding.
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
