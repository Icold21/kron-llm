# The 10M model: purpose, contract and compute plan

This document defines *why* the next model exists, *what* it must do, and *how*
it will be trained. It is the reference for every decision about the 10M run:
if a change does not serve the goals below, it does not go in.

## 1. Purpose

Kron is a **compact bilingual (English/Russian) language model with Kronecker
factorised weights**, built to answer one question:

> How much language ability can a model with ~10M parameters reach when its
> weights are stored as sums of Kronecker products instead of dense matrices?

The practical target is a model that fits and runs on hardware people actually
own: a 6 GB consumer GPU (GTX 1660 Ti class) and, in int8, a CPU. The 10M size
is the largest point on this curve that still satisfies the deployment budget:
40 MB in fp32, 10 MB in int8, 2 bytes per parameter in the shipped artifact.

The model is a **research artifact**, not a product:

* it is not a chatbot, an assistant, or a general-purpose reasoner;
* it does not follow instructions outside the small SFT slice;
* it writes short English and Russian text, answers simple questions about
  itself, and solves school-level arithmetic and single-file code snippets;
* it is allowed to be wrong, and says so when asked about itself.

## 2. Why 10M and not 100M

| constraint | consequence |
|---|---|
| 6 GB VRAM deployment target | ≤10M params in fp32 with room for the KV cache and activations |
| int8 artifact for CPU use | ~10 MB download, loads in under a second |
| Kaggle T4 (16 GB, 30 GPU-h/week) | 1.6B tokens is a multi-session job, not a multi-week one |
| Kronecker weights | 3.5× fewer stored parameters per unit of compute than dense at the same width |

Scaling past 10M breaks the first two constraints and turns the project into a
generic small-LM training run, which is not the point. The interesting claim is
the *ratio*: quality per stored parameter, at a size where the artifact still
fits in a pocket.

## 3. Success criteria

Measured on a held-out split that is disjoint from training (196 documents in
the pilot; the final split is 2 % of the mix):

| # | criterion | threshold |
|---|---|---|
| G1 | validation bpb on the bilingual mix | ≥10 % better than the 2.85M model trained on the same mix (baseline to be measured) |
| G2 | English vs Russian bpb gap | ≤ 15 % relative |
| G3 | bits-per-byte beats the 2.85M model at equal tokens | required |
| G4 | int8 round-trip degradation | ≤ 0.005 bpb |
| G5 | identity probe ("Who are you?") in EN and RU | names itself Kron in ≥ 8/10 samples at T=1.0, top_k=8 |
| G6 | arithmetic probe (two-digit addition/subtraction) | ≥ 60 % exact |
| G7 | generation at T=1.0, top_k=5–10 | fluent, on-topic, no repetition loops in 9/10 samples |

G1 is deliberately relative: the 2.85M model's 0.7542 is a TinyStories number
and cannot be compared with the bilingual mix. The baseline is the same
architecture at 2.85M on the same 1.6B-token mix, and it runs before the 10M
model so the comparison is honest.

## 4. Architecture fixed for the 10M run

The scaling decision follows the ablation results (E8r/E27–E30) and the
architecture log in `docs/results.md`. The starting point:

| component | choice | why |
|---|---|---|
| composition | product, `K_win` per ablation | product beat sum by 4.9 % at 190K params |
| σ | learned per module (unless E27/E28 say otherwise) | one scalar per module, no channel specialisation |
| FFN | gated (Q8w/Q8g) | −1.5 % bpb at equal params and tokens |
| attention | MHA + qk_norm, RoPE | GQA gave no win at this size |
| embedding | factorised, rank ≈ d/2, tied head | 0.57M parameters saved at d=320 |
| tokenizer | 8K BPE, bilingual, byte-fallback | 3.34 bytes/token measured on the pilot mix |
| depth | 25 layers, d=512 | the smallest width that keeps the Kron factors square-ish |

The reference configuration is **9,938,886 parameters** (d=512, 25 layers,
n1×n2 = 16×32, K=10, L=4, vocab 8192, factorised embedding rank 256, gated FFN).
`python -m kronllm.cli budget` verifies it; the constraint is
**9.5M ≤ params ≤ 10.5M**.

The 6 GB fallback is **10,062,036 parameters** (d=384, 30 layers, 16×24,
K=14, embedding rank 192) — the same architecture, narrower and deeper, because
d=512 does not fit the 1660 Ti's memory (see §6).

**Is one more layer worth it?** A layer costs 308,406 parameters (+3.1 %),
5 MFLOP/token (+3.6 %) and 0.3 MB in int8:

| layers | parameters | MFLOP/token | int8 size | 2.251B tokens |
|---:|---:|---:|---:|---:|
| 25 | 9,938,886 | 138 | 9.9 MB | 3.0×10¹⁷ |
| 26 | 10,247,292 | 143 | 10.2 MB | 3.1×10¹⁷ |
| 27 | 10,555,698 | 148 | 10.6 MB | 3.2×10¹⁷ |

At this size a 3 % parameter increase is worth roughly 0.2–0.3 % bpb under a
standard scaling slope — below the ~1 % seed noise we measure. Depth is the
*right* axis when parameters are held fixed (E20: −3.5 % bpb against E8 at
2× compute), but adding a layer is not that experiment. The reference run
therefore stays at 25 layers; 26 is the fallback if the corpus grows past
2.4B tokens, where the extra capacity is at least matched by extra data.

## 5. Corpus contract

| property | value |
|---|---|
| tokens | **2,250,807,719 measured** (8.62 GB mixed, 949k documents, 2 % held out) |
| bytes | 8.45 GB of train text at 3.69 bytes/token |
| languages | **EN 46 % / RU 54 % of bytes**; the token share is **EN 61.9 % / RU 38.1 %** because Russian costs 4.96 bytes/token against 2.92 for English — the contract is bytes, the consequence is stated |
| domains | encyclopedic 30.3 %, literature 21.2 %, math 19.1 %, science 11.1 %, code 7.5 %, ML 5.4 %, logic 3.2 %, CS/engineering 2.3 %, identity 0.19 % |
| dedup | MinHash, 0.8 Jaccard threshold, applied before the mix |
| held-out | 2 % per language, disjoint by document |
| provenance | every shard has source, licence, URL and SHA-256 in `manifest.jsonl` |

Domain labels for Wikipedia and FineWeb text are **keyword heuristics**, not a
curated taxonomy; this is recorded in the manifest note for those shards. The
identity slice is synthetic and prompt-constrained (see
`docs/data_plan.md`).

## 6. Compute plan

FLOPs per token ≈ `6N + 12·L·seq·d` = 60M + 79M ≈ **138 MFLOP/token** at
d=512, L=25, seq=512. For 2.251B tokens: **3.1 × 10^17 FLOPs**.

| hardware | assumed throughput | 2.251B tokens | notes |
|---|---|---|---|
| Kaggle T4 (16 GB, fp16) | 4–6× the 1660 Ti (tensor cores, batch 24×512) | **40–70 h** | 2–3 sessions at 30 GPU-h/week |
| GTX 1660 Ti (6 GB, fp32) | **1,927 tokens/s measured** | **12 days** | d=384, batch 8×256, 4.59 GB peak |

The 1660 Ti number is now measured, not extrapolated:
`scripts/bench_10m_384.py` runs 60 steps at 1.06 s/step (2,048 tokens), peak
4.59 GB, and the loss falls from 9.01 to 6.00 over those steps. Two hard
constraints were found while measuring:

* the d=512 reference configuration **does not fit 6 GB**: `KronBank` stacks the
  depth state `(modules, L, d, d)`, which is 630 MB at d=512 before gradients.
  Both `d=512, batch 8×192` and `d=448, batch 8×256` OOM;
* the 6 GB fallback is **d=384, 30 layers, K=14 → 10,062,036 parameters**,
  which fits at batch 8×256; batch 16×256 OOMs.

So the 1660 Ti is a valid fallback only as a multi-day checkpointed run
(≈13 days for 2.251B), while the T4 is the intended path. A
prelude-checkpointing rewrite of `KronBank` would cut the memory enough to run
d=512 on 6 GB; it is not on the critical path until the T4 is unavailable.

Training uses `save_every` in the run config and resumes from `last.pt`, so a
multi-day 1660 Ti run is restartable.

## 7. Non-goals

* instruction following beyond the SFT slice;
* long context (512 tokens is the training length; the model extrapolates
  poorly past it);
* factual reliability without retrieval;
* beating dense models of the same size on general benchmarks — the claim is
  about stored parameters and deployment, not about winning a leaderboard.

## 8. Stages and deliverables

There is **one pretrain** in this cycle, followed by SFT. The order is fixed:

| stage | what | deliverable |
|---|---|---|
| 1. Pretrain | 10M model on the bilingual corpus, 2.251B tokens | checkpoint, int8 artifact, held-out bpb, generation samples |
| — | article #1 (Habr) written immediately after pretrain | published post with the bpb table and the negative results |
| 2. SFT / QA | structured Q/A, math word problems, short code, guarded identity slice, 10–20 % pretrain replay | SFT checkpoint, before/after bpb and probe table |
| — | article #2 (Habr) written immediately after QA, if there is something to report | published post with the SFT results |

Distillation is **out of scope for now**. It is the one remaining idea with a
plausible gain (soft top-k labels from a larger teacher as a final annealing
phase), but every convenient teacher is instruct-tuned, and its refusal and
disclaimer style would leak into the logits. It is shelved until a base or
otherwise unaligned teacher is available; the plan then is a bounded 20–50M
token annealing experiment, mixed with hard-label cross-entropy, never a
full-corpus teacher pass.

The SFT stage keeps the pretrained checkpoint untouched, writes to its own run
directory, and gates on: held-out pretrain bpb (≤3 % regression), identity
probe, and generation at T=1.0, top_k=5–10.

## 9. Run mechanics (implemented and verified)

| piece | state |
|---|---|
| launch | 	orchrun --nproc_per_node=2 -m kronllm.cli train --exp Q10 --data-dir /kaggle/input/kron-corpus-2b |
| DDP | rank-0 logging and checkpoints, per-rank batches, identical validation; verified on two CPU ranks |
| epoch sampler | one shared permutation per epoch, ranks take 
ank::world, so an epoch is a full pass (not the ~63 % of random sampling) |
| checkpoints | save_every=500, last.pt carries model + optimizer + step, --resume tested |
| steps | 137,377 x 16,384 tokens = 2.251B = one epoch (one T4, batch 32, fp16) |
| Kaggle | code dataset icold21/kron-llm-code; bins icold21/kron-corpus-2b |

The first epoch is the first segment, not a hard target: if validation bpb is
still falling at step 137,377, the run resumes for a second epoch with a fresh
cosine cycle. At 10M parameters and 1.6B tokens, a second pass is cheap and
unlikely to overfit.

