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
| Kaggle T4 (16 GB, 30 GPU-h/week) | 2B tokens is a multi-session job, not a multi-week one |
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
architecture at 2.85M on the same 2B-token mix, and it runs before the 10M
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

**Is one more layer worth it?** A layer costs 308,406 parameters (+3.1 %),
5 MFLOP/token (+3.6 %) and 0.3 MB in int8:

| layers | parameters | MFLOP/token | int8 size | 2B tokens |
|---:|---:|---:|---:|---:|
| 25 | 9,938,886 | 138 | 9.9 MB | 2.8×10¹⁷ |
| 26 | 10,247,292 | 143 | 10.2 MB | 2.9×10¹⁷ |
| 27 | 10,555,698 | 148 | 10.6 MB | 3.0×10¹⁷ |

At this size a 3 % parameter increase is worth roughly 0.2–0.3 % bpb under a
standard scaling slope — below the ~1 % seed noise we measure. Depth is the
*right* axis when parameters are held fixed (E20: −3.5 % bpb against E8 at
2× compute), but adding a layer is not that experiment. The reference run
therefore stays at 25 layers; 26 is the fallback if the corpus overshoots
2.2B tokens, where the extra capacity is at least matched by extra data.

## 5. Corpus contract

| property | value |
|---|---|
| tokens | 2.0B (2.2B fetched, ~10 % lost to dedup and the held-out split) |
| bytes | ~6.7 GB of text at 3.34 bytes/token |
| languages | 60/40 EN/RU **by bytes** (the token share is ~55/45 because Cyrillic costs 2 bytes per character; the contract is bytes, stated explicitly) |
| domains | math 25 %, code 15 %, science 14 %, ML 12 %, literature 12 %, encyclopedic 11 %, logic 10 %, identity 1 % |
| dedup | MinHash, 0.8 Jaccard threshold, applied before the mix |
| held-out | 2 % per language, disjoint by document |
| provenance | every shard has source, licence, URL and SHA-256 in `manifest.jsonl` |

Domain labels for Wikipedia and FineWeb text are **keyword heuristics**, not a
curated taxonomy; this is recorded in the manifest note for those shards. The
identity slice is synthetic and prompt-constrained (see
`docs/data_plan.md`).

## 6. Compute plan

FLOPs per token ≈ `6N + 12·L·seq·d` = 60M + 79M ≈ **138 MFLOP/token** at
d=512, L=25, seq=512. For 2B tokens: **2.8 × 10^17 FLOPs**.

| hardware | assumed throughput | 2B tokens | notes |
|---|---|---|---|
| Kaggle T4 (16 GB, fp16) | 15–30 % MFU ≈ 10–20 TFLOPS | **4–8 h** | one or two sessions; T4 has tensor cores |
| GTX 1660 Ti (6 GB, fp32) | 90–300 GFLOP/s | **5–8 days** | no tensor cores; launch-bound; needs checkpoint/resume |

The 1660 Ti number is extrapolated from the measured Q8g throughput
(5.15K tokens/s at 2.85M params, ≈90 GFLOP/s, 1.6 % of the card's 5.4 TFLOPS).
It is a range because the 10M configuration has 35× more tokens per step, which
amortises the per-launch overhead that dominates Q8g. A 200-step benchmark of
the actual 10M configuration replaces this estimate before the run is
scheduled. Training uses gradient checkpointing per block, `save_every` in the
run config, and resumes from `last.pt`, so a multi-day 1660 Ti run is
restartable.

## 7. Non-goals

* instruction following beyond the SFT slice;
* long context (512 tokens is the training length; the model extrapolates
  poorly past it);
* factual reliability without retrieval;
* beating dense models of the same size on general benchmarks — the claim is
  about stored parameters and deployment, not about winning a leaderboard.
