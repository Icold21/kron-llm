# Roadmap: from architecture search to a 10M bilingual Q/A model

This is the long-term plan behind the current ablation series. It is a living
document: the architecture search in the README is Phase 0, and every later
phase has a gate that must be passed before the next one starts.

## 1. Target

| property | target |
|---|---|
| Parameters | **~10M soft ceiling** (current base: 2.47M at 0.849 bpb) |
| Training hardware | one **6 GB** consumer GPU (currently a GTX 1660 Ti) |
| Languages | **English + Russian**, one shared tokenizer |
| Data | curated, deduplicated, licence-checked pretraining corpus |
| Post-training | **structured Q/A** supervised fine-tuning |
| Optional | distillation from a larger teacher |
| Deployment | fp32 / int8, CPU or one consumer GPU, no custom kernels |
| Quality metric | bits per byte on held-out EN and RU, plus Q/A task metrics |

The point of the Kronecker parameterisation is not to beat a dense model of the
same size on raw quality; it is to make a **10M model trainable inside 6 GB**
and small enough to inspect, quantise and run on ordinary hardware.

### TinyStories is a development bench, not the product corpus

Every architecture comparison so far (the E and Q series) runs on TinyStories
because it is small, fast, and gives a clean bits-per-byte signal. That is its
entire role:

- **now:** cheap, comparable ablations (one run = tens of minutes);
- **later:** a regression/sanity bench for the frozen architecture;
- **never:** part of the final pretraining mix.

The production corpus is the curated bilingual English–Russian set in Phase 1.
Once that set exists, every freeze/scale decision moves to its held-out splits;
TinyStories is kept only to catch regressions in the code path (same config,
expected bpb), not to choose the final model.

## 2. Where we are (2026-10-26)

| result | status |
|---|---|
| **Q8g-long** (2.85M, BPE-4096, gated FFN) | **shipped**, **0.7542 bpb** at 61.4M tokens, 4.09 GB |
| Q6@10k (2.47M, BPE-4096) | previous best, 0.7644 at the same budget |
| Q6 (BPE-4096) | win: 0.8489 bpb at 24.6M tokens, 4.28 GB peak |
| Q4 (full backprop) | win: 0.9248 vs Q2 0.962 at step 4,000 |
| Q8w (gated, 2.47M, 20 layers) | win: 0.8359 vs Q6 0.8489 — the FFN gain is architectural |
| Q9 / Q9k (bilinear GLU FFN) | **negative**: 0.8740 / 0.8673 vs Q6 0.8489 |
| E25/E26/E26b (nonlinear composition) | **negative**: 1.519 / 1.568 / 1.591 vs E8 1.479 |
| Q5 (`K=24`, 7.75M) | **negative**: 0.9530 at 4k, layer collapse (`cos_prev_max` 0.99999) |
| Q7 (Kronecker embedding) | **negative**: 0.9533 at 4k vs Q6 0.8489 |
| E14 (sum composition) | negative: rank is not the quality bottleneck |
| E19-E22 (weight sharing / recursion) | negative as a default; usable as adaptive compute |

The architecture is **frozen**: product composition, learned sigma with a
two-layer carry, gated Kronecker FFN, BPE tokenizer with a reserved chat
template. The 2026-10-25 ablation set (E8r/E27-E30) showed the depth carry is
worth 1.1 %, freezing sigma 0.8 %, and widening the window *costs* 1.4-2.8 %.

**Corpus and run (2026-10-26).** The bilingual corpus is published as
`kron-corpus`: 8.62 GB mixed, **2.251B tokens** measured, English 61.9 % /
Russian 38.1 % of tokens (see README §14 for why bytes and tokens disagree, and
for the Russian top-up that closed the gap). The identity slice is final: 800
plain + 850 pastiche per language, plain first-person prose, no digits, no
dialogue scaffolding, 0 defects under the generator's own gate. Q10
(9,938,886 params) runs on one T4 (DDP is blocked by the 32 GB host cgroup) with a sharded epoch sampler and
checkpoint/resume every 500 steps; one epoch is 137,377 steps.

Everything that broke the single-GEMM fusion lost quality, and the one
structural win (the gate) was confirmed at matched parameters. Remaining levers
are data, context length and scale toward the 10M ceiling.

## 3. Phase 0 — architecture freeze (gate A)

Freeze one recipe when it satisfies all of:

1. Best validation bpb at matched tokens among the ≤10M candidates.
2. Stable across at least two seeds (Δ bpb < 0.01).
3. Peak VRAM ≤ 6 GB with batch ≥ 16 at `seq_len` 192–256.
4. Every winning component has a trained ablation behind it, not an intuition.

Open questions to settle before the gate:

- allocation of the 10M budget: width/depth vs more Kronecker summands;
- embedding storage at an 8K–16K bilingual vocabulary: the Kron embedding
  (Q7) lost 12 % at V=4096, so 8K with a dense factorised table is the default
  until a better compression or initialisation exists;
- context length: 192 today; 256 may be needed for Q/A paragraphs;
- whether the product composition remains the default (E14 says yes for now).

**Exit:** `kronllm/config.py` gets a `B1` config, a model card, and a seed
sweep. No further architecture work until Phase 3 produces a scaling point.

## 4. Phase 1 — bilingual data and tokenizer (gate B)

### 4.1 Corpus

| source | role |
|---|---|
| Wikipedia EN / RU | encyclopedic prose, clean, licence-checked |
| FineWeb-edu / CulturaX | filtered web text, quality-scored |
| public-domain books EN / RU | long-form narrative |
| instruction sets (later) | Q/A fine-tuning only, never pretraining |

Pipeline: language ID → URL/domain filtering → MinHash near-dedup →
quality filter (length, repetition, punctuation, perplexity) → PII and licence
checks → decontamination against the evaluation sets → fixed mixing ratio.
TinyStories is **not** part of the pretraining shards; it is kept only as the
development bench described above.

The concrete 60/40 byte mix, domain quotas (math, science, logic, literature,
encyclopedic), source licences and the pilot pipeline are fixed in
[data_plan.md](data_plan.md).

Start with **50/50 by bytes**, then tune on held-out bpb per language. Target
**0.5–2B tokens**; the compute-optimal point for 10M parameters is ~200M, but
small models usually keep improving past it, so the run is checkpointed and the
bpb-vs-tokens curve decides where to stop.

### 4.2 Tokenizer

Train a byte-level BPE on a mixed sample of both languages.

- vocabulary **8K by default**, 16K only if the dense factorised embedding
  still fits the 10M budget (Q7's Kronecker table is not adopted: it lost 12 %
  bpb at V=4096);
- target fertility: **≈4.0 bytes/token EN**, **≈3.0–3.5 bytes/token RU**;
- special tokens reserved for the Q/A format (user, assistant, end);
- evaluate on held-out text in both languages, not on the training sample.

**Exit:** tokenizer artifacts + meta, fertility report, contamination report,
and a fixed 10M-token held-out set per language.

## 5. Phase 2 — pretraining at 10M (gate C)

Recipe inherited from the frozen Phase-0 config: product composition, full
backprop, RMSNorm, qk-norm, BPE, `seq_len` 256.

| item | plan |
|---|---|
| parameters | 8–10M |
| tokens | 0.5–2B (checkpointed; curve decides) |
| throughput | ~7k tokens/s at 2.5M; expect 4–7k at 10M |
| wall clock | ~20–80 h on the 1660 Ti |
| checkpoints | 25 / 50 / 100 % + best-bpb |
| evaluation | bpb EN/RU, generation samples, VRAM, tokens/s; TinyStories only as a regression check |

**Kill criterion:** if the 10M model is not better than the 2.5M model at
matched tokens, the budget allocation is wrong; stop and return to Phase 0
rather than spending more GPU hours.

**Exit:** a bilingual base model, its scaling curve, and a reproducible config.

## 6. Phase 3 — structured Q/A SFT (gate D)

Format (special tokens, loss masked on the prompt):

```
<|user|> ... <|assistant|> ... <|end|>
```

Data: translated Dolly-15k, open Russian instruction sets, and
teacher-generated Q/A pairs, all filtered for format, duplication and answer
quality. Start with 10–50k examples; the model is 10M parameters, so quality
and consistency matter more than volume.

Three task families are in scope:

| family | example | metric |
|---|---|---|
| structured Q/A | question about a short passage | format compliance, answer bpb, token F1 |
| math word problems | one-step arithmetic with a numeric answer | exact match on the final number |
| short code | function from a docstring, in a fenced block | `pass@1` on unit tests, syntax validity |

Math data: GSM8K-style problems (licence-checked) plus synthetic arithmetic
with controlled templates. Code data: docstring→function pairs from
permissively licensed repositories and small exercises; code prompts keep the
fence and indentation, and the answer is executed in a sandbox for evaluation.

Metrics:

- **format compliance** (does it emit the assistant turn and stop?);
- exact match / token F1 on extractive tasks;
- exact match on the final number for math;
- `pass@1` on unit tests and `ast.parse` validity for code;
- answer bpb on held-out Q/A;
- generation checks at `T=1.0`, `top_k=5–10`.

Honest scope: a 10M model can learn format, style, translation patterns,
one-step arithmetic and short functions; it cannot solve general math or write
non-trivial programs. The SFT goal is a reliable interface and a measurable
baseline, not an assistant that knows the world.

**Identity slice (optional, guarded).** A small set of examples where the model
introduces itself as **Kron / KronLLM**, in English and Russian, with varied
phrasings (`What is your name?`, `Кто ты?`, `What model are you?`). Keep it at
**0.5–1 % of the SFT mix** (roughly 100–300 examples) and keep the answers
factual and modest: a ~10M-parameter compact model, not an oracle.

Guardrails:

- evaluate identity accuracy on held-out phrasings in both languages;
- measure **leakage**: the rate at which "Kron" appears on unrelated prompts;
  if it exceeds ~1–2 %, shrink or drop the slice;
- never let the persona claim capabilities the model does not have;
- keep identity examples out of the math/code subsets so they do not pollute
  those metrics.

**Exit:** SFT checkpoint + evaluation table + failure gallery.

## 7. Phase 4 — SOTA comparison and Habr article (gate E)

Before distillation, the fine-tuned model must be positioned against the
strongest available Kronecker/efficient small models. This phase produces the
final pre-production article for **Habr**; distillation comes after it, so the
article is not blocked on an optional experiment.

**What is compared**

- our frozen 10M model, base and SFT;
- published Kronecker-structured and efficient small LMs with runnable
  checkpoints, evaluated on the same held-out EN/RU bytes;
- dense controls at matched parameter count and matched FLOPs/token;
- the 2.5M model from Phase 0, to show what the 10M scale buys.

**Protocol**

- bits per byte on raw held-out text, so different tokenizers remain comparable;
- identical hardware and evaluation harness; batch-1 generation and training
  throughput at the same batch size;
- report parameters, FLOPs/token, peak VRAM, tokens/s, fp32/int8 checkpoint
  size, and the Phase-3 metrics (Q/A, math exact match, code `pass@1`);
- no cherry-picking: include the negative results (Q5, Q7, E14) and the cases
  where our model loses.

**Deliverables**

- `docs/comparison.md` with the table and the exact commands behind every row;
- the Habr article draft (Russian): architecture, 6 GB capability, all
  ablations including failures, comparison table, honest limitations, roadmap;
- an artifact bundle: int8 checkpoint, config, evaluation script.

**Gate:** every claim in the article maps 1:1 to a row in the comparison table,
and the table is reproducible from a clean checkout. If the model is not
competitive, the article says so — the negative results are part of the value.

**Exit:** final pre-production article + reproducible comparison table.

## 8. Phase 5 — distillation (optional, gate F)

Teacher: a 1.5–8B open model, run through an API or 4-bit locally.

1. Generate answers on curated prompts; filter by format and teacher confidence.
2. Sequence-level distillation: train the student on teacher outputs.
3. Optional logit distillation if teacher and student share a tokenizer or a
   mapping is cheap.
4. Compare against the Phase-3 SFT model at equal tokens and equal inference cost.

**Exit:** a student that is better than SFT-only on the Q/A metrics, or a
documented negative result.

## 9. Phase 6 — packaging

- int8 post-training quantisation (near-lossless on Q2: +0.0004 bpb);
- CPU and single-GPU inference through the Hugging Face wrapper;
- model card with data, licence, evaluation and known failure modes;
- keep only the final artifacts (no near-duplicate checkpoints).

## 10. Risks

| risk | mitigation |
|---|---|
| 10M is too small for useful Q/A | keep expectations at format/skills; distillation and retrieval are the escape hatches |
| Russian tokenizer fertility is poor | train on mixed text, measure bytes/token per language, 8K default |
| data licensing | only permissive sources; record provenance per shard |
| contamination | decontaminate against all evaluation sets before training |
| catastrophic forgetting in SFT | small LR, short schedule, mix a fraction of pretraining text |
| 6 GB ceiling | fixed activation footprint, chunked loss, int8 evaluation |
| architecture overfitting to TinyStories | validate the final recipe on the bilingual held-out sets |
| SOTA comparison is unfair or incomplete | fixed protocol, matched bytes/FLOPs, all commands published, negative results included |

## 11. Rough timeline

| phase | wall clock |
|---|---|
| Phase 0 (architecture freeze) | 1–2 weeks of GPU runs |
| Phase 1 (data + tokenizer) | 3–5 days, mostly CPU |
| Phase 2 (10M pretraining) | 3–4 days GPU |
| Phase 3 (Q/A SFT) | 1 day |
| Phase 4 (SOTA comparison + Habr article) | 2–3 days |
| Phase 5 (distillation, optional) | 1–2 days |
| Phase 6 (packaging) | 1–2 days |

These are research estimates, not commitments; every phase can send the project
back one step, and that is a valid outcome as long as it is recorded.
