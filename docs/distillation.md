# Distillation from math teachers — design note

Status: **deferred on 2026-10-28** by the project owner ("пока отложим").  The
note stays in the repository so the analysis does not have to be redone: it
records the measured tokenizer mismatch, the cost accounting and the staged plan
with its gates.  Nothing here is scheduled; revisit only if the model is scaled
past ~40M parameters or if the SFT baseline leaves math clearly on the table.

Nothing here has been trained yet.  The note was written 2026-10-28
while the SFT corpus was being fetched, so that the decision to run (or not run)
the distillation stage rests on numbers instead of on "the idea sounds good".

The proposal this note evaluates: after SFT, distil from math-specialised models
(DeepSeekMath, Qwen2.5-Math, and similar).  Their alignment is thin to
non-existent, so the student receives a distribution that has not been pushed
towards refusal or politeness boilerplate — "soft labels for free".  Mix SFT
samples into the distillation batch to keep the format.

The idea is sound, with two caveats that decide whether it works on a 10M model:
the teacher's tokens are not our tokens, and a 700x larger teacher is not
necessarily a useful one for a student this small.

## 1. Three different things are called "distillation"

| form | what the student sees | cost per token | what it teaches |
|---|---|---|---|
| sequence-level (hard labels) | teacher's sampled solution, possibly verified | one teacher generation | the teacher's *path*, not its uncertainty |
| token-level (soft labels) | teacher's top-k distribution at each step | one teacher forward | the teacher's *ranking* of alternatives, including "this is close but wrong" |
| hidden-state / feature | teacher activations | forward + storage of d-dim vectors | internal representation, not output behaviour |

"Soft labels" only exist in the second row.  Most published "math distillation"
(OpenMathInstruct, NuminaMath, Orca-Math) is the first row: generate, filter,
fine-tune.  That is already in our SFT mix and it is cheap.  The interesting
part of the proposal is the second row, and that is where the tokenizer problem
below bites.

## 2. The advantage of an unaligned teacher is real, and testable

A refusal-tuned model puts mass on "I can't help with that", on hedging, and on
restating the question.  On a math prompt that mass is pure contamination: the
student learns to spend its first tokens on an apology.  A base-model teacher
spends those tokens on the derivation instead.

This is measurable rather than rhetorical, and it should be measured before
committing:

1. take 200 held-out math prompts from the SFT mix;
2. run the teacher's first-token distribution on the answer prefix;
3. report the mass on a refusal/boilerplate token list ("I", "Sorry", "As",
   "Не", "К сожалению", plus the chat-template tokens);
4. compare base vs instruct checkpoints of the same teacher.

If the base model's refusal mass is not lower, the whole "no alignment" premise
is wrong and the plan loses its main argument.

## 3. The obstacle: the teacher does not speak our tokens

Our model uses an 8192-token byte-level BPE with four chat specials.  Qwen2.5-Math
uses a ~150K vocabulary with its own normalisation; DeepSeekMath uses a 100K
SentencePiece vocabulary.  A KL divergence between two categorical
distributions over *different* alphabets is undefined, so "just take the
teacher's logits" does not work.

Measured with `scripts/tokenizer_alignment.py` on our own SFT arithmetic text
(300 sequences per language, teacher = Qwen2.5-Math-1.5B):

| language | student B/token | teacher B/token | our token == one teacher token | partial overlap | spans several teacher tokens | student tokens per teacher token |
|---|---:|---:|---:|---:|---:|---:|
| English | 1.71 | 2.49 | **17.6 %** | 64.8 % | 17.6 % | 1.19 |
| Russian | 2.52 | 4.65 | **4.7 %** | 79.9 % | 15.4 % | 1.03 |

On English, fewer than one token in five lines up exactly; on Russian, fewer
than one in twenty.  (Arithmetic is the easy case — digits tokenise the same way
in both vocabularies.  Prose would be worse.)  So a token-level KL is only
available through one of three workarounds:

**a. Segment scoring (exact, recommended).**  Our token `t` covers a byte string
`s`.  Run the teacher once over the text, then compute

  `P_T(t | context) = Π_j P_T(u_j | u_<j)`  over the teacher's own sub-tokens
  `u_1..u_k` that cover `s`.

This is a prefix product over a contiguous run of teacher tokens, so one forward
pass is enough; the result is a distribution over *our* vocabulary and KL is
well defined.  Two honesty requirements: tokens that straddle a teacher
boundary have to be handled explicitly (their mass leaks out of the product), and
the reconstructed distribution sums to slightly less than 1 — the deficit is the
measurement of how much the two tokenizations disagree, and it must be reported
rather than normalised away silently.

**b. Byte-level projection (robust, low resolution).**  Reduce both
distributions to the next-*byte* distribution (sum the mass of every token
starting with that byte).  No alignment work at all, vocabulary-free, and it
degrades gracefully — but it throws away most of the teacher's information: at
4 bytes per word, byte-level targets are nearly uniform, and the student already
has them exactly.

**c. Same-tokenizer teacher (no problem at all).**  Take a teacher that shares
our tokenizer — our own larger Kronecker model, or a teacher whose head is
retrained on our vocabulary.  This is self-distillation with extra steps, but it
is the only variant with zero alignment risk, and it is the natural second step
if a 10M model ever has a 40M sibling.

## 4. Is a 7B teacher useful to a 10M student?

Capacity gap: 700x for DeepSeekMath-7B, 150x for Qwen2.5-Math-1.5B.  The
distillation literature is consistent that gains shrink as the gap grows: a
student cannot represent a distribution that is far sharper and structured
differently from anything in its own function class, and the KL objective then
just pushes it towards the teacher's *most likely* token — which is what a
hard-label SFT on the same data already does.

Consequence for the plan: start with the smallest teacher that is clearly better
than the student on the task (Qwen2.5-Math-1.5B, or a 1-3B DeepSeekMath family
member), and treat the 7B teacher as an optional upgrade only if the 1.5B run
shows a signal.  A 10M student matched against a 7B teacher risks buying
nothing but GPU hours.

## 5. Cost, measured against what we actually have

Per 20M-token distillation subset, one teacher forward pass, fp16, T4:

| teacher | VRAM | throughput (order) | GPU-hours per 20M tokens | top-50 logits on disk |
|---|---|---:|---:|---:|
| 1.5B | ~3 GB | 4-6k tok/s | 1-1.5 h | ~4 GB |
| 7B | ~14 GB (tight) | 400-800 tok/s | 8-14 h | ~4 GB |

Storing top-50 logits as `(uint16 id, float16 logit)` costs 200 bytes per token;
20M tokens is 4 GB, 50M is 10 GB.  Our weekly Kaggle budget is 30 GPU-hours, so
the 1.5B teacher fits inside a week even with the pretraining epoch running; the
7B teacher does not, not without displacing a pretraining session.

A local pilot is possible without Kaggle at all: Ollama can return top-k
log-probabilities (`/api/chat` with `logprobs`), and a 1.5B math model fits on
the GTX 1660 Ti.  At a few hundred tokens per second a 100k-token pilot takes
under ten minutes — enough to validate the segment-scoring code and the loss on
real logits before spending GPU hours.

## 6. Staged plan with gates

| stage | what | gate to continue |
|---|---|---|
| 0. baseline | SFT on the 0.5B mix, no distillation | — |
| 1. hard-label distillation | teacher generates solutions for the math slice, verified against the arithmetic answer; mix 30-50 % of these into SFT | +5 points exact match on the final answer, no loss on format compliance |
| 2. soft-label pilot | segment-scored KD on 100k local tokens (Ollama, 1.5B); loss = α·KL(T=2) + (1-α)·CE, α = 0.5, top-50 | KL on held-out math prompts falls more than the same compute of extra CE |
| 3. soft-label run | 20-50M tokens, 1.5B teacher on Kaggle, 20-30 % SFT/pretrain replay mixed in | exact match beats stage 1 by more than its confidence interval |
| 4. optional 7B | only if stage 3 shows a real gain | — |

Stage 1 is the highest value per GPU-hour and can start as soon as SFT has a
baseline.  Stage 2 is cheap enough to run on the local GPU tomorrow.  Stage 3 is
where the "soft labels for free" claim actually gets tested.

The user's suggestion of mixing SFT samples into the distillation batch is part
of the recipe, not an optimisation: KL against a teacher that was never
fine-tuned on our chat template will otherwise pull the student off the
`<|assistant|> … <|eot|>` format, and the format is the whole interface.

## 7. Failure modes worth naming in advance

- **Teacher arithmetic is wrong.**  Math-specialised models still fail; a wrong
  CoT with a confident number teaches confident wrongness.  Every generated
  solution whose final answer can be checked (all our synthetic arithmetic, all
  GSM8K-style problems) must be checked, and unchecked sources must be a small
  share of the mix.
- **Reasoning-length mismatch.**  DeepSeekMath writes 300-token derivations; our
  context is 1,024 tokens and our model is 10M parameters.  Long CoT is not
  learnable at this size and will teach the student to ramble; truncate or
  prefer short-CoT teachers.
- **Alignment bugs are silent.**  A one-token offset in segment scoring shifts
  every target by one step and the model still trains — it just learns garbage.
  The unit test is: reconstruct the teacher's distribution on text the teacher
  tokenises identically (digits, spaces) and compare against the teacher's own
  logits; they must match to float precision.
- **Forgetting.**  Distillation on math only will cost prose and Russian unless
  replay is mixed in; measure bpb on the pretraining held-out before and after.
- **Capacity ceiling.**  If the student cannot fit the teacher's distribution,
  the KL stops falling while exact match stops improving; that is the signal to
  stop, not to add epochs.

## 8. Honest expectation

At 10M parameters the student will learn *format, short derivations and the
arithmetic it can already do*.  Soft labels may buy a fraction of a point of
exact match; they will not buy general math.  The value of the stage is
therefore twofold: a measurable comparison of hard vs soft labels at a size
where both are cheap, and a pipeline that is ready if the model is later scaled
to 40-100M, where the capacity gap argument flips in favour of the teacher.

That is worth doing after SFT, in the order staged above — but stage 1 before
stage 3, and the local pilot before either.
