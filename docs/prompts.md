# Domain system prompts in pretraining

Status: implemented 2026-10-30.  The decay bin (`data/bilingual/anneal_bin`)
carries prompts since the 2026-10-30 rebuild; the epoch bin picks them up at
its next rebuild.  Code: `kronllm/domain_prompts.py`; wiring:
`scripts/prepare_bilingual_bin.py`; tests: `tests/test_domain_prompts.py`.

## 1. Why

Two problems, one mechanism.

**The chat tokens were cold.**  A scan of the 2.25 B-token epoch bin found
0 / 5 / 6 / 0 occurrences of `<|system|>` / `<|user|>` / `<|assistant|>` /
`<|eot|>`.  SFT and every sampler open with `<|system|>…<|eot|>`, so the first
SFT steps would have been spent teaching the model four markup tokens instead of
answers, and any prompt written by a user would land in a region of the token
space the model has never seen used.

**The stream said nothing about what follows.**  A window of pretraining text
could be a chemistry textbook, a novel or Python, and nothing in the stream
distinguished them.  A 10M-parameter model cannot afford to infer the register
from the first fifty tokens every time; labelling the document is the cheapest
signal available.

## 2. The mechanism

A share of the training documents is prefixed, in place, with the chat
template's system turn:

```
<|system|>Ты — помощник по естественным наукам: химия, физика, биология,
астрономия. Отвечай формулами, единицами и числами.<|eot|>Ниже — учебник по
естествознанию. Опиши механизм, величины и единицы измерения…
```

* **Six paraphrases per (language, domain)**, three for `humour` and
  `identity`, written by hand in English and Russian — 108 prompts in total.
  The variant is chosen by `sha256(document) % len(variants)`, so a rebuild
  produces the same bin byte for byte and every variant appears equally often.
* **Every document is annotated** (`--prompt-rate`, default 1.0).  The first
  version left a sixth bare, on the theory that the held-out bin has no prompts
  and the promptless mode therefore had to stay in distribution.  That was
  defensive and empty: a training window is cut at a random offset inside a
  document, so a bare *document start* — the only place the tag could appear —
  reaches the model in a small fraction of windows either way.  The partial rate
  bought nothing measurable and cost a second distribution to explain.
* **Dialogue is skipped.**  Documents that already open with `<|system|>` (the
  chat pools of the decay pass) are left alone: wrapping a rendered
  conversation in a second system turn would teach the model that a system
  prompt can appear in the middle of a dialogue.
* **The held-out set is untouched.**  The pass runs after the train/val split,
  so `val.bpe.bin` is the same file the earlier checkpoints were scored on and
  bpb stays comparable across the whole run.

## 3. Why paraphrases and not one sentence

A single fixed sentence per domain is a string to memorise: the model learns to
continue *that sentence*, and the association dies the moment a user writes
their own system prompt in SFT.  A family of paraphrases forces the model to
treat the tag as an announcement of register, which is the part that has to
survive into SFT.

The variants deliberately differ in what they promise: some name the field, some
name the expected form of the answer ("code first, then the explanation"), some
name the behaviour ("ask a clarifying question when the request is ambiguous").
That is a first, cheap approximation of instruction following: the model sees
the same tag introduce different *contracts* over different text.

## 4. What it costs

Measured on the two bins rebuilt 2026-10-31 (`tokenizer_meta.json` →
`domain_prompts`).  "Added" and "already chat" are separate counters on purpose:
a chat document opens with `<|system|>` already, so it is covered without being
annotated, and `added + already` is what must equal the document count.

| quantity | decay bin | epoch bin |
|---|---|---|
| training documents | 397,504 | 1,028,370 |
| prompts added | 346,895 | 964,626 |
| already chat (skipped) | 50,609 | 63,744 |
| covered | **397,504 (100 %)** | **1,028,370 (100 %)** |
| prompts, mean length | 21.3 tokens | 21.3 tokens |
| train tokens | 655,857,221 | building |
| held-out documents | 8,101 (untouched) | 20,978 (untouched) |

The 87 % and 94 % the build log prints are "prompts added over documents", not
coverage: the remainder is the chat pools, which already carry a system turn.
Coverage per (language, domain) is in `by_domain` in the same file, and every
domain of the mix appears in both languages — that is the check the counter
exists for.

The cost is paid once per document, not per window: a document split across
several 512-token windows carries its prompt only at the start.

The trade is deliberate.  At 10M parameters every token of context is scarce,
and a prompt that names the domain spends ~1 % of the budget on labels.  What it
buys is a warm chat template, a domain signal, and an SFT stage that starts from
"system turn understood" instead of "what is this token".

## 5. What it is not

* Not a chat format: there is no `<|user|>` and no `<|assistant|>` turn, the
  document text follows the system turn directly.  The model learns
  "system turn → text of this kind", not "system turn → answer".
* Not a substitute for SFT: SFT still has to teach the three-turn structure,
  answer formatting and instruction following.  It starts warmer, not finished.
* Not a change to the tokenizer: `<|system|>` and `<|eot|>` already existed
  (ids 8187 and 8190); nothing about the vocabulary moves.

## 6. How to check it worked

```powershell
python -c "import numpy as np; a=np.fromfile('data/bilingual/anneal_bin/train.bpe.bin', dtype=np.uint16); print((a==8187).sum(), 'system tokens of', len(a))"
```

Measured on the 2026-10-31 decay bin: **397,504 `<|system|>` ids in
655,857,221 tokens** — exactly one per document, which is the whole point: 346,895
added by the pass plus 50,609 chat documents that already opened with a system
turn.  A single extra or missing tag shows up immediately in that count.

Decode the first window with `kronllm.tokenizer.BPETokenizer` and it opens with
`<|system|>`, a domain prompt, `<|eot|>` and then the document.  The build log
prints coverage per (language, domain), so a domain that silently got no prompt
is visible in the log rather than inferred from a flat loss curve.

## 7. Removed: 2026-10-31

The prompts are out of the corpus.  Measured on both bins end to end:

| bin | tokens | `<\|system\|>` | documents | tags/doc |
|---|---|---|---|---|
| prompted | 2,368,625,438 | 1,028,370 | 1,028,370 | 1.000 |
| prompt-free | 2,347,352,486 | 63,744 | 1,028,370 | 0.062 |

21,272,952 tokens (0.90 %) of the stream were our own boilerplate, and exactly
one tag per document means every one of them was ours.  The prompt-free rebuild
keeps 63,744 tags -- the chat documents that open with a system turn of their
own, which is data, not boilerplate.

Why: step 35,115 was trained without prompts, and the run resumes there.
Prefixing 74 % of documents changes what the model is asked to predict without
changing the step count, and the mask only hid that from the loss, not from the
context.  With 10 M parameters, spending 1 % of the budget on 108 repeated
strings to teach "this register follows this sentence" is not worth a format
that SFT then has to unlearn.  The held-out bin was never prompted, so every
historical bpb stays comparable -- it was always measured on the bare text.

The mask (`kronllm/loss_mask.py`, `model.ignore_prompt_targets`) stays on: it
still removes the system turn of those 63,744 chat documents, and it is what
makes a future prompt-bearing corpus safe by construction.

Rebuild recipe:

```bash
python -u scripts/prepare_bilingual_bin.py \
  --out-dir E:/kron_bins/epoch_np \
  --reuse-tokenizer data/bilingual/tokenizer.json \
  --reuse-val-dir E:/kron_bins/epoch \
  --phase epoch --no-domain-prompts
```

`--reuse-val-dir` is what keeps `val.bpe.bin` byte-identical (sha256 a45bbb03…).
Note that `--no-domain-prompts` used to leave `rate: 0.85` in the metadata; it
now writes 0.0, and the 2026-10-31 bin was patched by hand for the same reason.
