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
* **85 % of documents are annotated** (`--prompt-rate`), the rest stay bare.
  With every document annotated the model would never see a continuation
  without a prompt — and the held-out bin, which every bpb number we quote is
  measured on, has no prompts.  Keeping a sixth of the stream bare keeps both
  modes in distribution.
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

Measured on the decay bin, built 2026-10-30 (`data/bilingual/anneal_bin`,
`tokenizer_meta.json` → `domain_prompts`):

| quantity | value |
|---|---|
| training documents | 404,277 |
| documents with a prompt | 300,744 (74 %) |
| prompts, mean length | 21.1 tokens |
| tokens added | 6,354,032 (650,472,331 → 656,826,363, +0.98 %) |
| held-out documents | 8,242 (untouched) |

Coverage per (language, domain), prompted documents: en/math 33,363,
en/ml 43,500, en/encyclopedic 42,135, en/identity 41,198, en/code 12,251,
en/science 8,033, en/logic 2,715, en/literature 1,760; ru/identity 46,899,
ru/literature 34,855, ru/encyclopedic 23,595, ru/math 4,452, ru/science 3,147,
ru/cs 1,689, ru/ml 774, ru/logic 378.  Every domain of the mix is represented in
both languages — that is the check the counter exists for.

The 74 % is not a bug in the rate.  Twelve and a half per cent of the documents
in this mix are the chat pools (51,641 of 412,519), and those are skipped
because they already open with a rendered `<|system|>` turn; of the remaining
87.5 %, the rate annotates 85 %: `0.875 × 0.85 = 0.744`.

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

Measured on the 2026-10-30 decay bin: **351,353 `<|system|>` ids in
656,826,363 tokens** — 300,744 prompt prefixes plus 50,609 chat documents that
already opened with a system turn.  `<|user|>` / `<|assistant|>` appear 90,218 /
90,342 times, all of them inside the chat pools: the prompts themselves add no
dialogue turns.

Decode the first window with `kronllm.tokenizer.BPETokenizer` and it opens with
`<|system|>`, a domain prompt, `<|eot|>` and then the document.  The build log
prints coverage per (language, domain), so a domain that silently got no prompt
is visible in the log rather than inferred from a flat loss curve.
