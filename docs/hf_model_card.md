---
language:
  - en
  - ru
license: mit
library_name: pytorch
pipeline_tag: text-generation
tags:
  - language-model
  - kronecker
  - small-language-model
  - bilingual
  - parameter-efficiency
---

# KronLLM

A compact bilingual (English/Russian) language model whose matrix layers are
built entirely from **Kronecker sums of rank-1 factors**. Every attention and
FFN projection is a composition of `L` factors instead of a dense `d × d`
weight: a layer that would need 102,400 numbers needs 15,753.

This repository hosts the model card, the code and the shipped int8
checkpoints. The training corpus is separate: see
[Icold21/kron-corpus](https://huggingface.co/datasets/Icold21/kron-corpus) and
the research log at [github.com/Icold21/kron-llm](https://github.com/Icold21/kron-llm).

## Main model: Q10 (10M)

**Q10 is the model this project is about**: 9,938,886 parameters (d=512, 25
layers, 8K bilingual BPE, Kronecker K=10), trained on 2.78B tokens of bilingual
English/Russian text. It is mid-pretraining — **step 36,549 of 169,884 (21.5 %)**
of the first epoch — and reaches **1.1828 bits/byte** mixed (EN 1.4327 /
RU 0.9411) on a held-out bin that never changes; GPT-2-124M scores 1.554 on the
same blocks. The checkpoint lives on Kaggle
([`icold21/kron-q10-ckpt`](https://www.kaggle.com/datasets/icold21/kron-q10-ckpt))
and will be mirrored here when the epoch finishes.

## Experimental checkpoints (the development bench)

The two small checkpoints below are **not** the product. They are the TinyStories
bench that made the 10M configuration cheap to choose: each isolates one
decision (tokenizer, gated FFN, depth carry), and their bits/byte is not
comparable with Q10's bilingual number.

| file | parameters | size | quality |
|---|---:|---:|---|
| `q8g_int8.pt` | 2,851,280 | 3.14 MB | **0.7542 bits/byte** (TinyStories val, 61.4M tokens seen) |
| `q2_int8.pt` | 2,079,992 | 2.34 MB | previous bench model, same corpus |

Int8 quantisation costs +0.0003 bpb on Q8g. The fp32 checkpoint is 11.4 MB and
is not shipped.

## Usage (bench checkpoint)

```python
import torch
from kronllm.model import KronTransformer
from kronllm.quantization import load_int8

ckpt = torch.load("q8g_int8.pt", map_location="cpu")
model = KronTransformer(ckpt["cfg"]).eval()
load_int8(model, ckpt["model"])
```

The bench model is a research artifact: it writes short English text and is not
an instruct-tuned assistant.

## Architecture of the bench model

| | |
|---|---|
| layers / width | 24 / 320, 8 heads, seq 192 |
| composition | product over L=4 factors, K=6 terms, depth carry `K_win=2` |
| FFN | gated (SwiGLU-style) |
| tokenizer | BPE-4096, tied factorised embedding (rank 128) |

The main model keeps the same block but scales to d=512, 25 layers, K=10 and the
8K bilingual BPE, with pre-norm QK and a gated FFN.

Measured, not assumed: the depth carry is worth 1.1 % bpb, freezing its scalar
0.8 %, and widening the window to four layers *costs* 1.4 %. Details in the
[research log](https://github.com/Icold21/kron-llm/blob/main/docs/results.md).

## Training budget

At the measured 2.552 s/step on a single Kaggle T4 (batch 32, seq 512), one epoch of the 2.82B-token corpus is **122 GPU-hours**; the decay pass (30 % of the epoch, quality-selected slice plus 5 % chat) adds **36**, and SFT 4,000 steps adds **2.8**. Roughly **131 GPU-hours remain** from step 36,549 -- about 12 sessions of quota. Derivation in `docs/compute_budget.md` of the code repository.

## Next

Q10 continues on [KronCorpus](https://huggingface.co/datasets/Icold/kron-corpus)
— 2.78B tokens of bilingual English/Russian text with a science accent — on one
Kaggle T4. After the epoch: a decay pass (29 % of the epoch, quality-selected,
plus 5 % chat) and SFT. The weights will be mirrored here when the epoch
finishes; the corpus card documents every source and licence.

## Licence

Code and weights: MIT. The corpus keeps its sources' licences (see the dataset
card).
