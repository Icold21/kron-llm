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

## Shipped checkpoints

| file | parameters | size | quality |
|---|---:|---:|---|
| `q8g_int8.pt` | 2,851,280 | 3.14 MB | **0.7542 bits/byte** (TinyStories val, 61.4M tokens seen) |
| `q2_int8.pt` | 2,079,992 | 2.34 MB | previous model, same corpus |

Int8 quantisation costs +0.0003 bpb on Q8g. The fp32 checkpoint is 11.4 MB and
is not shipped.

## Usage

```python
import torch
from kronllm.model import KronTransformer
from kronllm.quantization import load_int8

ckpt = torch.load("q8g_int8.pt", map_location="cpu")
model = KronTransformer(ckpt["cfg"]).eval()
load_int8(model, ckpt["model"])
```

The model is a research artifact: it writes short English and Russian text and
is not an instruct-tuned assistant.

## Architecture at a glance

| | |
|---|---|
| layers / width | 24 / 320, 8 heads, seq 192 |
| composition | product over L=4 factors, K=6 terms, depth carry `K_win=2` |
| FFN | gated (SwiGLU-style) |
| tokenizer | BPE-4096, tied factorised embedding (rank 128) |

Measured, not assumed: the depth carry is worth 1.1 % bpb, freezing its scalar
0.8 %, and widening the window to four layers *costs* 1.4 %. Details in the
[research log](https://github.com/Icold21/kron-llm/blob/main/docs/results.md).

## Next

The 10M model (Q10: d=512, 25 layers, K=10, 9,938,886 parameters) trains on
[KronCorpus](https://huggingface.co/datasets/Icold/kron-corpus) — 2.25B tokens of
bilingual English/Russian text with a science accent — on two Kaggle T4s. The
checkpoint will appear here when the first epoch finishes; the corpus card
documents every source and licence.

## Licence

Code and weights: MIT. The corpus keeps its sources' licences (see the dataset
card).
