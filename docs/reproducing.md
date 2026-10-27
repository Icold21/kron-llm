# Reproducing the numbers

Every claim in [results.md](results.md) is a command plus a seed, not a
checkpoint. This page is the list of those commands, rendered from the
verbatim `results.json` dump in [retired_runs.json](retired_runs.json),
so it survives the deletion of the weights it describes.

## Policy: what is kept, and why

| artefact | kept? | reason |
|---|---|---|
| code, config, seed | yes, in git | the only things a re-run needs |
| `results.json` of every run | yes, `docs/retired_runs.json` | the numbers are the result; `runs/` is git-ignored, so this dump is the copy |
| `log.csv` per run | yes, local | curves are cheap and useful for diffing |
| 190K–2.8M experiment weights | **no** | 4,000 steps ≈ 28 min on the reference GPU; storing them buys nothing |
| shipped artefacts (`artifacts/q8g_int8.pt`, `q2_int8.pt`) | yes, in git | a reader can load them offline |
| Q10 weights | yes, at the end of the epoch | 95 GPU-h is not reproducible on a laptop; publish weights-only + SHA256 |

A 190K model that takes 28 minutes to retrain does not need a binary
attached: the config is the artefact. The opposite is true for the 10M
run, where nobody can afford the retrain — that one gets published in
full.

## What is *not* claimed

* **Bit-exactness.** `deterministic=True` is set, but cuDNN and the
  reduction order in `baddbmm` are not bit-stable across driver versions.
  Expect the same bpb to ±0.01, not to the fourth digit.
* **Seed-robust deltas below ~0.03 bpb.** One control run (hybrid
  compositions) reversed a 1.3 % win on a second seed; that is the
  measured noise floor and it applies to every single-seed number here.
* **Anything on TinyStories as a statement about the target corpus.** The
  bench is the bench.

## Commands

Family prerequisites: byte → TinyStories bytes: `python scripts/prepare_data.py`; BPE → TinyStories BPE: `python scripts/prepare_bpe.py --vocab 4096`; WikiText → WikiText-103 bytes: `python scripts/prepare_wikitext.py`.

| run | params | steps | expected bpb | runtime | command |
|---|---:|---:|---:|---:|---|
| D1 | 1,068,544 | 4,200 | **0.9770** | 12 min | `python -m kronllm.cli train --exp D1 --steps 4200` |
| E14 | 190,348 | 4,000 | **1.5508** | 29 min | `python -m kronllm.cli train --exp E14 --steps 4000` |
| E14w | 190,348 | 4,000 | **2.2118** | 28 min | `python -m kronllm.cli train --exp E14w --steps 4000` |
| E15 | 221,308 | 4,000 | **1.5091** | 72 min | `python -m kronllm.cli train --exp E15 --steps 4000` |
| E16 | 193,468 | 4,000 | **1.5504** | 75 min | `python -m kronllm.cli train --exp E16 --steps 4000` |
| E17w | 190,348 | 4,000 | **2.1561** | 33 min | `python -m kronllm.cli train --exp E17w --steps 4000` |
| E17w_s43 | 190,348 | 4,000 | **2.1624** | 33 min | `python -m kronllm.cli train --exp E17w_s43 --steps 4000` |
| E18w | 190,348 | 4,000 | **2.1578** | 34 min | `python -m kronllm.cli train --exp E18w --steps 4000` |
| E19 | 111,502 | 4,000 | **1.7172** | 27 min | `python -m kronllm.cli train --exp E19 --steps 4000` |
| E20 | 190,108 | 8,000 | **1.4271** | 65 min | `python -m kronllm.cli train --exp E20 --steps 8000` |
| E21 | 111,758 | 4,000 | **1.7180** | 21 min | `python -m kronllm.cli train --exp E21 --steps 4000` |
| E22 | 111,502 | 8,000 | **1.5214** | 63 min | `python -m kronllm.cli train --exp E22 --steps 8000` |
| E25 | 190,108 | 4,000 | **1.5189** | 58 min | `python -m kronllm.cli train --exp E25 --steps 4000` |
| E26 | 190,108 | 4,000 | **1.5675** | 44 min | `python -m kronllm.cli train --exp E26 --steps 4000` |
| E26b | 190,108 | 4,000 | **1.5906** | 43 min | `python -m kronllm.cli train --exp E26b --steps 4000` |
| E27 | 190,108 | 4,000 | **1.5455** | 28 min | `python -m kronllm.cli train --exp E27 --steps 4000` |
| E28 | 190,108 | 4,000 | **1.5409** | 28 min | `python -m kronllm.cli train --exp E28 --steps 4000` |
| E29 | 190,108 | 4,000 | **1.5496** | 28 min | `python -m kronllm.cli train --exp E29 --steps 4000` |
| E30 | 190,108 | 4,000 | **1.5712** | 29 min | `python -m kronllm.cli train --exp E30 --steps 4000` |
| E8r | 190,108 | 4,000 | **1.5282** | 29 min | `python -m kronllm.cli train --exp E8r --steps 4000` |
| E8w | 190,108 | 4,000 | **2.1848** | 28 min | `python -m kronllm.cli train --exp E8w --steps 4000` |
| E8w_s43 | 190,108 | 4,000 | **2.1559** | 28 min | `python -m kronllm.cli train --exp E8w_s43 --steps 4000` |
| Q2 | 2,079,992 | 10,000 | **0.8391** | 176 min | `python -m kronllm.cli train --exp Q2 --steps 10000` |
| Q4 | 2,079,992 | 4,000 | **0.9248** | 55 min | `python -m kronllm.cli train --exp Q4 --steps 4000` |
| Q6 | 2,473,208 | 4,000 | **0.8489** | 58 min | `python -m kronllm.cli train --exp Q6 --steps 4000` |
| Q6_10k | 2,473,208 | 10,000 | **0.7644** | 141 min | `python -m kronllm.cli train --exp Q6_10k --steps 10000` |
| Q8g | 2,851,280 | 5,333 | **0.8318** | 93 min | `python -m kronllm.cli train --exp Q8g --steps 5333` |
| Q8g_long | 2,851,280 | 13,333 | **0.7542** | 199 min | `python -m kronllm.cli train --exp Q8g_long --steps 13333` |
| Q8w | 2,470,328 | 5,333 | **0.8359** | 64 min | `python -m kronllm.cli train --exp Q8w --steps 5333` |
| Q9 | 2,463,776 | 5,333 | **0.8740** | 58 min | `python -m kronllm.cli train --exp Q9 --steps 5333` |
| Q9k | 2,462,468 | 5,333 | **0.8673** | 55 min | `python -m kronllm.cli train --exp Q9k --steps 5333` |
| S1 | 2,079,992 | 400 | **1.1102** | 6 min | `python -m kronllm.cli train --exp S1 --steps 400` |

The seed is `config.seed` (default 42), recorded in every
`results.json`; nothing else about a run is implicit.

## Reproducing the Q10 validation number

The training-time `val_bpb` is measured on a fixed slice; the honest
number is measured per bucket with real byte counts. Both are
reproducible offline against a downloaded checkpoint:

```bash
# per-slice bpb with exact UTF-8 byte counts (one 10M forward pass)
python scripts/eval_reference.py --source ckpt \
    --ckpt data/kaggle_ckpt/last.pt --blocks 1024 \
    --out data/baseline_q10.json

# the GPT-2 reference on exactly the same blocks
python scripts/eval_reference.py --source gpt2 --blocks 1024 \
    --out data/baseline_gpt2.json
```

Both write the byte count, the token count and every block they scored,
so the comparison can be audited rather than trusted.
