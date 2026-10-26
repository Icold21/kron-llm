# Q10 pretraining launched on the final corpus

## Corpus (digit tokenizer v2)

| metric | value |
|---|---|
| train tokens | **2,250,807,719** (gate ≥ 2.0B — pass) |
| val tokens | 46,197,636 |
| bytes/token | 3.69 |
| EN / RU by tokens | 61.9 % / **38.1 %** (gate ≥ 30 % — pass) |
| EN / RU fertility | 2.92 / 4.96 bytes/token |
| documents | 948,976 train, 19,358 val |

The token count grew from 2.184B to 2.251B (+3.1 %) because every digit is now a
separate token — that is the price of digit-level arithmetic, and it is the only
price: Cyrillic fertility stayed at 4.96 B/t.

## Tokenizer fix

The first attempt put `Digits(individual_digits=True)` **after** ByteLevel. That
split UTF-8 sequences and Russian fell back to raw bytes: 2.72 B/t instead of
4.96, i.e. the whole Russian half of the corpus became ~2× more expensive and
semantically poorer. I caught it before publishing the bins.

The correct order is to split on `\d` **before** ByteLevel. UTF-8 stays intact,
digits are still isolated:

```
'Москва, 12 октября 2026 года.' -> ['М','о','ск','ва',',',' ','1','2',' ','октяб','ря',' ','2','0','2','6', ...]
'1234 + 5678 = 6912'            -> ['1','2','3','4',' +',' ','5','6','7','8',' =',' ','6','9','1','2']
```

Round-trip is exact on both languages.

## Identity slice

The epoch sampler shuffles 512-token windows across the whole stream, so the
identity block was never seen as one chunk — but its *share* was ~0.014 % of
tokens, which is too rare for a 10M model. I upsampled the slice ×8 (exact
copies, 1,650 docs per language, 13,200 docs per language in the mix): now
**0.111 %** of tokens (2,493,752 over the epoch), still spread uniformly because
the windows are shuffled.

## Run

| item | value |
|---|---|
| config | Q10, product composition, terms 10, carry_window 2, d 512, 25 layers |
| params | 9,938,886 |
| batch × seq | 32 × 512 = 16,384 tokens/step |
| steps per epoch | **137,377** (derived from the bin, `max_steps=0`) |
| measured throughput | 6,540 tok/s (batch 32, 15.2 GB VRAM, single T4) |
| GPU-hours per epoch | ~95.6 |
| wall clock at 30 GPU-h/week | **~3.2 weeks** |
| session budget | 500 min, checkpoint every 500 steps |

Started from scratch: no checkpoint is attached, the previous 2,000-step run was
on the broken tokenizer and is discarded.

## What happens next

1. I verify the first ~500 steps (loss, tok/s, VRAM) and the first validation bpb.
2. At the 500-minute mark the session stops itself; `last.pt` is published as a
   dataset and the next session resumes from it with the same code (v10).
3. The profile pass runs once at the start of this session: it shows where the
   ~2.5 s/step goes, which is the next place to cut Python overhead.

## Also done

Renamed the config keys that were all called `K`:

- `model.kron.terms` (10) — Kronecker terms of the weight composition
- `model.kron.carry_window` (2) — the learned-σ carry window
- `model.embed_kron.terms` (32) — terms of the factorised embedding
- `model.routed.active_rows` (10) — rows mixed per token in the routed variant
- `model.kron.qkv_terms` — extra terms on Q only (ablation E4)

Numeric behaviour is unchanged; 110 tests pass, 3 skipped.
