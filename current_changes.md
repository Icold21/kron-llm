# Current research state

Last updated: 2026-10-23

## Goal

KronLLM is a compact Kronecker-parameterised language model for weak consumer
GPUs. The practical target is a **6 GB VRAM** training/inference budget and a
soft ceiling of **about 10M parameters**. Quality is measured in bits per byte
so that byte-level and BPE models can be compared.

## Current reference

**Q2 remains the shipped quality model:** 2,079,992 parameters, val bpb
**0.839**, PPL **5.95**, 8.55 MB fp32 checkpoint, peak **4.22 GB** VRAM.
Q3 is experimental; the warm-started Q3 is not evidence of from-scratch
convergence.

## Completed evidence

- **Q1:** 926,468 params, bpb **0.976**.
- **D1:** dense control on the same BPE corpus, bpb **0.9770**; Q1 is 0.13 %
  better with 15.3 % fewer parameters.
- **E8 (full backprop):** 190,108 params, best bpb **1.479** at step 4,000;
  at the matched 2,500-step point **1.586** vs E2 **1.659**. Quality improves,
  but `eff_rank(P) = 60.3` vs `eff_rank(sum) = 101.2`; the product still
  compresses rank and H3 remains unconfirmed.
- **Q3long (from scratch):** stopped at 2,000 steps at bpb **1.277**, with no
  phase transition. Q3warm reached **0.8236** but is an initialisation test.
- **E14 (sum composition):** 190,348 params, bpb **1.551** at step 4,000 vs
  E8's **1.479**. `eff_rank(P)` rises from 60.3 to **97.7**, but quality is
  worse and `α_l` stays near 1.0 (mean 0.993, std 0.044). Removing rank
  compression does not by itself improve quality; the product remains the
  default. E14 is a negative result, not a shipped model.

## In progress

**Q4 (Q2 + full backprop):** 2,079,992 params, 4,000 steps. It tests the E8
win at quality scale against Q2's **0.962 bpb** at step 4,000. Live state is in
`runs/Q4/train.log` and `current_progress.md`.

The sum path remains implemented in `KronCompose`, `KronBank`, the block config
and the diagnostics. The CPU suite passes (66 passed, 2 skipped).

## Next experiments

1. **Q5: scale toward 8-10M at fixed activation memory.** Start with Q2's
   `d=320, N=24, batch=32, seq=192` geometry and raise `K` (for example
   `K=24`, about 7.75M parameters). This tests whether more Kronecker summands
   buy quality while staying inside 6 GB.
2. **If Q4 wins:** use full backprop in Q5 and compare against Q2 at matched
   steps. If Q4 loses, keep the stop-gradient in the scaled run.
3. **Orthogonal anchor** remains a separate, higher-risk rank experiment; it is
   not part of the current shipped path. The E14 result lowers its priority:
   rank compression alone is not the quality bottleneck.

## Constraints

- Keep the product as the default until a trained ablation beats it.
- Report bpb at `T=1.0`, `top_k=5-10` for generation checks.
- Keep only a few best artifacts; no near-duplicate checkpoints.
- Every result goes into `current_progress.md`; this file is the short overall
  research snapshot.
