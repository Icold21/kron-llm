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

## In progress

**E14 (sum composition):** `P = sI + Σ α_l D_l`, learnable `α_l`, full
backprop, 190,348 parameters, 11.5 MFLOP/token. Running for 4,000 steps to
match E8's best checkpoint. Live state is in `runs/E14/train.log` and
`current_progress.md`; the final result is not in yet.

The sum path is implemented in `KronCompose`, `KronBank`, the block config and
the diagnostics. The CPU suite passes (66 passed, 2 skipped).

## Next experiments

1. **Q4: Q2 + full backprop.** The cheapest test of the only clear E8 win at
   quality scale; 2.08M params, 4,000 steps, compare against Q2's 0.962 bpb at
   step 4,000.
2. **Q5: scale toward 8-10M at fixed activation memory.** Start with Q2's
   `d=320, N=24, batch=32, seq=192` geometry and raise `K` (for example
   `K=24`, about 7.75M parameters). This tests whether more Kronecker summands
   buy quality while staying inside 6 GB.
3. **Only if E14 is competitive:** a sum-composition run at Q scale. The sum
   costs `L·d²` per token instead of an amortised product, so it must justify
   its FLOPs.
4. **Orthogonal anchor** remains a separate, higher-risk rank experiment; it is
   not part of the current shipped path.

## Constraints

- Keep the product as the default until a trained ablation beats it.
- Report bpb at `T=1.0`, `top_k=5-10` for generation checks.
- Keep only a few best artifacts; no near-duplicate checkpoints.
- Every result goes into `current_progress.md`; this file is the short overall
  research snapshot.
