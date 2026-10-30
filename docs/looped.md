# Looped transformers and mHC: what we measured

Status: experiment done 2026-10-30 (night).  Code: `kronllm/model.py`
(`MHCRouter`, `plan`/`pass_index`, `loop_start`/`loop_end`,
`KronBlock.forward_mhc`).  Tests: `tests/test_loop.py`, `tests/test_mhc.py`.
Runs: `runs/L1`…`runs/L4`, configs `L1`–`L4` in `kronllm/config.py`.

## 1. Why this was worth a night

SenseNova released SenseNova-Loop-9B-Preview on 2026-10-30: 32 physical
layers, the middle band `[4,28)` executed twice (56 executions), a KV cache per
*execution*, and **mHC** — two residual lanes inside the loop, mixed by a
matrix projected onto the doubly stochastic polytope with Sinkhorn iterations.
It reports ~9.66B stored parameters against ~14.6B compute-equivalent.

Two things made this relevant rather than interesting:

* We already had the looping half and had measured it.  `recurse` reuses
  `n_unique_blocks` blocks, `pass_bias` is the Universal-Transformer step
  signal, and E19–E22 in `docs/results.md` §12 are the receipts.
* Our own E19 lost 16 % bpb.  If mHC fixes *that*, it is worth knowing before
  the next pretraining run is designed.

## 2. The measurement that was missing

E19/E20/E22 compared arms at **matched tokens**.  For a project whose binding
constraint is GPU-hours, that is the wrong axis: the question is what a fixed
compute budget should buy.  L1–L4 hold `layer-executions × tokens` constant at
E8's budget (12 × 131M layer-token units) and differ only in how it is spent.

All four use batch 64, not E8's 128: a 20-execution arm with two residual lanes
does not fit in the 6 GB card at batch 128 — the 256-token attention scores
alone are 134 MB per execution slot.  Peak measured 4.63 GB.

| arm | executions | steps | tokens | params | val bpb | wall clock |
|---|---:|---:|---:|---:|---:|---:|
| **L1** tokens, E8 depth | 12 | 8 000 | 131M | 190 108 | **1.395** | 35 min |
| **L3** middle-band loop | 20 | 4 800 | 78.6M | 190 108 | 1.510 | 33 min |
| **L2** whole-stack loop | 24 | 4 000 | 65.5M | 190 108 | 1.563 | 33 min |
| **L4** L3 + mHC | 20 | 4 800 | 78.6M | 206 556 | see §5 | — |

The three finished arms cost the same wall clock (35/33/33 min), so the
matching is real and not a paper exercise.

## 3. What the numbers say

**At a fixed budget, data beats depth.**  L1 spends everything on tokens and
wins by 0.115 bpb over the best depth arm — 8.2 %.  This is the direction the
old matched-tokens comparison obscured: E20 gained 3.5 % over E8 *while being
given twice the compute*, and halving the tokens costs more than doubling the
depth buys.

Cross-check: L2 is E20's architecture at half the tokens (66M vs 131M) and
scores 1.563 against E20's 1.427.  Half the data costs 0.136 bpb; doubling
depth at equal tokens bought 0.052.  Data is worth about 2.6× depth per unit of
compute at this scale.

**Within the looped family, SenseNova's shape is right.**  L3 (middle band)
beats L2 (whole stack) by 3.4 % at equal budget.  The head turns tokens into
features and the tail turns features into logits; neither is iterative work, and
sharing them forces one block to be both.  That part of the design transfers.

**The conclusion for us:** do not loop.  Q10 is a plain 25-layer stack
(`recurse=1`, asserted by `test_q10_is_untouched`) and stays that way.  The
config knobs remain, tested, for the one case where loops make sense — trading
compute for *memory* on-device, which is not our constraint.

## 4. What mHC cost, and the two bugs it exposed

The first L4 run took **1.194 s/step** against L3's 0.417 — a 2.9× slowdown for
8.6 % more parameters, which makes the arm incomparable (at matched time it
would reach ~1 675 of its 4 800 steps).  Two causes, both ours:

1. `read()` and `merge()` each called `_route(lanes)` on the **same** lanes, so
   every router was computed twice per block.  Fixed by computing the routing
   once and passing it to both.
2. Twenty Sinkhorn iterations on a 2×2 matrix is forty kernel launches per
   router per step.  A 2×2 doubly stochastic matrix has one degree of freedom
   and the limit is available in closed form,
   `x = 1 / (1 + sqrt(bc/ad))` — exact, no iteration.  `sinkhorn2` does that;
   `sinkhorn` stays for four lanes.

After both fixes: **0.609 s/step**, a 2× improvement, and mHC is now cheaper
than the block it routes.

A third finding, from re-deriving the constraint: twenty Sinkhorn iterations
leave the row sums off by up to 2.5 % at two lanes and 4 % at four, and
`1.025**48` over a 24-layer double pass is a 3× norm leak — exactly what the
projection exists to prevent.  `sinkhorn` therefore ends with a row
normalisation: every output lane becomes an exact convex combination of the
input lanes, and the column sums stay within ~2 % of one.  Boundedness is the
property the loop needs; preserving the sum of the lanes is the part that can
afford to be approximate.

## 5. L4 (mHC) — result

**1.4779** against L3's 1.5102: mHC is worth **0.032 bpb, 2.1 %** at matched
steps, the same budget and the same parameter count (plus 8.6 % parameters for
the routers).  The gap opens at step 1000 and widens monotonically:

| step | L3 no mHC | L4 mHC | Δ |
|---:|---:|---:|---:|
| 1000 | 2.0124 | 1.9899 | −0.022 |
| 2000 | 1.7014 | 1.6747 | −0.027 |
| 3000 | 1.5878 | 1.5551 | −0.033 |
| 4800 | 1.5102 | **1.4779** | −0.032 |

So the mechanism is not decoration: it recovers about two thirds of the 3.4 %
the middle band already bought over the whole-stack loop.  It is also still
0.083 bpb behind L1, which spent the same budget on tokens.

The honest reading is therefore narrower than "mHC does not work" and wider
than "loops are back": **at 190k parameters a fixed compute budget is better
spent on data, and mHC makes the looped family better without making it
competitive.**  The next 10M-parameter run stays a plain stack; mHC is kept,
tested, and documented for a larger model where parameters bind instead of
tokens.

## 6. What we would test next, if anyone revisits this

* A larger parameter budget.  At 190k parameters and a 131M-token budget the
  model is data-limited; the depth trade could invert when parameters bind.
* `recurse_random` (E21) is the only variant with a second use: one model at
  several depths, i.e. adaptive compute at inference.  That is a product
  feature, not a quality one.
* Nothing about mHC at this scale.  The mechanism is implemented, tested, and
  costs 8.6 % parameters for a comparison that data already wins.
