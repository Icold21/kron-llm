# Repetition probe: how many epochs does this model actually want?

Status: designed, not run (2026-11-15). Cost ~6.7 h of GPU, fits in one evening.

## Why

The SFT curve is flattening on the metric that matters (SFT bpb 0.9773 -> 0.8310
-> 0.8084 for 4.5k -> 12.2k steps) while the only axis still moving is the bare
form (3 % -> 12 % -> 17 % at n=200), and that one moves with the number of
*distinct* examples, not with repetition.

A second full epoch over the 504.6M corpus costs 57,642 steps (about 32 h, more
than a week's quota) and would mostly re-read what the model has seen.  The
question the article needs answered is not "what happens after two epochs" but
**"where does repetition stop buying skill and start buying memorisation"**, and
that shape does not need the full corpus to be measured: on a small slice the
same thing happens in hours.

## Design

* Slice: **50M tokens** of the SFT corpus (3,052 steps per epoch at 16,384
  tokens/step), drawn the same way `scripts/subset_mix.py` draws the decay mix.
* Split the slice **before** training: 90 % train / 10 % held out from the same
  sources.  The held-out part is the "unseen" side of every comparison.
* Also set aside **500 trained examples** by hash.  They are in the training
  stream and are asked back verbatim at every checkpoint -- that is the
  memorisation side.
* Start from the S10b checkpoint (step 12,198), LR 1e-5 -> 1e-6 over one epoch,
  restarted each epoch (four short anneals, one per epoch), checkpoints every 250
  steps.
* **Four epochs**: 12,200 steps, 6.7 h at the measured 1.99 s/step.

## What is measured at each checkpoint

| metric | what it says |
|---|---|
| bpb on the slice's **held-out** 10 % | generalisation: is the third epoch still teaching anything |
| bpb on the **trained** 500 examples | memorisation: how far to zero it goes |
| exact reproduction of those 500 answers | memorisation in the form a reader understands: "it emits the answer it saw" |
| drills (28 items) | is the procedure still intact, or is it being overwritten |
| bare form, n=200, Wilson CI | does repetition move the one axis that was moving |

The gap between the first two curves is the answer, and it is a figure the
article can print: *capability flat from epoch 2, memorisation still falling*.

## What would make this probe wrong

* If the slice's held-out set is not from the same source mix as its training
  part, the "generalisation" curve measures domain shift, not repetition.
* If the LR schedule is restarted identically each epoch, the drops at epoch
  boundaries are schedule artefacts.  Run the four epochs as one continuous
  schedule and place the epoch marks on the x axis instead.
* 50M tokens is 10 % of the corpus: the absolute numbers will not match the
  full-corpus run, only the shape should be read.
