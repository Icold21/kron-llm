"""Which target positions belong to us, not to the corpus.

The rebuilt corpora open every document with ``<|system|>`` + a hand-written
instruction + ``<|eot|>``.  At inference the harness writes that prefix and the
model only has to continue, so those tokens are never something the model must
produce -- yet they sat inside the training loss, which cost two things:

* ~1 % of the gradient went into predicting our own boilerplate.  The prompt is
  23.3 tokens of a 2,303-token document, so it is small, but it is pure waste:
  the same 108 strings, repeated, learned as if they were text.
* the reported loss was flattered.  A cheap 1 % of tokens drags the mean down,
  and it does so *more* as the model learns the tag, so the curve showed a drop
  at the corpus switch that was partly bookkeeping rather than progress.

Masking makes the training loss mean the same thing it meant before the prompts
existed: cross-entropy on corpus text.  That is what makes the two phases of the
run comparable, and it is the same convention SFT uses (loss on the answer, not
on the prompt).

``<|system|>`` itself stays supervised on purpose.  It is the document boundary
-- the model does have to know where a document starts, because it has to stop
somewhere -- and it is one token per document.
"""
from __future__ import annotations

import torch

from .tokenizer import SYSTEM_ID, TURN_EOT_ID


def prompt_target_mask(idx: torch.Tensor,
                       system_id: int = SYSTEM_ID,
                       eot_id: int = TURN_EOT_ID) -> torch.Tensor:
    """Boolean mask, True where the target is our own prompt and is not trained.

    ``idx`` is the model *input*; the target at position ``t`` is ``idx[t + 1]``.
    The prompt text occupies ``idx[s + 1 .. e]`` (``s`` is ``<|system|>``, ``e``
    the closing ``<|eot|>``), so it is predicted at positions ``s .. e - 1`` and
    that is the range masked.  ``<|system|>`` at ``s`` is the last *context*
    token of the preceding document and stays supervised.

    A window that begins inside a prompt has no marker and nothing to mask; a
    chat document's first ``<|eot|>`` closes its system turn, and any later turn
    ends in the window are left alone -- they are conversation, not boilerplate.
    """
    mask = torch.zeros_like(idx, dtype=torch.bool)
    if idx.numel() == 0:
        return mask
    is_sys = idx == system_id
    if not bool(is_sys.any()):
        return mask
    is_eot = idx == eot_id
    width = idx.shape[1]
    for b in range(idx.shape[0]):
        starts = is_sys[b].nonzero(as_tuple=False).flatten().tolist()
        if not starts:
            continue
        ends = is_eot[b].nonzero(as_tuple=False).flatten().tolist()
        for s in starts:
            later = [e for e in ends if e > s]
            # No closing <|eot|> in the window: the prompt runs to its edge, and
            # the last position's target is still prompt, so mask to ``width``.
            e = later[0] if later else width
            mask[b, s:e] = True
    return mask
