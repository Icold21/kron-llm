"""Which target positions belong to us, not to the corpus.

Two kinds of text are ours and must not be taught:

* the rebuilt corpora open every document with ``<|system|>`` + a hand-written
  instruction + ``<|eot|>``;
* a drill or chat document also carries ``<|user|>`` + a question + ``<|eot|>``
  + ``<|assistant|>``.

At inference the harness writes both prefixes and the model only has to
continue, so those tokens are never something the model must produce -- yet
they sat inside the training loss, which cost two things:

* ~1 % of the gradient went into predicting our own boilerplate (23.3 tokens of
  a 2,303-token document), and for drills the question is another 40 % of the
  document;
* the reported loss was flattered.  Cheap tokens drag the mean down, and they
  do so *more* as the model learns them, so the curve showed drops that were
  partly bookkeeping rather than progress.

Masking makes the training loss mean the same thing it meant before the
prompts existed: cross-entropy on text the model has to produce.  It is also
the same convention SFT uses -- ``sft_index.npy`` stores ``prompt_tokens`` and
the answer starts after ``<|assistant|>`` -- so the two phases optimise the
same objective on the same rendered text.

``<|system|>`` and ``<|user|>`` themselves are masked too.  One token per
document and one per turn is nothing, but the rule is then exact: the loss
covers only text the model has to produce, which on a drill is the worked
answer and on a chat document is each assistant reply.  That is precisely the
span ``prompt_tokens`` marks in SFT, so the two phases optimise the same
objective on the same rendered text -- a window in which the pretrainer trained
on the question and SFT did not would have been an argument waiting to happen.
"""
from __future__ import annotations

import torch

from .tokenizer import ASSISTANT_ID, SYSTEM_ID, TURN_EOT_ID, USER_ID


def prompt_target_mask(idx: torch.Tensor,
                       system_id: int = SYSTEM_ID,
                       eot_id: int = TURN_EOT_ID,
                       user_id: int = USER_ID,
                       assistant_id: int = ASSISTANT_ID) -> torch.Tensor:
    """Boolean mask, True where the target is our own prompt and is not trained.

    ``idx`` is the model *input*; the target at position ``t`` is ``idx[t + 1]``.
    A system prompt occupies ``idx[s .. e]`` (``s`` is ``<|system|>``, ``e`` the
    closing ``<|eot|>``), so those tokens are predicted at positions
    ``s - 1 .. e - 1`` and that is the range masked.

    For a ``<|user|>`` turn the same arithmetic runs against the ``<|assistant|>``
    marker at ``a``: the question, its ``<|eot|>`` and the marker are tokens
    ``u .. a``, predicted at positions ``u - 1 .. a - 1``, so the first *trained*
    target is the first token of the answer -- exactly the span ``prompt_tokens``
    covers in SFT.

    A window that begins inside a prompt has no marker and nothing to mask; a
    user turn with no assistant marker after it (a truncated window) is masked
    to the edge, because everything that follows is still the prompt.
    """
    mask = torch.zeros_like(idx, dtype=torch.bool)
    if idx.numel() == 0:
        return mask
    is_sys = idx == system_id
    is_user = idx == user_id
    if not bool(is_sys.any()) and not bool(is_user.any()):
        return mask
    is_eot = idx == eot_id
    is_asst = idx == assistant_id
    width = idx.shape[1]
    for b in range(idx.shape[0]):
        starts = is_sys[b].nonzero(as_tuple=False).flatten().tolist()
        if starts:
            ends = is_eot[b].nonzero(as_tuple=False).flatten().tolist()
            for s in starts:
                later = [e for e in ends if e > s]
                # No closing <|eot|> in the window: the prompt runs to its edge,
                # and the last position's target is still prompt, so mask to
                # ``width``.
                e = later[0] if later else width
                mask[b, max(0, s - 1):e] = True
        turns = is_user[b].nonzero(as_tuple=False).flatten().tolist()
        if not turns:
            continue
        answers = is_asst[b].nonzero(as_tuple=False).flatten().tolist()
        for u in turns:
            later = [a for a in answers if a > u]
            if later:
                mask[b, max(0, u - 1):later[0]] = True
            else:
                mask[b, max(0, u - 1):width] = True
    return mask

