"""Which target positions belong to us, not to the corpus.

Two kinds of text are ours and must not be taught:

* the rebuilt corpora may open every document with ``<|system|>`` + a
  hand-written instruction + ``<|eot|>``;
* a drill or chat document also carries ``<|user|>`` + a question + ``<|eot|>``
  + ``<|assistant|>``.

Which of those belongs in the loss depends on the phase, and the difference is
deliberate:

* **pretraining** trains a text model.  The system turn is ours (we wrote the
  persona), so it is masked; the user turn -- a question or a human reply -- is
  data, and it stays.  Learning to write questions is not a defect: it is what a
  text model does with a corpus, and it is why the drills carry questions at all.
* **SFT** trains an assistant whose prompt is supplied by the harness, so
  everything up to ``<|assistant|>`` is context: ``prompt_tokens`` in
  ``sft_index.npy`` marks exactly that span.

Masking the system turn makes the training loss mean the same thing it meant
before the prompts existed: cross-entropy on text the model has to produce.
``<|system|>`` itself is masked rather than left as a boundary token, because one
token per document of gradient is not worth a rule that differs between the two
phases.
"""
from __future__ import annotations

import torch

from .tokenizer import ASSISTANT_ID, SYSTEM_ID, TURN_EOT_ID, USER_ID


def prompt_target_mask(idx: torch.Tensor,
                       system_id: int = SYSTEM_ID,
                       eot_id: int = TURN_EOT_ID,
                       user_id: int = USER_ID,
                       assistant_id: int = ASSISTANT_ID,
                       mask_user_turns: bool = False) -> torch.Tensor:
    """Boolean mask, True where the target is our own prompt and is not trained.

    ``idx`` is the model *input*; the target at position ``t`` is ``idx[t + 1]``.
    A system prompt occupies ``idx[s .. e]`` (``s`` is ``<|system|>``, ``e`` the
    closing ``<|eot|>``), so those tokens are predicted at positions
    ``s - 1 .. e - 1`` and that is the range masked.  The system turn is always
    masked: it is the persona *we* write, not corpus text.

    ``mask_user_turns`` decides what happens to ``<|user|>`` turns, and the two
    phases want different answers:

    * **pretraining** (default, ``False``): a user turn is data.  The run is a
      text model -- at inference it continues a document -- so the question or
      the human reply is exactly the kind of text it should be able to produce.
      Masking it there threw away 15 % of every drill and half of every chat
      document for no benefit.
    * **SFT** (``True``): the prompt is what the harness supplies, so everything
      up to and including ``<|assistant|>`` is context and the answer is the only
      target -- the span ``prompt_tokens`` records in ``sft_index.npy``.

    A window that begins inside a prompt has no marker and nothing to mask; a
    user turn with no assistant marker after it (a truncated window) is masked
    to the edge when ``mask_user_turns`` is on.
    """
    mask = torch.zeros_like(idx, dtype=torch.bool)
    if idx.numel() == 0:
        return mask
    is_sys = idx == system_id
    is_user = idx == user_id
    if not bool(is_sys.any()) and not (mask_user_turns and bool(is_user.any())):
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
        if not mask_user_turns:
            continue
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


