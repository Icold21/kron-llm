"""The prompt prefix is context, not a target.

Three things can go wrong with a loss mask and only one of them is visible:

* it masks too much -- the model silently trains on less text than we think;
* it masks the wrong positions -- off by one, so the prompt's last token is
  still trained and the first document token is not;
* the two loss paths disagree -- the chunked path is what Q10 actually uses
  (``loss_chunk: 1024``), so a mask threaded only through the unchunked branch
  would pass every unit test here and do nothing in the run.

The equivalence test is the one that catches the third.
"""
import torch

from kronllm.loss_mask import prompt_target_mask
from kronllm.model import KronTransformer
from kronllm.testutils import make_cfg
from kronllm.tokenizer import SYSTEM_ID, TURN_EOT_ID

DOC = 8191          # <|endoftext|>, the document separator
VOCAB = 8192        # the reserved ids live above 8186, so the tiny vocab of
                    # make_cfg() is too small for a model-level test


def _ids(seq):
    return torch.tensor([seq], dtype=torch.long)


# --------------------------------------------------------------- mask geometry

def test_bare_text_is_not_masked():
    idx = _ids([10, 11, 12, 13, DOC, 14, 15])
    assert not prompt_target_mask(idx).any()


def test_prompt_is_masked_and_system_token_is_not():
    # <|system|> p p p <|eot|> then the document.
    idx = _ids([10, SYSTEM_ID, 20, 21, 22, TURN_EOT_ID, 30, 31, DOC])
    mask = prompt_target_mask(idx)[0].tolist()
    # Target t is idx[t + 1], so the prompt text at idx[2..5] is predicted at
    # positions 1..4.
    assert mask == [False, True, True, True, True, False, False, False,
                    False], mask
    # <|system|> itself is context for the first prompt token and stays
    # supervised: it is the document boundary.
    assert not mask[0]


def test_chat_document_masks_only_the_system_turn():
    idx = _ids([SYSTEM_ID, 20, 21, TURN_EOT_ID, 40, 41, TURN_EOT_ID, 50, DOC])
    mask = prompt_target_mask(idx)[0].tolist()
    # The system turn is predicted at 0..2 (its <|eot|> at index 3 is the target
    # of position 2); the user turn at 3.. is conversation and stays supervised.
    assert mask[:3] == [True, True, True], mask
    assert not any(mask[3:]), mask


def test_window_starting_inside_a_prompt_masks_nothing():
    # No <|system|> in the window: there is no way to know a prompt is running.
    idx = _ids([20, 21, 22, TURN_EOT_ID, 30, 31])
    assert not prompt_target_mask(idx).any()


def test_prompt_without_closing_eot_runs_to_the_end():
    idx = _ids([10, SYSTEM_ID, 20, 21, 22])
    assert prompt_target_mask(idx)[0].tolist() == [False, True, True, True, True]


def test_two_prompts_in_one_window_are_both_masked():
    idx = _ids([SYSTEM_ID, 20, TURN_EOT_ID, 30, DOC,
                SYSTEM_ID, 21, TURN_EOT_ID, 31, DOC])
    mask = prompt_target_mask(idx)[0].tolist()
    # Each prompt is predicted up to (not including) the position of its <|eot|>.
    assert mask[:2] == [True, True], mask
    assert not mask[2] and not mask[3] and not mask[4]
    assert mask[5:7] == [True, True], mask
    assert not mask[7] and not mask[8] and not mask[9]


# ----------------------------------------------------------------- the model

def _model(**over):
    cfg = make_cfg(**{"data.vocab_size": VOCAB, "model.d_model": 64,
                      "model.loss_chunk": 4, **over})
    torch.manual_seed(0)
    return KronTransformer(cfg)


def _batch():
    """One bare row and one prompted row; targets shifted by one."""
    x = torch.tensor([
        [10, 11, 12, 13, 14, 15, 16, 17],
        [10, SYSTEM_ID, 20, 21, 22, TURN_EOT_ID, 30, 31],
    ], dtype=torch.long)
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = DOC
    return x, y


def test_chunked_and_unchunked_losses_agree_with_a_mask():
    """The path Q10 uses must equal the path a reader would test."""
    x, y = _batch()
    plain = _model(**{"model.loss_chunk": 0})
    chunked = _model()
    chunked.load_state_dict(plain.state_dict())
    with torch.no_grad():
        a = plain(x, y)["loss"]
        b = chunked(x, y)["loss"]
    assert torch.allclose(a, b, atol=1e-5), (a.item(), b.item())


def test_disabling_the_flag_restores_the_unmasked_loss():
    """The flag must switch behaviour, not merely exist."""
    x, y = _batch()
    on = _model()
    off = _model(**{"model.ignore_prompt_targets": False})
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        masked = on(x, y)["loss"]
        unmasked = off(x, y)["loss"]
    assert not torch.allclose(masked, unmasked), "the mask changed nothing"


def test_bare_batch_gives_identical_loss_either_way():
    """On the held-out bare bin the flag is a no-op, so val_bpb stays
    comparable with every earlier session."""
    x = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17],
                      [20, 21, 22, 23, 24, 25, 26, 27]], dtype=torch.long)
    y = torch.roll(x, -1, dims=1)
    on = _model()
    off = _model(**{"model.ignore_prompt_targets": False})
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        a = on(x, y)["loss"]
        b = off(x, y)["loss"]
    assert torch.allclose(a, b, atol=1e-6), (a.item(), b.item())


def test_a_masked_target_cannot_change_the_loss():
    """Perturbing a token inside the prompt must leave the loss untouched."""
    x, y = _batch()
    m = _model()
    with torch.no_grad():
        a = m(x, y)["loss"]
        y2 = y.clone()
        y2[1, 2] = (y2[1, 2] + 7) % VOCAB      # inside the prompt, masked
        b = m(x, y2)["loss"]
        # ... while perturbing a document token must move it.
        y3 = y.clone()
        y3[1, 6] = (y3[1, 6] + 7) % VOCAB      # document text, supervised
        c = m(x, y3)["loss"]
    assert torch.allclose(a, b), (a.item(), b.item())
    assert not torch.allclose(a, c), "a supervised target did not move the loss"


def test_masking_keeps_gradients_finite():
    x, y = _batch()
    m = _model()
    m.train()
    out = m(x, y)
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no gradients reached the parameters"
    assert all(torch.isfinite(g).all() for g in grads)
