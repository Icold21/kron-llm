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
from pathlib import Path

import pytest
import torch

from kronllm.config import EXPERIMENTS
from kronllm.loss_mask import prompt_target_mask
from kronllm.model import KronTransformer
from kronllm.testutils import make_cfg
from kronllm.tokenizer import (ASSISTANT_ID, BPETokenizer, SYSTEM_ID,
                               TURN_EOT_ID, USER_ID)

DOC = 8191          # <|endoftext|>, the document separator
VOCAB = 8192        # the reserved ids live above 8186, so the tiny vocab of
                    # make_cfg() is too small for a model-level test
TOK_PATH = Path(__file__).resolve().parents[1] / "data" / "bilingual" / "tokenizer.json"

DRILL = ("<|system|>You are Kron, a bilingual assistant.<|eot|>"
         "<|user|>Compute 605 * 8 and show the carries.<|eot|>"
         "<|assistant|>Multiply one digit at a time, starting from the units:\n"
         "units: 5 * 8 = 40 -> write 0, carry 4\n"
         "tens: 0 * 8 + 4 = 4 -> write 4\n"
         "hundreds: 6 * 8 = 48 -> write 8, carry 4\n"
         "carry: 4 -> write 4\n"
         "So 605 * 8 = 4840.")


def _ids(seq):
    return torch.tensor([seq], dtype=torch.long)


# --------------------------------------------------------------- mask geometry

def test_bare_text_is_not_masked():
    idx = _ids([10, 11, 12, 13, DOC, 14, 15])
    assert not prompt_target_mask(idx).any()


def test_the_prompt_is_masked_including_its_opening_marker():
    # <|system|> p p p <|eot|> then the document.
    idx = _ids([10, SYSTEM_ID, 20, 21, 22, TURN_EOT_ID, 30, 31, DOC])
    mask = prompt_target_mask(idx)[0].tolist()
    # Target t is idx[t + 1].  The prompt is idx[1..5], predicted at positions
    # 0..4, so <|system|> itself (the target of position 0) goes with it: the
    # model never has to emit the tag, the harness does.
    assert mask == [True, True, True, True, True, False, False, False,
                    False], mask


def test_chat_document_masks_only_the_system_turn_in_pretraining():
    """Pretraining is a text model: the question is data and stays."""
    idx = _ids([SYSTEM_ID, 20, 21, TURN_EOT_ID,
                USER_ID, 40, 41, TURN_EOT_ID, ASSISTANT_ID, 50, DOC])
    mask = prompt_target_mask(idx)[0].tolist()
    # The system turn is ours (the persona), so it is masked: positions 0..2.
    assert mask[:3] == [True, True, True], mask
    # Everything from <|user|> on is corpus text -- question, answer, both ends.
    assert not any(mask[3:]), mask


def test_sft_masks_the_whole_prompt_including_the_question():
    """SFT is an assistant: the harness writes the prompt, so only the answer."""
    idx = _ids([SYSTEM_ID, 20, 21, TURN_EOT_ID,
                USER_ID, 40, 41, TURN_EOT_ID, ASSISTANT_ID, 50, DOC])
    mask = prompt_target_mask(idx, mask_user_turns=True)[0].tolist()
    assert mask[:8] == [True] * 8, mask
    assert mask[8:] == [False, False, False], mask


def test_the_two_phases_disagree_on_user_turns_by_config():
    """Wiring mistake check: the flag must come from the experiment, not a default.

    The mask is one function with a parameter precisely because the two phases
    want opposite answers, so a config that forgets to set it silently trains the
    wrong objective -- the kind of error that costs a week of quota.
    """
    def flag(name):
        return EXPERIMENTS[name]["overrides"].get("model.mask_user_turns")

    assert not flag("Q10"), "pretraining must keep the user turn"
    assert not flag("Q10A"), "the decay pass still trains text"
    assert flag("S10") is True, "SFT must mask the whole prompt"


def test_the_model_reads_the_flag():
    """A parameter nothing passes is a parameter that does nothing."""
    x, y = _batch()
    strict = _model(**{"model.mask_user_turns": True})
    loose = _model()
    loose.load_state_dict(strict.state_dict())
    assert strict.mask_user_turns is True and loose.mask_user_turns is False


def test_the_first_answer_token_is_a_target():
    """Off by one here would train the model to repeat the question's marker."""
    idx = _ids([SYSTEM_ID, 20, TURN_EOT_ID, USER_ID, 40, TURN_EOT_ID,
                ASSISTANT_ID, 50, 51, TURN_EOT_ID, DOC])
    mask = prompt_target_mask(idx, mask_user_turns=True)[0].tolist()
    # Position 6 targets idx[7], the first answer token: must not be masked.
    assert not mask[6], mask
    # Position 5 targets idx[6], the marker: must be masked.
    assert mask[5], mask


def test_a_question_without_an_answer_marker_masks_to_the_edge():
    idx = _ids([USER_ID, 40, 41, TURN_EOT_ID])
    assert prompt_target_mask(idx, mask_user_turns=True)[0].tolist() == [True] * 4
    # ... and in pretraining the same document is entirely data.
    assert not prompt_target_mask(idx).any()


def test_a_window_starting_inside_a_question_masks_nothing():
    """No marker in the window, so there is nothing to identify as prompt."""
    idx = _ids([40, 41, TURN_EOT_ID, 50])
    assert not prompt_target_mask(idx).any()


@pytest.mark.skipif(not TOK_PATH.exists(), reason="the 8K tokenizer is not in the repo")
def test_in_pretraining_only_the_system_turn_is_masked():
    """The masked tokens decode to exactly the persona we wrote."""
    tok = BPETokenizer(TOK_PATH, vocab_size=VOCAB, bytes_per_token=1.0,
                       eot_id=DOC)
    ids = tok.encode(DRILL) + [DOC]
    mask = prompt_target_mask(_ids(ids))[0]
    masked = [ids[t + 1] for t in range(len(ids) - 1) if mask[t]]

    start = len("<|system|>")
    end = DRILL.index("<|eot|>") + len("<|eot|>")
    assert tok.decode(masked) == DRILL[start:end], tok.decode(masked)
    # the question is still a target
    trained = [ids[t + 1] for t in range(len(ids) - 1) if not mask[t]]
    assert "Compute 605 * 8" in tok.decode(trained)


@pytest.mark.skipif(not TOK_PATH.exists(), reason="the 8K tokenizer is not in the repo")
def test_in_sft_the_masked_tokens_decode_to_exactly_the_prompt():
    """The masked span must be the prompt, character for character.

    Position arithmetic is easy to get almost right: one token of the answer
    dragged into the mask, or the ``<|assistant|>`` marker left out, and every
    count-based test still passes.  This one decodes the masked targets and
    compares the text with the rendered prompt.
    """
    tok = BPETokenizer(TOK_PATH, vocab_size=VOCAB, bytes_per_token=1.0,
                       eot_id=DOC)
    ids = tok.encode(DRILL) + [DOC]
    mask = prompt_target_mask(_ids(ids), mask_user_turns=True)[0]
    masked = [ids[t + 1] for t in range(len(ids) - 1) if mask[t]]

    end = DRILL.index("<|assistant|>") + len("<|assistant|>")
    # The first token of a window is never a target -- there is nothing before it
    # to predict it from -- so the leading <|system|> is absent by construction.
    expected = DRILL[len("<|system|>"):end]
    assert tok.decode(masked) == expected, tok.decode(masked)
    # and the answer itself is untouched
    trained = [ids[t + 1] for t in range(len(ids) - 1) if not mask[t]]
    assert "Multiply one digit" in tok.decode(trained)


@pytest.mark.skipif(not TOK_PATH.exists(), reason="the 8K tokenizer is not in the repo")
def test_a_second_document_in_the_window_masks_its_system_tag_too():
    """Mid-window the tag *is* predictable, so it must be masked."""
    tok = BPETokenizer(TOK_PATH, vocab_size=VOCAB, bytes_per_token=1.0,
                       eot_id=DOC)
    one = tok.encode(DRILL) + [DOC]
    ids = one + tok.encode(DRILL) + [DOC]
    mask = prompt_target_mask(_ids(ids))[0]
    second = len(one)
    assert ids[second] == SYSTEM_ID
    # The target of position second - 1 is the second document's <|system|>.
    assert bool(mask[second - 1]), (second, mask.tolist()[:second + 2])
    # The first answer token of the second document is still a target: it is
    # predicted at the position before it.
    answer_at = second + ids[second:].index(ASSISTANT_ID) + 1
    assert not bool(mask[answer_at - 1]), (answer_at, mask.tolist())


def test_window_starting_inside_a_prompt_masks_nothing():
    # No <|system|> in the window: there is no way to know a prompt is running.
    idx = _ids([20, 21, 22, TURN_EOT_ID, 30, 31])
    assert not prompt_target_mask(idx).any()


def test_prompt_without_closing_eot_runs_to_the_end():
    idx = _ids([10, SYSTEM_ID, 20, 21, 22])
    assert prompt_target_mask(idx)[0].tolist() == [True, True, True, True, True]


def test_two_prompts_in_one_window_are_both_masked():
    idx = _ids([SYSTEM_ID, 20, TURN_EOT_ID, 30, DOC,
                SYSTEM_ID, 21, TURN_EOT_ID, 31, DOC])
    mask = prompt_target_mask(idx)[0].tolist()
    # Each prompt is masked from its own <|system|> up to its <|eot|>: the first
    # covers positions 0..1, the second 4..6, and the corpus text between them
    # stays supervised.
    assert mask[:2] == [True, True], mask
    assert not mask[2] and not mask[3]
    assert mask[4:7] == [True, True, True], mask
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
