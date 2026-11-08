"""The bare form exists in both phases, and its answer is computed, not templated.

The probe that scored 1/60 feeds ``67 + 75 = `` with no chat frame at all.  The
corpus had 8,192 bare equations and not one of them as a question, so the model
was never asked the short form.  These tests pin the two additions: a family that
renders question -> number for SFT, and plain equation lines for pretraining.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def arith():
    spec = importlib.util.spec_from_file_location(
        "gen_arithmetic_sft_test", ROOT / "scripts" / "gen_arithmetic_sft.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_arithmetic_sft_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bare_family_is_registered_for_both_languages(arith):
    for lang in ("en", "ru"):
        names = [f.__name__ for f, _ in arith.FAMILIES[lang][0]]
        assert "en_bare" in names, lang


def test_bare_case_has_no_chain_of_thought(arith):
    rng = random.Random(1)
    for _ in range(50):
        question, cot, answer = arith.en_bare(rng)
        assert cot == "", "a direct answer must not carry a procedure"
        assert arith.is_direct(cot) is True
        assert answer.lstrip("-").isdigit()


def test_bare_answer_is_the_arithmetic_result(arith):
    rng = random.Random(2)
    for _ in range(500):
        question, _, answer = arith.en_bare(rng)
        left, op, right = question.rstrip("=").split()
        a, b = int(left), int(right)
        want = a + b if op == "+" else (a - b if op == "-" else a * b)
        assert int(answer) == want, f"{question} -> {answer}, want {want}"


def test_bare_covers_three_digit_and_multiplication(arith):
    rng = random.Random(3)
    seen = {"3digit": False, "mul": False, "two": False}
    for _ in range(2000):
        question, _, _ = arith.en_bare(rng)
        left, op, right = question.rstrip("=").split()
        if op == "*":
            seen["mul"] = True
        elif len(left) >= 3 or len(right) >= 3:
            seen["3digit"] = True
        else:
            seen["two"] = True
    assert all(seen.values()), seen


def test_render_of_a_bare_case_ends_with_the_number(arith):
    spec = importlib.util.spec_from_file_location(
        "build_sft_render_test", ROOT / "scripts" / "build_sft.py")
    build_sft = importlib.util.module_from_spec(spec)
    sys.modules["build_sft_render_test"] = build_sft
    spec.loader.exec_module(build_sft)

    rng = random.Random(4)
    question, cot, answer = arith.en_bare(rng)
    text, prompt = build_sft.render([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer if not cot else "x"},
    ])
    assert prompt.endswith("<|assistant|>")
    assert f"<|user|>{question}<|eot|>" in prompt
    # the assistant turn is the number and nothing else
    body = text.split("<|assistant|>", 1)[1]
    assert body == f"{answer}<|eot|>", body
