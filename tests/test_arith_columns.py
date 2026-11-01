"""Column drills: the printed steps must reconstruct the answer.

The point of these families is the algorithm (carry, borrow, partial product)
written out digit by digit, so the test parses the *rendered text* back into a
number and compares it with Python's own arithmetic.  A tautological check would
be to call the same helper twice; this one reads what the model will read.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gen_arithmetic_sft as g  # noqa: E402

WRITTEN = re.compile(r"(?:write|пишем)\s+(\d+)")


def rebuild(rows: list[str]) -> int:
    """Read the 'write N' values back, least-significant row first."""
    value = 0
    for pos, row in enumerate(rows):
        m = WRITTEN.search(row)
        assert m, f"row without a written digit: {row}"
        value += int(m.group(1)) * 10 ** pos
    return value


def test_addition_rows_reconstruct_the_sum():
    rng = random.Random(1)
    for _ in range(400):
        a, b = rng.randint(120, 989), rng.randint(120, 989)
        rows, value = g.add_columns(a, b, g.PLACE_EN)
        assert value == a + b
        assert rebuild(rows) == a + b


def test_subtraction_rows_reconstruct_the_difference():
    rng = random.Random(2)
    number = re.compile(r"=\s*(\d+)(?:\s*,|\s*$)")
    for _ in range(400):
        a = rng.randint(320, 989)
        b = rng.randint(120, a - 100)
        rows, value = g.sub_columns(a, b, g.PLACE_EN)
        assert value == a - b
        digits = []
        for row in rows:
            m = number.search(row)
            assert m, f"row without a result: {row}"
            digits.append(int(m.group(1)))
        assert sum(d * 10 ** i for i, d in enumerate(digits)) == a - b


def test_multiplication_rows_reconstruct_the_product():
    rng = random.Random(3)
    for _ in range(400):
        a, b = rng.randint(120, 989), rng.randint(3, 9)
        rows, value = g.mul_columns(a, b, g.PLACE_EN)
        assert value == a * b
        assert rebuild(rows) == a * b


def test_every_drill_exercises_a_carry_or_a_borrow():
    rng = random.Random(4)
    # the families retry until the algorithm is actually needed, so the text
    # always shows the rule: a drill without a carry teaches nothing
    for _ in range(100):
        _q, cot, _a = g.en_add_columns(rng)
        assert "carry 1" in cot, cot
        _q, cot, _a = g.ru_add_columns(rng)
        assert "запоминаем 1" in cot, cot
        _q, cot, _a = g.en_sub_columns(rng)
        assert "borrow 1" in cot, cot
        _q, cot, _a = g.ru_sub_columns(rng)
        assert "заём 1" in cot, cot


def test_russian_rows_are_not_english():
    rng = random.Random(5)
    for family in (g.ru_add_columns, g.ru_sub_columns, g.ru_mul_columns):
        q, cot, answer = family(rng)
        assert not re.search(r"\b(write|carry|borrow|units|tens|hundreds)\b", cot), cot
        assert re.search(r"(пишем|запоминаем|занимаем|заём)", cot), cot
        assert answer.isdigit()


def test_the_answer_matches_the_last_sentence():
    rng = random.Random(6)
    for family in (g.en_add_columns, g.en_sub_columns, g.en_mul_columns,
                   g.ru_add_columns, g.ru_sub_columns, g.ru_mul_columns):
        for _ in range(20):
            _q, cot, answer = family(rng)
            assert cot.rstrip().endswith(answer + "."), (family.__name__, cot)
