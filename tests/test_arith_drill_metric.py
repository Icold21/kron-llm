"""The drill evaluator's two dead metrics: both read 0/60 no matter what.

Found on 2026-11-08 while explaining the drill format: the checkpoint produced
"So 583 + 693 = 1276. Answer: 1276" and the evaluator still scored it wrong,
because

* ``first_row`` compared the first ``=`` value with the final answer -- for a
  column drill those are the units digit and the whole number, so it could never
  match;
* ``answer_line`` compared ``int(matched) == answer`` while the answer families
  return the answer as a formatted string, so it could never match either.

Between them, every arithmetic number we ever quoted from the drill format was
zero by construction.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_arith_drill.py"


@pytest.fixture(scope="module")
def drill():
    spec = importlib.util.spec_from_file_location("eval_arith_drill_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_answer_value_takes_the_formatted_string(drill):
    assert drill.answer_value("1276") == 1276
    assert drill.answer_value("1,276") == 1276
    assert drill.answer_value(1276) == 1276
    assert drill.answer_value(" -7 ") == -7


@pytest.mark.parametrize("text,ok", [
    ("units: 3 + 3 = 6 -> write 6", True),
    ("units: 3 + 3 = 5 -> write 5", False),          # the failure mode we chase
    ("units: 1 - 0 is less than 4 -> borrow: 11 - 4 = 7, borrow 1", True),
    ("units: 3 * 7 = 21 -> write 1, carry 2", True),
    ("units: 4 * 7 + 2 = 27 -> write 7, carry 2", False),  # 4*7+2 = 30, not 27
    ("no equation at all", False),
])
def test_first_row_checks_the_arithmetic_not_the_answer(drill, text, ok):
    assert drill.first_row_ok(text) is ok


def test_first_row_ignores_the_units_digit_is_not_the_answer(drill):
    # The old metric: the first "=" is 6, the answer is 1276 -> "wrong".  The
    # row itself is correct, and that is what this reports.
    text = "units: 3 + 3 = 6 -> write 6\ntens: 8 + 9 = 17 -> write 7, carry 1"
    assert drill.first_row_ok(text) is True
