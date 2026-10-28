"""Generate synthetic arithmetic SFT data in English and Russian.

Arithmetic is the one part of the math slice that can be made *exactly* right:
the question is built from a template, the answer is computed by Python, and the
solution is the computation written out.  No teacher model, no verification
gamble, no licence question (CC0).

The generators are deliberately boring and varied in wording: twelve families,
each with many phrasings and random numbers, so the model learns the *pattern*
("read the numbers, do the steps, state the answer") rather than a fixed
sentence.  Division is generated only when it is exact, and every answer is
checked against the printed steps.

Usage::

    python scripts/gen_arithmetic_sft.py --per-lang 200000 --out E:/kron_sft/raw
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

Case = Tuple[str, str, str]      # question, chain-of-thought, answer


# --------------------------------------------------------------------------
# English templates
# --------------------------------------------------------------------------


def en_add(rng: random.Random) -> Case:
    a, b = rng.randint(2, 999), rng.randint(2, 999)
    q = rng.choice([
        f"What is {a} + {b}?",
        f"Compute {a} plus {b}.",
        f"Add {a} and {b}.",
        f"Calculate the sum of {a} and {b}.",
        f"Find {a} + {b}.",
    ])
    cot = (f"Write the addition in columns:\n"
           f"{a} + {b} = {a + b}\n\n"
           f"Check: {a + b} - {b} = {a}, so the sum is consistent.")
    return q, cot, str(a + b)


def en_sub(rng: random.Random) -> Case:
    a = rng.randint(20, 1000)
    b = rng.randint(1, a - 1)
    q = rng.choice([f"What is {a} - {b}?", f"Subtract {b} from {a}.",
                    f"Compute {a} minus {b}.", f"Find {a} - {b}."])
    cot = (f"{a} - {b} = {a - b}\n\n"
           f"Check: {a - b} + {b} = {a}.")
    return q, cot, str(a - b)


def en_mul(rng: random.Random) -> Case:
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    if rng.random() < 0.5:
        cot = (f"{a} * {b} = {a * b}\n\n"
               f"Break it up: {a} * {b} = {a} * {b // 10 * 10} + {a} * {b % 10}"
               f" = {a * (b // 10 * 10)} + {a * (b % 10)} = {a * b}.")
    else:
        cot = (f"{a} * {b} = {a * b}\n\n"
               f"Repeated addition: {b} copies of {a} is {a * b}.")
    q = rng.choice([f"What is {a} * {b}?", f"Multiply {a} by {b}.",
                    f"Compute {a} times {b}.", f"Find the product of {a} and {b}."])
    return q, cot, str(a * b)


def en_div(rng: random.Random) -> Case:
    b = rng.randint(2, 40)
    ans = rng.randint(2, 200)
    a = b * ans
    q = rng.choice([f"What is {a} / {b}?", f"Divide {a} by {b}.",
                    f"Compute {a} divided by {b}.",
                    f"How many times does {b} fit into {a}?"])
    cot = (f"{a} / {b} = {ans}\n\n"
           f"Check: {ans} * {b} = {a}, so the quotient is exact.")
    return q, cot, str(ans)


def en_mul_add(rng: random.Random) -> Case:
    a, b, c = rng.randint(2, 30), rng.randint(2, 30), rng.randint(1, 200)
    q = rng.choice([f"What is {a} * {b} + {c}?",
                    f"Compute {a} times {b}, then add {c}.",
                    f"Evaluate {a} * {b} + {c}."])
    cot = (f"Order of operations: multiplication first.\n"
           f"{a} * {b} = {a * b}\n"
           f"{a * b} + {c} = {a * b + c}")
    return q, cot, str(a * b + c)


def en_paren(rng: random.Random) -> Case:
    a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 12)
    q = f"What is ({a} + {b}) * {c}?"
    cot = (f"Brackets first: {a} + {b} = {a + b}\n"
           f"Then multiply: {a + b} * {c} = {(a + b) * c}")
    return q, cot, str((a + b) * c)


def en_percent(rng: random.Random) -> Case:
    pct = rng.choice([5, 10, 12, 15, 20, 25, 40, 50, 60, 75])
    base = rng.choice([40, 80, 120, 200, 240, 300, 400, 500, 640, 900])
    value = base * pct / 100
    q = rng.choice([f"What is {pct}% of {base}?",
                    f"Find {pct} percent of {base}.",
                    f"Compute {pct}% of {base}."])
    cot = (f"{pct}% means {pct}/100, so multiply {base} by {pct}/100.\n"
           f"{base} * {pct} / 100 = {value:g}")
    return q, cot, f"{value:g}"


def en_avg(rng: random.Random) -> Case:
    n = rng.randint(3, 6)
    values = [rng.randint(1, 100) for _ in range(n)]
    total = sum(values)
    if total % n:
        values[0] += n - total % n
        total = sum(values)
    mean = total // n
    q = (f"What is the average of {', '.join(map(str, values[:-1]))} "
         f"and {values[-1]}?")
    cot = (f"Sum: {' + '.join(map(str, values))} = {total}\n"
           f"Count: {n}\n"
           f"Average: {total} / {n} = {mean}")
    return q, cot, str(mean)


def en_fraction(rng: random.Random) -> Case:
    b, d = rng.choice([2, 3, 4, 5, 6, 8, 10, 12]), rng.choice([2, 3, 4, 5, 6, 8, 10, 12])
    a, c = rng.randint(1, b - 1), rng.randint(1, d - 1)
    total = Fraction(a, b) + Fraction(c, d)
    q = f"What is {a}/{b} + {c}/{d}?"
    cot = (f"Common denominator: {b} * {d} / gcd({b}, {d}) = "
           f"{b * d // gcd(b, d)}\n"
           f"{a}/{b} = {a * (d // gcd(b, d))}/{b * d // gcd(b, d)}, "
           f"{c}/{d} = {c * (b // gcd(b, d))}/{b * d // gcd(b, d)}\n"
           f"Sum: {total.numerator}/{total.denominator} = {float(total):.4g}")
    ans = (f"{total.numerator}/{total.denominator}"
           if total.denominator != 1 else str(total.numerator))
    return q, cot, ans


def en_speed(rng: random.Random) -> Case:
    speed = rng.choice([40, 50, 60, 70, 80, 90, 100, 120])
    hours = rng.randint(1, 9)
    dist = speed * hours
    q = rng.choice([
        f"A train travels at {speed} km/h for {hours} hours. How far does it go?",
        f"A car drives {speed} km each hour for {hours} hours. "
        f"What distance does it cover?",
    ])
    cot = (f"Distance = speed * time\n"
           f"{speed} * {hours} = {dist} km")
    return q, cot, f"{dist} km"


def en_money(rng: random.Random) -> Case:
    price = rng.choice([15, 20, 25, 30, 45, 50, 60, 75, 90, 120])
    count = rng.randint(2, 12)
    paid = price * count + rng.choice([0, 0, 5, 10, 20, 50])
    total = price * count
    q = (f"A notebook costs {price} rubles. Anna buys {count} notebooks "
         f"and pays with {paid} rubles. How much change does she get?")
    cot = (f"Cost of the notebooks: {price} * {count} = {total} rubles\n"
           f"Change: {paid} - {total} = {paid - total} rubles")
    return q, cot, f"{paid - total} rubles"


def en_gcd(rng: random.Random) -> Case:
    a, b = rng.randint(12, 400), rng.randint(12, 400)
    g = gcd(a, b)
    q = f"What is the greatest common divisor of {a} and {b}?"
    cot = (f"Prime factorisation or Euclid's algorithm gives gcd({a}, {b}) = {g}.\n"
           f"Check: {a} / {g} = {a // g} and {b} / {g} = {b // g}, "
           f"and those two are coprime.")
    return q, cot, str(g)


def en_sequence(rng: random.Random) -> Case:
    start, step, n = rng.randint(1, 20), rng.randint(2, 12), rng.randint(5, 15)
    last = start + step * (n - 1)
    total = n * (start + last) // 2
    q = (f"An arithmetic sequence starts at {start} and grows by {step} each "
         f"term. What is the sum of its first {n} terms?")
    cot = (f"The {n}-th term is {start} + {step} * {n - 1} = {last}\n"
           f"Sum = n * (first + last) / 2 = {n} * ({start} + {last}) / 2 = {total}")
    return q, cot, str(total)


# --------------------------------------------------------------------------
# Russian templates (the same families, phrased independently)
# --------------------------------------------------------------------------


def ru_add(rng: random.Random) -> Case:
    a, b = rng.randint(2, 999), rng.randint(2, 999)
    q = rng.choice([f"Сколько будет {a} + {b}?",
                    f"Вычисли {a} плюс {b}.",
                    f"Сложи {a} и {b}.",
                    f"Найди сумму чисел {a} и {b}."])
    cot = (f"Складываем по разрядам:\n"
           f"{a} + {b} = {a + b}\n\n"
           f"Проверка: {a + b} - {b} = {a}.")
    return q, cot, str(a + b)


def ru_sub(rng: random.Random) -> Case:
    a = rng.randint(20, 1000)
    b = rng.randint(1, a - 1)
    q = rng.choice([f"Сколько будет {a} - {b}?", f"Вычти {b} из {a}.",
                    f"Вычисли {a} минус {b}.", f"Найди разность {a} и {b}."])
    cot = (f"{a} - {b} = {a - b}\n\n"
           f"Проверка: {a - b} + {b} = {a}.")
    return q, cot, str(a - b)


def ru_mul(rng: random.Random) -> Case:
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    q = rng.choice([f"Сколько будет {a} * {b}?", f"Умножь {a} на {b}.",
                    f"Вычисли произведение {a} и {b}.",
                    f"Найди {a} умножить на {b}."])
    cot = (f"{a} * {b} = {a * b}\n\n"
           f"Раскладываем: {a} * {b} = {a} * {b // 10 * 10} + "
           f"{a} * {b % 10} = {a * (b // 10 * 10)} + {a * (b % 10)} = {a * b}.")
    return q, cot, str(a * b)


def ru_div(rng: random.Random) -> Case:
    b = rng.randint(2, 40)
    ans = rng.randint(2, 200)
    a = b * ans
    q = rng.choice([f"Сколько будет {a} / {b}?", f"Раздели {a} на {b}.",
                    f"Вычисли {a} делить на {b}.",
                    f"Сколько раз {b} помещается в {a}?"])
    cot = (f"{a} / {b} = {ans}\n\n"
           f"Проверка: {ans} * {b} = {a}, деление точное.")
    return q, cot, str(ans)


def ru_mul_add(rng: random.Random) -> Case:
    a, b, c = rng.randint(2, 30), rng.randint(2, 30), rng.randint(1, 200)
    q = rng.choice([f"Сколько будет {a} * {b} + {c}?",
                    f"Вычисли {a} умножить на {b}, затем прибавь {c}.",
                    f"Найди значение выражения {a} * {b} + {c}."])
    cot = (f"Сначала умножение:\n"
           f"{a} * {b} = {a * b}\n"
           f"{a * b} + {c} = {a * b + c}")
    return q, cot, str(a * b + c)


def ru_paren(rng: random.Random) -> Case:
    a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 12)
    q = f"Сколько будет ({a} + {b}) * {c}?"
    cot = (f"Сначала скобки: {a} + {b} = {a + b}\n"
           f"Затем умножение: {a + b} * {c} = {(a + b) * c}")
    return q, cot, str((a + b) * c)


def ru_percent(rng: random.Random) -> Case:
    pct = rng.choice([5, 10, 12, 15, 20, 25, 40, 50, 60, 75])
    base = rng.choice([40, 80, 120, 200, 240, 300, 400, 500, 640, 900])
    value = base * pct / 100
    q = rng.choice([f"Сколько составляет {pct}% от {base}?",
                    f"Найди {pct} процентов от числа {base}.",
                    f"Вычисли {pct}% от {base}."])
    cot = (f"{pct}% — это {pct}/100, значит умножаем {base} на {pct}/100.\n"
           f"{base} * {pct} / 100 = {value:g}")
    return q, cot, f"{value:g}"


def ru_avg(rng: random.Random) -> Case:
    n = rng.randint(3, 6)
    values = [rng.randint(1, 100) for _ in range(n)]
    total = sum(values)
    if total % n:
        values[0] += n - total % n
        total = sum(values)
    mean = total // n
    q = (f"Найди среднее арифметическое чисел "
         f"{', '.join(map(str, values[:-1]))} и {values[-1]}.")
    cot = (f"Сумма: {' + '.join(map(str, values))} = {total}\n"
           f"Количество чисел: {n}\n"
           f"Среднее: {total} / {n} = {mean}")
    return q, cot, str(mean)


def ru_fraction(rng: random.Random) -> Case:
    b, d = rng.choice([2, 3, 4, 5, 6, 8, 10, 12]), rng.choice(
        [2, 3, 4, 5, 6, 8, 10, 12])
    a, c = rng.randint(1, b - 1), rng.randint(1, d - 1)
    total = Fraction(a, b) + Fraction(c, d)
    q = f"Сколько будет {a}/{b} + {c}/{d}?"
    cot = (f"Общий знаменатель: {b * d // gcd(b, d)}\n"
           f"Сумма: {total.numerator}/{total.denominator} = {float(total):.4g}")
    ans = (f"{total.numerator}/{total.denominator}"
           if total.denominator != 1 else str(total.numerator))
    return q, cot, ans


def ru_speed(rng: random.Random) -> Case:
    speed = rng.choice([40, 50, 60, 70, 80, 90, 100, 120])
    hours = rng.randint(1, 9)
    dist = speed * hours
    q = rng.choice([
        f"Поезд едет со скоростью {speed} км/ч {hours} часов. "
        f"Какое расстояние он проедет?",
        f"Машина проезжает {speed} км за час и едет {hours} часов. "
        f"Сколько километров она проедет?",
    ])
    cot = (f"Расстояние = скорость * время\n"
           f"{speed} * {hours} = {dist} км")
    return q, cot, f"{dist} км"


def ru_money(rng: random.Random) -> Case:
    price = rng.choice([15, 20, 25, 30, 45, 50, 60, 75, 90, 120])
    count = rng.randint(2, 12)
    paid = price * count + rng.choice([0, 0, 5, 10, 20, 50])
    total = price * count
    q = (f"Тетрадь стоит {price} рублей. Анна покупает {count} тетрадей "
         f"и платит {paid} рублей. Сколько сдачи она получит?")
    cot = (f"Стоимость тетрадей: {price} * {count} = {total} рублей\n"
           f"Сдача: {paid} - {total} = {paid - total} рублей")
    return q, cot, f"{paid - total} рублей"


def ru_gcd(rng: random.Random) -> Case:
    a, b = rng.randint(12, 400), rng.randint(12, 400)
    g = gcd(a, b)
    q = f"Найди наибольший общий делитель чисел {a} и {b}."
    cot = (f"По алгоритму Евклида НОД({a}, {b}) = {g}.\n"
           f"Проверка: {a} / {g} = {a // g}, {b} / {g} = {b // g}, "
           f"эти числа взаимно просты.")
    return q, cot, str(g)


def ru_sequence(rng: random.Random) -> Case:
    start, step, n = rng.randint(1, 20), rng.randint(2, 12), rng.randint(5, 15)
    last = start + step * (n - 1)
    total = n * (start + last) // 2
    q = (f"Арифметическая прогрессия начинается с {start} и каждый следующий "
         f"член больше на {step}. Чему равна сумма первых {n} членов?")
    cot = (f"{n}-й член: {start} + {step} * {n - 1} = {last}\n"
           f"Сумма = n * (первый + последний) / 2 = "
           f"{n} * ({start} + {last}) / 2 = {total}")
    return q, cot, str(total)


FAMILIES: Dict[str, Tuple[List[Callable[[random.Random], Case]], str]] = {
    "en": ([en_add, en_sub, en_mul, en_div, en_mul_add, en_paren,
            en_percent, en_avg, en_fraction, en_speed, en_money, en_gcd,
            en_sequence], "en"),
    "ru": ([ru_add, ru_sub, ru_mul, ru_div, ru_mul_add, ru_paren,
            ru_percent, ru_avg, ru_fraction, ru_speed, ru_money, ru_gcd,
            ru_sequence], "ru"),
}

SOURCE_NAME = {"en": "arith_en", "ru": "arith_ru"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=200_000,
                    help="maximum examples per language")
    ap.add_argument("--out", default="E:/kron_sft/raw")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for lang, (families, _) in FAMILIES.items():
        rng = random.Random(args.seed + (0 if lang == "en" else 1))
        seen: set[int] = set()
        path = out / f"{SOURCE_NAME[lang]}.jsonl.gz"
        kept = 0
        total_bytes = 0
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=3) as fh:
            for i in range(args.per_lang * 3):
                if kept >= args.per_lang:
                    break
                family = families[i % len(families)]
                question, cot, answer = family(rng)
                text = f"{question}\x00{cot}\x00{answer}"
                key = int.from_bytes(
                    hashlib.blake2b(text.encode("utf-8"),
                                    digest_size=8).digest(), "big")
                if key in seen:
                    continue
                seen.add(key)
                record = {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant",
                         "content": f"{cot}\n\nAnswer: {answer}"
                         if lang == "en" else f"{cot}\n\nОтвет: {answer}"},
                    ],
                    "lang": lang, "domain": "math", "task": "math_arith",
                    "source": SOURCE_NAME[lang], "licence": "CC0-1.0",
                    "family": family.__name__,
                }
                record["bytes"] = sum(len(m["content"].encode("utf-8"))
                                      for m in record["messages"])
                total_bytes += record["bytes"]
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
        marker = out / f"{SOURCE_NAME[lang]}.done.json"
        marker.write_text(json.dumps({
            "source": SOURCE_NAME[lang], "hf": "synthetic",
            "licence": "CC0-1.0", "lang": lang, "domain": "math",
            "task": "math_arith", "rows_seen": kept, "rows_kept": kept,
            "bytes": total_bytes, "note": "templated arithmetic, exact answers",
        }, indent=1), encoding="utf-8")
        print(f"[{SOURCE_NAME[lang]}] {kept:,} examples, "
              f"{total_bytes / 1e6:.1f} MB -> {path}")


if __name__ == "__main__":
    main()
