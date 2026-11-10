# Q10: сессия 9 завершена

Старт был с шага **58,213**, бюджет 660 минут.

| параметр | значение |
|---|---|
| шаг чекпоинта | **73636** |
| пройдено за сессию | 15,423 шагов (0.25 млрд токенов) |
| скорость | 2.54 с/шаг |
| loss (последняя запись) | 2.7177928256988526 |
| **val_bpb_mixed** | **1.1971918741542666** |
| val_bpb_en / ru | 1.4495834077938337 / 0.9530735821170988 |
| до конца эпохи | 35,871 шагов (32.8%) |

## Кривые

Валидация за сессию (`val_bpb_mixed` — единственная честная метрика: held-out бин не меняется с первого шага, поэтому серия сопоставима несмотря на пересборки корпуса):

| шаг | val_bpb_mixed | EN | RU |
|---:|---:|---:|---:|
| 69,000 | 1.2009053924802011 | 1.4532221240845005 | 0.9568594505128911 |
| 69,500 | 1.1994002935023877 | 1.451321499241905 | 0.9557369123015774 |
| 70,000 | 1.2002103238216564 | 1.4518811487705556 | 0.9567891160808 |
| 70,500 | 1.197611262125161 | 1.450375170263025 | 0.9531328018002077 |
| 71,000 | 1.1969308366034905 | 1.4468753673714378 | 0.9551793361659286 |
| 71,500 | 1.1975160018326563 | 1.448603053122087 | 0.9546594320042949 |
| 72,000 | 1.1981138348508542 | 1.4502005618937535 | 0.9542903580347324 |
| 72,500 | 1.1955933217105545 | 1.4488694568985678 | 0.9506194248279063 |
| 73,000 | 1.1939950136408057 | 1.4445575551868308 | 0.9516457604441502 |
| 73,500 | 1.1971918741542666 | 1.4495834077938337 | 0.9530735821170988 |

График приложен к письму (`curve_q10.png`): сверху train loss, снизу `val_bpb_mixed` с языками и референсами GPT-2-124M (1.554) и soup на шаге 35k (1.133). Вертикальные линии — смены корпуса; ступени на train loss в этих местах ожидаемы, на валидации их нет.

## Тестовая генерация — сессия 9, шаг 73636

Чекпойнт `last.pt`, шаг **73,636**, 9,938,886 параметров, по 60 токенов. Жадное: `T=1e-05`, `top_k=1`, штраф за повторы 1.1. Сэмплирование: `T=0.7`, `top_k=40`, `top_p=0.95`, штраф за повторы 1.2.

### EN, продолжение прозы

Промпт: `The Roman Empire was one of the largest and longest-lasting empires in history, and`

Жадно:
```
the Middle East of the Middle East was the first to be a greatly recognized and undoubtedly religious.

In 1967, the Middle East was renamed the Middle East. The Middle E
```
Сэмплирование:
```
I have never heard from any other publication or government, but the main reason is that the murder of the university and the institution is a native and member of the Municipality Board.

I have been an extension to the greatest
```

### RU, продолжение прозы

Промпт: `Фотосинтез — это процесс, при котором растения используют энергию света, чтобы`

Жадно:
```
получить необходимые условия для здоровья.
Важно отметить, что в процессе развития болезни происходит не только в организме, но и в организме, а также в организме. В качестве этого можно выделить следующие:
- Симптомы:
```
Сэмплирование:
```
получить импульсивную форму и уменьшить их размер.
Важно отметить, что экологические условия могут быть определены как в виде мелких коктейнов, так и в виде специальных ломок (например, наружный фарф
```

### EN, код

Промпт: `def add(a, b):\n    """Return the sum of two numbers."""\n    return`

Жадно:
```
b


def add_order(a, b):
    """Return the order of the sum of two numbers."""
    return b


def add_order(a, b):
    """Return the order of the sum of two numbers."""
    return b
```
Сэмплирование:
```
a
```python
import rendev as np:
    # Note that the number of times is equal to the number of times.
    # Use the sum of one number of times.
    # Number of times.
    # ON tag - a list of
```

### EN, арифметика столбиком

Промпт: `<|user|>Compute 47 + 38 and show the carries.<|eot|><|assistant|>`

Жадно:
```
Add the columns from the units:
units: 6 + 3 = 10 -> write 0, carry 1
tens: 7 + 8 + 1 = 16 -> write 6, carry 1
hundreds
```
Сэмплирование:
```
Add the columns from the units:
units: 6 + 3 = 10 -> write 0, carry 1
tens: 7 + 8 + 1 = 16 -> write 6, carry 1
hundreds
```

### RU, арифметика столбиком

Промпт: `<|user|>Вычисли 800 - 385 и покажи заёмы.<|eot|><|assistant|>`

Жадно:
```
Вычитаем по разрядам, начиная с единиц:
единицы: 0 - 0 меньше 5 -> занимаем: 10 - 5 = 4, заём 1
десятки: 0 - 1 меньше 8 -
```
Сэмплирование:
```
Вычитаем по разрядам, начиная с единиц:
единицы: 0 - 0 меньше 5 -> занимаем: 10 - 5 = 4, заём 2
десятки: 0 - 1 меньше 8 -
```

### чат-формат (цель SFT)

Промпт: `<|system|>You are KronLLM, a compact bilingual model.<|eot|><|user|>Who are you?<|eot|>...`

Жадно:
```
Yes, the bird is a military and is a military. Here's an example:

"I'm a military, and I'm a military, and I'm a military, and I'm a m
```
Сэмплирование:
```
I do not have access to information about the topic or online content that you can identify and manage.

Absolute value: 1 - 2 = 3
Sum: 9 - 8 = 1
Sum: 10 -
```

_Сгенерировано за 18 с на cpu._


Выгрузка: `E:\kron_out\session9`. Чекпоинт опубликован версией `icold21/kron-q10-ckpt`.
