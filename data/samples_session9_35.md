## Тестовая генерация — шаг 73636, те же 35 токенов

Чекпойнт `last.pt`, шаг **73,636**, 9,938,886 параметров, по 35 токенов. Жадное: `T=1e-05`, `top_k=1`, штраф за повторы 1.1. Сэмплирование: `T=0.7`, `top_k=40`, `top_p=0.95`, штраф за повторы 1.2.

### EN, продолжение прозы

Промпт: `The Roman Empire was one of the largest and longest-lasting empires in history, and`

Жадно:
```
the Middle East of the Middle East was the first to be a greatly recognized and undoubtedly religious.

In
```
Сэмплирование:
```
the Government of Australia was also authorized by the Irish Catholics. In 1934, the Bug
```

### RU, продолжение прозы

Промпт: `Фотосинтез — это процесс, при котором растения используют энергию света, чтобы`

Жадно:
```
получить необходимые условия для здоровья.
Важно отметить, что в процессе развития болезни происходит не только в организме, но и в организме
```
Сэмплирование:
```
снять необходимые меры и улучшить его. К тому же, они могут быть использованы для облегчения или удаляют изоляцию.
```

### EN, код

Промпт: `def add(a, b):\n    """Return the sum of two numbers."""\n    return`

Жадно:
```
b


def add_order(a, b):
    """Return the order of the sum of two numbers."""
    return b


def add_
```
Сэмплирование:
```
a + b


def _add_final(j = (b, b)):
  """Return the number of elements in the list.

  Ar
```

### EN, арифметика столбиком

Промпт: `<|user|>Compute 47 + 38 and show the carries.<|eot|><|assistant|>`

Жадно:
```
Add the columns from the units:
units: 6 + 3 = 10 -> write 0, carry 1
tens:
```
Сэмплирование:
```
Add the columns from the units:
units: 6 + 3 = 10 -> write 0, carry 1
tens:
```

### RU, арифметика столбиком

Промпт: `<|user|>Вычисли 800 - 385 и покажи заёмы.<|eot|><|assistant|>`

Жадно:
```
Вычитаем по разрядам, начиная с единиц:
единицы: 0 - 0 меньше 5 -> занимаем: 10
```
Сэмплирование:
```
Вычитаем по разрядам, начиная с единиц:
единицы: 0 - 1 = 0
десятки: 0 -
```

### чат-формат (цель SFT)

Промпт: `<|system|>You are KronLLM, a compact bilingual model.<|eot|><|user|>Who are you?<|eot|>...`

Жадно:
```
Yes, the bird is a military and is a military. Here's an example:

"I'm a milit
```
Сэмплирование:
```
Yes, the other is the historical site of the Royal House.

I have been a member of the Royal House
```

_Сгенерировано за 18 с на cuda._
