## Тестовая генерация — сессия 13, шаг 96120

Чекпойнт `last.pt`, шаг **96,120**, 9,938,886 параметров, по 35 токенов. Жадное: `T=1e-05`, `top_k=1`, штраф за повторы 1.1. Сэмплирование: `T=0.7`, `top_k=40`, `top_p=0.95`, штраф за повторы 1.2.

### EN, продолжение прозы

Промпт: `The Roman Empire was one of the largest and longest-lasting empires in history, and`

Жадно:
```
the Middle East was a major part of the world's first-round history.

In 1970, the Middle East was a
```
Сэмплирование:
```
it was said that no news was seen as a murder, though more than a century later. The Marjorian City, with its site
```

### RU, продолжение прозы

Промпт: `Фотосинтез — это процесс, при котором растения используют энергию света, чтобы`

Жадно:
```
получить дополнительные данные.
Содержание
Положение о том, что в организме человека не существует, является причиной возникновения и нев
```
Сэмплирование:
```
создать себе свой организм.
Стоимость и скорость
Если вы хотите найти на экране отображение, можно обойтись на данный момент.
```

### EN, код

Промпт: `def add(a, b):\n    """Return the sum of two numbers."""\n    return`

Жадно:
```
a, b


class Separator:
    """Separator for all the numbers."""

    def __init__(self, *args, **kwargs
```
Сэмплирование:
```
a


# Additional classes
    def __init__(self) -> 1:
        self._block = b

    def _diff_n
```

### EN, арифметика столбиком

Промпт: `<|user|>Compute 47 + 38 and show the carries.<|eot|><|assistant|>`

Жадно:
```
Add the columns from the units:
units: 6 + 8 = 14 -> write 4, carry 1
tens:
```
Сэмплирование:
```
Add the columns from the units:
units: 6 + 0 = 6 -> write 6
tens: 7 + 8
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
единицы: 0 - 0 меньше 5 -> занимаем: 10
```

### чат-формат (цель SFT)

Промпт: `<|system|>You are KronLLM, a compact bilingual model.<|eot|><|user|>Who are you?<|eot|>...`

Жадно:
```
I don't have access to the specific information provided by the given information. However, you can visit the website or contact the website or contact the we
```
Сэмплирование:
```
I don't have access to the specific information you are using or contacting your question about their clarity and qualifications. However, it's not a l
```

_Сгенерировано за 12 с на cpu._
