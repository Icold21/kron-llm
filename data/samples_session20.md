## Тестовая генерация — сессия 20, шаг 12198

Чекпойнт `last.pt`, шаг **12,198**, 9,938,886 параметров, по 35 токенов. Жадное: `T=1e-05`, `top_k=1`, штраф за повторы 1.1. Сэмплирование: `T=0.7`, `top_k=40`, `top_p=0.95`, штраф за повторы 1.2.

### EN, продолжение прозы

Промпт: `The Roman Empire was one of the largest and longest-lasting empires in history, and`

Жадно:
```
the Romans were also known as the Roman Empire.

In 1863, the Roman Empire was awarded the
```
Сэмплирование:
```
the Maritas City was 80-94, and the Kaiserians were 25,475,460
```

### RU, продолжение прозы

Промпт: `Фотосинтез — это процесс, при котором растения используют энергию света, чтобы`

Жадно:
```
определить, сколько времени вы можете использовать.<|eot|>
```
Сэмплирование:
```
определить, сколько и больше.<|eot|>
```

### EN, код

Промпт: `def add(a, b):\n    """Return the sum of two numbers."""\n    return`

Жадно:
```
a + b
```<|eot|>
```
Сэмплирование:
```
a + b
```<|eot|>
```

### EN, арифметика столбиком

Промпт: `<|user|>Compute 47 + 38 and show the carries.<|eot|><|assistant|>`

Жадно:
```
Add the columns from the units:
units: 7 + 8 = 15 -> write 5, carry 1
tens:
```
Сэмплирование:
```
Add the columns from the units:
units: 7 + 0 = 7 -> write 7
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
I am Kron, a small language model created by the KronLLM project. My architecture relies on Kronecker products to ensure that my weights are not unique. This means that my weights are
```
Сэмплирование:
```
These are the greatest common divisor of 10 and 872, but they are not as fast as 389.
Since
```

_Сгенерировано за 10 с на cpu._
