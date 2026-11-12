## Тестовая генерация — сессия 16, шаг 4521

Чекпойнт `last.pt`, шаг **4,521**, 9,938,886 параметров, по 35 токенов. Жадное: `T=1e-05`, `top_k=1`, штраф за повторы 1.1. Сэмплирование: `T=0.7`, `top_k=40`, `top_p=0.95`, штраф за повторы 1.2.

### EN, продолжение прозы

Промпт: `The Roman Empire was one of the largest and longest-lasting empires in history, and`

Жадно:
```
the Romans were also known as the Roman Empire.

In the 19th century, the Roman Empire was one of the
```
Сэмплирование:
```
the Korean Palace's Laws were part of the 13th century. It was now known as the Dominican Home.
```

### RU, продолжение прозы

Промпт: `Фотосинтез — это процесс, при котором растения используют энергию света, чтобы`

Жадно:
```
найти общее количество растений, которые вы можете использовать для лечения.<|eot|>
```
Сэмплирование:
```
создать виды, которые могут быть называться гладкими, в то время как тропа становится очень высоким.<|eot|>
```

### EN, код

Промпт: `def add(a, b):\n    """Return the sum of two numbers."""\n    return`

Жадно:
```
a * b
```<|eot|>
```
Сэмплирование:
```
sum([i for j in a if a] + 1
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
units: 6 + 8 = 14 -> write 4, carry 1
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
единицы: 0 - 0 меньше 5 -> занимаем: 10
```

### чат-формат (цель SFT)

Промпт: `<|system|>You are KronLLM, a compact bilingual model.<|eot|><|user|>Who are you?<|eot|>...`

Жадно:
```
I'm not aware of any other information about the world's history, but I can provide you with a detailed explanation of how the world is being used to
```
Сэмплирование:
```
It's like a great deal of pain and a smooth story that makes it easier for you to feel comfortable and ready to fill with
```

_Сгенерировано за 9 с на cpu._
