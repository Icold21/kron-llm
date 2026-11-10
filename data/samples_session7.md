## Тестовая генерация — сессия 7, шаг 58213

Чекпойнт `last.pt`, шаг **58,213**, 9,938,886 параметров. Декодирование: жадное (`top_k=1`) и сэмплирование (`T=0.8`, `top_k=8`, `top_p=0.9`, штраф за повторы 1.15), по 60 токенов.

### EN, продолжение прозы

Промпт: `The history of the Roman Empire begins`

Жадно:
```
to the

passage of the Roman Empire. The Roman Empire is a very good

capital, and the Roman Empire is a very good, and the Roman Empire

is a very good, and the R
```
Сэмплирование:
```
in

the Pope of Barbara, and the Population of Ministers; but the

critics of the Populs, and the Populs, and the Pope of the

Posts, and the Cath
```

### RU, продолжение прозы

Промпт: `История России начинается`

Жадно:
```
с 1990-х годов.

В 1990-х годах в России было создано 110000000000000000000000000000000
```
Сэмплирование:
```
с 1957 по 2018 годы, когда 30 лет (по данным на 42 %) прошёл в Европе.
Среди современных исторических проектов, в том числе и в России. Память — Алексей В
```

### EN, знание

Промпт: `Photosynthesis is the process by which`

Жадно:
```
the phenomena of the phenomena of the phenomena of the phenomena of the phenomena of the phenomena of the phenomena of the phenomena of the phenomena of the phenomena of
```
Сэмплирование:
```
the reaction is used to be a function of the form . In this paper we discuss the properties of the form and the form . The form of a function of the form is defined as follows:

The form of the form , where , , and the form

( )
```

### RU, знание

Промпт: `Фотосинтез — это процесс, при котором`

Жадно:
```
в данном случае он будет не только не только в случае, но и в случае, если в данном случае он будет не только не только в случае, но и в случае, если в данном случае он будет не только не только в случае, но и в случае, если в случае,
```
Сэмплирование:
```
в данном моменте, как и на основе, а также в данном случае.
В этом случае, когда на основе, где на основе, в этот момент, можно было бы использовать в качестве основного объекта. В этом случае, когда на основе объекта, в данном случае
```

### EN, код

Промпт: `def add(a, b):\n    return`

Жадно:
```
a, b<|eot|>
```
Сэмплирование:
```
a[b]
#    #

#   `c`->csv.add(a)`
##
#    `a`->csv.add(a, b)`
##
#    `c`, `f`.
#
```

### EN, арифметика столбиком

Промпт: `<|user|>Compute 47 + 38 and show the carries.<|eot|><|assistant|>`

Жадно:
```
Add the columns from the units:
units: 6 + 3 = 10 -> write 0, carry 1
tens: 7 + 8 + 1 = 14 -> write 4, carry 1
hundreds
```
Сэмплирование:
```
Add the columns from the units:
units: 6 + 3 = 10 -> write 0, carry 1
tens: 7 + 38 + 1 = 92 -> write 2, carry 9
hundred
```

### RU, арифметика столбиком

Промпт: `<|user|>Вычисли 800 - 385 и покажи заёмы.<|eot|><|assistant|>`

Жадно:
```
Складываем по разрядам, начиная с единиц:
единицы: 0 - 0 меньше 5 -> занимаем: 10 - 5 = 5, заём 1
десятки: 0 - 1 меньше 8
```
Сэмплирование:
```
Складываем по разрядам, начиная с единиц:
единицы: 6 + 5 = 14 -> пишем 4, запоминаем 1
десятки: 0 + 8 = 2 -> пишем 2
```

### чат-формат

Промпт: `<|system|>You are KronLLM, a compact bilingual model.<|eot|><|user|>Who are you?<|eot|>...`

Жадно:
```
You are given by:
Step 1: 50px
Step 2: 50px
Step 3: 50px
Step 4: 50px
Step 5:
```
Сэмплирование:
```
She is 700 * 4 and the borrows: 18 / 30 = 52
Since 700 + 52 = 5400, the borrow is 54 * 4 = 54
```

_Сгенерировано за 25 с на cpu._
