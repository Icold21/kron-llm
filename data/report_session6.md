# Q10: сессия 6 завершена

Старт был с шага **36,549**, бюджет 660 минут.

| параметр | значение |
|---|---|
| шаг чекпоинта | **52205** |
| пройдено за сессию | 15,656 шагов (0.26 млрд токенов) |
| скорость | 2.51 с/шаг |
| loss (последняя запись) | 2.972463388442993 |
| **val_bpb_mixed** | **1.1827283320993998** |
| val_bpb_en / ru | 1.417878953078488 / 0.9552858060482986 |
| до конца эпохи | 120,068 шагов (69.7%) |

## Кривые

Валидация за сессию (`val_bpb_mixed` — единственная честная метрика: held-out бин не меняется с первого шага, поэтому серия сопоставима несмотря на пересборки корпуса):

| шаг | val_bpb_mixed | EN | RU |
|---:|---:|---:|---:|
| 43,000 | 1.1896880573372899 | 1.426412204289425 | 0.9607235845414744 |
| 44,000 | 1.187947369607971 | 1.4265309583904073 | 0.9571844063588113 |
| 45,000 | 1.1889096923915743 | 1.4275677868249808 | 0.9580746657415563 |
| 46,000 | 1.1881902892432803 | 1.4271203854867485 | 0.9570921768377862 |
| 47,000 | 1.1867870841557024 | 1.4256434814471564 | 0.9557602548952142 |
| 48,000 | 1.1872459819367978 | 1.4228974870920312 | 0.9593189903894669 |
| 49,000 | 1.1845752420381233 | 1.4207265956835575 | 0.9561647867313483 |
| 50,000 | 1.183307510213962 | 1.4205738907175256 | 0.9538185779541103 |
| 51,000 | 1.1832115342246288 | 1.419167394386958 | 0.9549901642429908 |
| 52,000 | 1.1827283320993998 | 1.417878953078488 | 0.9552858060482986 |

График приложен к письму (`curve_q10.png`): сверху train loss, снизу `val_bpb_mixed` с языками и референсами GPT-2-124M (1.554) и soup на шаге 35k (1.133). Вертикальные линии — смены корпуса; ступени на train loss в этих местах ожидаемы, на валидации их нет.

## Тестовая генерация — сессия 6, шаг 52205

Чекпойнт `last.pt`, шаг **52,205**, 9,938,886 параметров. Декодирование: жадное (`top_k=1`) и сэмплирование (`T=0.8`, `top_k=8`, `top_p=0.9`, штраф за повторы 1.15), по 60 токенов.

### EN, продолжение прозы

Промпт: `The history of the Roman Empire begins`

Жадно:
```
to the

political and political part of the Roman Empire.

The Roman Empire was a part of the Roman Empire, and the

Scotlands, and the Roman Empire, and the R
```
Сэмплирование:
```
to the

democratic party. Their wise and wounded-bearer were

certain from the military of the Sabbath. Among them, the

Essay, the Civil War,
```

### RU, продолжение прозы

Промпт: `История России начинается`

Жадно:
```
с 1970-х годов.

В 1970-х годах в России было 1970-х годов.

В 1970-х годах в России было 1970-х годов.

В
```
Сэмплирование:
```
в 1968 году.

Память
Все сенаторы, основанные на переживании, в 1973 году были опубликованы в журнале «Русский дом» (после 1974 года). В 2
```

### EN, знание

Промпт: `Photosynthesis is the process by which`

Жадно:
```
the Chemnians are the most commonly used.

The Chemnians are the most commonly used in the Chemnians.

The Chemnians are the most commonly used in the Chemni
```
Сэмплирование:
```
the Chemistry of the Marine Company is a fair and fair-destruction of the Pacific Social Council. The Chemistry of the Marine Company is the only part of the Chemist
```

### RU, знание

Промпт: `Фотосинтез — это процесс, при котором`

Жадно:
```
он может быть не менее 100000000000000000000000000000000000000000000000000000
```
Сэмплирование:
```
вы можете использовать ключ и выглядеть так:
[code]
    public static string S = "{";}"
    void s = s;
    void s = s;
    void s = s.get().value();
    void
```

### EN, код

Промпт: `def add(a, b):\n    return`

Жадно:
```
a.get_a(b)

def add(a, b):
    return a.get_a(b)

def add(a, b):
    return a.get_a(b)

def add(a, b):
    return a.get
```
Сэмплирование:
```
b


def add_packet(a,b):
    print "Simple adding the packet from the packets.  The packet is used to remove
    packet from the packet's packet and then use the p
```

### EN, арифметика столбиком

Промпт: `<|user|>Compute 47 + 38 and show the carries.<|eot|><|assistant|>`

Жадно:
```
Let's denote the carries of the carries.

The carries are the carries of the carries.

The carries are the carries of the carries.

The carries are the carries of the carries.

The carries
```
Сэмплирование:
```
Let's take a chance that's a 2D.
All these steps are:

1. If you want to do this, then you need to do it.
2. If you want to do this, then press the carries.
3. If you want
```

### RU, арифметика столбиком

Промпт: `<|user|>Вычисли 800 - 385 и покажи заёмы.<|eot|><|assistant|>`

Жадно:
```
Следует отметить, что в 1000 - 385 + 385 = 1000 - 385 = 1000 - 385 = 1000 - 385 = 10
```
Сэмплирование:
```
Здравствуйте. Учитывая, что в 2016 году 90000000000000000000000000000000000000000
```

### чат-формат

Промпт: `<|system|>You are KronLLM, a compact bilingual model.<|eot|><|user|>Who are you?<|eot|>...`

Жадно:
```
When you have a bunch of 100000000000000000000000000000000000000000000000000
```
Сэмплирование:
```
Order: 72px

Anything is a whole number of bats in the same way, so that there is some bats.

8. August 19, 2013

### Check

The first time I
```

_Сгенерировано за 30 с на cpu._


Выгрузка: `E:\kron_out\session6`. Чекпоинт опубликован версией `icold21/kron-q10-ckpt`.
