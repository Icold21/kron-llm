# Примеры генерации на новых параметрах — и статус сессии

Коротко: **вы были правы про `top_p`.** Из 31 измеренной конфигурации лучшей
оказалась `T=1.0, top_k=8, top_p=0.9, rp=1.1`, и на 8 сидах она выигрывает у
чистого `top_k=8` по-английски на 0.109 bpb (2.1σ) — то есть это не шум и не
вкусовщина. Ниже — живые примеры на этих параметрах и рядом те же промпты на
старых, чтобы разницу можно было увидеть, а не принять на веру.

## Что изменилось в цифрах

| рецептура | GPT-2 bpb (EN) | ruGPT bpb (RU) | петель |
|---|---:|---:|---:|
| **T=1.0, k=8, p=0.9, rp=1.1 (новая)** | **1.630** | **0.906** | 0.84 |
| T=1.0, k=8, rp=1.1 (была) | 1.739 | 0.989 | 0.80 |
| T=1.0, k=8, p=0.95, rp=1.1 | 1.763 | 0.979 | 0.79 |
| T=1.0, k=8, rp=1.2 | 1.973 | 1.008 | 0.82 |
| T=1.0, только top_p=0.9, rp=1.1 | 2.206 | 1.141 | 0.64 |
| T=1.0, без обрезки вовсе | 2.239 | 1.480 | 0.57 |

Почему `top_p` в одиночку проигрывает: на 26 % эпохи распределение плоское,
топ-1 токен держит 0.34 массы, топ-8 — 0.61, а до порога 0.9 набирается
**~135 токенов**. То есть «порог по вероятности» здесь не обрезает хвост, а
пропускает его. А `top_k=8` + `top_p=0.9` — это не украшение: порог срабатывает
на **17 % шагов** (почти все — в русском тексте, где чаще есть доминирующий
токен) и именно там сужает выбор.

Честная оговорка, которую надо будет повторить в статье: качество текста на
26 % эпохи — во многом заслуга декодера. `top_k=8` отбрасывает 39 % массы
модели на каждом шаге, и без этой отсечки получается заметно хуже.

## Генерация на новых параметрах

Чекпоинт шага 35115 (26 % эпохи), 64 токена, 8 промптов. **Новые параметры:** T=1.0, top_k=8, **top_p=0.9**, repetition penalty 1.1. Рядом — тот же промпт со **старыми** параметрами (T=1.0, top_k=8, без top_p), чтобы разница была видна глазами, а не только в таблице судей.

⏎ = перенос строки в выводе модели.

### EN prose

`She opened the door and`

**новые (top_p=0.9)**

> twitched with the streets of the firm, a man, and the bells of the ⏎  ⏎ white hot gallant. The gallant was sorryly, and it seemed a ⏎  ⏎ stripping-room in a fair

> died to the next door. The door was now a fair and sleeping, with his hand ⏎  ⏎ and his eyes. ⏎  ⏎ Then he was now a girl-holder, and the door of his hand came toward the ⏎  ⏎ master’s face.

**старые (без top_p)**

> twitched with the streets of the firm, a man, and the bells of the ⏎  ⏎ white hot gallant. The gallant was sordered by the bells of the hot gallant ⏎  ⏎ the firms of

### EN science

`The Moon orbits the Earth at a distance of`

**новые (top_p=0.9)**

> 20 kpc, but the Park is not too late. ⏎  ⏎ See also ⏎  ⏎ References ⏎  ⏎ 1953 births ⏎ 1964 deaths ⏎ Boys of Western Roman Emperorating ⏎ Muslims in Brande Count

> about 485 m. ⏎ • The Schwartz equation of the Earth is: ⏎  ⏎ \begin{align} \frac{\alpha}{1-\left ( 1-x+3 \right) + 1+\frac{4x-1}{2} + \frac{\

**старые (без top_p)**

> 20 kpc, but the Park is not too late. ⏎  ⏎ During the Park, David Richard Hotel and Walter Williamson were later built on the Billbook of the Days of New Jersey and the C

### RU prose

`Он открыл дверь и`

**новые (top_p=0.9)**

> был вынужден был уволить на свою сторону. Вскоре после этого, когда он не пригрозил ему «небольшого» и был взят от ноги. Следствие, когда он стал бежать, он вновь приказал ей отпу

> был взволнован с нею. Он стал первым из пяти пятнадцатилетней, но в результате его называли «самой пятнадцатилетни». Стоит ли это сделать в качестве гран-при второго пятнадц

**старые (без top_p)**

> был вынужден был уволежить в парижском доме. В 1839 году был приглашён, затем в 1853 году он был вынужден возвратиться в бухту, чтобы отвориться, что он не хот

### RU history

`В 1961 году`

**новые (top_p=0.9)**

> был выпущен 8-й гимн в Париже. В настоящее время, когда банкетки не превышали 50 тысяч человек, его трудности и внедрении были вынесены 100 тысяч человек. ⏎  ⏎ Сюжет ⏎ Старик

> была образована СПГ (СО) и была переименована в Кировскую область. ⏎  ⏎ В 2010—2013 годах — Спорный дом, Спорный дом — в Кировскую область в составе «Нового»

**старые (без top_p)**

> был вынуждён израильский и вторые должности, внесё себя в список союзов с В. И. Козиновым и В. Н. Алексеевым. С началом 1970-х — первой заместитель начальника отделения по

### code

`def add(a, b): ⏎     return`

**новые (top_p=0.9)**

> a.__init__(b, b) ⏎  ⏎  ⏎ def add_csrf_function(a, b): ⏎     """Represents an csrf_function in a class of bifurcations.""" ⏎     if a is not None: ⏎         return a.__init__(

> a + a ⏎  ⏎     def __init__(self, *args, **kwargs): ⏎         self._b_args = {} ⏎  ⏎     def _multi_args(self): ⏎         self._b_args.append([[1, 0]) ⏎  ⏎     def _multi_args(self): ⏎         self._

**старые (без top_p)**

> a.__init__(['a', 'b'), ''' ⏎     a = 0 ⏎   a += 1 ⏎  ⏎   def __init__(self, a): ⏎     """Calls a list of a new list of all the values, and iterates with ⏎     a value. ⏎  ⏎     This is an inte

### arithmetic

`17 + 25 =`

**новые (top_p=0.9)**

> 80 . in addition , the first term is an integer of a number . it follows that the two terms are not zero and can be written for a given string . the last term in the second term is not the same . in addition , the last term is equivalent to the second term , which is

> 3.8 + 0.01001 + 1.26903 = 100000.02411 = 0.5 + 3.5 = 0.1 + 4.0 = 0.5 * 3

**старые (без top_p)**

> 80 . in addition , the first term is an integer of a number 9 and a positive integer . the second term is called a string for each string . the last term is a number , and hence it has not been written @xcite for every string and the last term is an inte

### identity

`KronLLM is`

**новые (top_p=0.9)**

> not a very small language, so there are no natural numbers in this sense that they will not be able to do so. ⏎  ⏎ To make sure, I am going to do some more complex things and I think that I have the following code, and I have to do some more complex things, so I can do

> not necessarily trivial. ⏎  ⏎ The second one can be a mathematical function. ⏎  ⏎ Since this is an open set of sets of functions in the same space, and is not a mathematical function. The first one can be an ordinal function of the form a subtraction of the subtraction of

**старые (без top_p)**

> not. It has been noticed that the first part of this program should be aware of a very simple example, but rather the most important case is aware of the second part of this program. ⏎  ⏎ Another important part of this program is an example of an example which uses the same function as the

### chat template

`<|system|>You are KronLLM, a compact bilingual model.<|eot|><|user|>Who are you?<|eot|><|assistant|>`

**новые (top_p=0.9)**

> who, what is what you meaning? entirely, but not to help you!

> who, you all have to be a fair and welcome. You can do that, but they aren't too small for the entire time to get a good idea to understand the world of fairs, so that it might be very much more difficult to think about me, I don't

**старые (без top_p)**

> who, what is there, my architecture? I am not too easy to understand your designers about the nature of the architecture. My architecture, I am not too easy to understand your designers about its nature of the architecture. My architecture, I am not too easy to understand your designers about

## Где сейчас обучение

Последняя сессия закрылась нормально — не аварийно, а по бюджету 500 минут:

| | значение |
|---|---|
| шаг | **35 100** (25.6 % эпохи) |
| время | 8.33 ч, 2.54 с/шаг, 6 452 ток/с |
| legacy val bpb | 0.813 |
| **val bpb mixed** | **1.180** (EN 1.435 / RU 0.934) |
| эталонный пересчёт, 1024 блока | 1.176 |
| для сравнения, GPT-2-124M | 1.554 |

Дальше — **упёрлись в недельную квоту Kaggle**: 30 часов выбраны полностью,
пуш четвёртой сессии отклонён с `Maximum weekly GPU quota of 30.00 hours
reached`. Цепочка не упала: она спит по 3 часа и повторяет пуш (следующая
попытка около 12:20), а квота сбрасывается **в субботу 00:00 UTC, это 03:00
МСК 31 октября**.

Что осталось: ~102 тыс. шагов ≈ 72 GPU-часа ≈ 8.7 сессий, то есть до конца
эпохи примерно девять ночей после сброса квоты.

Отдельно: ночью цепочка отработала не полностью — отчёт и пуш ломались на
четырёх багах после обновления клиента (рабочая папка в job'е, заголовки CSV,
обязательные `--session`/`--model` у мейлера, отказ по квоте выглядел как
«сессия закончилась»). Всё четыре исправлены и закоммичены (`365f643`), так что
первая же сессия после сброса квоты должна пройти без ручного вмешательства.
Отчёты, которые созревают раньше 08:00, теперь не будятся в три ночи, а
складываются в очередь и уходят в 08:00; аварии по-прежнему уходят сразу.

## Что дальше

1. Суббота, 03:00 МСК — первая сессия после сброса квоты; в отчёте 08:00
   проверить, что появились колонки `val_bpb_mixed` и что опубликованный
   чекпоинт новее шага 35 100.
2. В конце эпохи — перемерить декодинг: у обученной модели распределение
   острее, и оптимум обрезки сдвинется (вероятно, в сторону более широкого
   `top_k`, а `top_p` станет осмысленнее).
3. Дальше — decay-проход Q10A (40 649 шагов, LR 3e-4 → 1e-6), затем статья на
   Habr по претрейну, затем SFT/Q&A и вторая статья.

Подробности измерений — в `docs/decoding.md` в репозитории: там метод, таблицы,
и отдельный раздел «что эти числа не показывают».
