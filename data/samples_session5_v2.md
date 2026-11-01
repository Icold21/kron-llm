# Q10: примеры генерации, шаг 36 549 — настройки по вашему запросу

`temperature 1.0`, `top_p 0.95`, `top_k 10`, до 90 новых токенов, seed 42, чекпоинт шага **36 549** (9 938 886 параметров, 26 % эпохи).

Ваше замечание про `top_k 40` при словаре 8 192 верно: это отсекало всего 40 кандидатов из восьми тысяч, то есть практически ничего не делало, и распределение оставалось почти сырым. `top_p 0.95` с `top_k 10` — ограничение осмысленное.

## Вопрос про EOT

В корпусе **1,028,370** токенов EOT (id 8191, конец документа) — ровно по одному на документ, и он **всегда** был целью обучения: `encode_docs` дописывает его в конец каждого документа, а цель — следующий токен, поэтому последний токен документа предсказывает EOT. Так было и в сессиях 1–5, это не новое.

`<|eot|>` (id 8190) — другой токен, конец реплики. В корпусе их **296,661**, и они приходят только из чат-документов (<|system|> — 63,744 штук). В сессии 4 их добавлял ещё и наш вставленный промпт — именно это мы и убрали.

**Но на генерации EOT не появляется.** Измерено по токенам, бюджет 300: **0 из 6** выборок остановились сами, все упёрлись в лимит. При 300 токенах бюджета модель тоже ни разу не выдала EOT. Причина арифметическая: EOT — это 0.044 % всех целей (один на ~2 300 токенов), и на 26 % эпохи такой редкий класс ещё не закрепился. Практический вывод: останавливать генерацию по EOT пока нельзя, нужен `max_new_tokens`.

## Примеры

### Once upon a time  
*EN · художественный*

```
Once upon a time,

and a few years, and we are not aware of that,

and that’s not the same thing,

and the more,

After all, it’s just the same thing’s notion,

And if you’re not, you’ll have been aware of that,

and I’ll be the same thing as your own.

To all,
```

### The derivative of x^2 is  
*EN · математика*

```
The derivative of x^2 is a function with the derivative, and we are not the derivative of x^2, which means that for x\in x\cap\in x\otimes y^2\in x\otimes z^2\otimes y^2\otimes \otimes \cap(x\cap\cap(x\cap\cap(x\cap\cap(x\
```

### def quicksort(arr):  
*EN · код*

```
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        return asts.findsort(arr)


def test_canvass_frame_frame_frame_frame_frame_frame_frame_frame_frame_frame_frame_frame_frame_frame_frame_frame_
```

### The photo of a black hole was made by  
*EN · энциклопедическое*

```
The photo of a black hole was made by a brief moon, and a black hole was the first time a glass of the black hole to the sun's face. The sun was now a black hole, which is called a black hole. In some cases, the black hole had two white holes.

From 2 to 8 of the black hole, and the black hole would be the most massive of its black hole. It is not the case
```

### What is the capital of France?  
*EN · вопрос-ответ*

```
What is the capital of France?
The capital of France is a partner of a
such that this capital is an important partner of an
civilist painter, the person who has been being a
general partner of the world.
A girl who has not been being an important partner of
the world, and the person who has been being a
general partner of the world, was a partner of
the partner of the world
```

### Однажды в маленьком городе  
*RU · художественный*

```
Однажды в маленьком городе, в которой в 1056 килитмами в Балтийском училище было временно захвачено по городам, по которому они находились в северном монастыре.

Устройство

С 1954 года до н. э. в Балтийском училище в северном монастыре, с 22
```

### Производная функции x^2 равна  
*RU · математика*

```
Производная функции x^2 равна, если она не имеет значения, так что она не будет иметь значения, если она является в зависимости от других.

Современные значения
Современные значения
Если же они не делается, то в данной области можно получить значение для того, что все равны. Такое отношение можно быть для того, что они не являются значениями:

Все все равно, а на них не могут быть
```

### Чёрная дыра — это  
*RU · энциклопедическое*

```
Чёрная дыра — это наследственное название герба, который при этом уверен, что в этой местности и как на этом гербе, не имея на этом скотках в этом иных условиях, он не был задумчивым, что наследственное название герба.

Существование герба
Герб был использован в качестве герба для обозначения герба, в котором использовался
```


## Поправка к предыдущему письму

В прошлом письме я написал, что остановились сами 8 из 8 примеров. Это неверно: я сравнивал длину в символах, а при бюджете 90 токенов так не отличить законченный текст от упёршегося в лимит. Правильное измерение — по токенам при бюджете 300: **0 из 6**. Извините за неточность.

## Итог

- язык и орфография держатся, связность — на уровне фразы;
- настроения/фактов нет; код синтаксически похож, но не работает;
- с `top_p 0.95` / `top_k 10` текст стал заметно менее случайным, но и беднее — модель чаще повторяет одну конструкцию.
