# Q10: примеры генерации, шаг 36 549 — temp 1.0, top_p 0.9, top_k 40

`temperature 1.0`, `top_p 0.9`, `top_k 40`, до 90 новых токенов, seed 42. Чекпоинт шага **36 549**, 9 938 886 параметров, 26 % эпохи.

Проверка остановки по EOT при бюджете 300 токенов: **0 из 6** выборок остановились сами — то есть EOT модель по-прежнему не выдаёт, и генерацию надо ограничивать `max_new_tokens`.

## Примеры

### Once upon a time  
*EN · художественный*

```
Once upon a time, where he was a
such that he would be a bed with a glass of
canvard brown hair.

His face, he could hardly be found to himself, and we might
break at any point or to it, and,
for he was not a man, he thought to be of his own. But the
human
husband would have seen
```

### The derivative of x^2 is  
*EN · математика*

```
The derivative of x^2 is a function with the derivative on x^2. Then the derivative is called the vector space with respect to x^2 . For example, a linear operator, with the same eigenvalues of y^2, is called the Liebremand function of the Liebremand function. This makes a special direction to find that in this case, it takes the first step to find that the Liebremand function is
```

### def quicksort(arr):  
*EN · код*

```
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        return asts.images[arr]
    try:
        return asts.images[arr]


def make_filters(latest, x, y):
    """Set of a Latest x in the y direction
    """
    line = y_elevation
    line = y_elevation
    line = y_line
    line = line

    lines = line
```

### The photo of a black hole was made by  
*EN · энциклопедическое*

```
The photo of a black hole was made by Riabati Richard Harley Barney Rosaud, which later he was declared to the Louisiano military.

Early life
Army was born on 6 December 1969, where he had a population of 7,200,800,600. After the fleet of the D
```

### What is the capital of France?  
*EN · вопрос-ответ*

```
What is the capital of France?
The capital of France is a partner of a
syncial account and a number of guys
Apart in France is the second to the United
Education, which has not always
published to give the capital for
breaking the account in France, and the policies of
education and ability to the United
Austrians are of interest.
The fact that these accounts
```

### Однажды в маленьком городе  
*RU · художественный*

```
Однажды в маленьком городе, переименовал к ним, и примкнул к государя, бледный, спокойно, чуть позже, по которому они и отдавались мне. Тот вон очаровывает, не только гребнулась, и примкнул к ним: гребнулись утолщики с голубым слеом, подозревки
```

### Производная функции x^2 равна  
*RU · математика*

```
Производная функции x^2 равна k , где можно привести к нему. При этом снаружи и дозирования этого порядка на этом пределе нет, чтобы это можно сделать в полотно-конкурендах. Здесь мы вторичная функция y^2 равна k, где можно сделать к нему. Приведите к гемброкеровым и хроническим условиям к вторичной функция y^
```

### Чёрная дыра — это  
*RU · энциклопедическое*

```
Чёрная дыра — это пыль, кроме дыра. Причина — это самая дыра — и каким-либо дыра. Внутри утром, в свою очередь, как и в нижнем чёрном дыре, но не только гребнеется, и при этом можно придать к гребне.

Только и с приветом, а так как с нелеп
```

## Наблюдение

- с `top_k 40` и `top_p 0.9` повторов меньше, чем при `top_k 10`, но фразы длиннее и рыхлее;
- язык и орфография держатся, содержание — нет;
- код сохраняет форму (`def`, отступы, докстринги), но не работает.
