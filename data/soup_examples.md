# Генерация: `last.pt` против усреднённых весов

Одинаковые промпты, одинаковые сиды, рецепт T = 1.0, top_k = 8, top_p = 0.9, rp = 1.1. Номер сида — это один и тот же бросок кубика для обеих моделей, поэтому пары сравнимы построчно.

| | bpb судьи EN | bpb судьи RU | циклы | distinct-3 |
|---|---:|---:|---:|---:|
| last.pt (шаг 35 115) | 1.661 | 0.929 | 0.87 | 0.763 |
| soup (50/50, шаг 35 000 + 35 115) | 1.578 | 0.845 | 0.93 | 0.754 |

## `en_prose` — промпт `She opened the door and`

**сид 0**

*last.pt (шаг 35 115):*

```
 went to the stairs.

On the same time he was now a few years later, and he was not to be a faster

and a gaze. In his first place he was in the middle, but he had been in the

```

*soup (50/50, шаг 35 000 + 35 115):*

```
 went to the

glasses, and was still in the house.

"When I saw that I came to my bow, I had been at the door.

"Here was my daughter," I said, "and he
```

**сид 1**

*last.pt (шаг 35 115):*

```
 was opened, the next morning.

The next day, after the night, when she had been opened to the door, he

was closed, and went away from the background of the street and a few minutes,

and then came
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 was a fool.

“I'm not going to do,” she answered.

“Well, my dear mother is a very young maid.”

They stretched into a loud, flicking down the
```

## `en_science` — промпт `The Moon orbits the Earth at a distance of`

**сид 0**

*last.pt (шаг 35 115):*

```
 120 kg.

References

Geographical in the United States
Geography of the South Mountain
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 120 kg.

In the 1940s the Metropolitan River was awarded the Democratic Football League. It was also the first of the 1950s,
```

**сид 1**

*last.pt (шаг 35 115):*

```
 72,640. According to the Salamann-Wangsch theorem and the Dirac-Elliotti equation of the Hamiltonian (1098) and the Hamiltonian (1070).

```

*soup (50/50, шаг 35 000 + 35 115):*

```
 12,600. According to the Schrödinger equation, a series of the earth is given by a given number, and the two terms are the same; the earth's length is 150 degrees (i.
```

## `ru_prose` — промпт `Он открыл дверь и`

**сид 0**

*last.pt (шаг 35 115):*

```
 был уволен, но уже в 1940 году был выставлен на Востоке.

После 2021 года вновь получил своё название, а также в составе Добровольческой Государственной Депар
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 был удостоен своей женитьбы на прежней службе в 1940 году.

Книга была опубликована в 1956 году, когда в 1963 году вышла книга, а 1
```

**сид 1**

*last.pt (шаг 35 115):*

```
 в том числе, что, как и в конце концов, он начал «вотмировать» в свою «собственную и сдержанную работу.
В течение первых 1680 года на 2000 году он переме
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 в 2006 году был приглашён.

В 1977 году в Караганды, где он стал первым заместителем кандидата физических наук в Венеции, было принято решение на 13
```

## `ru_history` — промпт `В 1961 году`

**сид 0**

*last.pt (шаг 35 115):*

```
 была образована Майкл Гаррисон, затем основана в 1957 году в Виленском институте математики (в том числе в Вилимской школе, а также в Ленинградском университете).

В 1
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 была образована Майкл Гаррисон, затем учредившаяся в 1974 году.

Учреждения и развитие
Карьера
Карьера — это один из крупнейших учебных заведений. На
```

**сид 1**

*last.pt (шаг 35 115):*

```
 в Северной Германии было создано 137389 человек (в 1963 г. 5039 человек), но не было 1248 человек. Вскоре после смерти 1964 года 3
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 в городе был переименован в Бразилию, где он работал на Бразилии.

Достопримечательности
Состояние отечественной иконописи — 1000 000 3
```

## `code` — промпт `def add(a, b):\n    return`

**сид 0**

*last.pt (шаг 35 115):*

```
 a.group(b)

# Take a subgroup of a group, but this is a group that contains a group of groups. The group of groups in the group is
    # the groups in which it is the group of groups.
 
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 a.__index__.get('b')


class MetaType:
    """Reply a new class that allows you to create a new class that hasn't been built
    from the middle-size of the old name.

    Reply
```

**сид 1**

*last.pt (шаг 35 115):*

```
 a.get(b, c)
    if b:
        a = b.get(a)
    assert a[0] + b[1]
    assert a[1] + b[2]

    def test_package_failure_s
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 a.get(b, c)
    for i, b in range(b):
        print(b, b), b = c

    assert_equal(c, c, a)
    assert_equal(b, a, b)
    assert_equ
```

## `arith` — промпт `17 + 25 =`

**сид 0**

*last.pt (шаг 35 115):*

```
 11000 # 41000
# # 43100 - 4020 = 21010 # 68100
# # 63000 + 190
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 126 + 25 = 140
1.15 = 100 - 25 = 25 = 130
0.15 = 30 = 30 + 15 = 4
```

**сид 1**

*last.pt (шаг 35 115):*

```
 620 + 620 .
we show that for the @xmath40-bound and for the 25-bits , we show the two different types of @xmath38-bit pairs in fig .
[ fig : b
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 620 + 620.5 = 34968.0 + 150.5 + 30 + 620.5 = 50680.0 + 150.5 + 
```

## `identity` — промпт `KronLLM is`

**сид 0**

*last.pt (шаг 35 115):*

```
 a small family of such that its own, or, if it's not.

This is a simple example of the following:

• If we don't know what the product is, you have to say, the product of the product and the product of the products
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 a small family of such kinds, and there are no real numbers.
• ##### Statistics: 02-23-10-10

The most common example is the first-order method, the first is the first-order
```

**сид 1**

*last.pt (шаг 35 115):*

```
 highly efficient and efficient.
The STRP is designed to operate a large system of STRPs. This is only a few years of development in the US and the STRP is designed to operate a large system of STR
```

*soup (50/50, шаг 35 000 + 35 115):*

```
 a compact function, and it is useful to say if there is a compact function that can be defined in terms of the derivatives.

If you have a particular set of functions, then the product is called a linear function. This is just an elementary function.

```

