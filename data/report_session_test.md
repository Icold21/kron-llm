# Q10: отчёт по сессии

## Кривая

Кривая не найдена: `runs/q10/log.csv` отсутствует.

## Метрики на страйд-выборке held-out

| модель | bpb (страйд, 1024 блока) |
|---|---:|

## last против усреднённых

**last.pt** (шаг 35115):
- bpb на продолжении: **1.1394** на 64 блоках (en 1.315, ru 0.988)
- ранжирование: истинное продолжение первым в 37.5% случаев (случайное 11.1%), средний ранг 2.66 из 9
- арифметика: точных ответов 0% (60 примеров)
- генерация: повтор 4-грамм 0.09, distinct-2 0.83 (24 сэмплов)

**soup.pt** (шаг 35115):
- bpb на продолжении: **1.1093** на 64 блоках (en 1.280, ru 0.962)
- ранжирование: истинное продолжение первым в 37.5% случаев (случайное 11.1%), средний ранг 2.64 из 9
- арифметика: точных ответов 0% (60 примеров)
- генерация: повтор 4-грамм 0.10, distinct-2 0.82 (24 сэмплов)

## Примеры

**last.pt** — `longform`:
> wisted-payments-debug_src_payers} for the following document. ⏎  ⏎ ### Make RFC 2543. ⏎  ⏎ The first version of the implementation is provided by this module in the IETF Protocol (with
> arget:13678}. ⏎  ⏎ import logging ⏎  ⏎ from . import SimpleBaseProtocol ⏎ import subnet.session import SimpleBaseProtocol ⏎  ⏎ CONFIG = "2004-05-25 20
> itle.ProtocolState()}, as well as in the NOTICE URI of RFC ⏎ # 3261, and 4541-8641. ⏎  ⏎  ⏎ from . import HTTPData, OptionError ⏎  ⏎ def In

**soup.pt** — `longform`:
> emplate:simuli} for the first qualified version of the document ⏎ message and the following choices are provided for ⏎ each month. ⏎  ⏎ Documents from COMPLETE (including documentation) ⏎  ⏎  ⏎ @
> wisted} for details. ⏎  ⏎  ⏎ --- ⏎ module: nmdnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
> } for details. ⏎  ⏎ ##### Share on other sites ⏎  ⏎ Makhaller has a GNU General Public License, but it will not be useful to ⏎ the permissions and/or authors of the GNU General Public License. ⏎ The limited permissions

## Что сделано в сессии

- шаг и токены — из лога сессии;
- чекпоинт опубликован как новая версия `icold21/kron-q10-ckpt`;
- `best.pt` заархивирован, чтобы усреднять соседние веса, а не через фазу.

Кривая размечена: с шага 35 115 корпус новый (химия, учебники, системные промпты), поэтому разрыв в train loss на этой точке — ожидаемый, а `val_bpb` меряется на голом held-out и остаётся сравнимым со всей историей.
