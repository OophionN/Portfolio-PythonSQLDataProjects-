Проект "Тестирование замены шрифтов" (Анализ пользовательского поведения в мобильном приложении (Project - Analysis of user behavior in a mobile application))


**Установочное описание задачи**

На основе данных использования мобильного приложения для продажи продуктов питания проанализировать воронку продаж, а также оценить результаты A/A/B-тестирования 


**Основные задачи**

  - Необходимо изучить воронку продаж.
  - Узнайть, как пользователи доходят до покупки.
  - Сколько пользователей доходит до покупки, а сколько — «застревает» на предыдущих шагах? На каких именно?
  - После этого необходимо исследовать результаты A/A/B-эксперимента.
  - Дизайнеры захотели поменять шрифты во всём приложении, а менеджеры испугались, что пользователям будет непривычно. Договорились принять решение по результатам A/A/B-теста.
  - Пользователей разбили на 3 группы: 2 контрольные со старыми шрифтами и одну экспериментальную — с новыми. Выясните, какой шрифт лучше.
  - Создание двух групп A вместо одной имеет определённые преимущества. Если две контрольные группы окажутся равны, вы можете быть уверены в точности проведенного тестирования. Если же между значениями A и A будут существенные различия, это поможет обнаружить факторы, которые привели к искажению результатов. Сравнение контрольных групп также помогает понять, сколько времени и данных потребуется для дальнейших тестов.
  - В случае общей аналитики и A/A/B-эксперимента работайте с одними и теми же данными. В реальных проектах всегда идут эксперименты. Аналитики исследуют качество работы приложения по общим данным, не учитывая принадлежность пользователей к экспериментам.


**Описание данных**

Каждая запись в логе — это действие пользователя, или событие:

  - EventName — название события;
  - DeviceIDHash — уникальный идентификатор пользователя;
  - EventTimestamp — время события;
  - ExpId — номер эксперимента: 246 и 247 — контрольные группы, а 248 — экспериментальная.


**Используемые инструменты и методики**

  - A/B-тестирование
  - Pandas
  - Matplotlib
  - Seaborn
  - событийная аналитика
  - продуктовые метрики
  - Plotly
  - проверка статистических гипотез
  - визуализация данных

--------------------------------

**Проект завершен**

**Вывод** 

По результатам исследования значимой разницы между контрольными группами и тестовой не выявлено, корректировка шрифта существенного эффекта не оказала. Опасения менеджеров не подтвердились, тестирование предлагается признать успешным.

Развернутый вывод и рассуждения в блокноте проекта. 
