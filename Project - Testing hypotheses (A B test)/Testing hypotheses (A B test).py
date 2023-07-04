#!/usr/bin/env python
# coding: utf-8

# # Описание проекта (вводная часть)
# Контекст
# Вы — аналитик крупного интернет-магазина. Вместе с отделом маркетинга вы подготовили список гипотез для увеличения выручки.
# Приоритизируйте гипотезы, запустите A/B-тест и проанализируйте результаты. 

# ## Этап 1. Приоритизация гипотез.
# 
# 
# В файле /datasets/hypothesis.csv 9 гипотез по увеличению выручки интернет-магазина с указанными параметрами Reach, Impact, Confidence, Effort.
# 
# ### Задачи на 1 Этап
#     Применить фреймворк ICE для приоритизации гипотез. Отсортировать их по убыванию приоритета.
#     Применить фреймворк RICE для приоритизации гипотез. Отсортировать их по убыванию приоритета.
#     Указать, как изменилась приоритизация гипотез при применении RICE вместо ICE. Объяснить, почему так произошло.
# 

# ## Этап 2. Анализ A/B-теста
# Вы провели A/B-тест и получили результаты, которые описаны в файлах /datasets/orders.csv и /datasets/visitors.csv.
# 

# ### Задачи на 2 Этап
# 
# 
# Проанализировать A/B-тест:
# 
#     - Построить график кумулятивной выручки по группам. Сделать выводы и предположения.
#     - Построить график кумулятивного среднего чека по группам. Сделать выводы и предположения.
#     - Построить график относительного изменения кумулятивного среднего чека группы B к группе A. Сделать выводы и предположения.
#     - Построить график кумулятивного среднего количества заказов на посетителя по группам. Сделать выводы и предположения.
#     - Построить график относительного изменения кумулятивного среднего количества заказов на посетителя группы B к группе A. Сделать выводы и предположения.
#     - Построить точечный график количества заказов по пользователям. Сделать выводы и предположения.
#     - Посчитать 95-й и 99-й перцентили количества заказов на пользователя. Выбрать границу для определения аномальных пользователей.
#     - Построить точечный график стоимостей заказов. Сделать выводы и предположения.
#     - Посчитать 95-й и 99-й перцентили стоимости заказов. Выберать границу для определения аномальных заказов. 
#     - Посчитать статистическую значимость различий в среднем количестве заказов на посетителя между группами по «сырым» данным. Сделать выводы и предположения.
#     - Посчитать статистическую значимость различий в среднем чеке заказа между группами по «сырым» данным. Сделать выводы и предположения.
#     - Посчитать статистическую значимость различий в среднем количестве заказов на посетителя между группами по «очищенным» данным. Сделать выводы и предположения.
#     - Посчитать статистическую значимость различий в среднем чеке заказа между группами по «очищенным» данным. Сделать выводы и предположения.
#     
#     
# По итогам анализа принять решение по результатам теста и объяснить его. Варианты решений:
# 
# 1. Остановить тест, зафиксировать победу одной из групп.
# 2. Остановить тест, зафиксировать отсутствие различий между группами.
# 3. Продолжить тест.

# **Общий план**

# Перед выполнением любой из задач, в первую очередь необходимо подготовить набор библиотек, познакомится с данными.
# 
# Сам процесс исследования будет проведен в следующем порядке (основные этапы):
# 
#     - загрузка библиотек и знакомство с данными
#     - предобработка данных
#     - формирование временных таблиц, включая новые сводные (если потребуется)
#     - проведение 1 Этапа - приоретизация гипотез ( с использованием фреймворков ICE и RICE)
#     - проведение 2 Этапа (Анализ А/В тестов)
#     - Общие выводы по результатам
# 
# *Итоговая цель анализа* - оценка результатов текущего этапа тестирования, отработка гипотез, сформированных с отделом маркетинга, определение целесообразности проведения дальнейшего тестирования.

# # Подготовка, знакомство и предобработка 

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
from matplotlib.dates import DateFormatter
from matplotlib import colors
import seaborn as sns
get_ipython().system('pip install tabulate')
from tabulate import tabulate
from scipy import stats as st


# In[2]:


try:
    hypothesis = pd.read_csv (r'C:\Users\PC_Maks\Desktop\study\S 13 (11) business solutions\Project\hypothesis.csv')
    orders = pd.read_csv (r'C:\Users\PC_Maks\Desktop\study\S 13 (11) business solutions\Project\orders.csv',                         parse_dates=["date"])
    visitors = pd.read_csv (r'C:\Users\PC_Maks\Desktop\study\S 13 (11) business solutions\Project\visitors.csv',                           parse_dates=["date"])
except:
    hypothesis = pd.read_csv ('/datasets/hypothesis.csv')
    orders = pd.read_csv ('/datasets/orders.csv', parse_dates=["date"])
    visitors = pd.read_csv ('/datasets/visitors.csv', parse_dates=["date"])
    
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)


# ## Знакомство с данными

# In[3]:


display (hypothesis.shape)
display (hypothesis.head ())
display (hypothesis.info ())


# In[4]:


num_columns = len(hypothesis.columns)
for i in range(1, num_columns):
    col_name = hypothesis.columns[i]
    print(f"Максимальное значение в столбце '{col_name}': {hypothesis[col_name].max()}")
    print(f"Минимальное значение в столбце '{col_name}': {hypothesis[col_name].min()}")
    print ()


# In[5]:


display (orders.shape)
display (orders.head ())
display (orders.info ())


# In[6]:


display (orders['date'].max ())
display (orders['date'].min ())


# In[7]:


display (visitors.shape)
display (visitors.head ())
display (visitors.info ())


# In[8]:


display (visitors['date'].max ())
display (visitors['date'].min ())


# In[9]:


display (orders ['group'].unique ())
display (visitors ['group'].unique ())


# По результатам первичного знакомства с данными, предварительно, данные без пропусков, безотлагательных корректировок не требуют. формат даты мы заменили на этапе загрузки таблиц, дополнительная корректировка пока не требуется.
# 
# Даты корректны, период с 1 по 31 августа 2019 года.
# В группах только 2 значения.
# 

# ## Отработка пропусков, дубликатов

# стоит проверить на явные дубликаты и итог на пропуски

# In[10]:


print (f' Явные дублика во всех строках и столбцах {orders.duplicated ().sum ()}')
print ()
duplicates = orders.duplicated()
duplicate_rows = orders.loc[duplicates]
print(duplicate_rows.info())
print(duplicate_rows)


# In[11]:


print (f' Явные дублика во всех строках и столбцах {visitors.duplicated ().sum ()}')
print ()
duplicates = visitors.duplicated()
duplicate_rows = visitors.loc[duplicates]
print(duplicate_rows.info())
print(duplicate_rows)


# In[12]:


display (hypothesis.isna ().sum ())
display (orders.isna ().sum ())
display (visitors.isna ().sum ())


# In[13]:


orders_a = orders.query ('group=="A"')
orders_b = orders.query ('group=="B"')
duples = orders_a.merge (orders_b, on='visitorId')
duples ['visitorId'].value_counts ().count ()


# In[14]:


orders [orders['visitorId']==2378935119]


# ## Предварительно посмотрим на выбросы
# 
# В группе В есть аномально высокие значения в категории revenue, но детальный анализ и решение о дальнейших действиях будем принимать в ходе проведения основного анализа. Прим. Данная оценка для внутреннего использования, не для презентации

# In[15]:


orders_group = orders.groupby('group', as_index=False)['revenue'].agg({'revenue': 'sum'})

orders_group

group_counts = orders_group.groupby('group').sum()


total_count = group_counts['revenue'].sum()


group_counts['percentage'] = group_counts['revenue'] / total_count * 100


difference = group_counts.loc['B', 'percentage'] - group_counts.loc['A', 'percentage']

print(group_counts)
print(f'Группа B на {difference:.2f}% больше группы A')


# In[16]:


display(orders['revenue'].describe())
plt.figure(figsize=(20,10))

plt.grid(axis='y', alpha=1)

sns.boxplot(x='group', y='revenue', data=orders, palette='Spectral')
plt.title('Диаграмма размаха. Доходы по группам', fontsize=16)
plt.xlabel('группы', fontsize=14)
plt.ylabel('доходы', fontsize=14)
plt.show()


# Проведем аналогичную процедуру в части оценки на выбросы в файле visitors, в части количество посетителей.
# Предварительно, аномальных выбросов нет, группы в общем и целом близки по размерам, но стоит помнить, что они совсем равные

# In[17]:


visitors_group = visitors.groupby('group', as_index=False)['visitors'].agg({'visitors': 'sum'})

visitors_group

group_counts = visitors_group.groupby('group').sum()


total_count = group_counts['visitors'].sum()


group_counts['percentage'] = group_counts['visitors'] / total_count * 100


difference = group_counts.loc['B', 'percentage'] - group_counts.loc['A', 'percentage']

print(group_counts)
print(f'Группа B на {difference:.2f}% больше группы A')


# In[18]:



display(visitors['visitors'].describe())
plt.figure(figsize=(20,10))

plt.grid(axis='y', alpha=1)

sns.boxplot(x='group', y='visitors', data=visitors, palette='Spectral')
plt.title('Диаграмма размаха. Доходы по группам', fontsize=16)
plt.xlabel('группы', fontsize=14)
plt.ylabel('доходы', fontsize=14)
plt.show()


# Обобщение по результатам знакомства:
# 
#     - аномалий не выявлено
#     - пропусков не выявлено
#     - значения в части определения параметров таблиц корректны (2 группы)
#     - в таблице с гипотезами используется диапазон значений от 1 до 10, ошибок (меньше 0 или больше 10) нет
#     - временной интервал идентичный - с 1 по 31 августа 2019 года
#     - есть выбросы в суммах заказа в группе В, а в процентном соотношении в группе В на 18.24% общая сумма покупок больше, чем в группе А
#     - размер выборок групп по количестве покупателей/посетителей примерно равны, но не идентичны, разница 0.48% - принимаем за некритичную разницу.
#     - 58 клиентов попали сразу в обе группы, некоторые клиенты были очень активными и совершили более 10 заказов за месяц, но для анализа данные попадания в обе группы не критичны.

# # Этап 1.

# ## Приоритизация гипотез
# 
# ### Используем фреймворк ICE для приоритизации гипотез. Отсортируем их по убыванию приоритета.

# In[19]:


hypothesis


# ice score = (Impact*Confidence)/Efforts

# In[20]:


hypothesis ['ICE'] = (hypothesis['Impact']*hypothesis ['Confidence'])/hypothesis ['Efforts']
hypothesis['ICE'] = hypothesis['ICE'].round(2) 
display (hypothesis [['Hypothesis', 'ICE']].sort_values (by='ICE', ascending = False).head (10))


# По результатам расчетам скоринга веса гипотез, выделяются три лидера - гипотезы под номерами:
# 
#     - №8 - Запустить акцию, дающую скидку на товар в день рождения
#     - №0 - Добавить два новых канала привлечения трафика, что позволит привлекать на 30% больше пользователей
#     - №7 - Добавить форму подписки на все основные страницы, чтобы собрать базу клиентов для email-рассылок

# ### Используем фреймворк RICE для приоритизации гипотез. Отсортируем их по убыванию приоритета.

# In[21]:


hypothesis ['RICE'] = (hypothesis['Reach']*hypothesis['Impact']*hypothesis ['Confidence'])/hypothesis ['Efforts']
hypothesis['RICE'] = hypothesis['RICE'].round(2) 
display (hypothesis [['Hypothesis', 'RICE']].sort_values (by='RICE', ascending = False).head (10))


# **Вывод** 
# 
# По результатам расчета скорингового балла RICE лидеры немного скорректированы, теперь это:
#     
#     - № 7 - Добавить форму подписки на все основные страницы, чтобы собрать базу клиентов для email-рассылок
#     - № 2 - Добавить блоки рекомендаций товаров на сайт интернет магазина, чтобы повысить конверсию и средний чек заказа	
#     - № 0 - Добавить два новых канала привлечения трафика, что позволит привлекать на 30% больше пользователей 
#     
#     
# Гипотеза № 8 (Запустить акцию, дающую скидку на товар в день рождения) потеряла позиции, так как имеет небольшой охват, и касается только именинников среди клиентов, а новообретенные "лидеры" коснутся всех клиентов, что соответствует значению Reach  в гипотезе № 8=1, а гипотеза № 2, несмотря на низкое значение влияния (всего 3), имеет высокий охват и низкий уровень сложности реализации, что с высокой долей вероятности позволит ее проверить значительно быстрее.  
# Учитывая результаты расчета скорингового балла, рекомендации по тестированию будут отражать необходимость отработки 7 гипотезы, и возможно 2-ой
# 

# # Этап 2

# ## График кумулятивной выручки по группам. 

# Создадим массив уникальных пар значений дат и групп теста методом drop_duplicates(): 

# In[22]:


datesGroups = orders[['date','group']].drop_duplicates() 


# In[23]:


ordersAggregated = datesGroups.apply(lambda x: orders[np.logical_and(orders['date'] <= x['date'],                                                                     orders['group'] == x['group'])].                                     agg({'date' : 'max',
                                          'group' : 'max',
                                          'transactionId' : 'nunique',
                                          'visitorId' : 'nunique',
                                          'revenue' : 'sum'}),
                                     axis=1).sort_values(by=['date','group'])


# In[24]:


# получаем агрегированные кумулятивные по дням данные о посетителях интернет-магазина 
visitorsAggregated = datesGroups.apply(
    lambda x: visitors[np.logical_and(visitors['date'] <= x['date'],\
                                      visitors['group'] == x['group'])].\
    agg({'date' : 'max', 'group' : 'max', 'visitors' : 'sum'}),\
    axis=1).sort_values(by=['date','group']
)


# In[25]:


# объеденим кумулятивные данные в одной таблице и присвоим ее столбцам новые названия
cumulativeData = ordersAggregated.merge(visitorsAggregated, left_on=['date', 'group'], right_on=['date', 'group'])
cumulativeData.columns = ['date', 'group', 'orders', 'buyers', 'revenue', 'visitors']


# ### датафрейм с кумулятивным количеством заказов и кумулятивной выручкой по дням в группе А

# In[26]:


cumulativeRevenueA = cumulativeData[cumulativeData['group']=='A'][['date','revenue', 'orders']]


# ### датафрейм с кумулятивным количеством заказов и кумулятивной выручкой по дням в группе B

# In[27]:


cumulativeRevenueB = cumulativeData[cumulativeData['group']=='B'][['date','revenue', 'orders']]
cumulativeRevenueB


# ### Строим графики 2-х выборок 

# In[28]:


plt.figure(figsize=(10, 6))
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue'], label='A', color='blue', alpha=0.8)
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue'], label='B', color='orange', alpha=0.8)
plt.grid(True)
plt.title("Кумулятивная выручка по группам")
plt.xlabel("Дата")
plt.ylabel("Выручка")
date_form = DateFormatter("%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)
plt.legend(loc='upper left')
plt.show()


# **Вывод**
# 
# Как видно из графика, примерно до 5-го августа кумулятивная выручка примерно одинаково растет, как в группе А, так и в группе В. Однако, начиная с 13 августа, группа В показывает куда более существенный рост. В районе 18-19 авугста наблюдается резкий скачок в группе В, с последующим равномерным ростом. Группа А, также увеличивается со временнем (все таки расчет идет кумулятивной выручки), но заметно меньшими темпами, чем группа В.
# 
# Стоит сразу разобраться, что произошло в группе В в период 18-19 августа

# In[29]:


# сделаем срез данных нашей сгруппированной таблицы по периоду 18-19 августа
cumulativeRevenueB_18_19 = cumulativeRevenueB.query('date >= "2019-08-18" and date <= "2019-08-19"')
diff = (cumulativeRevenueB_18_19.loc[37, 'revenue'] - cumulativeRevenueB_18_19.loc[35, 'revenue']) / cumulativeRevenueB_18_19.loc[35, 'revenue'] * 100

display (cumulativeRevenueB_18_19)
display (diff)


# в кумулятивной выруче скачок на 53%, выброс, аномально крупный заказ/заказы? посмотрим внимательнее на 19 августа

# In[30]:


orders_19_В = orders.query ('date=="2019-08-19" and group=="B"')
display (orders_19_В.sort_values (by = 'revenue', ascending=False))
display (orders_19_В[orders_19_В['revenue'] != 1294500]['revenue'].mean())


# Вы видим очень большой заказ на сумму 1 294 500, хотя за весь период среднее значение составило 8348,  за 19 августа меньше - 6771. Пока расчеты и корректировки делать не будем, так как у нас впереди будет обособленный анализ "очищенных" данных, но мы уже понимаем, что в группе В есть аномально крупные заказы.

# ## График кумулятивного среднего чека по группам. 

# In[31]:


plt.figure(figsize=(10, 6))
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue']/cumulativeRevenueA['orders'], label='A')
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue']/cumulativeRevenueB['orders'], label='B')
plt.grid(True)
plt.title("Средний чек по группам")
plt.xlabel("Дата")
plt.ylabel("Средний чек")
date_form = DateFormatter("%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)
plt.legend(loc='upper left')
plt.show()


# На графике мы опять наблюдаем за скачком среднего чека в отмеченный ранее день - 19 августа, но даже без "аномалии", группа В показывает больший рост начиная с 15 августа. Стоит отметить, что ранее на общей сумме выручки мы отмечали, что после 13-го августа группа В показывала стабильно более высокий рост выручки, но на среднем чеке мы видим, что 13-го августа контрольная группа вырывалась вперед, данные всплески мы оценим в следующих этапах анализа.
# 
# В части скачка группы В, однозначного выхода на плато мы не наблюдаем, наоборот есть некоторое падение среднего чека после скачка, что логично, если принять за допущение, что скачок произошел по причине аномально крупного заказа, и "истинный" средний чек может себя показать, или после чистки данных, или после большего периода наблюдений.  
# 

# ## График относительного изменения кумулятивного среднего чека группы B к группе A

# In[32]:


mergedCumulativeRevenue = cumulativeRevenueA.merge(cumulativeRevenueB, left_on='date',
                                                   right_on='date',
                                                   how='left',
                                                   suffixes=['A', 'B'])
plt.figure(figsize=(10, 6))
plt.plot(mergedCumulativeRevenue['date'],
         (mergedCumulativeRevenue['revenueB']/mergedCumulativeRevenue['ordersB'])/\
         (mergedCumulativeRevenue['revenueA']/mergedCumulativeRevenue['ordersA'])-1)
plt.axhline(y=0, color='red', linestyle='--') 
plt.grid(True)
plt.title("График относительного изменения кумулятивного среднего чека")
plt.xlabel("Дата")
plt.ylabel("отношение")
date_form = DateFormatter("%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)

plt.show()


# Группа В, после 15 августа показывает стабильно большее значение, всплеск 19 августа виден и тут конечно, но пока общая тенденция просматривается- показатели группы В значительно лучше.

# ## График кумулятивного среднего количества заказов на посетителя по группам. 

# In[33]:


cumulativeData['convers'] = ((cumulativeData['orders']/cumulativeData['visitors'])*100).round (4)


# In[34]:


plt.figure(figsize=(10, 6))
cumulativeDataA = cumulativeData [cumulativeData['group']=='A']
cumulativeDataB = cumulativeData [cumulativeData['group']=='B']
plt.plot (cumulativeDataA ['date'], cumulativeDataA ['convers'], label='A')
plt.plot (cumulativeDataB ['date'], cumulativeDataB ['convers'], label='B')
plt.grid(True)
plt.title("Кум.среднее количество заказов по группам")
plt.xlabel("Дата")
plt.ylabel("Среднее количество заказов")
date_form = DateFormatter("%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)
plt.legend(loc='upper right')
plt.show()


# Если в общем и целом, обе группы показывают сопостовимые показатели среднего количества заказов на посетителя, в диапазоне от 3,0 до 3,7, да, значение и в группе А и в группе В, меньше 4, но если опираться на конкретные цифры, показатель среднего количества заказов в группе В лучше, в период с 1 по 5 августа обе группы показывают всплеск, причем группа А показывает всплеск почти до 3.7, но уже после 6 августа группы меняются местами, и группа В имеет стабильный рост количества заказов на посетителя с выходом на потенциальное плато в районе 3,3-3,4, а группа А наоборот проседает и закрепляется около 3,0

# ## График относительного изменения кумулятивного среднего количества заказов на посетителя группы B к группе A.

# In[35]:


mergedCumulativeconvers = cumulativeDataA.merge(cumulativeDataB, left_on='date',
                                                   right_on='date',
                                                   how='left',
                                                   suffixes=['A', 'B'])
plt.figure(figsize=(10, 6))
plt.plot(mergedCumulativeconvers['date'],
         (mergedCumulativeconvers['ordersB']/mergedCumulativeconvers['visitorsB'])/\
         (mergedCumulativeconvers['ordersA']/mergedCumulativeconvers['visitorsA'])-1)
plt.axhline(y=0, color='red', linestyle='--')
plt.axhline(y=0.1, color='black', linestyle='--')
plt.grid(True)
plt.title("График относительного изменения кум. среднего количества заказов")
plt.xlabel("Дата")
plt.ylabel("отношение")
date_form = DateFormatter("%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)

plt.show()


# Как и отмечалось на прошлом этапе, начиная с 6 августа кум.сред.количество заказов группы В значительно лучше контрольной, после указанной даты, отношение переходит границу 0, и далее к концу наблюдаемого периода закрепляется в диапазоне 0,10-0,15, без тенденции на снижение. 

# ## Точечный график количества заказов по пользователям

# In[36]:


ordersByUsers = orders.groupby ('visitorId', as_index=False).agg ({'transactionId':'nunique'})
ordersByUsers.columns = ['userId', 'orders']
display (ordersByUsers.sort_values (by ='orders', ascending=False).head(10))


# In[37]:


x_values = pd.Series(range(0, len(ordersByUsers)))
plt.figure(figsize=(10, 6))
plt.scatter(x_values, ordersByUsers['orders'], marker='o', s=25, color='blue')
plt.grid(True)
plt.title("Точечный график количества заказов", fontsize=16)

plt.ylabel("Количество заказов", fontsize=14)

z = np.polyfit(x_values, ordersByUsers['orders'], 1)
p = np.poly1d(z)
plt.plot(x_values,p(x_values),"r--", linewidth=2)



plt.rcParams.update({'font.size': 14})
plt.show()


# Подавляющее большинство клиентов делает только 1 заказ(линия тренда на графике), стоит отметить, что имеется слабый тренд на смещение среднего количества заказов в сторону 2-х (очень малый). Часть клиентов делает более 1-го заказа, но после значения 2 - такие заказы единичны. Имеются и аномально активные клиенты с 8 и более заказов на клиента

# ## 95-й и 99-й перцентили количества заказов на пользователя.

# In[38]:


percentiles = np.percentile(ordersByUsers['orders'], [95,96,97,98,99])
display('Процентили количества заказов на одного кользователя 95,96,97,98,99:', percentiles)

percentiles_98 = np.percentile(ordersByUsers['orders'], [98])
display('Процентили количества заказов на одного кользователя - 98%:', percentiles_98)


# 95% пользователей вошли в 2 заказа, 99% включают пользователей с заказами до 4-х, но мы видим, что для 98% граница в районе 3-х заказов на пользователя. Оставляет для расчетов 98%

# ## Точечный график стоимостей заказов.

# In[39]:


x_values = pd.Series(range(0, len(orders)))
plt.figure(figsize=(10, 6))
plt.scatter(x_values, orders['revenue'], marker='o', s=25, color='blue')
plt.grid(True)
plt.title("Точечный график стоимости заказов", fontsize=16)

plt.ylabel("Стоиомсть заказов", fontsize=14)

z = np.polyfit(x_values, orders['revenue'], 1)
p = np.poly1d(z)
plt.plot(x_values,p(x_values),"r--", linewidth=2)



plt.rcParams.update({'font.size': 14})
plt.show()


# Как уже отмечалось ранее подавляющее большинство заказов находится в пределах 0,1 (читать 100 000), а общая тенденция в стоимости заказов еще ниже - около 0,01 (читать 10 000), имеются аномально большие заказы, которые ранее мы уже отмечали на прошлых этапах анализа, в частности был отмечен заказ на сумму около 1 250 000.  Посмотрим на этот же параметр без отмеченных на графике аномалий

# In[40]:


orders_drop_anomaly = orders.query ('revenue <= 150000')


# In[41]:


x_values = pd.Series(range(0, len(orders_drop_anomaly)))
plt.figure(figsize=(10, 6))
plt.scatter(x_values, orders_drop_anomaly['revenue'], marker='o', s=25, color='blue')
plt.grid(True)
plt.title("Точечный график стоимости заказов", fontsize=16)

plt.ylabel("Стоиомсть заказов", fontsize=14)

z = np.polyfit(x_values, orders_drop_anomaly['revenue'], 1)
p = np.poly1d(z)
plt.plot(x_values,p(x_values),"r--", linewidth=2)



plt.rcParams.update({'font.size': 14})
plt.show()


# Теперь график стал более наглядный, мы видим, что ранее осуществленные расчеты, согласно которым средняя стоимость заказов около 8 000 рублей, нашел свое подтверждение в графике (линия тенденции)

# ## 95-й и 99-й перцентили стоимости заказов

# In[42]:


percentiles = np.percentile(orders['revenue'], [95,96,97,98,99])
display('Процентили количества заказов на одного кользователя 95,96,97,98,99:', percentiles)
percentiles_99 = np.percentile(orders['revenue'], [99])
display('Процентили количества заказов на одного кользователя 95,96,97,98,99:', percentiles_99)


# В диапазон до 95% вошли заказы с стоимостью до 28000, 98 процентов заказов в диапазоне до 58322. Учитывая результаты анализа графика и определения процентелей, останавливаемся на 99%

# ## Статистическая значимость различий в среднем количестве заказов на посетителя между группами по «сырым» данным.

# **Вводные**
# 
# В расчетах используется непараметрический тест Уилкоксона-Манна-Уитни, с оценкой при уровне значимости 0,05

# Мы уже отмечали на прошлых этапах, что наблюдаем выбросы, но для определений их значимости необходимо провести дополнительное исследование.
# 
# Для начала, сформулируем гипотезы. **Нулевая: различий в среднем количестве заказов между группами нет. Альтернативная: различия в среднем между группами есть.**
# 
# 

# In[43]:


visitorsADaily = visitors[visitors['group'] == 'A'][['date', 'visitors']]
visitorsADaily.columns = ['date', 'visitorsPerDateA']


# In[44]:


visitorsBDaily = visitors[visitors['group'] == 'B'][['date', 'visitors']]
visitorsBDaily.columns = ['date', 'visitorsPerDateB']


# In[45]:


ordersByUsersA = (
    orders[orders['group'] == 'A']
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.nunique})
)
ordersByUsersA.columns = ['userId', 'orders'] 

ordersByUsersB = (
    orders[orders['group'] == 'B']
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.nunique})
)
ordersByUsersB.columns = ['userId', 'orders'] 


# Объявим переменные sampleA и sampleB

# In[46]:


sampleA = pd.concat([ordersByUsersA['orders'],pd.Series(0, index=np.arange(visitorsADaily['visitorsPerDateA'].sum() - len(ordersByUsersA['orders'])), name='orders')],axis=0)

sampleB = pd.concat([ordersByUsersB['orders'],pd.Series(0, index=np.arange(visitorsBDaily['visitorsPerDateB'].sum() - len(ordersByUsersB['orders'])), name='orders')],axis=0) 


# In[47]:


print  ('alpha =',0.05)
display ("{0:.3f}".format(st.mannwhitneyu(sampleA, sampleB)[1]))

display ("{0:.3f}".format(sampleB.mean() / sampleA.mean() - 1))


# **Вывод**
# 
# По "сырым" данным наблюдается статистическая значимая разница, так как p-value меньше 0,05. Относительный выигрыш группы В к группе А около 13,8%.
# 
# нулевую гипотезу отвергаем, статистическая значимость различий по "сырым" данным есть.
# 

# ## Статистическую значимость различий в среднем чеке заказа между группами по «сырым» данным

# **Нулевая: различий в среднем чеке заказа между группами нет. Альтернативная: различия в среднем чеке между группами есть.**

# In[48]:


print  ('alpha =',0.05)
display ('{0:.3f}'.format(st.mannwhitneyu(orders[orders['group']=='A']['revenue'],                                             orders[orders['group']=='B']['revenue'])[1]))
display ('{0:.3f}'.format(orders[orders['group']=='B']['revenue'].mean()/orders[orders['group']=='A']['revenue'].mean()-1)) 


# Результат теста - p-value больше 0,05 - нулевую гипотезу не отвергаем. 

# ## Статистическая значимость различий в среднем количестве заказов на посетителя между группами по «очищенным» данным

# **Нулевая гипотеза: различий в конверсии между группами нет. Альтернативная гипотеза: различия в конверсии между группами есть.**

# Ранее мы отпределили границы аномалий в части количества и суммы заказов в 3 и 58234 соответсвенно

# In[49]:


usersWithManyOrders = pd.concat(
    [
        ordersByUsersA[ordersByUsersA['orders'] >= percentiles_98[0]]['userId'],
        ordersByUsersB[ordersByUsersB['orders'] >= percentiles_98[0]]['userId'],
    ],
    axis=0,
)
usersWithExpensiveOrders = orders[orders['revenue'] > percentiles_99[0]]['visitorId']
abnormalUsers = (
    pd.concat([usersWithManyOrders, usersWithExpensiveOrders], axis=0)
    .drop_duplicates()
    .sort_values()
)
display (abnormalUsers.head(5))
display (abnormalUsers.shape[0]) 


# всего 31 аномальный пользователь
# 
# Оценим различия по "очищенным данным"

# In[50]:


sampleAFiltered = pd.concat(
    [
        ordersByUsersA[
            np.logical_not(ordersByUsersA['userId'].isin(abnormalUsers))
        ]['orders'],
        pd.Series(
            0,
            index=np.arange(
                visitorsADaily['visitorsPerDateA'].sum() - len(ordersByUsersA['orders'])
            ),
            name='orders',
        ),
    ],
    axis=0,
)

sampleBFiltered = pd.concat(
    [
        ordersByUsersB[
            np.logical_not(ordersByUsersB['userId'].isin(abnormalUsers))
        ]['orders'],
        pd.Series(
            0,
            index=np.arange(
                visitorsBDaily['visitorsPerDateB'].sum() - len(ordersByUsersB['orders'])
            ),
            name='orders',
        ),
    ],
    axis=0,
) 
print  ('alpha =',0.05)
display ('{0:.3f}'.format(st.mannwhitneyu(sampleAFiltered, sampleBFiltered)[1]))
display ('{0:.3f}'.format(sampleBFiltered.mean()/sampleAFiltered.mean()-1)) 


# Результаты по среднему количеству заказов изменились. Нулевую гипотезу отвергаем, выгрыш группы В после очистки данных, по сравнению с группой А около 17,4%

# ## Статистическая значимость различий в среднем чеке заказа между группами по «очищенным» данным.

# In[51]:


print  ('alpha =',0.05)
display (
    '{0:.3f}'.format(
        st.mannwhitneyu(
            orders[
                np.logical_and(
                    orders['group'] == 'A',
                    np.logical_not(orders['visitorId'].isin(abnormalUsers)),
                )
            ]['revenue'],
            orders[
                np.logical_and(
                    orders['group'] == 'B',
                    np.logical_not(orders['visitorId'].isin(abnormalUsers)),
                )
            ]['revenue'],
        )[1]
    )
)

display (
    "{0:.3f}".format(
        orders[
            np.logical_and(
                orders['group'] == 'B',
                np.logical_not(orders['visitorId'].isin(abnormalUsers)),
            )
        ]['revenue'].mean()
        / orders[
            np.logical_and(
                orders['group'] == 'A',
                np.logical_not(orders['visitorId'].isin(abnormalUsers)),
            )
        ]['revenue'].mean()
        - 1
    )
) 


# Результат: Нулевую гипотезу не отвергаем, значение p-value больше 0,05, проигрыш группы В, после очистки данных, в среднем чеке составляет около 2%

# # Выводы и рекомендации

# По итогам изучения результатов теста, получены следующие данные:
# 
# 1.	На «сырых» данных:
# 
#     a.	Отказ от основной гипотезы в пользу альтернативной гипотезы – имеются статистически значимые различия в среднем количестве заказов между группами, выигрыш группы В составляет 13,8% 
#     b.	Основную гипотезу не отвергаем – статистически значимых различий в среднем чеке нет, хоть выигрыш группы В по среднему чеку составляет 25,9%, но в ходе исследования пришли к пониманию, что данные значения получены благодаря случайному выбросу в виде аномально большого заказа.
#     
# 2.	На «очищенных» данных:
# 
#     a.	Отказ от основной гипотезы в пользу альтернативной гипотезы – имеются статистически значимые различия в среднем количестве заказов между группами, выигрыш группы В составляет 17,4% 
#     
#     b.	Основную гипотезу не отвергаем – статистически значимых различий в среднем чеке нет.
# 
#     Как можем увидеть, после очистки от аномальных значений, оценка гипотез не изменилась, а вот результаты отношений претерпели существенные изменения.
#     
#     Так, после очистки уровень конверсии в группе В стал 17,4% против 13,8% до.
#     
#     Средний чек, после очистки сильно стал меньше в группе В, до очистки разница была 25,9%, а после очистки -2%. Данный момент предполагался на всем этапе работы с данными, так как были выявлены аномально большие заказы, в частности заказ на сумму более 1,294 млн., против среднего чека около 8 348. 
# 
# **Итог:** Проведение теста рекомендуется остановить, по результатам тестирования - группа В показала себя значительно более эффективной, с выигрышем в 17,4% в части среднего количества заказов на посетителя. Таким образом, на каждого посетителя из группы В приходится на 17.4% больше заказов чем на каждого посетителя из группы А, что в свою очередь может увеличить общую выручку относительно посетителей из группы В.
