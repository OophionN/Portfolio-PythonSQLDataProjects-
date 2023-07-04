#!/usr/bin/env python
# coding: utf-8

# # Проект "Рынок заведений общественного питания Москвы" 
# Инвесторы из фонда «Shut Up and Take My Money» решили попробовать себя в новой области и открыть заведение общественного питания в Москве. Заказчики ещё не знают, что это будет за место: кафе, ресторан, пиццерия, паб или бар, — и какими будут расположение, меню и цены.
# В этой связи, нам необходимо подготовить исследование рынка Москвы, найти интересные особенности и презентовать полученные результаты, которые в будущем помогут в выборе подходящего инвесторам места.
# 
# В анализе используется датасет с заведениями общественного питания Москвы, составленный на основе данных сервисов Яндекс Карты и Яндекс Бизнес на лето 2022 года. Информация, размещённая в сервисе Яндекс Бизнес, могла быть добавлена пользователями или найдена в общедоступных источниках (могут иметь место неточности и некорректные данные). 
# 

# # Описание данных
# 
# Предоставленный датасет содержит следующие сведения:
#     
#     - name — название заведения;
#     - address — адрес заведения;
#     - category — категория заведения, например «кафе», «пиццерия» или «кофейня»;
#     - hours — информация о днях и часах работы;
#     - lat — широта географической точки, в которой находится заведение;
#     - lng — долгота географической точки, в которой находится заведение;
#     - rating — рейтинг заведения по оценкам пользователей в Яндекс Картах (высшая оценка — 5.0);
#     - price — категория цен в заведении, например «средние», «ниже среднего», «выше среднего» и так далее;
#     - avg_bill — строка, которая хранит среднюю стоимость заказа в виде диапазона, например:
#         
#         - «Средний счёт: 1000–1500 ₽»;
#         - «Цена чашки капучино: 130–220 ₽»;
#         - «Цена бокала пива: 400–600 ₽».
#         - и так далее;
#         
#     - middle_avg_bill — число с оценкой среднего чека, которое указано только для значений из столбца avg_bill, начинающихся с подстроки «Средний счёт»:
#         
#         - Если в строке указан ценовой диапазон из двух значений, в столбец войдёт медиана этих двух значений.
#         - Если в строке указано одно число — цена без диапазона, то в столбец войдёт это число.
#         - Если значения нет или оно не начинается с подстроки «Средний счёт», то в столбец ничего не войдёт.
# 
#     - middle_coffee_cup — число с оценкой одной чашки капучино, которое указано только для значений из столбца avg_bill, начинающихся с подстроки «Цена одной чашки капучино»:
#         
#         - Если в строке указан ценовой диапазон из двух значений, в столбец войдёт медиана этих двух значений.
#         - Если в строке указано одно число — цена без диапазона, то в столбец войдёт это число.
#         - Если значения нет или оно не начинается с подстроки «Цена одной чашки капучино», то в столбец ничего не войдёт.
#     
#     - chain — число, выраженное 0 или 1, которое показывает, является ли заведение сетевым (для маленьких сетей могут встречаться ошибки):
#         - 0 — заведение не является сетевым
#         - 1 — заведение является сетевым
# 
#     -district — административный район, в котором находится заведение, например Центральный административный округ;
#     - seats — количество посадочных мест.

# <a id='выполнение_проекта'></a>
# # Выполнение проекта
# 
# 
# <a id='этап_1'></a>
# 
# 
# 
# **Этап 1.** Подготовка данных
# 
#     - Изучение данных (что хранят, объемы, типы);
#     - Проверим пропуски и типы данных. Откорректируем, если нужно;
#     - Создадим столбец street с названиями улиц из столбца с адресом.
#     - Создатим столбец is_24/7 с обозначением, что заведение работает ежедневно и круглосуточно (24/7):
#         - логическое значение True — если заведение работает ежедневно и круглосуточно;
#         - логическое значение False — в противоположном случае.;
# 
# <a id='этап_2'></a>
# 
# 
# 
# 
# **Этап 2.** Анализ данных
#     В ходе этапа ответим на следующие вопросы и проведем исследования:
#         
#     - Какие категории заведений представлены в данных? Исследуем количество объектов общественного питания по категориям: рестораны, кофейни, пиццерии, бары и так далее. Постройте визуализации. Ответьте на вопрос о распределении заведений по категориям.
#     - Исследуем количество посадочных мест в местах по категориям: рестораны, кофейни, пиццерии, бары и так далее. Построим визуализации. Проанализируем результаты и сформулируем предварительные выводы.
#     - Рассмотрим и изобразим соотношение сетевых и несетевых заведений в датасете. Выясним каких заведений больше?
#     - Какие категории заведений чаще являются сетевыми? Исследуем данные и постараемся ответить на вопросы графиком.
#     - Сгруппируем данные по названиям заведений и найдем топ-15 популярных сетей в Москве (под популярностью понимается количество заведений этой сети в регионе). Построим подходящую для такой информации визуализацию.
#     - Какие административные районы Москвы присутствуют в датасете? Необходимо отобразить общее количество заведений и количество заведений каждой категории по районам. 
#     - Визуализируем распределение средних рейтингов по категориям заведений. Сильно ли различаются усреднённые рейтинги в разных типах общепита?
#     - Построим фоновую картограмму (хороплет) со средним рейтингом заведений каждого района.
#     - Отобразим все заведения датасета на карте с помощью кластеров средствами библиотеки folium.
#     - Найдем топ-15 улиц по количеству заведений. Построим график распределения количества заведений и их категорий по этим улицам.
#     - Найдем улицы, на которых находится только один объект общепита. Предоставим ему оценку.
#     - Необходимо посчитать медиану столбца middle_avg_bill для каждого района. Используем это значение в качестве ценового индикатора района. Построим фоновую картограмму (хороплет) с полученными значениями для каждого района. Проанализируем цены в центральном административном округе и других. Как удалённость от центра влияет на цены в заведениях?
#     - Проиллюстрируем другие взаимосвязи, которые вы нашли в данных. Например, по желанию исследуйте часы работы заведений и их зависимость от расположения и категории заведения. Также можно исследовать особенности заведений с плохими рейтингами, средние чеки в таких местах и распределение по категориям заведений.
#     - Обобщим результаты исследования в общий вывод.
# 
# [Перейти к разделу проекта "Этап 2"](#этап_2_1)
# 
# 
# 
# <a id='этап_3'></a>
# 
# 
# 
# 
# **Этап 3.**  Детализируем исследование: открытие кофейни.
#     
#     В данном этапе необходимо ответить на следующие вопросы:
#     
#     - Сколько всего кофеен в датасете? В каких районах их больше всего, каковы особенности их расположения?
#     - Есть ли круглосуточные кофейни?
#     - Какие у кофеен рейтинги? Как они распределяются по районам?
#     - На какую стоимость чашки капучино стоит ориентироваться при открытии и почему?
#     
# [Перейти к разделу проекта "Этап 3"](#этап_3_1)
# 
# <a id='этап_4'></a>
# 
# 
# 
# 
# **Этап 4.** Подготовка презентации
# 
#     Необходимо подготовить презентацию исследования для инвесторов.  
#     Отправить презентацию нужно обязательно в формате PDF. Необходимо приложить ссылку на презентацию в markdown-ячейке в формате:
#     Презентация: <ссылка на облачное хранилище с презентацией> 

# **Общий план выполнения проекта** 
# 
# Перед выполнением любой из задач, в первую очередь необходимо подготовить набор библиотек, познакомится с данными.
# 
# Сам процесс исследования будет проведен в следующем порядке (основные этапы):
# 
#     - загрузка библиотек и знакомство с данными
#     - предобработка данных (Этап 1)
#     - формирование временных таблиц, включая новые сводные (если потребуется)
#     - проведение 2 Этапа - самый трудозатратный этап, так как необходимо провести многостронний анализ, и визуализировать полученные данные (пока для внутреннего использования)
#     - проведение 3 Этапа, на котором разбере5м детально вопросы в срезе кофейн.
#     - 4 Этап - подготовка итоговой презентации.
#     
#     - Общие выводы по результатам
# 
# *Итоговая цель анализа* - подготовить презентацию исследования для инфвесторов, как в общем срезе общепита МСК, так и в срезе кофейн, так как у инвесторов есть идея кофейни по аналогии с сериалом "Друзья", предоставить рекомендации

# ## Этап. Предварительная обработка данных и предварительная обработка

# [Перейти к разделу содержания "Этап 1"](#этап_1)

# ### Загрузка необходимого набора библиотек

# In[1]:


import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
from matplotlib.dates import DateFormatter
from matplotlib import colors
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
get_ipython().system('pip install tabulate')
from tabulate import tabulate
from scipy import stats as st
import math as mth
from statsmodels.stats.multitest import multipletests
import statsmodels.stats.multitest as smm
import plotly.graph_objs as go
import plotly.express as px
import plotly.subplots as sp
get_ipython().system('pip install folium ')
import folium 
from folium import Map, Choropleth, Marker
from folium.plugins import MarkerCluster
import json
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.style.use('ggplot')
plt.style.use('default')

import warnings
warnings.filterwarnings("ignore")


# In[2]:


pip install folium --upgrade


# ### Загрузка DS 

# In[3]:


try:
    data = pd.read_csv (r'C:\Users\PC_Maks\Desktop\study\S15 vision\project\moscow_places.csv', sep=',')
except:
    data = pd.read_csv ('/datasets/moscow_places.csv', sep=',') 
    
    
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['figure.figsize'] = (15, 8) 
 


# ### Загрузка файла json

# In[4]:


try:
    with open(r'C:\Users\PC_Maks\Desktop\study\S15 vision\project\admin_level_geomap.geojson', 'r', encoding='utf-8') as f:
        geo_json = json.load(f)
except:
    with open('/datasets/admin_level_geomap.geojson', 'r', encoding='utf-8') as f:
        geo_json = json.load(f)
   


# In[5]:


data


# In[6]:


data.info()


# ### Предварительная обработка (пропуски, дубли, переименование)
# 
# Воспользуемся заготовленными ранее функциями для изучения и первичной обработки DS 

# In[7]:


def check(data):
    try:
        
        display('Проверка на дубликаты:')
        duplicates = data.duplicated()
        duplicate_rows = data.loc[duplicates]
        display(duplicate_rows.info())
        display(duplicate_rows)
        display('----------------------')
        display('Пропуски:')
        display(data.isna().sum())
        display('Пропуски в процентном отношении к всему датасету:')
        display(data.isna().sum() / len(data) * 100)

        num_rows_before = len(data)
        data.drop_duplicates(inplace=True)
        num_rows_after = len(data)
        num_rows_deleted = num_rows_before - num_rows_after
        percent_deleted = round(num_rows_deleted / num_rows_before * 100, 2)
        display(f'Удалено дубликатов: {num_rows_deleted} строк ({percent_deleted}% от всего датасета)')
        
    except Exception as e:
        print(f'ERROR: {e}')


# In[8]:


check (data=data)


# Предварительно дубликатов нет. Но нужно проверить еще раз после приведения названий заведений к нижнему регистру, так как в названиях могут встречаться разные способы написания заведений.

# In[9]:


columns_to_lower = ['name', 'category', 'address']
data.loc[:, columns_to_lower] = data.loc[:, columns_to_lower].apply(lambda x: x.str.lower())


# ### Глубокая чистка дублей.
# Но важно проверить на неочевидные дубли, единственный пока вариант который приходит на ум, проверить по координатам, так как они являются максимально уникальными в данном случае.

# найдено повторные комбинации, посмтрим внимательнее что скрывается по этим координатам

# In[10]:


duplicate_rows = data[data.duplicated(subset=['lat', 'lng'],keep=False)]
duplicate_rows


# как видно из результатов, в основной своей массе объекты разные, и расположены по одному адресу. С высокой долей вероятности, мы имеем дело с торговыми центрами. Не критично. Но выявлены и несколько фактических дублей, например кафе леон/leon, чайхона doner кафе, и more poke - их удаление ломает датасет (после консультации с куратором, получил рекомендацию их оставить, так как количество выявленных потенциально проблемных объектов менее 10, и они сильной роли не играют)

# После замены букв во всем датасете на нижний регистр, дубликаты также не выявлены. 

# In[11]:


duplicate_rows = data[data.duplicated(subset=['name', 'address'],keep=False)]
duplicate_rows


# In[12]:


data = data[~data.duplicated(subset=['name', 'address'], keep='first')]


# In[13]:


data[['name', 'address']].duplicated().sum()


# Выявлено 4 не очивидных дубликата (связка название+адрес), удалены с сохранением первой строки. Есть довольно большое количество пропусков в разделах - price(60.5%), avg_bill(54.6%), middle_avg_bill (62.5%), middle_coffee_cup (93.6%), seats (42.9%), hours (6.3%). Согласно установочным данным у нас в полях - middle_avg_bill , middle_coffee_cup данные формируются на данных среднего чека, но мы видим, что в процентном соотношении пропуски в разделе avg_bill и указанных средних диапазонах - разные, можем попробовать дополнить пропуски (все не устраним, но улучшить попробуем)

# In[14]:


def extract_avg_values(data):
    avg_values = data['avg_bill'].apply(lambda x: re.findall(r'\d+', str(x)))

    # Извлечение среднего значения диапазона для столбца middle_avg_bill
    avg_values_avg = avg_values.apply(lambda x: (int(x[0]) + int(x[1])) / 2 if len(x) > 1 else None)
    data['middle_avg_bill'] = np.where(data['avg_bill'].str.contains('средний счёт') & data['middle_avg_bill'].                                       isnull(), avg_values_avg, data['middle_avg_bill'])

    # Извлечение среднего значения диапазона для столбца middle_coffee_cup
    avg_values_coffee = avg_values.apply(lambda x: (int(x[0]) + int(x[1])) / 2 if len(x) > 1 else None)
    data['middle_coffee_cup'] = np.where(data['avg_bill'].str.contains('цена чашки капучино') & data['middle_coffee_cup'].                                         isnull(), avg_values_coffee, data['middle_coffee_cup'])

    return data


# In[15]:


dataset = extract_avg_values(data)


# In[16]:


check (data=dataset)


# К сожалению данные не изменились, старались зря, но функция пригодится в будущем 

# In[17]:


data1=dataset [dataset ['middle_avg_bill'].isnull()]
data1.head (50)


# Изучим содержание столбцов

# In[18]:


def check_unique(data):
    for col in data.select_dtypes(include=['object']):
        print(f"Уникальные значения в столбце {col}:")
        print(data[col].unique())
        print('---------------------')

    for col in data.select_dtypes(include=['datetime64']):
        print(f"Диапазон значений в столбце {col}:")
        print(f"Минимальное значение: {data[col].min()}")
        print(f"Максимальное значение: {data[col].max()}")
        print('---------------------')

    for col in data.select_dtypes(include=['int64', 'float64']):
        if len(data[col].unique()) > 20:
            print(f"В столбце {col} более 20 уникальных значений")
        else:
            print(f"Уникальные значения в столбце {col}:")
            print(data[col].unique())
        print('---------------------')


# In[19]:


check_unique(data)


# ### Создадим столбец, в котором будет отмечен признак круглосуточной работы заведения

# In[20]:


data['is_24/7'] = data['hours'].apply(lambda x: True if 'круглосуточно' in str(x) else False)


# ### Создадим столбец с названием улиц 

# In[21]:


# посмотрим для начала какие варианты есть в адресах.
address= data['address'].value_counts().index.tolist()
display (address)


# Настало время новых эксперетментов (раньше не использовал). Попробуем выделить нужную информацию в несколько этапов через регулярное выражение

# In[22]:


data['street_raw'] = data['address'].str.replace(r'москва,|площадь\s*|проспект\s*|улица\s*|шоссе|проезд|переулок|мкад|^\s*,|^[,\s]+', '',                                                 regex=True, flags=re.IGNORECASE)
data['street_raw'] = data['street_raw'].str.replace(r'москва,|площадь\s*|проспект\s*|улица\s*|шоссе|переулок|мкад|^\s*,|,\s*.*\s*$|^[,\s]+', '',                                                 regex=True, flags=re.IGNORECASE)


# In[23]:


# повторим выгрузку названий и посмотрим что получилось и что осталось,
#будем повторять чистку, пока не получим нужный результат
address= data['street_raw'].value_counts().index.tolist()
display (address)


# In[24]:


# вроде бы получилось, осталось только убрать пробелы в конце и в начале значений в некоторых полях
data['street_raw'] = data['street_raw'].str.replace(r'^\s+|\s+$', '', regex=True)
address= data['street_raw'].value_counts().index.tolist()
display (address)


# ### Корректировка очевидных ошибок в названиях

# In[25]:


# Выгрузим все названия ресторанов которые есть
with pd.option_context('display.max_rows', None):
    display(data['name'].value_counts())


# Перебор всех названий довольно долгая процедура, но проверка нужна, так например кафе хинкали-gaли, это одна сеть, но написания названия разные.

# In[26]:


# посмотрим на явно проблемное название - хинкали гали
selected_rows = data[data['name'].str.contains(r'\bgали\b', case=False, regex=True)]
selected_rows


# In[27]:


# заменим на одинаковое название

рег_выражение = r'.*[gг][аa][лl][иi].*'
новое_значение = 'хинкали-gaли'


data['name'] = data['name'].str.replace(рег_выражение, новое_значение, regex=True)

# и сразу проверим
selected_rows = data[data['name'].str.contains(r'\bgaли\b', case=False, regex=True)]
selected_rows


# ### Корректировка признака сети

# Остался еще один этапа предварительной обработки - проверка признака сети. По логике сетевой ресторан - минимум 2 объекта
# 

# In[28]:


data_raw=data


# In[29]:


pivot_table = data.pivot_table(values='chain', index='name', aggfunc={'chain': 'sum', 'name': 'count'})

pivot_table.columns = ['chain','count_rest']


# In[30]:


pivot_table_filt= pivot_table.query ('(count_rest==1) & (chain==1)')
# а вот и все замаскированные "сетевые" рестораны - 59 объектов
pivot_table_filt


# Допускаю, что некоторые рестораны могут иметь некорректное название, и поэтому по факту являются сетевыми, возможно некоторые рестораны на момент формирования выгрузки открыли только 1 объект, но имеют амбиции на сеть, но что имеем то имеем, заменим статус этих 59 ресторанов из списка на не сетевой.

# In[31]:


restaurants_to_correct = pivot_table_filt.index.tolist()

data.loc[data_raw['name'].isin(restaurants_to_correct), 'chain'] = 0


# In[32]:


# проверим все ли скорректировали верно - итог - все корректно. 
pivot_table = data.pivot_table(values='chain', index='name', aggfunc={'chain': 'sum', 'name': 'count'})
pivot_table.columns = ['chain','count_rest']
pivot_table_filt = pivot_table.query ('(count_rest==1) & (chain==1)')


# Но стоит проверить и обратную сторону - несколько ресторанов, но указано, что они не сетевые

# In[33]:


pivot_table_filt_raw = pivot_table.query ('(count_rest>1) & (chain==0)')
pivot_table_filt_raw


# Если с названиями шаурма, ресторан, столовая и т.п. - понятно - просто общее название объекта, но вот остальные требуют или индивидуального разбора, или оставления как есть. Я считаю, что второй вариант лучше, так как если оставить варианты в диапазоне от 2 до 10 объектов, не станет яснее - просто совпадает название или это сеть, Москва большой город, ресторанов очень много. Оставляем как есть.

# ### Пропуски в категории количества посадочных мест
# 
# 42.95% записей в столбце количества мест - пропуски (выясняли это ранее), забить все просто 0 не очень хорошая идея. На мой взгляд, лучше взять медиану по категориям и заполнить все ей, но итоговое решение и корректировку произведем на этапе анализа количества посадочных место Этапа 2

# ### Итого этапа 1
# Предварительная обработка окончена, нужные столбцы добавили, дубли и пропуски отработали, там, где было нужно или возможно.
# На выходе получили существенное количество пропусков в разделах средних чеков, как диапазонов, так и средних из диапазонов, а также в разделах средней стоимости чашки кофе. Видимо не все рестораны указывают данный параметр при заполнении карточек. 
# Попробовали отработать не очевидные дубли, но объем работы не соразмерен с результатом корректировки, оставим в рамках "допустимой погрешности".
# Скорректировали статус сетевого ресторана, по тем объектам, где несколько ресторанов, а статус - не сетевой (59 корректировок), рестораны с обратной ситуацией, оставили как есть, так как очень много не уникальных названий.
# Неочевидные дубли и некорректные названия выявили, но часть оставили без изменений, как допустимые, скорректировали только крупную сеть - ханкали-gaли. 
# 
# Аномалии в категорийных столбцах не выявлено.  
# Исследования на предмет выявления аномалий в других разделах будем проводить и принимать окончательные решения на этапе анализа.
# 
# 
# Продолжим
# 

# <div class="alert alert-success" style="border-radius: 10px; box-shadow: 2px 2px 2px; border: 1px solid; padding: 10px ">
# <b>Комментарий ревьюера v.1</b> 
#     
# 👍 
# Здесь все корректно. Двигаемся дальше.
# </div>

# ## Этап 2. Исследование
# 

# [Перейти к разделу содержания "Этап 2"](#этап_2)
# 
# <a id='этап_2_1'></a>

# ### Категории заведений.
# 
# - Какие категории заведений представлены в данных? Исследуем количество объектов общественного питания по категориям: рестораны, кофейни, пиццерии, бары и так далее. Постройте визуализации. Ответьте на вопрос о распределении заведений по категориям.

# In[34]:


# всего в датасете 8 категорий 
display (data['category'].nunique())
data['category'].unique()


# In[35]:


# посчитаем количество объектов в каждой категории.
category_table = data.pivot_table(values='name', index='category', aggfunc={'name': 'count'}).reset_index()
category_table.columns = ['category','count']
category_table_sorted = category_table.sort_values('count', ascending=False).reset_index(drop=True)
# Ожидаемый результат - кафе/рестораны/кофейня - самые популярный тип объектов.
total = category_table_sorted ['count'].sum ()
category_table_sorted ['ration'] = (category_table_sorted ['count']/total*100).round ()
category_table_sorted


# #### График "Количество заведений в Москве по категориям"

# In[36]:


plt.figure(figsize=(15, 8))
sns.barplot(x='category', y='count', data=category_table_sorted)

plt.xlabel('Категория', fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.title('Количество заведений в Москве по категориям', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=12)

for index, row in category_table_sorted.iterrows():
    plt.text(index, row['count'], row['count'], ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()


# **Промежуточный вывод**
# 
# Как видно из графика топ-3 по количеству занимают - кафе, рестораны и кофейни, при этом только кафе почти столько же сколько остальных заведений за пределами топ-3 (2378 против 2572). Да, конкуренция существенная

# <div class="alert alert-success" style="border-radius: 10px; box-shadow: 2px 2px 2px; border: 1px solid; padding: 10px ">
# <b>Комментарий ревьюера v.1</b> 
#     
# 👍 
# Визуализация отличная,  тип графика подобран корректно. Все подписи, заголовки присутствуют. Вывод корректный.
# </div>

# ### Посадочные места
# 
# Мы помним, что у нас 42.95% в этом столбце пропуске. удаление не приемлемое решение, забить просто 0 - тоже не выход, лучшее решение, чтобы и не потерять в данных, и заполнить пропуски - медиана по категориям, она менее чувствительна к выбросам
# 

# In[37]:


# построим таблицу с общим количеством посадочных мест и медиапной значений на каждую категорию
category_table_counts = data.pivot_table(values='name', index='category', aggfunc='count')
category_table_seats = data.pivot_table(values='seats', index='category', aggfunc='median')

category_table_with_seats = category_table_counts.merge(category_table_seats, on='category').reset_index()
category_table_with_seats.columns = ['category', 'count', 'median_seats']
category_table_with_seats ['median_seats']=category_table_with_seats['median_seats'].round(0).astype(int)
category_table_with_seats=category_table_with_seats.sort_values (by='median_seats', ascending=False) 
category_table_with_seats


# Лидеры по медиативному количеству посадочных мест среди типов заведений немного изменились, на первом месте рестораны, бары и кофейни, потом столовые. В этом списке меня смущает что кофейни имеют якобы 80 посадочных мест, если с пабами, где часто стойки и большие столы, могу допустить около 10-12 столов (каждый на 6 человек и более), то с кофейнями такой результат выглядит странным. Посмотрим на значения в исходных данных, возможно имеем дело с аномалиями. Но перед следующим шагом анализа - заполним пропуски медиативными значениями.

# In[38]:


category_seats_mapping = category_table_with_seats.set_index('category')['median_seats']
data['seats'] = data['seats'].fillna(data['category'].map(category_seats_mapping))


# In[39]:


# Но есть еще 136 объектов в которых нет посадочных мест
#в некоторых я допускаю отсуствие посадки (например кофейня), но ресторан и кафе маловероятно, 
data[data['seats']==0]


# In[40]:


# Посмотрим на размер потерь - 1.62% приемлимо, удаляем
data_new = data.query ('seats >0')
lost_data = data.shape[0] - data_new.shape[0]  
percent_lost_data = round((lost_data / data.shape[0]) * 100, 2) 
percent_lost_data


# In[41]:


data=data_new
data ['seats'].describe()


# In[42]:


# интересный результат, есть заведения с количесвтом посадочных место 1288, банкетные залы, фудкорты? 
#Посмотрим список заведений от 86 мест и выше (75%+)

data_top_75_proc_seats = data.query ('seats>86')
data_top_75_seats_table= data_top_75_proc_seats.pivot_table (values = 'seats', index = 'category',                                                        aggfunc = {'seats':'median'}).reset_index ()
display (data_top_75_proc_seats.sort_values(by = ['seats'], ascending = False))
data_top_75_seats_table


# Как видно из результатов, часть заведений действительно имеет общий адрес, с высокой долей вероятности футкорты, в таких случаях сложно назвать все 86+ мест рассадки местами самого заведения, медиана по каждой категории, после среза - в районе 150+, до среза 76, посмотрим какие могут быть потери при срезе разных показателей рассадки, и попробуем принять решение о допустимости среза того или иного процента.
# 

# На графике мы видим, что все категории укладываются в рассадку до 150 мест, далее выбросы. Посмотрим сколько процентов составляют заведения с местами 150+

# In[43]:


plt.ylim(0, 450)
sns.boxplot(x='category', y='seats', data=data, palette='Spectral')
plt.xlabel('Категории', fontsize=14)
plt.ylabel('Количество посадочных мест', fontsize=14)
plt.title('Диаграмма размаха посадочных мест по категориям', fontsize=16)
plt.grid(True)
plt.show()


# In[44]:


percentile = np.percentile(data['seats'], [80, 85, 90, 95, 97, 98, 99]).astype(int)
display('Процентиль количества действий на одного пользователя (оценка 80, 85, 90, 95, 97, 98, 99):', percentile)


# Учитывая банкетные залы, футкорты, пологая допустимо "отрезать" 3%, оставить 97% сета, с удалением аномально больших рассадок. 

# In[45]:


percentile_97 = np.percentile(data['seats'], 97).astype(int)


# In[46]:


plt.hist(x=data['seats'], bins=50, range=[0, 1288], edgecolor='white', color='blue', alpha=0.7)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Количество мест')
plt.ylabel('Количество ресторанов')
plt.title('Распределение количества мест по ресторанам')
plt.axvline(x=percentile_97, color='red', label='97% percentile')
plt.legend()
plt.show()


# In[47]:


data = data.query ('seats<=@percentile_97')
# отделались "малой кровью" теперь можем приступить к визуализации 


# In[48]:


category_table_seats = data.pivot_table(values='seats', index='category', aggfunc='median').reset_index()
category_table_seats = category_table_seats.sort_values (by = 'seats', ascending=False)
category_table_seats ['seats'] = category_table_seats ['seats'].astype(int)
category_table_seats = category_table_seats.reset_index(drop=True)


# #### График "Количество посадочных мест заведений Москвы"

# In[49]:


plt.figure(figsize=(15, 8))
sns.barplot(x='category', y='seats', data=category_table_seats)

plt.xlabel('Категория', fontsize=14)
plt.ylabel('Количество мест', fontsize=14)
plt.title('Количество посадочных мест в заведениях Москвы', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=12)

for index, row in category_table_seats.iterrows():
    plt.text(index, row['seats']+0.5, row['seats'], ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()


# **Промежуточный вывод** 
# Рестораны, бары, кофейни и столовые имеют самое больше количество посадочных мест. Если с ресторанами, барами и столовыми более-менее понятно, для них важна посадка - гости приходят туда или полноценно поесть и посидеть, или отдохнуть (например, в барах), то количество посадочных место 50+ в остальных категориях странно, видимо я не был в таких заведениях, или не считал места. Хотя помним, что большое количество посадочных мест дают ТЦ

# ### Сетевые/несетевые заведения
# - Рассмотрим и изобразим соотношение сетевых и несетевых заведений в датасете. Выясним каких заведений больше?

# In[50]:


data_chain = data.pivot_table(values='name', index='category', columns='chain', aggfunc='count', fill_value=0).reset_index(col_level=1)
data_chain.columns = ['category', 'non_chain', 'chain']

data_chain


# In[51]:


sum_row = data_chain[['non_chain', 'chain']].sum()
total_row = pd.Series(['Total', sum_row['non_chain'], sum_row['chain']], index=data_chain.columns)

data_chain_with_sum = pd.concat([data_chain, total_row], ignore_index=True)
data_chain = data_chain_with_sum


# In[52]:


#Добавим столбцы с отношением каждой категории к своей подкатегории (сетвой или нет)

# сохраним общее количество заведений
all_rest = data ['chain'].count ()

# сохраним для расчетов количество не сетевых заведений и сетевых
total_no_chain = data [data['chain']==0]['name'].count ()
total_chain = data [data['chain']==1]['name'].count ()

# расчитаем отношение отдельно взятых категорий к количеству заведений в сетевых и не сетевых
data_chain ['rat_non_chain'] = (data_chain ['non_chain']/total_no_chain*100).round (2)
data_chain ['rat_chain'] = (data_chain ['chain']/total_chain*100).round (2)

# расчитаем отношение отдельных категорий к общему количеству заведений в датасете
data_chain ['rat_all_non_chain'] = (data_chain ['non_chain']/all_rest*100).round (2)
data_chain ['rat_all_chain'] = (data_chain ['chain']/all_rest*100).round (2)

# расчитаем отношение категории заведения к количетву всего таких заведений 
data_chain ['ration_category'] = (data_chain ['chain']/(data_chain ['chain']+data_chain ['non_chain'])*100).round (2)
data_chain = data_chain.sort_values (by='ration_category', ascending = False)

# удалим ранее созданный столбец для расчетов, чтобы сохранить итоговую таблицу и выведем ее для просмотра  
data_chain = data_chain.drop(data_chain[data_chain['category'] == 'Total'].index)
data_chain


# In[53]:


data_chain = data_chain.sort_values (by='non_chain', ascending=False).reset_index (drop=True)


# #### Графики распределения сетевых и не сетевых заведений

# In[54]:


plt.figure(figsize=(15, 8))
ax = sns.barplot(x=data_chain['category'], y=data_chain['non_chain'], data=data_chain)

plt.xlabel('Категория', fontsize=14)
plt.ylabel('Количество несетевых заведений', fontsize=14)
plt.title('Распределение несетевых заведений в Москве по типам', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()


# In[55]:


data_chain = data_chain.sort_values (by='chain', ascending=False).reset_index (drop=True)


# In[56]:


plt.figure(figsize=(15, 8))
ax = sns.barplot(x=data_chain['category'], y=data_chain['chain'], data=data_chain)

plt.xlabel('Категория', fontsize=14)
plt.ylabel('Количество не сетевых заведений', fontsize=14)
plt.title('Распределение не сетевых заведений в Москве по типам', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()


# In[57]:


values = [total_no_chain, total_chain]
labels = ['не сетевые заведения', 'сетевые заведения']

fig = sp.make_subplots(rows=1, cols=2,
                       specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                       subplot_titles=['Распределение заведений', 'Тип заведения'])

fig.add_trace(go.Pie(labels=labels, values=values, textinfo='percent', hole=0.4),
              row=1, col=1)

fig.update_traces(marker=dict(colors=['#70bf5d', '#ff7f0e']), row=1, col=1)

fig.update_layout(title='Распределение заведений между сетевыми и не сетевыми',
                  height=500, width=1000)

fig.update_layout(
    annotations=[
        dict(text='Тип заведения:', x=0.5, y=0.98, xref='paper', yref='paper', align='right',
             showarrow=False, font=dict(size=14))
        
    ],
    legend=dict(x=0.4, y=1)  
)
fig.show()


# **Вывод**
# - Как видно из результатов исследования, 62,7% (5034 позиций в датасете) всех заведений в Москве не сетевые, соответсвенно 37,3% (2994)- сетевые. 
# 
# - Среди сетвых топ-3:
#         - кафе - 23,91% (716)
#         - кофейни - 22,71% (680)
#         - рестораны - 22,58% (676)
# - Топ-3 несетевых типа:
#         - кафе - 31,05% (1563)
#         - рестораны - 25,29% (1273)
#         - кофейни - 13,31% (670)
#         
# 

# In[58]:


data_chain = data_chain.sort_values (by='rat_all_chain', ascending = False)


# In[59]:


plt.figure(figsize=(15, 8))
ax = sns.barplot(x=data_chain['category'], y=data_chain['rat_all_chain'], data=data_chain)

plt.xlabel('Категория', fontsize=14)
plt.ylabel('Отношение', fontsize=14)
plt.title('Отношение сетевых заведений ко всем заведениям датасета', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()


# In[60]:


data_chain = data_chain.sort_values (by='ration_category', ascending = False)


# In[61]:



plt.figure(figsize=(15, 8))
ax = sns.barplot(x=data_chain['category'], y=data_chain['ration_category'], data=data_chain)

plt.xlabel('Категория', fontsize=14)
plt.ylabel('Отношение', fontsize=14)
plt.title('Отношение сетевых к заведениям такого же типа', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()


# Как видно из результатов и графика, самые распространённые категории среди сетевых заведений по отношению ко всем заведениям: кафе, рестораны и кофейни. Стоит отметить, данные категории существенно отрываются от остальных, но нужно понимать, что этих категорий в принципе больше, поэтому стоит посмотреть еще и на отношение внутри одной категории. 
# 
# А вот если давать оценку в части распространённости сетевых заведений внутри своей категории, то первое место занимают булочные, за ними идут пиццерии и только потом один из лидеров общей оценки – кофейни

# ### ТОП-15 сетевых заведений (по названию сети)
# 
# Для расчетов необходимо сделать срез ДС по признаку сетевой или нет объект

# In[62]:


data_chain=data.query ('chain == 1')
# сгруппируем данные, для более удобного изучения
chain_group=data_chain.groupby ('name').agg({'name':'count', 'rating': 'mean', 'category': pd.Series.mode})
chain_group ['rating'] = round (chain_group ['rating'], 1)


# In[63]:


chain_group.columns = ['count', 'rating', 'category']
chain_group.reset_index (inplace=True)


# In[64]:


chain_group.sort_values ('count', ascending = False )


# Яндекс лавка сервис хороший, но не заведение питания в текущем понимании, уберем ее из списка

# In[65]:


chain_group=chain_group.query ('name !="яндекс лавка"')


# In[66]:


top_15 = chain_group.sort_values ('count', ascending = False).head (15).reset_index(drop=True)
top_15


# #### График распределения заведений среди ТОП-15

# In[67]:


values = top_15 ['count']
labels = top_15 ['name']

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

# Лучшая практика: добавляем проценты к значениям на сегментах графика
fig.update_traces(textinfo='percent', textposition='inside', hole=0.3)

fig.update_layout(
    title='Распределение количества заведений среди ТОП-15 Москвы',
    height=600,
    width=800,
    legend=dict(
        x=1,
        y=0.65,
        title='Заведения',
        title_font=dict(size=14),
        font=dict(size=12)
    )
)

fig.show()


# Явный лидер среди Топ-15 сетевых заведений Москвы - сеть "Шоколадница" (15,5%, 115 заведений), замыкает список "Drive cafe" - 3,23% (24 объекта)

# In[68]:


plt.figure(figsize=(15, 8))
ax = sns.barplot(x=top_15['count'], y=top_15['name'], data=top_15)

plt.xlabel('Количество заведений', fontsize=14)
plt.ylabel('Категория', fontsize=14)
plt.title('Количество заведений у Топ-15 сетей', fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.7) 
plt.xticks(rotation=45)
plt.tick_params(axis='y', labelsize=12)  

ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

for p in ax.patches:
    ax.annotate(f'{p.get_width():.1f}', (p.get_width(), p.get_y() + p.get_height() / 2), ha='left', va='center', fontsize=12)  # Изменение позиции аннотаций

plt.tight_layout()
plt.show()


# ### Административные районы.
# Проведем исследование распределения по общему количеству заведений и заведений по категориям среди административных районов Москвы
# 
# - Какие административные районы Москвы присутствуют в датасете? Необходимо отобразить общее количество заведений и количество заведений каждой категории по районам. 

# In[69]:


district_list = data ['district'].value_counts().reset_index()
district_list = district_list.sort_values (by='district', ascending=False)
district_list.columns = ['district', 'count']
district_list


# #### График распределения объектов по округам

# In[70]:


fig, ax = plt.subplots(figsize=(15, 8))

sns.barplot(x='count', y='district', data=district_list, ci=None, ax=ax)
ax.set_title("Распределение заведений по административным районам", fontsize=15)
ax.set_xlabel("Количество заведений")
ax.set_ylabel("Районы")


ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)


for index, row in district_list.iterrows():
    plt.text(row['count'] + 50, index, str(row['count']), ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()


# In[71]:


district_tab = data.pivot_table (values='name', index='category', columns='district', aggfunc = 'count',margins=True, margins_name='Total')


# In[72]:


district_tab.sort_values(by='Total', ascending=False)


# **Вывод** Всего в датасете представлено 9 административных районов Москвы. Практически во всех районах, по количеству заведений, лидируют - кафе, однако в ЦАО - рестораны. Стоит отметить, что в ЦАО концентрация заведений более чем в 2 раза выше в абсолютном значении, чем концентрация в любом другом районе.

# ### Распределение средних рейтингов по категориям заведений
# 
# 

# In[73]:


rating_table = data.groupby('category')['rating'].agg(['mean', 'median'])
rating_table.reset_index(inplace=True)
rating_table ['mean'] = round(rating_table['mean'],1)


# In[74]:


rating_table = rating_table.sort_values (by='mean', ascending=False)
rating_table


# #### График среднего рейтинга заведений по округам

# In[75]:


fig = go.Figure()
fig.add_trace(go.Bar(x=rating_table['category'], y=rating_table['mean'], name='Среднее', marker_color='#1f77b4'))
#fig.add_trace(go.Bar(x=rating_table['category'], y=rating_table['median'], name='Медиана', marker_color='#ff7f0e'))

fig.update_layout(
    title='Рейтинг заведений по категориям',
    xaxis=dict(title='Категории'),
    yaxis=dict(title='Рейтинг'),
    barmode='group',
)

fig.update_traces(texttemplate='%{y}', textposition='outside')



fig.show()


# **Вывод**
# Как видно из результатов исследования - отличие медианы и среднего наблюдается только у категорий: "столовые", "быстрое питание" и "кафе", Медиана у указанных категорий выше. 
# Если оценивать просто средний рейтинг, то у всех заведений он выше 4 баллов, самый высокий у баров, самый низкий у заведений быстрого питания и кафе (видимо после посещения баров, гости более лояльны к заведениям))))) ), кроме этого, могу предположить, что не все кафе и кофейни получили оценку, их намного больше чем баров, и времени обычно в них проводят меньше, мало кто решит поставить оценку, если "забежал" купить чашку кофе, ну если не негативный конечно отзыв, их по практике ставят охотнее.

# ### Фоновая картограмму (хороплет) со средним рейтингом заведений каждого района.

# In[76]:


rating_tab = data.groupby('district', as_index=False)['rating'].agg('mean').round(2)

rating_tab


# In[77]:


moscow_lat, moscow_lng = 55.751244, 37.618423
# создаём карту Москвы
m = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=rating_tab,
    columns=['district', 'rating'],
    key_on="feature.name",
    fill_color='YlGn',
    fill_opacity=0.7,
    legend_name='Медианный рейтинг заведений по районам',
).add_to(m)

m


# ### Все заведения датасета на карте

# In[78]:


marker_cluster = MarkerCluster().add_to(m)


# In[79]:


def create_cluster(row):
    try:
        Marker(
            [row['lat'], row['lng']],
            popup=f"{row['name']} {row['rating']}",
        ).add_to(marker_cluster)
    except Exception as e:
        print(f'ERROR: {e}')     


# In[80]:


data.apply(create_cluster, axis=1)


# #### Кластерная карта Москвы со всеми заведениями

# In[81]:


m


# **Вывод**
# Как было отмеченно ранее, самый высокий рейтинг у заведений в ЦАО

# ### Топ-15 улиц по количеству заведений. Построим график распределения количества заведений и их категорий по этим улицам.

# In[82]:


group_data_street = data.groupby ('street_raw') ['category'].agg({'count'})


# In[83]:


group_data_street.reset_index(inplace=True)
group_data_street=group_data_street.sort_values(by='count',ascending=False)


# In[84]:


# сохраняем топ-15 улиц по количеству заведений 
top_15_streets = group_data_street.head (15).reset_index(drop=True)
#создадим список улиц в отдельной переменной, для фильтрации потом в срезе
list_top_streets = top_15_streets['street_raw'].tolist()
list_top_streets


# In[85]:


data_top_streets = data.query ('street_raw in @list_top_streets')


# #### График распределения заведений Москвы среди Топ-15 по популярности  улиц

# In[86]:


plt.figure(figsize=(12, 8))
sns.countplot(data=data_top_streets, x='street_raw', hue='category', saturation=1)
plt.xlabel('TOP STREET', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Количество заведений', fontsize=14)
plt.title('Количество заведений по категориям на топ-15 улиц Москвы', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.legend(title='Категория', fontsize=10)
plt.show()


# In[87]:


# стоит посмотреть и на тепловой карте распределение
heatmap_data_top = data_top_streets.pivot_table(values='rating', index='street_raw', columns='category', aggfunc='mean')

plt.figure(figsize=(15, 8))
sns.heatmap(heatmap_data_top, cmap='YlGnBu', annot=True, fmt=".1f")


plt.xlabel('Категория', fontsize=14)
plt.ylabel('Улица', fontsize=14)
plt.title('Тепловая карта оценок по категориям заведений (топ-15 улиц)', fontsize=16)

plt.show()


# In[88]:


moscow_lat, moscow_lng = 55.751244, 37.618423
# создаём карту Москвы
m_top_street = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=data_top_streets,
    columns=['district', 'rating'],
    key_on="feature.name",
    fill_color='YlGn',
    fill_opacity=0.7,
    legend_name='Топ-15 улиц с концентрацией заведений',
).add_to(m_top_street)

def create_marker(row):
    Marker([row['lat'], row['lng']],
        popup=f"{row['name']} {row['rating']}"
    ).add_to(m_top_street)

data_top_streets.apply(create_marker, axis=1)
m_top_street


# **Вывод**
# 
# Больше всего заведений на проспекте Мира, с явным отрывом кафе, рестораны и кофейни, не удивительно, центр города, где больше всего гуляет людей, спрос есть. Даже без визуализации самих улиц на карте, можно отметить один важный момент - в рейтинге присутствуют просто длинные улицы, но проспект Мира все равно имеет высокую концентрацию безотносительно длины улицы

# ### Одиночные заведения на улицах Москвы
# 
# - Найдем улицы, на которых находится только один объект общепита. Предоставим ему оценку.

# In[89]:


# у нас уже есть список улиц с количеством заведений, возьмем его, чтобы не повторятся
group_data_street.sort_values(by='count', ascending=True)
group_data_street_1 = group_data_street.query ('count == 1')
rest_only_1 = group_data_street_1['street_raw'].tolist()
data_only_1_streets = data.query ('street_raw in @rest_only_1')


# In[90]:


data_only_1_streets ['category']. value_counts()


# In[91]:


moscow_lat, moscow_lng = 55.751244, 37.618423
# создаём карту Москвы
m_only_1_street = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=data_only_1_streets,
    columns=['district', 'rating'],
    key_on="feature.name",
    fill_color='YlGn',
    fill_opacity=0.7,
    legend_name='Распределение одиночных заведений на карте',
).add_to(m_only_1_street)


# In[92]:


marker_cluster_only_1 = MarkerCluster().add_to(m_only_1_street)

def create_cluster(row):
    try:
        Marker(
            [row['lat'], row['lng']],
            popup=f"{row['name']} {row['rating']}",
        ).add_to(marker_cluster_only_1)
    except Exception as e:
        print(f'ERROR: {e}')  
        
data_only_1_streets.apply(create_cluster, axis=1)


# #### Графики и карта распределения одиночных объектов в Москве

# In[93]:


m_only_1_street


# In[94]:


# Для более объемной оценки, стоит посмотреть на рейтинги с привязкой по округам нашей выборки одиночных заведений
heatmap_data = data_only_1_streets.pivot_table(values='rating', index='district', columns='category', aggfunc='mean')

plt.figure(figsize=(15, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f")


plt.xlabel('Категория', fontsize=14)
plt.ylabel('Округ', fontsize=14)
plt.title('Тепловая карта по категориям заведений, одиночных на улицах по округам', fontsize=16)

plt.show()


# В датасете обнаружено, что 468 улиц имеют всего одно заведение. Большинство таких улиц находятся в категории кафе (166 случаев), а наименьшее количество улиц с одним заведением принадлежит категории булочных (7 случаев). Стоит отметить, что количество булочных в целом невелико. Улицы с единственным заведением встречаются в каждом округе, но наибольшая концентрация таких улиц наблюдается в Центральном административном округе (ЦАО). Вероятно, это небольшие проулки или проезды.
# 
# На тепловой карте мы видим, что есть округа, в которых хоть и находятся улицы из числа топ-15, в которых у некоторых заведений нет рейтинга, например бары в ЮАО и ЮЗАО, а также пиццерии там же, кроме этого, булочные в ЮВАО, СЗАО и СВАО. На текущий момент гипотез только 2 - в данных районах действительно очень мало заведений указанных категорий, и средний рейтинг не строится, и второй вариант - в этих округах есть несколько небольших заведений, у которых не выставлен рейтинг. И в первом и во-втором случае - можем утверждать о низкой концентрации данных категорий на Топ-улицах в этих округах. Стабильно высокий рейтинг по всем округам показывают кофейни и столовые.

# ### Медиана middle_avg_bill для каждого района.
# 

# In[95]:


data['middle_avg_bill'].sort_values (ascending = False)
# Интересно, а какие заведения скрываются за средним чеком 10000+


# In[96]:


big_bill = data [data['middle_avg_bill']>=10000]
big_bill
# три заведения, можно предположить что имеет место ошибка в данных, 
# так как ценник для таких заведения аномально высокий, удаляем из датасета, 
# при срезе у нас исчезнут и ресторана в которых не указан средний чек
data_with_mid_bill = data.query ('middle_avg_bill<10000')


# In[97]:


data_with_mid_bill['middle_avg_bill'].sort_values (ascending = False)


# In[98]:


# теперь посчитаем медиану среднего счета по округам для отфильтрованого датасета data_with_mid_bill

group_med_bill_district = data_with_mid_bill.groupby ('district')['middle_avg_bill'].agg('median').reset_index()
group_med_bill_district.columns = ['district', 'median_bill']
group_med_bill_district.sort_values (by = 'median_bill', ascending = False)


# #### Медиативная карта среднего чека Москвы

# In[99]:


moscow_lat, moscow_lng = 55.751244, 37.618423
# создаём карту Москвы
m_med_bill = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=group_med_bill_district,
    columns=['district', 'median_bill'],
    key_on="feature.name",
    fill_color='YlGn',
    fill_opacity=0.7,
    legend_name='медиативное значение среднего счета заведений на карте',
).add_to(m_med_bill)

m_med_bill


# Самые дорогие - ЦАО и ЗАО, самые низкие значения медиативного среднего чека в ЮАО, ЮВАО и СВАО. Учитывая результаты можно сделать предположение, что при формировании ценовой политики учитывается и престижность района, если с ЦАО ожидаемо (высокая конкуренция, большое количество заведений, высокая проходимость), то также мы видим в качестве "дорогого" района ЗАО, который по многим факторам считается достаточно дорогим районом в общем и целом (много крупных компаний там имеют офисы, дорогая, даже по меркам МСК, недвижимость). А вот районы спальной застройки попали вниз рейтинга по ценообразованию, что, тоже вполне объяснимо - если люди хотят отдохнуть или погулять, вероятнее всего они поедут в центральную часть города, где и отдыхать может быть приятнее, и выбор значительно больше, а в свой район они поедут спать. Если уже и захотят воспользоваться услугами заведений, вероятно это будет заказ доставки домой.

# ### Дополнительные проверки.
# 

# Для предоставления итоговых рекомендаций, посмотрим внимательнее на лучшие по рейтингу заведения.
# Мы уже знаем, что топова улица - проспект мира, пожалуй начнем с него, а также в выборке топы из топов - кафе, рестораны и кофейни.

# In[100]:


top_category = ['кафе', 'ресторан', 'кофейня']
top_data = data.query('(street_raw == "мира") & (category in @top_category)').reset_index(drop=True)

data_dict = {
    'top_data': top_data,
    'top_data_cafe': top_data.query('category == "кафе"').reset_index(drop=True),
    'top_data_coffee_shop': top_data.query('category == "кофейня"').reset_index(drop=True),
    'top_data_rest': top_data.query('category == "ресторан"').reset_index(drop=True)
}

for key, value in data_dict.items():
    print(key)
    display(value[['category', 'hours', 'rating', 'middle_avg_bill', 'seats']].describe())


# Таким образом мы получили усредненный образ среди топ-3 заведений:
#     
#     - кафе - рейтинг около 4.2, средний счет - 660, мест - 60
#     - кофейня - рейтинг около 4.2, средний счет  - 850, мест - 80
#     - ресторан - рейтинг 4.4, средний счет - 650, мест - 86
# 

# [Перейти к разделу содержания "Этап 3"](#этап_3)
# 
# <a id='этап_3_1'></a>

# ## Этап. Детализируем исследование: открытие кофейни.
# 

# In[101]:


data_coffee_full_dics = data.query ('category == "кофейня"')
data_coffee = data.query ('category == "кофейня"')


# ### Количество кофейн в районе

# In[102]:


# всего 1350 кофейн в датасете
data_coffee_full_dics ['address'].count ()


# На текущем этапе полные названия районов нам не нужны, а общее восприятие графиков они портят, заменим в этой таблице названия на аббревиатуры

# In[103]:


replacement_dict = {
    'Центральный административный округ': 'ЦАО',
    'Северный административный округ': 'САО',
    'Северо-Восточный административный округ': 'СВАО',
    'Западный административный округ': 'ЗАО',
    'Южный административный округ': 'ЮАО',
    'Восточный административный округ': 'ВАО',
    'Юго-Западный административный округ': 'ЮЗАО',
    'Юго-Восточный административный округ': 'ЮВАО',
    'Северо-Западный административный округ': 'СЗАО',
}


data_coffee['district'] = data_coffee_full_dics['district'].replace(replacement_dict)


# ### Распределение кофейн по округам

# In[104]:


data_coffee_area = data_coffee.groupby ('district') ['category'].count ()
data_coffee_area = data_coffee_area.reset_index ().sort_values (by = 'category', ascending = False).reset_index (drop = True)


# In[105]:


# Ожидаемо ЦАО впереди, с отрывом более чем в 2 раза
data_coffee_area


# <a id='этап_3_2'></a>

# In[106]:


# Предпологаю, что график в разных срезах по кофейням буду строить часто в этом этапе,
# поэтому сделаем функцию, которая будут строить его каждый раз

def plot_coffee_area(data, column, name_plot='График', name_x = 'ось Х'):

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='district', y=column, data=data, ci=None, ax=ax)  
    ax.set_title(f"Распределение кофейн {name_plot} по административным округам", fontsize=15)
    ax.set_ylabel(f"{name_x}")  
    plt.grid(axis='y', linestyle='--', alpha=0.7)   
    ax.set_xlabel("Районы")
    ax.tick_params(labelsize=12,rotation=25)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    for bar in ax.patches:
        ax.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 5),
                   textcoords='offset points')
    plt.tight_layout()
    plt.show()


# In[107]:


plot_coffee_area (data_coffee_area, 'category', '', 'Количество кофейн')


# ### Круглосуточные объекты

# In[108]:


data_coffee_24_7 = data_coffee [data_coffee['is_24/7'] == True]
data_coffee_24_7_area = data_coffee_24_7.groupby ('district') ['category'].count ()
data_coffee_24_7_area = data_coffee_24_7_area.reset_index ().sort_values (by = 'category', ascending = False).reset_index (drop = True)


# In[109]:


plot_coffee_area (data_coffee_24_7_area, 'category', '24/7', 'Количество кофейн')


# Интересные результаты, не в плане распределения, они как раз ожидаемые (ЦАО лидер по количеству кофеин в целом), а плане наличия кофеин в режиме 24/7. А в части концентрации в ЦАО стоит только дополнить, ЦАО - округ, где люди больше всего отдыхают, часто гуляют до поздней ночи, потребность видимо есть, но не большая.

# ### Рейтинг кофейн

# In[110]:


rating_table_coffee = data_coffee.groupby('district')['rating'].agg('mean').round(4)
rating_table_coffee = rating_table_coffee.reset_index().sort_values('rating', ascending=False). reset_index(drop=True)


# Можем отметить, что средний рейтинг кофейн не сильно отличается по округам, но все таки, самый высокий 4.34 в ЦАО, высокая конкуренция заставляет стараться лучше? 

# In[111]:


rating_table_coffee 


# In[112]:


plot_coffee_area (rating_table_coffee, 'rating', '', 'Средний рейтинг')


# In[113]:


# для вывода, нужно выделить топовый объект в ЗАО, как в сетевом сегменте так и в частном
zao_data_coffee = data_coffee_full_dics.query('district == "Западный административный округ"').reset_index()


# In[114]:


max_rating_row = zao_data_coffee[zao_data_coffee['chain'] == 1].nlargest(1, 'rating')
max_rating_row


# In[115]:


max_rating_row = zao_data_coffee[zao_data_coffee['chain'] == 0].nlargest(1, 'rating')
max_rating_row


# In[116]:


rating_table_coffee_full_dics = data_coffee_full_dics.groupby('district')['rating'].agg('mean').round(4)
rating_table_coffee_full_dics = rating_table_coffee_full_dics.reset_index().sort_values('rating', ascending=False). reset_index(drop=True)


# In[117]:


moscow_lat, moscow_lng = 55.751244, 37.618423
# создаём карту Москвы
m_coffe = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=rating_table_coffee_full_dics,
    columns=['district', 'rating'],
    key_on="feature.name",
    fill_color='YlGn',
    fill_opacity=0.7,
    legend_name='Медианный рейтинг кофейн по районам',
).add_to(m_coffe)

m_coffe


# С ЦАО уже было понятно на прошлом шаге, а вот на карте более наглядно получается - САО и СЗАО показывают чуть более лучшие результаты, в отличии от других округов Москвы (кроме ЦАО)

# ### Распределение посадочных мест

# In[118]:


seats_table_coffee = data_coffee.groupby('district')['seats'].agg('mean').round(1)
seats_table_coffee = seats_table_coffee.reset_index().sort_values('seats', ascending=False). reset_index(drop=True)


# In[119]:


seats_table_coffee


# In[120]:


plot_coffee_area (seats_table_coffee, 'seats', '', 'Среднее количество посадочных мест')


# ### Средний чек

# In[121]:


bill_table_coffee = data_coffee.groupby('district')['middle_avg_bill'].agg('mean').round(1)
bill_table_coffee = bill_table_coffee.reset_index().sort_values('middle_avg_bill', ascending=False). reset_index(drop=True)


# In[122]:


bill_table_coffee


# In[123]:


plot_coffee_area (bill_table_coffee, 'middle_avg_bill', '', 'Средний чек')


# ### Стоимость чашки кофе
# 
# На какую стоимость чашки капучино стоит ориентироваться при открытии и почему?

# Для начала стоит посмотреть на стоимость чашки кофе по всем районам. За ориентир возьмем медиану, так как пропусков очень много, а среднее будет искажать результаты

# In[124]:


data_coffee_price_cup = data_coffee.groupby('district')['middle_coffee_cup'].agg ('median').reset_index()


# In[125]:


data_coffee_price_cup = data_coffee_price_cup.sort_values(by = 'middle_coffee_cup', ascending=False).reset_index(drop=True)


# In[126]:


data_coffee_price_cup


# In[127]:


plot_coffee_area (data_coffee_price_cup, 'middle_coffee_cup', 'по цене за чашку кофе', 'Цена за чашку')


# Что мы имеет в результате - топ рейтинга и топ по количеству объектов - ЦАО, но самый дорогой кофе в ЮЗАО

# ### **Вывод этапа 3**
# 
#     - Всего в датасете 1350 кофеин, но они распределены по округам не равномерно, лидер- ЦАО, с отрывом более чем в 2 раза - 413 заведений против ближайшего района в 180 заведений, конкуренция высокая.
#     - Имеются круглосуточные заведения, которых опять больше всего в ЦАО - спрос рождает предложение, но важно понимать, что круглосуточные кофейни, в большей степени редкость, так как такой формат работы прерогатива баров. в ЦАО - 36, а вот в ЮВАО (спальный район) - всего 1
#     - распределение по рейтингу более или менее равномерное - с небольшим отрывом впереди ЦАО 4.33 средний рейтинг. Антилидер ЗАО - 4.19
#     - Средняя цена чашки кофе немного удивила - самый дорогой кофе в ЮЗАО (199 руб), второе место лидер всех исследований - ЦАО с 190 руб, а вот антилидер по рейтингу занимает 3-е место с 187 руб за чашку - отзывы самые плохие, а цена кофе стремится к топам. 
#     

# ## Общий вывод
# 
# 

# В рамках настоящего исследования мы провели анализ заведений Москвы и получили следующие результаты:
# 
#     1.Распределение заведений по категориям.
# 
#     Кафе является лидером с количеством заведений равным 2378, за ним следуют рестораны (2043) и кофейни (1413). Булочные представлены наименьшим количеством заведений – 256.
# 
#     2. Долевое распределение заведений по типам на рынке общественного питания:
# 
#     Топ-3 категории (кафе, рестораны и кофейни) занимают 69% рынка, остальные доли распределяются следующим образом:
#         
#         - Пабы/бары – 9%
#         - Пиццерии – 8%
#         - Заведения быстрого питания – 7%
#         - Столовые – 4%
#         - Булочные – 3%
# 
#     3. Распределение посадочных мест.
#     Максимальное количество посадочных мест отмечено в ресторанах (среднее значение – 86), за ними следуют пабы/бары и кофейни с примерно одинаковым числом мест – 82 и 80 соответственно. Меньше всего посадочных мест в пиццериях и булочных.
# 
#     4. Распределение сетевых и несетевых заведений.
# 
#     В Москве около 62.7% заведений являются несетевыми. Существуют характерные признаки распределения между сетевыми и несетевыми заведениями. Например, в категориях булочных, кофеен и пиццерий преобладают сетевые заведения. В то время как пабы/бары, кафе, рестораны и столовые в основном представлены несетевыми объектами.
# 
#     5. Топ-15 сетей на рынке:
# 
#     Первое место среди сетей занимает "Шоколадница" с 75 заведениями, на втором и третьем месте находятся "Домино’с Пицца" (71 заведение) и "Додо пицца" (24 заведения) соответственно.
# 
#     6. Распределение заведений по округам:
# 
#     Центральный административный округ (ЦАО) является безусловным лидером по количеству заведений с 2154 объектами. Самое близкое к ЦАО количество заведений имеет Северный административный округ (САО) с 1738 объектами. Затем следуют Южный административный округ (ЮАО) с 1300 заведениями и Восточный административный округ (ВАО) с 1140 заведениями. Меньше всего заведений находится в Северо-Восточном административном округе (СВАО) – 698 объектов.
# 
#     6. Среднее количество посадочных мест по округам:
# 
#     Максимальное среднее количество посадочных мест отмечается в Южном административном округе (ЮАО) – около 92 мест. За ним следуют ЦАО (89 мест), САО (86 мест) и ВАО (83 места). Наименьшее среднее количество мест отмечается в Северном административном округе (САО) – около 74 мест.
# 
#     7. Округ с наибольшей долей сетевых заведений:
# 
#     Южный административный округ (ЮАО) является лидером по доле сетевых заведений с более чем 47% сетевых объектов. Затем идут Северный административный округ (САО) и Восточный административный округ (ВАО) с долями около 43% и 42% соответственно. Наименьшая доля сетевых заведений отмечается в Западном административном округе (ЗАО) – около 35%.
# 
# 
#     8. Топ-15 улиц по размещению заведений:
# 
#     По результатам исследования, самая популярная улица для размещения заведений в Москве - это проспект Мира. Она занимает лидирующую позицию по всем категориям заведений. Также значительное количество заведений расположено на Ленинском проспекте, проспекте Вернадского и Профсоюзной улице. График показывает, что на проспекте Мира преобладают кафе, рестораны и кофейни, а на улице Миклухо-Маклая заведений меньше всего. Проспект Мира и другие популярные улицы в основном представляют туристическо-развлекательный сектор.
#     
#     9. Улицы с одним заведением:
# 
#     В датасете обнаружено, что 468 улиц имеют всего одно заведение. Большинство таких улиц находятся в категории кафе (166 случаев), а наименьшее количество улиц с одним заведением принадлежит категории булочных (7 случаев). Стоит отметить, что количество булочных в целом невелико. Улицы с единственным заведением встречаются в каждом округе, но наибольшая концентрация таких улиц наблюдается в Центральном административном округе (ЦАО). Вероятно, это небольшие проулки или проезды.
# 
#     10. Средний чек по округам:
#     Исследование показало, что самый высокий средний чек отмечается в Западном административном округе (ЗАО) и Центральном административном округе (ЦАО). Если обратиться к таблицам, то видно, что в этих округах средний чек превышает средний чек в других округах Москвы примерно в 1,5-2 раза.
#     
#     Эти результаты были получены в ходе исследования заведений общественного питания в Москве и отражают текущую ситуацию. Важно учесть, что данные могут незначительно измениться со временем, так как рынок заведений общественного питания динамичен и подвержен изменениям.
# 
# 
# *В рамках таргетированного исследования заведений кофеин, мы предоставляем следующие комментарии и рекомендации:*
# 
#     В административной черте Москвы насчитывается 1350 кофеин. Из них более 400 расположены в ЦАО, а затем следуют САО, СВАО и ЗАО с 180, 155 и 146 соответственно.
#     Усредненные ориентиры для категории "кофейня" включают рейтинг около 4.2, средний счет в размере 850 рублей и вместимость около 80 человек.
# 
#     При выборе локации для открытия нового объекта стоит обратить внимание на ЦАО и ЗАО по следующим причинам:
# 
#         Плюсы ЦАО:
# 
#             - Высокая проходимость и охват, а также статус туристической зоны.
#             - Второе место по стоимости чашки кофе – около 190 рублей.
#             - Высокая активность даже в ночное время.
# 
#         Минусы ЦАО:
# 
#             - Высокая конкуренция.
#             - Высокий средний рейтинг окружающих заведений.
#         
#         Плюсы ЗАО:
# 
#             - Достаточно высокая средняя стоимость чашки кофе – около 187 рублей.
#             - Низкий средний рейтинг конкурентов, что позволит выделиться на их фоне при поддержании уровня удовлетворенности гостей.
#             - Большое количество бизнес-центров и узловых станций, что привлечет компании для встреч после работы или в течение дня.
# 
#         Минусы ЗАО:
# 
#             - Меньшая проходимость и больше "спальных районов".
# 
#     Общие рекомендации для работы в ЦАО:
# 
#         - Рекомендуется иметь 80-90 посадочных мест.
#         - Стоимость чашки кофе должна быть в диапазоне 185-195 рублей. На начальном этапе можно рассмотреть возможность установки цены ниже 185 рублей, если это не повлияет на маржинальность.
#         - Следует проработать меню таким образом, чтобы средний чек составлял около 700-800 рублей.
#         - При выборе локации стоит обратить внимание на проспект Мира и Пятницкую улицу.
# 
#     Общие рекомендации для работы в ЗАО:
# 
#         - Рекомендуется иметь около 100 посадочных мест.
#         - Стоимость чашки кофе должна быть в диапазоне 180-190 рублей. На начальном этапе можно рассмотреть возможность установки цены ниже 185 рублей, если это не повлияет на маржинальность.
#         - Следует проработать меню таким образом, чтобы средний чек составлял около 600-700 рублей.
#         - Важным аспектом будет дополнительное изучение рынка в данном районе для выявления положительной практики. Низкий рейтинг в ЗАО можно использовать в свою пользу.
#         - Рекомендуется обратить внимание на топовые объекты ЗАО, такие как "Кофемания" на Осенней улице, 11 (сетевая кофейня), и "Дворик Невского" на проспекте Вернадского (не сетевая кофейня).
# 
# 
# 
# 
#     Кроме указанных рекомендаций, следующие аспекты также могут быть дополнительно рассмотрены:
# 
#     1. Исследование целевой аудитории: Проведите более подробное исследование целевой аудитории в выбранных районах (ЦАО и ЗАО). Узнайте о их предпочтениях, потребностях и привычках потребления кофе. Это поможет более точно адаптировать меню, цены и общий подход к организации кофейни.
# 
#     2. Анализ конкурентов: Проведите анализ конкурентов в выбранных районах. Определите их преимущества и слабые места, чтобы лучше понять, как выделиться на фоне других заведений. Обратите внимание на успешные практики конкурентов и возможности для инноваций.
# 
#     3. Уникальное предложение: Разработайте уникальное предложение, которое будет выделять вашу кофейню среди конкурентов. Это может быть особый вид кофе, уникальное меню, интересный дизайн интерьера или особая атмосфера. Убедитесь, что ваше предложение будет привлекательным и отличаться от того, что предлагают другие заведения.
# 
#     4. Маркетинговые и рекламные активности: Разработайте эффективные маркетинговые и рекламные стратегии для привлечения клиентов. Используйте социальные сети, рекламные кампании, участие в мероприятиях и сотрудничество с местными сообществами для повышения узнаваемости и привлечения новых посетителей.
# 
#     5. Обратная связь и уровень обслуживания: Обратите особое внимание на обратную связь от клиентов и поддержание высокого уровня обслуживания. Учтите комментарии и рекомендации гостей, чтобы постоянно улучшать качество предоставляемых услуг.
# 
#     6. Постоянное развитие и инновации: Следите за тенденциями в индустрии кофе и внедряйте инновации в вашу кофейню. Внимательно изучайте новые виды кофе, методы приготовления, тренды в дизайне и обслуживании клиентов. Постоянное развитие и адаптация помогут вашей кофейне
# 
# 
# Прим. рекомендация по тематическому интерьеру не озвучивается, так как, во-первых, исследование на данную тематику не проводилось, а во-вторых, из установочного задания и так следуюет, что заказчик планирует открыть кафе как в сериале "Друзья"- в стиле Central Perk.

# ## Презентация

# [Презентация на ЯД](https://disk.yandex.ru/i/gshb1D4VeY7E9A)
# 

# Презентация максимально лаконична, и предполагает пояснительные комментарии от докладчика.
