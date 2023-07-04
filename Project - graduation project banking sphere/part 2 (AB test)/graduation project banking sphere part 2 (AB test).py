#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Итоговый-сборный-проект-(этап-А/В-тестирование).-Описание" data-toc-modified-id="Итоговый-сборный-проект-(этап-А/В-тестирование).-Описание-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Итоговый сборный проект (этап А/В тестирование). Описание</a></span><ul class="toc-item"><li><span><a href="#Техническое-задание." data-toc-modified-id="Техническое-задание.-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Техническое задание.</a></span></li><li><span><a href="#Описание-датасетов" data-toc-modified-id="Описание-датасетов-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Описание датасетов</a></span></li><li><span><a href="#Общий-план-работы." data-toc-modified-id="Общий-план-работы.-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Общий план работы.</a></span></li></ul></li><li><span><a href="#Подготовка-инструментов-для-работы" data-toc-modified-id="Подготовка-инструментов-для-работы-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Подготовка инструментов для работы</a></span><ul class="toc-item"><li><span><a href="#Инструменты-для-исследования" data-toc-modified-id="Инструменты-для-исследования-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Инструменты для исследования</a></span></li><li><span><a href="#Первичное-знакомство-с-данными" data-toc-modified-id="Первичное-знакомство-с-данными-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Первичное знакомство с данными</a></span></li><li><span><a href="#Используем-инструменты-для-базовых-проверок" data-toc-modified-id="Используем-инструменты-для-базовых-проверок-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Используем инструменты для базовых проверок</a></span><ul class="toc-item"><li><span><a href="#Провека-на-очевидные-дуликаты-и-пропуски" data-toc-modified-id="Провека-на-очевидные-дуликаты-и-пропуски-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Провека на очевидные дуликаты и пропуски</a></span></li><li><span><a href="#Общее-изучение-наполненния-датасета" data-toc-modified-id="Общее-изучение-наполненния-датасета-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>Общее изучение наполненния датасета</a></span></li></ul></li></ul></li><li><span><a href="#Детальное-изучение-содержаний-датасетов,-и-подготовка-данных" data-toc-modified-id="Детальное-изучение-содержаний-датасетов,-и-подготовка-данных-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Детальное изучение содержаний датасетов, и подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Изучим-датасет-event" data-toc-modified-id="Изучим-датасет-event-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Изучим датасет event</a></span><ul class="toc-item"><li><span><a href="#Анализ-пропусков-в-столбце-details-датасета-event,-и-принятие-решение,-что-делаем-с-пропусками" data-toc-modified-id="Анализ-пропусков-в-столбце-details-датасета-event,-и-принятие-решение,-что-делаем-с-пропусками-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Анализ пропусков в столбце details датасета event, и принятие решение, что делаем с пропусками</a></span></li></ul></li><li><span><a href="#Изучим-датасет-participants" data-toc-modified-id="Изучим-датасет-participants-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Изучим датасет participants</a></span></li><li><span><a href="#Изучим-датасет-users" data-toc-modified-id="Изучим-датасет-users-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Изучим датасет users</a></span></li><li><span><a href="#Изучение-датасета-marketing_events" data-toc-modified-id="Изучение-датасета-marketing_events-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Изучение датасета marketing_events</a></span></li><li><span><a href="#Подготовка-объединенного-датасета-для-исследования" data-toc-modified-id="Подготовка-объединенного-датасета-для-исследования-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Подготовка объединенного датасета для исследования</a></span></li><li><span><a href="#Распределение-&quot;пересекающихся-клиентов&quot;" data-toc-modified-id="Распределение-&quot;пересекающихся-клиентов&quot;-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Распределение "пересекающихся клиентов"</a></span><ul class="toc-item"><li><span><a href="#Вывод-и-принятие-решения-в-части-отбора-пользователей-по-тестам" data-toc-modified-id="Вывод-и-принятие-решения-в-части-отбора-пользователей-по-тестам-3.6.1"><span class="toc-item-num">3.6.1&nbsp;&nbsp;</span>Вывод и принятие решения в части отбора пользователей по тестам</a></span></li></ul></li></ul></li><li><span><a href="#EDA" data-toc-modified-id="EDA-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>EDA</a></span><ul class="toc-item"><li><span><a href="#Формирование-итогового-датасета-для-исследований" data-toc-modified-id="Формирование-итогового-датасета-для-исследований-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Формирование итогового датасета для исследований</a></span></li><li><span><a href="#Посмотрим-на-распределение-регионов-среди-новых-пользователей" data-toc-modified-id="Посмотрим-на-распределение-регионов-среди-новых-пользователей-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Посмотрим на распределение регионов среди новых пользователей</a></span></li><li><span><a href="#Расчет-количество-событий-для-каждого-пользователя" data-toc-modified-id="Расчет-количество-событий-для-каждого-пользователя-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Расчет количество событий для каждого пользователя</a></span></li><li><span><a href="#Расчет-количества-событий-по-дням" data-toc-modified-id="Расчет-количества-событий-по-дням-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Расчет количества событий по дням</a></span></li><li><span><a href="#Отработка-продуктовой-гипотезы" data-toc-modified-id="Отработка-продуктовой-гипотезы-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Отработка продуктовой гипотезы</a></span><ul class="toc-item"><li><span><a href="#Воронки-продаж" data-toc-modified-id="Воронки-продаж-4.5.1"><span class="toc-item-num">4.5.1&nbsp;&nbsp;</span>Воронки продаж</a></span><ul class="toc-item"><li><span><a href="#общая-воронка" data-toc-modified-id="общая-воронка-4.5.1.1"><span class="toc-item-num">4.5.1.1&nbsp;&nbsp;</span>общая воронка</a></span></li><li><span><a href="#Воронки-по-группам-А-и-В" data-toc-modified-id="Воронки-по-группам-А-и-В-4.5.1.2"><span class="toc-item-num">4.5.1.2&nbsp;&nbsp;</span>Воронки по группам А и В</a></span></li></ul></li></ul></li><li><span><a href="#AB-Тесты" data-toc-modified-id="AB-Тесты-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>AB Тесты</a></span><ul class="toc-item"><li><span><a href="#Подготовка.-Кумулятивная-конверсия" data-toc-modified-id="Подготовка.-Кумулятивная-конверсия-4.6.1"><span class="toc-item-num">4.6.1&nbsp;&nbsp;</span>Подготовка. Кумулятивная конверсия</a></span></li><li><span><a href="#Конверсия-и-ее-графики" data-toc-modified-id="Конверсия-и-ее-графики-4.6.2"><span class="toc-item-num">4.6.2&nbsp;&nbsp;</span>Конверсия и ее графики</a></span></li><li><span><a href="#Статистическая-разница-(тесты)" data-toc-modified-id="Статистическая-разница-(тесты)-4.6.3"><span class="toc-item-num">4.6.3&nbsp;&nbsp;</span>Статистическая разница (тесты)</a></span></li><li><span><a href="#Вывод-этапа." data-toc-modified-id="Вывод-этапа.-4.6.4"><span class="toc-item-num">4.6.4&nbsp;&nbsp;</span>Вывод этапа.</a></span></li></ul></li></ul></li><li><span><a href="#Общий-вывод" data-toc-modified-id="Общий-вывод-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Общий вывод</a></span></li></ul></div>

# # Итоговый сборный проект (этап А/В тестирование). Описание
# 
# Наша задача — провести оценку результатов A/B-теста. В нашем распоряжении есть датасет с действиями пользователей, техническое задание и несколько вспомогательных датасетов.
# Необходимо оценить корректность проведения теста и проанализирувать его результаты.
#     
#     Чтобы оценить корректность проведения теста:
#     
#     - необходимо удостовериться, что нет пересечений с конкурирующим тестом и нет пользователей, участвующих в двух группах теста одновременно;
#     - необходимо проверить равномерность распределения пользователей по тестовым группам и правильность их формирования.
# 

# ## Техническое задание.
# 
#     - Название теста: recommender_system_test;
#     - Группы: А (контрольная), B (новая платёжная воронка);
#     - Дата запуска: 2020-12-07;
#     - Дата остановки набора новых пользователей: 2020-12-21;
#     - Дата остановки: 2021-01-04;
#     - Аудитория: 15% новых пользователей из региона EU;
#     - Назначение теста: тестирование изменений, связанных с внедрением улучшенной рекомендательной системы;
#     - Ожидаемое количество участников теста: 6000.
#     
#     
#     Ожидаемый эффект (продуктовая гипотеза): за 14 дней с момента регистрации в системе пользователи покажут улучшение каждой метрики не менее, чем на 10%:
#             - конверсии в просмотр карточек товаров — событие product_page
#             - просмотры корзины — product_cart
#             - покупки — purchase.

# ## Описание датасетов
# 
# Для проведения исследования нам предоставленны 4 датасета.
# 
#     1. /datasets/ab_project_marketing_events.csv — календарь маркетинговых событий на 2020 год;
#         
#     Структура файла:
#   
#     - name — название маркетингового события;
#     - regions — регионы, в которых будет проводиться рекламная кампания;
#     - start_dt — дата начала кампании;
#     - finish_dt — дата завершения кампании.
# 
# 
#     2. /datasets/final_ab_new_users.csv — все пользователи, зарегистрировавшиеся в интернет-магазине в период с 7 по 21 декабря 2020 года;
#     
#     Структура файла:
# 
#     - user_id — идентификатор пользователя;
#     - first_date — дата регистрации;
#     - region — регион пользователя;
#     - device — устройство, с которого происходила регистрация.
# 
#     
#     3. /datasets/final_ab_events.csv — все события новых пользователей в период с 7 декабря 2020 по 4 января 2021 года;
# 
#     Структура файла:
#     
#     - user_id — идентификатор пользователя;
#     - event_dt — дата и время события;
#     - event_name — тип события;
#     - details — дополнительные данные о событии. Например, для покупок, purchase, в этом поле хранится стоимость покупки в долларах.
# 
# 
#     4. /datasets/final_ab_participants.csv — таблица участников тестов.
# 
#     Структура файла:
#     
#     - user_id — идентификатор пользователя;
#     - ab_test — название теста;
#     - group — группа пользователя.
# 

# ## Общий план работы.
# 
#     1. Изучение и предобработка данных.
#         - Оценка типов данных, принятие решение о корректировке типов данных.
#         - Пропуски и дубликаты.
#         - оценка корректности проведения теста, с дополнительным изучением:
#             - соответствия данных техническому заданию.
#             - времени проведения теста (с учетом заявленного временного интервала теста, и выявлением других активностей)
#             - проверка на предмет пересечения с конкурируещим тестом, а также вхождения пользователей в несколько групп одновременно.
# 
#     2. Исследовательский анализ.
#     - Проверим распределение пользователей в выборках событий одинаково распределены.
#     - Распределение в выборках по дням.
#     - оценим изменение конверсии на разных этапах в каждой выборке.
#     
#     3. Проведеие оценку результатов A/B-тестирования:
#     - Что можно сказать про результаты A/B-тестирования?
#     - Проверим статистическую разницу долей z-критерием.
# 
#     4. Итоговые выводы. Сделаем общее заключение о корректности проведения теста.
# 

# # Подготовка инструментов для работы

# Загрузим все библиотеки, которые могут пригодится при выполнении проекта

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
from matplotlib.dates import DateFormatter
from matplotlib import colors
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import seaborn as sns
get_ipython().system('pip install tabulate')
from tabulate import tabulate
from scipy import stats as st
from scipy.stats import mannwhitneyu
from statistics import mode
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import math as mth

import warnings
warnings.filterwarnings('ignore')


# И обновления к ним

# In[2]:


#!pip install --upgrade pandas seaborn
#!pip install --upgrade pandas
#!pip install --upgrade seaborn


# In[3]:


try:
    event = pd.read_csv (r'C:\Users\PC_Maks\Desktop\study\final_project\AB_test_final\final_ab_events.csv',                         sep = ",", parse_dates = ['event_dt'])
    users = pd.read_csv (r'C:\Users\PC_Maks\Desktop\study\final_project\AB_test_final\final_ab_new_users.csv',                         sep = ",", parse_dates = ['first_date'])
    participants = pd.read_csv (r'C:\Users\PC_Maks\Desktop\study\final_project\AB_test_final\final_ab_participants.csv', sep = ",")
    marketing_events = pd.read_csv (r'C:\Users\PC_Maks\Desktop\study\final_project\AB_test_final\ab_project_marketing_events.csv',                                    sep = ",", parse_dates=['start_dt', 'finish_dt'])
except: 
    event = pd.read_csv ('/datasets/final_ab_events.csv', sep=',', parse_dates = ['event_dt']) 
    users = pd.read_csv ('/datasets/final_ab_new_users.csv', sep=',', parse_dates = ['first_date'])
    participants = pd.read_csv ('/datasets/final_ab_participants.csv', sep=',')
    marketing_events = pd.read_csv ('/datasets/ab_project_marketing_events.csv', sep=',', parse_dates=['start_dt', 'finish_dt'])
    
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['figure.figsize'] = (15, 8) 


# Из ТЗ и пописания мы знаем в каких столбцах в каждом из датасетов должны быть данные дата и время. при загрузке можем сразу изменить тип данных

# ## Инструменты для исследования

# На текущий момент в запасах появились различиные унифицированные функции  для проверки датасетов на пропуски и дубли. Не будем нарушать традицию, и используем заготовки

# In[4]:


def check(data):
    ''' Функция проверяет датасет на очивидные дубли, при выявленпии удлает их, 
    и сообщает сколько было удалено как в абсолютных значениях, так и в относительных'''
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


# In[5]:


def check_unique(data):
    '''Функция возращает уникальные значения столбцов, минимальную и максимальную дату (если есть), а также, 
    если уникальных значений в столбце меньше 10, возвращает перечень значений'''
    
    for col in data.select_dtypes(include=['object']):
        print(f"Уникальные значения в столбце {col}:")
        print(data[col].unique())
        print(f"Количество уникальных значений: {data[col].nunique()}")
        print('---------------------')

    for col in data.select_dtypes(include=['datetime64', 'float64', 'int64']):
        print(f"Диапазон значений в столбце {col}:")
        print(f"Минимальное значение: {data[col].min()}")
        print(f"Максимальное значение: {data[col].max()}")
        print('---------------------')

    for col in data.select_dtypes(include=['int64', 'float64']):
        if len(data[col].unique()) > 10:
            print(f"В столбце {col} более 10 уникальных значений")
        else:
            print(f"Уникальные значения в столбце {col}:")
            print(data[col].unique())
        print(f"Количество уникальных значений: {data[col].nunique()}")
        print('---------------------')


# ## Первичное знакомство с данными 

# In[6]:


display (event.info())
event


# In[7]:


display (users.info())
users


# In[8]:


display (participants.info())
participants


# In[9]:


display (marketing_events.info())
marketing_events


# Исходя из первичных наблюдений, можно отметить следующее:
# 
# Формирование тестовых групп происходило вне периода проведения других активностей.
# Однако само проведение теста приходилось на период с 7 декабря 2020 года по 4 января 2021 года, в котором происходили два дополнительных события: новогодний (с 25 декабря 2020 года по 3 января 2021 года) и новогодняя лотерея (с 30 декабря 2020 года по 7 января 2021 года), но только одно из них относится к EU.
# Стоит отметить, что в период формирования тестовых групп для проведения теста события новогоднего и новогодней лотереи не проводились.
# Данная информация важна для дальнейшего анализа результатов теста и понимания возможных влияний других активностей на его результаты. Однако, для получения полной картины и оценки влияния этих активностей на результаты теста, требуется дополнительный анализ и сбор более подробной информации о характере и длительности данных активностей.

# Типы данных скорректированны при загрузке ДС верно. Дополнительные изменения в данных не требуются

# ## Используем инструменты для базовых проверок

# для удобстава вызова функций, соберем все датасеты в один список

# In[10]:


all_datasets = [event, users, participants, marketing_events]
dataset_names = ['event', 'users', 'participants', 'marketing_events']


# ### Провека на очевидные дуликаты и пропуски 
# 

# In[11]:


for i, df in enumerate(all_datasets):
    dataset_name = dataset_names[i]
    print('Название датасета -', dataset_name)  
    check(df)  


# Очивидных дубликатов нет, выявлены пропуски в details более 377000 строк, а это около 85,75%

# ### Общее изучение наполненния датасета

# In[12]:


for i, df in enumerate(all_datasets):
    dataset_name = dataset_names[i]
    print('Название датасета -', dataset_name)  
    check_unique(df) 


# Изучение наполнения датасетов:
# 
#     1. датасет event:
#         - 58703 уникальных пользователей из 440317 строк записей
#         - столбец event_name - уникальные значения - 4 ('purchase' 'product_cart' 'product_page' 'login') - (покупка, корзина товаров, страница товара, вход в систему)
#         - столбец event_dt находится в диапазоне с 7 по 31 декабря, дальше данных нет
#         - details - содержит 4 уникальных значения  99.99/9.99/4.99/499.99, но конечно же и пропуски Nan (85%+)
#         
#     2. датасет users (пользователи зарегистрировались в магазине с 7 по 20 декабря)
#         - уникальных пользователей 61733
#         - 4 региона - ['EU' 'N.America' 'APAC' 'CIS']
#         - 4 устройства - ['PC' 'Android' 'iPhone' 'Mac']
#         - дата регистрации пользователей с 7 по 23 декабря - в ТЗ заявлено, что группа формировалась с 7 по 20 декабря (ошибка выгрузки? пока представляется 2 варианта - 1 отрезать данные после 20 декабря, и второй вариант тоже подразумевает отрезание аномальных дат, но есть нюанс, регионы у нас вся планета, с большой разницей в часовых поясах, подумаем поможем ли мы отработать данный столбец с учетом корректировки данных, хотя данные могли быть загружены без относительно регионального времени, а только по серверному стандарту UTC-0, но это мы не узнаем без обращения к коллегам, которые делали выгрузку, поэтому вероятен первый вариант развития событий). Но перед тем, как принимать решение, проверим, что именно наш тест попал в этот период
#         
#     3. participants (участники тестов)
#         - 16666 уникальных пользователя
#         - две группы А и В
#         - 2 типа тестов 'recommender_system_test' 'interface_eu_test', нас по ТЗ интересует только 'recommender_system_test'
#         
#     4. датасет marketing_events содержит
#         - 14 различных эвентов
#         - период проведения каждого эвента - часть эвентов пересекается с периодом проведения нашего теста, но учитывая ТЗ - важно чтобы формирование пула пользователей было вне эвентов, а данное условие выполняется.
#         

# # Детальное изучение содержаний датасетов, и подготовка данных

# ## Изучим датасет event

# ### Анализ пропусков в столбце details датасета event, и принятие решение, что делаем с пропусками

# In[13]:


missing_details = event.query("details.isnull()")


# In[14]:


check_unique(missing_details)


# теперь мы явно понимаем, что пропуски в столбце details это не пропуски в обычно понимании, и по сути details  содержит только информацию о покупках, и все пропуски только для действий отличных от покупки - просмотр корзины, вход в систему, просмотр страницы с товаром 

# In[15]:


event ['event_name'].value_counts ()


# Распределение шагов представляется корректным, больше всего авторизаций, потом просмотр страницы с товаром, а покупка и просмотр корзины с товаром примерно равны (некоторые видимо покупают без посещения корзины), специфику структуры сайта, можно уточнить у коллег. 

# ## Изучим датасет participants
# Проверим состав и выявим двоеные вхождения по граппам и тестам пользователей

# In[16]:


# функция проверяет двойные вхождения уникальных пользователей в разные тесты/регионы и т.п.
def find_users_in_both_tests(dataset, col1, col2):
    users_errors = dataset.groupby(col1)[col2].agg('nunique').reset_index().query(f"{col2} > 1")
    return users_errors


# In[17]:


test = participants[participants['ab_test'] == 'recommender_system_test']
find_users_in_both_tests(test, 'user_id', 'group')


# In[18]:


overlapping_clients = find_users_in_both_tests(participants, 'user_id', 'ab_test')


# В датасете participants 2 группы теста, нужная нам recommender_system_test и interface_eu_test. Пользователи в нужном нам тесте пересекаются с группой пользователей interface_eu_test - 1602, двойных распределений не выявлено (ситуация когда пользователь и в группе А и в группе В в тесте recommender_system_test). Всего в группе recommender_system_test 6701 пользователь. 

# In[19]:


# сохраним в отдельный датасет нашу выборку по нужной категории теста для работы.

test_group = participants [participants['ab_test']=='recommender_system_test']

# и в отдельную выборку сохраним конкурирующий тест, может пригодится
rival_test = participants [participants['ab_test']=='interface_eu_test']


# посмотрим на распределение внутри выборки групп А и В

# In[20]:


test_group ['group'].value_counts ()


# Группа А имеет размер 3824, группа В 2877. Размер группы A больше, чем размер группы B. Это может повлиять на статистическую мощность и достоверность результатов тестирования. Нужно будет обратить внимание на это при анализе результатов и сравнении метрик между группами

# In[21]:


check_unique (test_group)


# ## Изучим датасет users
# 
# Из предварительного изучения мы уже знаем, что датасет users содержит данные по пользователям, региону, устройству и дате первой авторизации. Дата первой авторизации в рамках нашего теста - 7 декабря 2020 года, а дата остановки набора новых пользователей: 2020-12-21, но в датасете users крайняя дата - 23 декабря. будем срезать 

# In[22]:


total_rows = len (users)
users_correct = users.query ('first_date< "2020-12-22"')


# In[23]:


correct_rows = len(users_correct)
removed_rows = total_rows - correct_rows
display(f"Количество удаленных строк: {removed_rows}")


# In[24]:


check_unique (users_correct)


# Один из пунктов ТЗ - изучение пользователей из региона EU, посмотрим на количество пользователей из каждого региона, заодно проверим на "странности" (Например 1 пользователь но разные регионы)

# In[25]:


users_correct ['region'].value_counts ()


# Большенство из EU

# In[26]:


find_users_in_both_tests (users_correct, 'user_id', 'region')


# In[27]:


find_users_in_both_tests (users_correct, 'user_id', 'device')


# Пока остановимся на выполненной корректировки, так как дальше с учетом ТЗ будем вычленять нужных пользователей, и для работы формировать объедененый датасет

# ## Изучение датасета marketing_events

# In[28]:


test_start_date = pd.to_datetime('2020-12-07')
test_end_date = pd.to_datetime('2021-01-04')
intersected_events = marketing_events[
    (marketing_events['start_dt'] <= test_end_date) &
    (marketing_events['finish_dt'] >= test_start_date)
]
intersected_events


# В период набора новых пользователей и проведения теста проходило 2 акции, 1 на территории СНГ, и вторая на территории Европы и Северной Америки. Хоть на набор акции и не повлияли (периоды иные), на активность они точно могли повлиять. Это плохо. 

# ## Подготовка объединенного датасета для исследования 

# In[29]:


merged_data = pd.merge(participants, event, on='user_id', how='left')
merged_data['recommender_system_test'] = merged_data.apply(
    lambda row: 1 if row['ab_test'] == 'recommender_system_test' else 0, axis=1
)
merged_data['interface_eu_test'] = merged_data.apply(
    lambda row: 1 if row['ab_test'] == 'interface_eu_test' else 0, axis=1
)
merged_data.drop(['ab_test'], axis=1, inplace=True)


# Немного пересоберем таблицу, для удобства изучения. 

# In[30]:


grouped_data = merged_data.groupby(['recommender_system_test', 'interface_eu_test', 'group', 'event_name'])['user_id'].nunique().reset_index()
grouped_data['test'] = np.where(grouped_data['recommender_system_test'] == 1, 'recommender_system_test', 'interface_eu_test')
grouped_data.drop(['recommender_system_test', 'interface_eu_test'], axis=1, inplace=True)
grouped_data.rename(columns={'user_id': 'unique_users_count'}, inplace=True)
grouped_data = grouped_data[['test', 'group', 'event_name', 'unique_users_count']]
grouped_data


# Объединенный датасет готов, теперь можем приступить к исследованию распределения тестовых групп нашего теста и конкурируещего

# ## Распределение "пересекающихся клиентов" 

# Ранее мы уже подготовили список не уникальных клиентов overlapping_clients, повторим создание объединенной таблицы с количеством клиентов по группам, но только на выборке таких клиентов, а потом на выборке таких клиентов и клиентов которые попали только в наш тест 

# In[31]:


overlapping_clients


# In[32]:


merged_table_overlapping = pd.merge(merged_data, overlapping_clients, on='user_id', how='inner')


# In[33]:


merged_table_overlapping ['user_id'].nunique ()
# объединение прошло корректно.


# In[34]:


grouped_data_overlapping = merged_table_overlapping.groupby(['recommender_system_test', 'interface_eu_test', 'group', 'event_name'])['user_id'].nunique().reset_index()

grouped_data_overlapping['test'] = np.where(grouped_data_overlapping['recommender_system_test'] == 1,                                            'recommender_system_test', 'interface_eu_test')
grouped_data_overlapping.drop(['recommender_system_test', 'interface_eu_test'], axis=1, inplace=True)
grouped_data_overlapping.rename(columns={'user_id': 'unique_users_count'}, inplace=True)
grouped_data_overlapping = grouped_data_overlapping[['test', 'group', 'event_name', 'unique_users_count']]


# In[35]:


grouped_data_overlapping['total_count'] = grouped_data_overlapping.groupby(['test', 'event_name'])['unique_users_count'].transform('sum')

grouped_data_overlapping['ratio_in_test'] = (grouped_data_overlapping['unique_users_count'] / grouped_data_overlapping['total_count'] * 100).round (0)
grouped_data_overlapping.drop(['total_count'], axis=1, inplace=True)


# Внутри теста interface_eu_test распределение происходит равномерно, а так как категории у нас на оба теста в части активностей едины, можем утверждать, что внутри евентов нашего теста, дублирующие значения распределены также равномерно.  

# In[36]:


merged_data_fin = merged_data[~merged_data['user_id'].isin(overlapping_clients['user_id'])]


# In[37]:


merged_data_fin [merged_data_fin ['recommender_system_test']==1] ['user_id'].nunique()


# в случае удаления всех дублирующихся клиентов, уровень выборки меньше требования в ТЗ. Удаление не целесообразно

# In[38]:


# сделаем выборку пользователей, которые попали в оба теста и при этом находятся в группе А.
test_a_group = participants.query ('group=="A"')
users_errors = test_a_group.groupby('user_id', )['ab_test'].agg('nunique').reset_index().query(f"{'ab_test'} > 1")


# In[39]:


unique_user_id_group_a = users_errors['user_id'].unique()
unique_user_id_group_a.shape[0]


# In[40]:


merged_data_whithout_group_a = merged_data.query ('(recommender_system_test==1) and (user_id not in @unique_user_id_group_a)')


# In[41]:


distribution_by_group = merged_data_whithout_group_a.groupby('group')['user_id'].nunique()
distribution_by_group


# In[42]:


grouped_data_whithout_group_a = merged_data_whithout_group_a.groupby(['recommender_system_test', 'interface_eu_test', 'group', 'event_name'])['user_id'].nunique().reset_index()

grouped_data_whithout_group_a['test'] = np.where(grouped_data_whithout_group_a['recommender_system_test'] == 1,                                            'recommender_system_test', 'interface_eu_test')
grouped_data_whithout_group_a.drop(['recommender_system_test', 'interface_eu_test'], axis=1, inplace=True)
grouped_data_whithout_group_a.rename(columns={'user_id': 'unique_users_count'}, inplace=True)
grouped_data_whithout_group_a = grouped_data_whithout_group_a[['test', 'group', 'event_name', 'unique_users_count']]


# In[43]:


grouped_data_whithout_group_a


# In[44]:


grouped_data_whithout_group_a['total_count'] = grouped_data_whithout_group_a.groupby(['test', 'event_name'])['unique_users_count'].transform('sum')

grouped_data_whithout_group_a['ratio_in_test'] = (grouped_data_whithout_group_a['unique_users_count'] / grouped_data_whithout_group_a['total_count'] * 100).round (0)
grouped_data_whithout_group_a.drop(['total_count'], axis=1, inplace=True)


# In[45]:


grouped_data_whithout_group_a


# ### Вывод и принятие решения в части отбора пользователей по тестам

# По итогу изучения датасета, можем сделать следующий вывод, просто удалять дублирующихся клиентов мы не можем, так как выборка снижается ниже порогового значения. Мы уже знаем, что в нашем тесте клиентов группы В почти на 30% меньше чем клиентов группы А, учитывая что группа А - контрольная для обоих тестов, мы можем удалить дублирующихся клиентов из группы А. Однако, оставив клиентов из групп В, которые пересекаются в тестах, данные могут быть искажены. Даже после удаления  дублирующихся в группе А клиентов и оставления гибридной группы В, общая картина в соотношении пользователей в группахне сильно изменилась, общее соотношение в итоге в части действий 3 к 1 (группа А к группе В). 
# 
# Таким образом, взяв во внимание распределение активности внутри групп в соотношении 75% к 25%, учитывая что у нас изначально группа В была меньше, чем группа А, пересечение с другим тестом, а также выбранный период тестирования - рождественские праздники на территории основной части пользователей из списка (католические страны), уже сейчас могу прийти к выводу о нецелесообразности проведения АВ теста, его результаты будут не корректными, так как подготовка данных проведена с нарушением базовых принципов проведения тестировани.
# 
# Таким образом, из-за перекрытия групп, нерепрезентативности выборки и других нарушений базовых принципов проведения тестирования, рекомендуется пересмотреть подход к проведению теста или внести необходимые корректировки в подготовку данных для обеспечения корректности результатов.

# # EDA
# З.Ы. дальнейшее исследование не несет прикладного характера, больше является рядовой практикой

# ## Формирование итогового датасета для исследований
# 
# Сейчас у нас есть просто список новых клиентов в файле users, которые зарегистрировались в период с 07 по 21 декабря включительно. И есть сводная таблица, в которой собраны все клиенты попавшие в выборки тестов, в ДС merged_data_whithout_group_a собраны клиенты нужного нам теста, группа А и группа В, со всеми своими действиями, нужно проверить, все ли наши клиенты из merged_data_whithout_group_a, стали новыми пользователями в период указанный в ТЗ

# In[46]:


new_users = users['user_id'].unique()
new_users.shape[0]


# In[47]:


main_data = merged_data_whithout_group_a.query ('user_id in @new_users ')
# готово


# In[48]:


# дополним нашу основную таблицу данными о регионе клиента. 
main_data = main_data.merge(users[['user_id', 'region']], on='user_id', how='left')


# In[49]:


main_data.drop (['recommender_system_test', 'interface_eu_test'], axis=1, inplace=True)


# ## Посмотрим на распределение регионов среди новых пользователей

# In[50]:


value_region_test = main_data.groupby('region')['user_id'].nunique().reset_index(inplace=False)
total = value_region_test ['user_id'].sum()
value_region_test ['ration'] = (value_region_test ['user_id']/total*100).round()


# In[51]:


value_region_test


# Среди участников теста 94% - это пользователи из Европы. Но нужно посмотреть на картину в общем среди новых пользователей. 

# In[52]:


value_region_all = users.groupby('region')['user_id'].nunique().reset_index(inplace=False)
total = value_region_all ['user_id'].sum()
value_region_all


# In[53]:


value_region_all ['ration'] = (value_region_all ['user_id']/total*100).round()
value_region_all


# В общей массе новых пользователей, пользователи из Европы 75%, среди участников теста, пользователи из Европы 94% 

# In[54]:


value_region_test = value_region_test.merge (value_region_all, on='region')


# In[55]:


value_region_test = value_region_test.rename(columns={'user_id_x': 'test_users', 'user_id_y': 'all_users'})


# In[56]:


value_region_test.drop (['ration_x', 'ration_y'], axis=1, inplace=True)


# In[57]:


value_region_test ['ration']= (value_region_test ['test_users']/value_region_test['all_users']*100).round ()


# In[58]:


value_region_test


# Как видно из результатов расчетов, в нашу тестовую группу попадает только 13% из всех пользователей из Европы (прим.После чисток и корректировок с конкурирующим тестом)

# ## Расчет количество событий для каждого пользователя 

# In[59]:


# Для группы А
group_a_event_count = main_data[main_data['group'] == 'A'].groupby('user_id')['event_name'].count().reset_index(drop=False)
# Для группы Б
group_b_event_count = main_data[main_data['group'] == 'B'].groupby('user_id')['event_name'].count().reset_index(drop=False)

# Посчитаем сколько пользователей совершили события в выборках
users_with_events_b = group_b_event_count[group_b_event_count['event_name'] > 0]
num_users_with_events_b = users_with_events_b['user_id'].count()

users_with_events_a = group_a_event_count[group_a_event_count['event_name'] > 0]
num_users_with_events_a = users_with_events_a['user_id'].count()

# общее количество событий в выборках
event_b = group_b_event_count ['event_name'].sum ()
event_a = group_a_event_count ['event_name'].sum ()

# количество пользоателей
num_users_b_total = group_b_event_count['user_id'].count()
num_users_a_total = group_a_event_count['user_id'].count()


# In[60]:


# Среднее количество событий на активных пользователей
mean_events_per_user_a = (event_a / num_users_with_events_a).round ()
print ('Среднее количество событий для активных пользователей в группе А - ', mean_events_per_user_a)
mean_events_per_user_b = (event_b / num_users_with_events_b).round ()
print ('Среднее количество событий для активных пользователей в группе B - ', mean_events_per_user_b)
print ()
# Среднее количество событий на всех пользователей
mean_events_per_user_total_a = (event_a / num_users_a_total).round ()
print ('Среднее количество событий для всех пользователей в группе А -', mean_events_per_user_total_a)
mean_events_per_user_total_b = (event_b / num_users_b_total).round ()
print ('Среднее количество событий для всех пользователей в группе B -', mean_events_per_user_total_b)


# У нас получается, что в группе А пользователи значительно активнее пользователей в группе В, так среди пользователей осуществивших хотя бы 1 событие - среднее число событий у пользователей группы А - 7, а в группе В - 6, но если рассмотреть аналогичный параметр для всех пользователей в выборке, то получается 5 к 2 в пользу группы А. 

# In[61]:


plt.hist(group_a_event_count['event_name'], bins=10, alpha=0.5, label='Group A')
plt.hist(group_b_event_count['event_name'], bins=10, alpha=0.5, label='Group B')

plt.xlabel('Число событий')
plt.ylabel('Количество клиентов')
plt.title('Распределение количества событий по клиентам групп А и В')
plt.legend()

plt.show()


# Учитывая результаты расчетов и график, складывается ощущение, что пользователи группы В, после определенных действий не возвращаются

# ## Расчет количества событий по дням

# In[62]:


event_by_date = main_data.groupby(['event_dt', 'group'])['user_id'].count().reset_index()
event_by_date['event_dt'] = pd.to_datetime(event_by_date['event_dt'])

plt.figure(figsize=(15, 4))
sns.set_style("whitegrid")
ax = sns.histplot(data=event_by_date, x='event_dt', hue='group', element='step', stat='count')
ax.set_title('Распределение количества событий в группах по дням')
plt.xticks(rotation=45)
plt.show()


# На графике мы наблюдаем сильный скачок группы А в перед рождеством, и резкий обрыв данных в районе 30 декабря, учитывая отсуствие данных в исходниках за период с 30 декабря по 4 января - можем предположить какой-то технических сбой.

# ## Отработка продуктовой гипотезы
# 
# Ожидаемый эффект (продуктовая гипотеза): за 14 дней с момента регистрации в системе пользователи покажут улучшение каждой метрики не менее, чем на 10%:
# 
#         - конверсии в просмотр карточек товаров — событие product_page
#         - просмотры корзины — product_cart
#         - покупки — purchase.

# In[63]:


main_data = main_data.merge(users[['user_id', 'first_date']], on='user_id', how='left')
main_data = main_data.rename(columns={'first_date': 'registration_dt'})


# добавим столбец с количеством дней от момента регистрации до события для каждого клиента

# In[64]:


main_data['days_since_registration'] = (main_data['event_dt'] - main_data['registration_dt']).dt.days


# In[65]:


main_data ['user_id'].nunique ()


# In[66]:


main_data = main_data.query ('days_since_registration <14')


# In[67]:


main_data ['days_since_registration'].value_counts()


# In[68]:


main_data ['user_id'].nunique ()


# ### Воронки продаж

# #### общая воронка

# In[69]:


# сгруппируем клиентов по шагам, посмотрим для начала на общие цифры

data_all_step = main_data.groupby('event_name') ['user_id'].nunique ().sort_values(ascending=False).reset_index ()
data_all_step = data_all_step.rename(columns={'user_id':'count'})
total_user = data_all_step.loc [0]['count']
data_all_step ['ration'] = (data_all_step ['count']/total_user*100).round()


# In[70]:


steps = data_all_step['event_name'].tolist()
counts = data_all_step['count'].tolist()

desired_order = ['login', 'product_page', 'product_cart', 'purchase']

# Прийдется создавать словарь для хранения соответствия между этапами и их количеством, 
# так как у нас оплата не последняя операция по количеству
data = dict(zip(steps, counts))

# Переупорядочиваем элементы в соответствии с желаемым порядком
steps_reordered = [step for step in desired_order if step in data]
counts_reordered = [data[step] for step in desired_order if step in data]

fig = go.Figure(go.Funnel(
    y = steps_reordered,
    x = counts_reordered,
    textposition = "inside",
    textinfo = "value+percent initial"))

fig.update_layout(
    title="Funnel Chart",
    xaxis_title="Количество пользователей",
    yaxis_title="События",
    funnelmode="stack",
    hovermode="x",
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False
)

fig.show()


# Мы помним, что еще на этапе изучения заметили, что количество событий оплаты товара больше чем просмотр корзины, что сейчас мы и наблюдаем в этом графике, разница не большая но она есть, как и ранее можем предположить что имеется какая-то специфическая настройка на сайте, или прямой лендинг на страницу оплаты (часто встречается кнопка - быстрая покупка)

# #### Воронки по группам А и В

# In[71]:


group_df = main_data.groupby('group')


# In[72]:


group_a_data = group_df.get_group('A')
group_b_data = group_df.get_group('B')


# In[73]:


data_a_step = group_a_data.groupby('event_name') ['user_id'].nunique ().sort_values(ascending=False).reset_index ()
data_a_step = data_a_step.rename(columns={'user_id':'count'})
total_user = data_a_step.loc [0]['count']
data_a_step ['ration'] = (data_a_step ['count']/total_user*100).round()


# In[74]:


steps = data_a_step['event_name'].tolist()
counts = data_a_step['count'].tolist()


# Прийдется создавать словарь для хранения соответствия между этапами и их количеством, 
# так как у нас оплата не последняя операция по количеству
data = dict(zip(steps, counts))

# Переупорядочиваем элементы в соответствии с желаемым порядком
steps_reordered = [step for step in desired_order if step in data]
counts_reordered = [data[step] for step in desired_order if step in data]

fig = go.Figure(go.Funnel(
    y = steps_reordered,
    x = counts_reordered,
    textposition = "inside",
    textinfo = "value+percent initial"))

fig.update_layout(
    title="Воронка продаж для группы А",
    xaxis_title="Количество пользователей",
    yaxis_title="События",
    funnelmode="stack",
    hovermode="x",
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False
)

fig.show()


# Для группы А процентные значения очень близки к общим

# In[75]:


data_b_step = group_b_data.groupby('event_name') ['user_id'].nunique ().sort_values(ascending=False).reset_index ()
data_b_step = data_b_step.rename(columns={'user_id':'count'})
total_user = data_b_step.loc [0]['count']
data_b_step ['ration'] = (data_b_step ['count']/total_user*100).round()


# In[76]:


steps = data_b_step['event_name'].tolist()
counts = data_b_step['count'].tolist()


# Прийдется создавать словарь для хранения соответствия между этапами и их количеством, 
# так как у нас оплата не последняя операция по количеству
data = dict(zip(steps, counts))

# Переупорядочиваем элементы в соответствии с желаемым порядком
steps_reordered = [step for step in desired_order if step in data]
counts_reordered = [data[step] for step in desired_order if step in data]



fig = go.Figure(go.Funnel(
    y = steps_reordered,
    x = counts_reordered,
    textposition = "inside",
    textinfo = "value+percent initial"))

fig.update_layout(
    title="Воронка продаж для группы B",
    xaxis_title="Количество пользователей",
    yaxis_title="События",
    funnelmode="stack",
    hovermode="x",
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False
)

fig.show()


# Процентное отношение шага в группе В хуже, а в части численного значения, мы уже знали, что в этой группе значительно меньше.

# ## AB Тесты

# ### Подготовка. Кумулятивная конверсия

# In[77]:


def cumulative_data(df, event):
    df_filtered_event = df[df['event_name'] == event]
    date_group = df_filtered_event[['event_dt', 'group']].drop_duplicates()

    df_purchase_agg = date_group.apply(lambda x: df_filtered_event[
        (df_filtered_event['event_dt'] <= x['event_dt']) & (df_filtered_event['group'] == x['group'])]
                                       .agg({'event_dt': 'max', 'group': 'max', 'user_id': 'nunique'}),
                                       axis=1).sort_values(by=['event_dt', 'group'])

    df_buyers_agg = date_group.apply(lambda x: df_filtered_event[
        (df_filtered_event['event_dt'] <= x['event_dt']) & (df_filtered_event['group'] == x['group'])]
                                     .agg({'event_dt': 'max', 'group': 'max', 'user_id': 'count'}),
                                     axis=1).sort_values(by=['event_dt', 'group'])

    result = df_purchase_agg.merge(df_buyers_agg, on=['event_dt', 'group'])
    result.columns = ['date', 'group', 'users', 'action']
    return result


# In[78]:


def cumulative_data_date(df, event):
    df_filtered_event = df[df['event_name'] == event]
    # Извлекаем только дату из столбца event_dt
    df_filtered_event['date'] = df_filtered_event['event_dt'].dt.date
    date_group = df_filtered_event[['date', 'group']].drop_duplicates()

    df_purchase_agg = date_group.apply(lambda x: df_filtered_event[
        (df_filtered_event['date'] <= x['date']) & (df_filtered_event['group'] == x['group'])]
                                       .agg({'date': 'max', 'group': 'max', 'user_id': 'nunique'}),
                                       axis=1).sort_values(by=['date', 'group'])

    df_buyers_agg = date_group.apply(lambda x: df_filtered_event[
        (df_filtered_event['date'] <= x['date']) & (df_filtered_event['group'] == x['group'])]
                                     .agg({'date': 'max', 'group': 'max', 'user_id': 'count'}),
                                     axis=1).sort_values(by=['date', 'group'])

    result = df_purchase_agg.merge(df_buyers_agg, on=['date', 'group'])
    result.columns = ['date', 'group', 'users', 'action']
    return result


# ### Конверсия и ее графики

# In[79]:


# Соберём кумулятивные значения для анализа относительных показателей и события "просмотр страниц"
cumulative_data_product_page = cumulative_data(main_data, 'product_page')

# столбец расчета конверсии
cumulative_data_product_page['conversion'] = cumulative_data_product_page['action']/cumulative_data_product_page['users']

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе А
cumulative_data_product_page_A = cumulative_data_product_page.query('group == "A"')

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе В
cumulative_data_product_page_B = cumulative_data_product_page.query('group == "B"')


# In[80]:


def plot_conversion(cumulativeData_A, cumulativeData_B, column, date, title, ylabel):
    """ 
    Функция для отображения кумулятивной конверсии в разбивке по группам
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set(title=title, xlabel='дата', ylabel=ylabel)
    
    ax.plot(cumulativeData_A[date], cumulativeData_A[column], label='A')
    ax.plot(cumulativeData_B[date], cumulativeData_B[column], label='B')
    
    ax.legend()
    plt.xticks(rotation=45)  
    
    plt.show()


# In[81]:


plot_conversion(cumulative_data_product_page_A, cumulative_data_product_page_B,
                'conversion', 'date', 'Кумулятивная конверсия просмотр карточек', 'кумулятивная конверсия')


# Начало примерно одинаковое в обеих группах. Примерно с 10 декабря, группа А начинает уступать группе В в росте, с резким падением кумулятивной конверсии 14 декабря (на 14 декабря в группе В значение в районе 2, в группе А 1.6). Но начиная с 22 декабря, кумулятивная конверсия в группе А начинает резко расти, и к 30 декабря достигает значений 3+, группа В фиксируется в районе 2.6.

# In[82]:


# Соберём кумулятивные значения для анализа относительных показателей и события "корзина"
cumulative_data_product_cart = cumulative_data(main_data, 'product_cart')

# столбец расчета конверсии
cumulative_data_product_cart['conversion'] = cumulative_data_product_cart['action']/cumulative_data_product_cart['users']

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе А
cumulative_data_product_cart_A = cumulative_data_product_cart.query('group == "A"')

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе В
cumulative_data_product_cart_B = cumulative_data_product_cart.query('group == "B"')


# In[83]:


plot_conversion(cumulative_data_product_cart_A, cumulative_data_product_cart_B,
                'conversion', 'date', 'Кумулятивная конверсия события "корзина"', 'кумулятивная конверсия')


# В общем и целом, общая картина с кумулятивной конверсией события "корзина" идентична просмотру карточек. Есть несколько отличий в деталях (например момент старта, несущественные отличия в кумулятивной конверсии на протяжении теста), но как уже отмечено, общая тенденция практически без изменений

# In[84]:


# Соберём кумулятивные значения для анализа относительных показателей и события "покупка"
cumulative_data_purchase = cumulative_data(main_data, 'purchase')

# столбец расчета конверсии
cumulative_data_purchase['conversion'] = cumulative_data_purchase['action']/cumulative_data_purchase['users']

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе А
cumulative_data_purchase_A = cumulative_data_purchase.query('group == "A"')

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе В
cumulative_data_purchase_B = cumulative_data_purchase.query('group == "B"')


# In[85]:


plot_conversion(cumulative_data_purchase_A, cumulative_data_purchase_B,
                'conversion', 'date', 'Кумулятивная конверсия события "корзина"', 'кумулятивная конверсия')


# Координальных изменений в общей картине нет.

# In[86]:


def plot_relative_change(df1, df2, date_column, value_column1, value_column2, title, xlabel, ylabel):
    """
    Функция для отображения графика относительного изменения в разбивке по датам

    Args:
        df1 (DataFrame): Первый DataFrame с данными
        df2 (DataFrame): Второй DataFrame с данными
        date_column (str): Название столбца с датами
        value_column1 (str): Название столбца для значения 1
        value_column2 (str): Название столбца для значения 2
        title (str): Заголовок графика
        xlabel (str): Название оси x
        ylabel (str): Название оси y
    """
    merged_data = df1.merge(df2, left_on=date_column, right_on=date_column, how='left', suffixes=['A', 'B'])
    relative_change = (merged_data[value_column2] / merged_data[value_column1]) - 1

    plt.figure(figsize=(10, 6))
    plt.plot(merged_data[date_column], relative_change)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.grid(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    date_form = DateFormatter("%m-%d")
    plt.gca().xaxis.set_major_formatter(date_form)

    plt.show()


# In[87]:


# Повторно соберём кумулятивные значения для анализа относительных показателей и события "просмотр страниц" только по дате
cumulative_data_product_page = cumulative_data_date(main_data, 'product_page')

# столбец расчета конверсии
cumulative_data_product_page['conversion'] = cumulative_data_product_page['action']/cumulative_data_product_page['users']

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе А
cumulative_data_product_page_A = cumulative_data_product_page.query('group == "A"')

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе В
cumulative_data_product_page_B = cumulative_data_product_page.query('group == "B"')


# In[88]:


plot_relative_change (cumulative_data_product_page_A, cumulative_data_product_page_B, 'date', 'conversionA', 'conversionB',
                      'График относительного изменения кумулятивной конверсии по событию "просмотр страницы" группы В к А', 'Дата', 'Отношение')


# In[89]:


# Повторно соберём кумулятивные значения для анализа относительных показателей и события "просмотр страниц" только по дате
cumulative_data_product_cart = cumulative_data_date(main_data, 'product_cart')

# столбец расчета конверсии
cumulative_data_product_cart['conversion'] = cumulative_data_product_cart['action']/cumulative_data_product_cart['users']

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе А
cumulative_data_product_cart_A = cumulative_data_product_cart.query('group == "A"')

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе В
cumulative_data_product_cart_B = cumulative_data_product_cart.query('group == "B"')


# In[90]:


plot_relative_change (cumulative_data_product_cart_A, cumulative_data_product_cart_B, 'date', 'conversionA', 'conversionB',
                      'График относительного изменения кумулятивной конверсии по событию "просмотр корзины" группы В к А', 'Дата', 'Отношение')


# In[91]:


# Повторно соберём кумулятивные значения для анализа относительных показателей и события "просмотр страниц" только по дате
cumulative_data_purchase = cumulative_data_date(main_data, 'purchase')

# столбец расчета конверсии
cumulative_data_purchase['conversion'] = cumulative_data_purchase['action']/cumulative_data_purchase['users']

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе А
cumulative_data_purchase_A = cumulative_data_purchase.query('group == "A"')

# датасет с кумулятивным количеством действий и кумулятивными уникальными пользователями по дням в группе В
cumulative_data_purchase_B = cumulative_data_purchase.query('group == "B"')


# In[92]:


plot_relative_change (cumulative_data_purchase_A, cumulative_data_purchase_B, 'date', 'conversionA', 'conversionB',
                      'График относительного изменения кумулятивной конверсии по событию "покупкам" группы В к А', 'Дата', 'Отношение')


# На всех графиках мы наблюдаем, как до 21 декабря 2020 года группа В показывала более хорошие результаты, а после 21 декабря мы видим резкое падение отношения группы В к группе А.

# ### Статистическая разница (тесты)

# Нам для отработки тестов необходимо проверить гипотезы различий в ключевые шаги (просмотр товара, корзина, покупка).
# 
# Для начала, сформулируем обобщенную гипотезу.
# 
# **Нулевая: различий в группах А и В в среднем количестве пользователей совершивших ключевое событие нет.** 
# 
# **Альтернативная: Есть различия в группах А и В в среднем количестве пользователей совершивших ключевое событие.**

# In[93]:


# выберем только тех клиентов, которые совершили нужный нам для исследования этап
events = ['product_page', 'purchase', 'product_cart']
selected_data = main_data[main_data['event_name'].isin(events)]


# In[94]:


# сгруппируем их
group_count_all = selected_data.groupby ('group') ['user_id'].agg ('nunique').reset_index(inplace=False).rename(columns={'user_id':'count'})
group_count_all


# In[95]:


all_groups = main_data.pivot_table(index='event_name', columns='group',values='user_id',aggfunc='nunique')
all_groups


# In[96]:


def z_test(val1, val2, event, alpha): 
    value_event_1 = all_groups.loc[event, val1]
    value_event_2 = all_groups.loc[event, val2] 
    value_users_1 = group_count_all.loc[0, 'count']
    value_users_2 = group_count_all.loc[1, 'count'] 
    p1 = value_event_1 / value_users_1 
    p2 = value_event_2 / value_users_2 
    difference = p1 - p2
    p_combined = (value_event_1 + value_event_2) / (value_users_1 + value_users_2) 
    z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1 / value_users_1 + 1 / value_users_2))
    distr = st.norm(0, 1)
    p_value = (1 - distr.cdf(abs(z_value))) * 2
    print('Проверка для A и B, событие: {}, p-значение: {p_value:.2f}'.format(event, p_value=p_value))
    if (p_value < alpha):
        print("Отвергаем нулевую гипотезу")
        print("Среднее количество пользователей дошедших до {} в группах A и B значимо различается".format(event))
    else:
        print("Не получилось отвергнуть нулевую гипотезу")
        print("Среднее количество пользователей дошедших до {} в группах A и B значимо не различается".format(event))


# In[97]:


a = .05

# для каждого события кроме события login вызываем функуию z_test
for event in all_groups.index:
    if event != 'login':
        z_test('A', 'B', event, a)
        print('')


# ### Вывод этапа.
# 
# 
# В начале теста обе группы показывали примерно одинаковые результаты, но с 10 декабря группа A начала уступать группе B в росте.
# Наблюдается резкое падение кумулятивной конверсии в группе A 14 декабря, в то время как в группе B она осталась стабильной.
# Однако начиная с 22 декабря, кумулятивная конверсия в группе A начала резко расти и превзошла группу B к 30 декабря.
# Общая картина с кумулятивной конверсией события "корзина" аналогична картины с кумулятивной конверсией события "просмотр карточек", за исключением нескольких деталей.
# В проведенном тесте не было обнаружено статистически значимого различия в среднем количестве пользователей дошедших до события "product_cart" и "purchase" между группами A и B.
# Однако было обнаружено статистически значимое различие в среднем количестве пользователей дошедших до события "product_page" между группами A и B.
# Таким образом, можно сделать вывод, что обе группы показывали схожие результаты до определенного момента, после чего произошли изменения в динамике конверсии. Группа A начала уступать группе B, но затем смогла догнать и превзойти ее. Однако статистически значимое различие было обнаружено только в случае события "product_page".

# # Общий вывод

# В результате анализа проведенного теста можно сделать следующие выводы:
# 
#     - Проведение теста в период праздников являлось некорректным выбором, так как в этот период пользовательское поведение может существенно отличаться от обычного.
#     - Одновременное проведение двух тестов и пересечение пользователей между группами делают невозможным оценку влияния тестов друг на друга и получение четких результатов.
#     - Промоакции, проводимые во время теста, могли исказить результаты, так как влияют на пользовательское поведение.
#     - Неравномерная формировка выборок, существенно меньшая размерность тестовой группы B по сравнению с контрольной группой A, создает дисбаланс и может повлиять на результаты теста.
#     - После фильрации датасета в соответствии с ТЗ, общее количество активных пользователей становится меньше требований ТЗ. В этой связи, данный фактор привел к неадекватным результатам и необъективности исследования.
# 
# 
# На основании вышеуказанных выводов, рекомендуется:
# 
#     - Пересмотреть период проведения теста и выбрать более репрезентативный период, исключающий факторы, которые могут исказить результаты.
#     - Проводить тесты отдельно, чтобы исключить взаимное влияние между ними и получить более четкие результаты.
#     - Исключить проведение промоакций во время теста, чтобы снизить искажение результатов и получить более чистую картину.
#     - Обеспечить равномерную формировку выборок, чтобы исключить дисбаланс между тестовой и контрольной группами.
#     - Обратить внимание на активность пользователей в тесте и увеличить усилия по привлечению активных участников для получения более надежных результатов.
#     - В целом, с учетом выявленных проблем и недостатков проведения теста, полученные результаты не могут быть использованы в качестве основы для принятия решений. Необходимо провести тестирование заново, устраняя выявленные проблемы, чтобы получить достоверные и адекватные результаты для принятия решений.
