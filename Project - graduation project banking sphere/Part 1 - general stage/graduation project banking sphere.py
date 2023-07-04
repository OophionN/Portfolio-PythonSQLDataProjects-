#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Итоговый-проект.-Описание" data-toc-modified-id="Итоговый-проект.-Описание-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Итоговый проект. Описание</a></span><ul class="toc-item"><li><span><a href="#Задачи." data-toc-modified-id="Задачи.-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Задачи.</a></span></li><li><span><a href="#Описание-датасетов" data-toc-modified-id="Описание-датасетов-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Описание датасетов</a></span></li><li><span><a href="#Общий-план-работы-(декомпозиция)" data-toc-modified-id="Общий-план-работы-(декомпозиция)-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Общий план работы (декомпозиция)</a></span></li></ul></li><li><span><a href="#Подготовка-инструментов-для-работы,-изучение-датасетов" data-toc-modified-id="Подготовка-инструментов-для-работы,-изучение-датасетов-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Подготовка инструментов для работы, изучение датасетов</a></span><ul class="toc-item"><li><span><a href="#Используем-стандартные-проверки,-с-использованием-шаблонных-функций-(заготовки-прошлых-периодов)" data-toc-modified-id="Используем-стандартные-проверки,-с-использованием-шаблонных-функций-(заготовки-прошлых-периодов)-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Используем стандартные проверки, с использованием шаблонных функций (заготовки прошлых периодов)</a></span><ul class="toc-item"><li><span><a href="#Провека-на-очевидные-дуликаты-и-пропуски" data-toc-modified-id="Провека-на-очевидные-дуликаты-и-пропуски-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Провека на очевидные дуликаты и пропуски</a></span></li><li><span><a href="#Изучение-наполненния-датасета" data-toc-modified-id="Изучение-наполненния-датасета-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>Изучение наполненния датасета</a></span></li><li><span><a href="#Проверка-на-неочевидные-дубликаты" data-toc-modified-id="Проверка-на-неочевидные-дубликаты-2.1.3"><span class="toc-item-num">2.1.3&nbsp;&nbsp;</span>Проверка на неочевидные дубликаты</a></span><ul class="toc-item"><li><span><a href="#Вывод-по-дубликатам." data-toc-modified-id="Вывод-по-дубликатам.-2.1.3.1"><span class="toc-item-num">2.1.3.1&nbsp;&nbsp;</span>Вывод по дубликатам.</a></span></li></ul></li><li><span><a href="#Корректировка-названий-столбцов" data-toc-modified-id="Корректировка-названий-столбцов-2.1.4"><span class="toc-item-num">2.1.4&nbsp;&nbsp;</span>Корректировка названий столбцов</a></span></li><li><span><a href="#Анализ-пропусков-в-столбце-Balance,-и-принятие-решение,-что-делаем-с-пропусками" data-toc-modified-id="Анализ-пропусков-в-столбце-Balance,-и-принятие-решение,-что-делаем-с-пропусками-2.1.5"><span class="toc-item-num">2.1.5&nbsp;&nbsp;</span>Анализ пропусков в столбце Balance, и принятие решение, что делаем с пропусками</a></span><ul class="toc-item"><li><span><a href="#Промежуточные-выводы-по-результатам-изучения-пропусков" data-toc-modified-id="Промежуточные-выводы-по-результатам-изучения-пропусков-2.1.5.1"><span class="toc-item-num">2.1.5.1&nbsp;&nbsp;</span>Промежуточные выводы по результатам изучения пропусков</a></span></li><li><span><a href="#Принятие-решения-по-пропускам,-через-оценку-остальных-столбцов" data-toc-modified-id="Принятие-решения-по-пропускам,-через-оценку-остальных-столбцов-2.1.5.2"><span class="toc-item-num">2.1.5.2&nbsp;&nbsp;</span>Принятие решения по пропускам, через оценку остальных столбцов</a></span></li></ul></li><li><span><a href="#Поиск-аномалий,-нарушений-логики" data-toc-modified-id="Поиск-аномалий,-нарушений-логики-2.1.6"><span class="toc-item-num">2.1.6&nbsp;&nbsp;</span>Поиск аномалий, нарушений логики</a></span></li></ul></li><li><span><a href="#Итоги-изучения-и-корректировки-исходного-датасета" data-toc-modified-id="Итоги-изучения-и-корректировки-исходного-датасета-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Итоги изучения и корректировки исходного датасета</a></span></li></ul></li><li><span><a href="#Исследовательский-анализ" data-toc-modified-id="Исследовательский-анализ-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Исследовательский анализ</a></span><ul class="toc-item"><li><span><a href="#Изучение-столбца-&quot;balance&quot;-и-&quot;estimated_salary&quot;" data-toc-modified-id="Изучение-столбца-&quot;balance&quot;-и-&quot;estimated_salary&quot;-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Изучение столбца "balance" и "estimated_salary"</a></span></li><li><span><a href="#Общее-изучение-столбцов-&quot;churn&quot;,-'gender',-'products'" data-toc-modified-id="Общее-изучение-столбцов-&quot;churn&quot;,-'gender',-'products'-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Общее изучение столбцов "churn", 'gender', 'products'</a></span></li><li><span><a href="#Общая-корреляция" data-toc-modified-id="Общая-корреляция-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Общая корреляция</a></span></li><li><span><a href="#Визуализация-относительных-величин-оттока-в-категориях" data-toc-modified-id="Визуализация-относительных-величин-оттока-в-категориях-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Визуализация относительных величин оттока в категориях</a></span></li><li><span><a href="#Общие-тенденции" data-toc-modified-id="Общие-тенденции-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Общие тенденции</a></span><ul class="toc-item"><li><span><a href="#По-категориям" data-toc-modified-id="По-категориям-3.5.1"><span class="toc-item-num">3.5.1&nbsp;&nbsp;</span>По категориям</a></span><ul class="toc-item"><li><span><a href="#Ростов-Великий" data-toc-modified-id="Ростов-Великий-3.5.1.1"><span class="toc-item-num">3.5.1.1&nbsp;&nbsp;</span>Ростов Великий</a></span></li><li><span><a href="#Ярославль" data-toc-modified-id="Ярославль-3.5.1.2"><span class="toc-item-num">3.5.1.2&nbsp;&nbsp;</span>Ярославль</a></span></li><li><span><a href="#Рыбинск" data-toc-modified-id="Рыбинск-3.5.1.3"><span class="toc-item-num">3.5.1.3&nbsp;&nbsp;</span>Рыбинск</a></span></li></ul></li></ul></li><li><span><a href="#Проверка-статистических-гипотез" data-toc-modified-id="Проверка-статистических-гипотез-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Проверка статистических гипотез</a></span><ul class="toc-item"><li><span><a href="#Формулировка-гипотез" data-toc-modified-id="Формулировка-гипотез-3.6.1"><span class="toc-item-num">3.6.1&nbsp;&nbsp;</span>Формулировка гипотез</a></span><ul class="toc-item"><li><span><a href="#Проверка-гипотезы-№-1---статистической-разницы-между-теми-клиентами,-которые-ушли-и-теми-которые-остались-в-доходах-нет." data-toc-modified-id="Проверка-гипотезы-№-1---статистической-разницы-между-теми-клиентами,-которые-ушли-и-теми-которые-остались-в-доходах-нет.-3.6.1.1"><span class="toc-item-num">3.6.1.1&nbsp;&nbsp;</span>Проверка гипотезы № 1 - статистической разницы между теми клиентами, которые ушли и теми которые остались в доходах нет.</a></span></li><li><span><a href="#Проверка-гипотезы-№-2---Статистической-разницы-в-уровне-дохода-между-клиентами-из-Ростова-и-клиентами-из-Рыбинска-нет." data-toc-modified-id="Проверка-гипотезы-№-2---Статистической-разницы-в-уровне-дохода-между-клиентами-из-Ростова-и-клиентами-из-Рыбинска-нет.-3.6.1.2"><span class="toc-item-num">3.6.1.2&nbsp;&nbsp;</span>Проверка гипотезы № 2 - Статистической разницы в уровне дохода между клиентами из Ростова и клиентами из Рыбинска нет.</a></span></li><li><span><a href="#Проверка-гипотезы-№-3---Статистической-разницы-в-уровне-оттока-между-клиентами-из-Ростова-и-Рынбинска-нет." data-toc-modified-id="Проверка-гипотезы-№-3---Статистической-разницы-в-уровне-оттока-между-клиентами-из-Ростова-и-Рынбинска-нет.-3.6.1.3"><span class="toc-item-num">3.6.1.3&nbsp;&nbsp;</span>Проверка гипотезы № 3 - Статистической разницы в уровне оттока между клиентами из Ростова и Рынбинска нет.</a></span></li></ul></li></ul></li><li><span><a href="#Выделение-сегментов" data-toc-modified-id="Выделение-сегментов-3.7"><span class="toc-item-num">3.7&nbsp;&nbsp;</span>Выделение сегментов</a></span><ul class="toc-item"><li><span><a href="#Общий-параметр-город---Ростов-Великий" data-toc-modified-id="Общий-параметр-город---Ростов-Великий-3.7.1"><span class="toc-item-num">3.7.1&nbsp;&nbsp;</span>Общий параметр город - Ростов Великий</a></span><ul class="toc-item"><li><span><a href="#Возраст" data-toc-modified-id="Возраст-3.7.1.1"><span class="toc-item-num">3.7.1.1&nbsp;&nbsp;</span>Возраст</a></span></li><li><span><a href="#Вывод-по-категории-&quot;Возраст&quot;" data-toc-modified-id="Вывод-по-категории-&quot;Возраст&quot;-3.7.1.2"><span class="toc-item-num">3.7.1.2&nbsp;&nbsp;</span>Вывод по категории "Возраст"</a></span></li><li><span><a href="#Количество-продуктов" data-toc-modified-id="Количество-продуктов-3.7.1.3"><span class="toc-item-num">3.7.1.3&nbsp;&nbsp;</span>Количество продуктов</a></span></li><li><span><a href="#Вывод-По-количеству-продуктов-у-клиентов." data-toc-modified-id="Вывод-По-количеству-продуктов-у-клиентов.-3.7.1.4"><span class="toc-item-num">3.7.1.4&nbsp;&nbsp;</span>Вывод По количеству продуктов у клиентов.</a></span></li><li><span><a href="#Комбинация-&quot;тонких-мест&quot;-Ростова" data-toc-modified-id="Комбинация-&quot;тонких-мест&quot;-Ростова-3.7.1.5"><span class="toc-item-num">3.7.1.5&nbsp;&nbsp;</span>Комбинация "тонких мест" Ростова</a></span></li><li><span><a href="#Выводы-по-комбинации-условий" data-toc-modified-id="Выводы-по-комбинации-условий-3.7.1.6"><span class="toc-item-num">3.7.1.6&nbsp;&nbsp;</span>Выводы по комбинации условий</a></span></li></ul></li></ul></li><li><span><a href="#Общие-выводы-по-результатам-исследования" data-toc-modified-id="Общие-выводы-по-результатам-исследования-3.8"><span class="toc-item-num">3.8&nbsp;&nbsp;</span>Общие выводы по результатам исследования</a></span></li><li><span><a href="#Презентация" data-toc-modified-id="Презентация-3.9"><span class="toc-item-num">3.9&nbsp;&nbsp;</span>Презентация</a></span></li><li><span><a href="#Дашборд" data-toc-modified-id="Дашборд-3.10"><span class="toc-item-num">3.10&nbsp;&nbsp;</span>Дашборд</a></span></li></ul></li></ul></div>

# # Итоговый проект. Описание
# "Банки - анализ оттока клиентов"
# 
# "Нашей главной задачей станет анализ оттока клиентов. Анализ покажет, какие клиенты уходят из банка, а так же поможет нам составить сегменты клиентов, которые склонны уходить из банка.
# «Метанпромбанк» — деньги не пахнут!"
# 

# ## Задачи.
# 
# Проанализировать клиентов регионального банка и выделите сегменты клиентов, которые склонны уходить из банка.
# 
# - Провести исследовательский анализ данных,
# - Вывести портреты клиентов, которые склонны уходить из банка,
# - Сформулировать и проверить статистические гипотезы.

# ## Описание датасетов
# 
# Датасет содержит данные о клиентах банка «Метанпром». Банк располагается в Ярославле и областных городах: Ростов Великий и Рыбинск.
# 
# Колонки:
# 
# - `userid` — идентификатор пользователя,
# - `score` — баллы кредитного скоринга,
# - `City` — город,
# - `Gender` — пол,
# - `Age` — возраст,
# - `Objects` — количество объектов в собственности / `equity`  — количество баллов собственности
# - `Balance` — баланс на счёте,
# - `Products` — количество продуктов, которыми пользуется клиент,
# - `CreditCard` — есть ли кредитная карта,
# - `Loyalty` / `last_activity` — активный клиент,
# - `estimated_salary` — заработная плата клиента,
# - `Churn` — ушёл или нет.
# 
# По итогам исследования необходимо подготовить презентацию. 
# 

# ## Общий план работы (декомпозиция)
# 
# Подготовительный этап:
# 
#     - Ознакомимся с предоставленными данными и проверим их целостность.
#     - Согласуем с тимлидом и точно определим цели исследования, аудиторию презентации и дашборда, а также требования к формату и содержанию.
# 
# Исследовательский анализ данных:
# 
#     - Проведем исследовательский анализ данных для лучшего понимания их структуры и особенностей. Включая:
#         - Расчет основных статистических показателей (среднее, медиана, стандартное отклонение) для числовых столбцов (например, "Age", "Balance", "Estimated_salary" и др.).
#         - Визуализацию данных для получения инсайтов, таких как распределение возрастов клиентов, баланса на счете и т.д.
#         - Анализ корреляции между различными переменными, чтобы выявить возможные связи или зависимости.
# 
# Проведем выделение сегментов клиентов:
# 
#     - Используя результаты исследовательского анализа, выделим сегменты клиентов, склонных к оттоку. Включая, но не ограничиваясь:
#         - Анализ оттока в зависимости от различных факторов, таких как возраст, пол, наличие кредитной карты и т.д.
# 
# Проверка статистических гипотез:
# 
#     - Сформулируем статистические гипотезы, связанные с оттоком клиентов, на основе предоставленных данных и по результатам встречи с тимлидом.
#     - Проведем соответствующие статистические тесты (например, t-тест, анализ дисперсии) для проверки этих гипотез.
#     - Дадим интерпретацию результатов и сделаем выводы.
# 
# Подготовка презентации и дашборда:
# 
#     - Соберем ключевые результаты и выводы из предыдущих этапов.
#     - Разработаем структуру презентации и дашборда, включая графики, таблицы и текстовые описания.
#     - Создадим презентацию с объяснением анализа, основными выводами и рекомендациями.
#     - Разработаем интерактивный дашборд, который позволит пользователям взаимодействовать с данными и исследованными показателями. (прим. Параметры презентации и дашборда, будут окончательно определены после встречи с тимлидом)
# 
# Финальный этап:
# 
#     - Проведем ревизию презентации и дашборда с тимлидом и получим обратную связь.
#     - Внесем необходимые корректировки и уточнения в презентацию и дашборд.
#     - Подготовим окончательные версии презентации и дашборда для представления заказчикам и заинтересованным сторонам.
# 

# Прим. Для анализа и исследования принято решение использовать файл bank_dataset.csv

# # Подготовка инструментов для работы, изучение датасетов

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


# И обновления к ним

# In[2]:


#!pip install --upgrade pandas seaborn
#!pip install --upgrade pandas
#!pip install --upgrade seaborn


# In[3]:


try:
    data = pd.read_csv (r'C:\Users\PC_Maks\Desktop\study\\final_project\bank_dataset.csv',sep = ",")
    
except: 
    data = pd.read_csv ('/datasets/bank_dataset.csv', sep=',') 
    
    
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['figure.figsize'] = (15, 8) 


# Познакомимся с данными

# In[4]:


data


# In[5]:


data.info()


# ## Используем стандартные проверки, с использованием шаблонных функций (заготовки прошлых периодов)

# ### Провека на очевидные дуликаты и пропуски 
# 

# In[6]:


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


# In[7]:


check(data)


# ### Изучение наполненния датасета

# In[8]:


def check_unique(data):
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


# In[9]:


check_unique(data)


# ### Проверка на неочевидные дубликаты

# In[10]:


def check_duplicates(data, columns):
    duplicates = data.duplicated(subset=columns, keep=False)
    if duplicates.any():
        duplicate_rows = data[duplicates]
        print("Найдены неочевидные дубликаты:")
        print(duplicate_rows)
        return duplicate_rows
    else:
        print("Не найдены неочевидные дубликаты.")
        return None


# In[11]:


columns_to_check = ["userid", "estimated_salary"]
check_duplicates(data, columns_to_check)


# In[12]:


columns_to_check = ["score", "Churn", "Gender", "Age", "Objects"]
raw_dupl = check_duplicates(data, columns_to_check)
if raw_dupl is not None:
    raw_dupl.head()


# In[13]:


raw_dupl = raw_dupl.sort_values (by='score', ascending=False)
raw_dupl.head (100)


# #### Вывод по дубликатам.
# Изучением датасета на предмет выявления неочевидных дубликатов установлено, что дубликатов в строках, при использовании относительно уникальных значений (баланс, доход) не выявлено. При проверке на дубликаты по категорированным полям (скорбал, возраст, пол, город), найдены "дубликаты", однако, в даным случае это фактически не дубликаты, а просто совпадение отдельных записей по ограниченному набору значений. 
# Итого - дубликатов нет.

# ### Корректировка названий столбцов
# Приведем названия столбцов к питоническому формату

# In[14]:


def data_preprocessing(data):
   
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    
    data = data.rename(columns=lambda x: x.lower().replace(' ', '_'))
    
    return data


# In[15]:


data_preprocessing(data)


# ### Анализ пропусков в столбце Balance, и принятие решение, что делаем с пропусками

# In[16]:


missing_balance = data.query("balance.isnull()")


# In[17]:


check_unique(missing_balance)


# Очевидных связей с пропусками нет, но есть интересное наблюдение - баланс не пропущен у клиентов из Ростова Великого, хм... думаем дальше

# In[18]:


# Создание бинарных переменных для столбца "city"
encoded_cities = pd.get_dummies(missing_balance['city'], prefix='city', drop_first=False)
encoded_cities = encoded_cities.astype(int)

encoded_gender = pd.get_dummies(missing_balance['gender'], prefix='gender', drop_first=False)
encoded_gender = encoded_gender.astype(int)

encoded_data = pd.concat([missing_balance, encoded_cities, encoded_gender], axis=1)

encoded_data


# In[19]:


encoded_data['balance'] = encoded_data ['balance'].fillna(1).astype(int)


# Удалим из таблицы для корреляции исходный столбец город

# In[20]:


data_without_city_gender = encoded_data.drop(['city', 'gender'], axis=1)


# In[21]:


data_without_city_gender


# создадим матрицу корреляции и построим хитмеп для лучшей визуализации

# In[22]:


label_encoder = LabelEncoder()
for column in data_without_city_gender.select_dtypes(include='object'):
    data_without_city_gender[column] = label_encoder.fit_transform(data_without_city_gender[column])

correlation = data_without_city_gender.corr()
correlation


# In[23]:


sns.heatmap(correlation, annot=True, cmap='coolwarm')

plt.title('Матрица корреляции значений в таблице пропусков по столбцу "Баланс"')
plt.xlabel('Значения столбцов')
plt.ylabel('Значения столбцов')
plt.show()


# #### Промежуточные выводы по результатам изучения пропусков 
# Очевидной корреляции в выборке с пропусками по балансу нет, немного выделяются пары возраст с значением оттока и возвраст-лояльность, а также лояльность-отток, и отток и продукт, но этот анализ мы будем проводить дальше в рамках основого исследования. 
# Таким образом, прихожу к выводу, что пропуски в столбце баланс имеют, с высокой долей вероятности, техническую ошибку в своей основе. Подумаем, что можно сделать с ними.

# #### Принятие решения по пропускам, через оценку остальных столбцов
# 
# Немного рассуждений. Удалять пропуски нельзя, их слишком много 36%, пока видится 2 решения, первое более простое - заглушки, но тогда в некоторых расчетах мы можем потерять наши 36%, второй вариат сложнее, можно попробовать изучить клиентов с пропусками, и попробовать заполнить медиативными значениями в разных группах. На мой взгляд, входной точкой для второго варианта должен быть доход клиентов, и возможно количество кредитов. 
# 
# Диапазон значений в столбце Balance:
# Минимальное значение: 3768.69
# Максимальное значение: 250898.09
# 
# Диапазон значений в столбце estimated_salary:
# Минимальное значение: 11.58
# Максимальное значение: 199992.48
# 

# Перед дальнейшей обработкой нужно скорректировать размер зп клиентов. До 1 января 2023 года минимальный размер оплаты труда в Ярославской области 13890 рублей (принимаем за факт, что размер ЗП в датасете в рублях). Посмотрим сколько клиентов у нас с такой ЗП и меньше.

# In[24]:


data_low_salary =  data.query ('estimated_salary < 13890')
data_low_salary.shape


# 694 клиентов у которых размер зп меньшее МРОТ, просто удалять 7% не стоит, учитывая, что часть клиентов могли предоставить данные о размере ЗП до 2022 года, а МРОТ меняется ежегодно, значит мы можем или поднять значения всех клиентов, у которых ЗП меньше МРОТ до 13890, но лучше будет чистку и корректировку провести в 2 этапа.
# 
# 1 этап, проверим корреляцию имеющехся значений баланса с значениями ЗП и количеством кредитов, попробуем скорректировать сначала их 
# 
# 2 этап -  заменим оставшиеся низкие значения на МРОТ

# In[25]:


# выберем из датасета сначала записи в которых есть баланс и посмотрим на корреляцию баланс-ЗП-количество кредитов
data_without_nan_balance = data.dropna(subset=['balance'])


# In[26]:


selected_columns = ['balance', 'products', 'estimated_salary']
correlation = data_without_nan_balance[selected_columns].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')

plt.title('Матрица корреляции ненулевого Баланса с ЗП и продуктами')
plt.xlabel('Значения столбцов')
plt.ylabel('Значения столбцов')
plt.show()


# Столбец "balance" не имеет сильной корреляции ни с одним из других столбцов. Коэффициент корреляции между "balance" и другими столбцами близок к нулю (-0.001027 и -0.001856), что указывает на отсутствие линейной зависимости между "balance" и другими переменными.
# 
# Столбец "products" имеет небольшую положительную корреляцию с "estimated_salary" (0.025769). Это может указывать на слабую связь между количеством продуктов, которыми пользуется клиент, и предполагаемой зарплатой. Однако, данная корреляция не является сильной. Наличие такого рода корреляции, даже незначительной, может быть обоснованно тем, что чем выше ЗП, тем больше клиент может получить кредитов (выше шанс получения продукта в общем и целом).
# 
# Столбцы "balance" и "products" не имеют значимой корреляции между собой (-0.001027). Это означает, что изменения в балансе счета клиента не связаны с количеством продуктов, которыми он пользуется.
# 
# **Общий вывод:** В предоставленной матрице корреляции наблюдается слабая или отсутствующая корреляция между столбцами "balance", "products" и "estimated_salary". Это говорит о том, что эти переменные не сильно влияют друг на друга. 
# Таким образом, переходим к этапу 2 - замена значений дохода на МРОТ, если он ниже него после вычисления квантилей.

# **Обработка пропусков:**
# 
# Можно попробоавать заполнить пропуски на основании группировки размера ЗП, например сделать 10 квантилей и посмотреть, какие средние значения баланса в процентном отношении к ЗП будут у клиентов в этих группах. Соответвенно оценка только по значениям Баланса, которые не пропущены. А потом заполнить пропуски в балансе на основе этих процентных отношений к размеру ЗП отдельной записи по клиенту.
# 
# Прим. Так как в расчете используются 10 квантилей и не берем пропуски, можем использовать среднее значение при вычислении отношений.

# In[27]:


data['category'] = pd.qcut(data['estimated_salary'], q=10, labels=False)

category_balance_ratio = (
    data.dropna()  
    .groupby('category')
    .apply(lambda x: (x['balance'] / x['estimated_salary'].mean()))
)

mean_category_balance_ratio = (
    category_balance_ratio.dropna()
    .groupby('category')
    .mean()
    .round(1)
)
mean_category_balance_ratio


# Интересная картина получается с накоплениями, у нас в 0 квантили накопления получаются в соотношении 11.8 к минимальной планке ЗП. хотя остальные квантили значительно меньше, могу предположить, что такое значение обусловлено, тем что в некоторых полях ЗП указан крайне низкий размер ЗП. Мне представляется следующе решение:
# 
#     - вычислить квантиль 0 через регрессию на основе квантилей 1-9, и только потом дозаполнить пропуски баланса. 

# In[28]:


quantiles = mean_category_balance_ratio.index.values
quantiles_1_to_9 = quantiles[1:10]
mean_size_1_to_9 = quantiles_1_to_9.mean()
def nonlinear_regression(x, a, b):
    return a * np.exp(-b * x)

# Выполнение нелинейной регрессии (подсказал друг как сделать, сам не изучал, 
# но идея, выводы и рассуждения мои, не мог найти инструмент для расчетов квантили 0 с учетом идеи, обратился за помощью)
popt, pcov = curve_fit(nonlinear_regression, quantiles_1_to_9, mean_size_1_to_9)
logical_size_0 = nonlinear_regression(0, *popt)
mean_category_balance_ratio[0] = logical_size_0
mean_category_balance_ratio


# In[29]:


data['balance'] = data.apply(lambda row: row['estimated_salary'] *                             mean_category_balance_ratio[row['category']] if pd.isnull(row['balance'])                             else row['balance'], axis=1).round(1)
data[data['category']==0]


# В рамках нашей логики заполнения, пропуски заполнились корректно, идем дальше

# In[30]:


data.loc[data['estimated_salary'] < 13890, 'estimated_salary'] = 13890
data[data['category']==0]


# ### Поиск аномалий, нарушений логики
# 
# Проверим, у всех ли клиентов, у которых указано наличие кредитной карты, стоит 1 или больше в категории продукты 

# In[31]:


data_anomaly = data.query ('(products == 0) and (creditcard>=1)')
data_anomaly


# Аномалий нет. 

# ## Итоги изучения и корректировки исходного датасета
# 
# Корректировка типов данных не требовалась.
# 
# Названия столбцов были скорректированы для удобства работы.
# 
# Была проведена оценка и поиск дубликатов - дубликаты не были обнаружены.
# 
# Выявлено, что в столбце "balance" имеется 3617 пропусков, что составляет около 36.17% от общего числа записей. Поскольку не было обнаружено явной корреляции между пропусками и размером ЗП/количеством кредитов, мы сделали предположение, что пропуски возникли из-за технической ошибки. Но один элемент можем с уверенностью отметить, пропуска баланса нет в Ростове, что, по сути, может является как раз связью, которую искали.
# 
# Для заполнения пропусков мы разделили клиентов на 10 квантилей на основе их заработной платы. Затем мы рассчитали отношение размера заработной платы к балансу для каждого клиента внутри своей квантили. Далее мы рассчитали средний коэффициент внутри каждой квантили. Обратили внимание, что коэффициент квантили 0 аномально большой, рассчитали квантиль 0 с помощью обратной регрессии. Наконец, мы заполнили пропуски в балансе, умножив размер заработной платы каждого клиента на соответствующий коэффициент для его квантилей.
# 
# Таким образом, проведенные шаги позволили учесть индивидуальные характеристики каждого клиента и заполнить пропуски в балансе, сохраняя баланс исходных данных. Также стоит отметить, что оценка и поиск дубликатов не выявили повторяющихся записей, что дает нам дополнительное подтверждение качества данных.
# 

# # Исследовательский анализ

# ## Изучение столбца "balance" и "estimated_salary" 
# Для начала посмотрим на распределение в столбцах дохода и баланса

# In[32]:


def plot_distribution_and_box(data, columns):
    for column in columns:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Histogram(x=data[column], nbinsx=25, histnorm='probability', name='Распределение', marker_color='steelblue'), secondary_y=False)
        fig.update_layout(title=f'Распределение значений в столбце {column}', xaxis_title=column, yaxis_title='Частота')
        fig.update_layout(xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'))
        
        kde = gaussian_kde(data[column])
        x_vals = sorted(data[column])
        y_vals = kde(x_vals)
        
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='red', width=2), name='KDE'), secondary_y=True)
        fig.update_yaxes(title_text="Частота", secondary_y=False)
        fig.update_yaxes(title_text="Плотность", secondary_y=True)
        
        fig.show()
        
        fig = go.Figure()
        fig.add_trace(go.Box(x=data[column], name='Ящик с усами', marker_color='steelblue'))
        fig.update_layout(title=f'Ящик с усами для столбца {column}', xaxis_title=column)
        fig.update_layout(xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'))
        
        fig.show()


# In[33]:


columns_to_plot = ['balance', 'estimated_salary', 'age']
plot_distribution_and_box(data, columns_to_plot)
# Графики интерактивные, для полноты восприятия, необходимо использовать активные значения легенды


# In[34]:


data[['balance', 'estimated_salary', 'age']].describe()


# In[35]:


def value_moda_func (data, columns):
    for column_name in columns:
        column_data = data[column_name]
        column_mode = mode(column_data)
        print (f' Значение моды {column_name} = {column_mode}')


# In[36]:


columns = ['balance', 'estimated_salary', 'age']
value_moda_func (data, columns)


# Мы видим, что в столбце баланс значения распределены нормально, в столбце зарплаты мы имеем всплеск на минимальных значениях (наша корректировка уровня зп до МРОТ), в дальнейшем размер заработной платы представлен однородно среди клиентов.
# 
# медиана баланса находится в районе 117725, а зарплаты в районе 100000 (видимо у нас довольно состоятельные клиенты, так как на текущий момент в Ярославле средняя зп около 45000), Критично аномальных значений/недопустимых выбросов в изучаемых столбцах - нет. 
# 
# В части возраста - средний возраст заемщиков около 39 лет, есть смещение к левому краю, что тоже объяснимо, так как самая активная часть населения в кредитовании обычно в пределах 30-45 лет. 
# 
# Прим. Изучив дополнительно значение моды для указанных значений, можем отметить, что в уровне баланса чаще всего встречается 114514, возраст - 37 лет (совпал с средним значением), а в размере дохода - 13890 - наша корректировка (что также очивидно, так как мода посчитала самое частое значение в сете, а мы почти 7% заменили фиксированным значением). 

# ## Общее изучение столбцов "churn", 'gender', 'products'

# In[37]:


def pie_plot(data, column_names):
    plt.figure(figsize=(10, 12)) 

    
    num_plots = len(column_names)
    num_cols = 2  
    num_rows = (num_plots + 1) // 2  

    for i, column_name in enumerate(column_names):
        category_counts = data[column_name].value_counts()

        plt.subplot(num_rows, num_cols, i + 1)

        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Распределение значений по ' + column_name)
        plt.axis('equal')

    plt.tight_layout()
    plt.show()


# In[38]:


column_names = ['churn', 'gender', 'products', 'objects', 'creditcard', 'loyalty']
pie_plot (data, column_names)


# Отток из датасета 20,4%, мужчин немного больше в общем количестве - 54,6% против 45,6% женщин, подавляющее большенство клиентов имеет 1 и 2 продукта - 50,8% и 45,9% соответсвенно.  По количеству объектов в собственности, распределение более равномерное, по кредитным картам 70,6% (есть карта) против 29,4% (карты нет), а лояльность среди всех клиентов практически равна - 51,5% и 48,5% (не лояльны)

# ## Общая корреляция 

# In[39]:


encoded_cities = pd.get_dummies(data['city'], prefix='city', drop_first=False)
encoded_cities = encoded_cities.astype(int)
encoded_gender = pd.get_dummies(data['gender'], prefix='gender', drop_first=False)
encoded_gender = encoded_gender.astype(int)
encoded_data = pd.concat([data, encoded_cities, encoded_gender], axis=1)
data_without_city_gender = encoded_data.drop(['city', 'gender'], axis=1)


# In[40]:


data_without_city_gender


# In[41]:


label_encoder = LabelEncoder()
for column in data_without_city_gender.select_dtypes(include='object'):
    data_without_city_gender[column] = label_encoder.fit_transform(data_without_city_gender[column])
correlation = data_without_city_gender.corr()
correlation


# In[42]:


sns.heatmap(correlation, annot=True, cmap='coolwarm')

plt.title('Матрица корреляции значений в таблице пропусков по столбцу "Баланс"')
plt.xlabel('Значения столбцов')
plt.ylabel('Значения столбцов')
plt.show()


# Явной корреляции в общем датасете не выявлено, немного прослеживается корреляция между оттоком и возрастом, оттоком и городом Ростов Великий, и обратная корреляция между оттоком и лояльностью. 
# 

# Образ отточного клиента начинает прорисовываться. Итак, у нас получается в описательном сегменте:
# 
#     - возраст 46 лет (по моде, 45 по среднему) - статистическая граница кредитной активности, совпадение? - не думаю)))
#     - доход в районе МРОТ (13890 в нашем скорректированном датасете или меньше 13890 в исходном)
#     - город - Ростов Великий, в меньшей степени Ярославль
#     - наличие 1 кредитного продукта (но помним, что в оттоке проявился рост клиентов с 3 и более продуктами)
#     
# Резюмируем:
#     - на отток больше свего влияют поля age, loyalty и город - Ростов Великий, опять Ростов в поле внимания, при проработке пропусков, только в этом городе не было пропусков в балансе. 
#     
# В части возраста нужно сделать остановку и предоставить развернутый комментарий:
#     Корреляция между "gender_Ж" и целевой переменной "churn" (отток) равна 0.106512, а корреляция между "gender_М" и "churn" также равна -0.106512. Это указывает на некоторую слабую корреляцию между гендером клиента и вероятностью оттока. Однако степень корреляции относительно невысока, что может указывать на то, что гендер сам по себе не является сильным предиктором оттока.    

# ## Визуализация относительных величин оттока в категориях

# In[43]:


### функиця построения графиков для нескольких столбцов
def plot_churn_comparison(data, category_columns):
    num_categories = len(category_columns)

    fig, axes = plt.subplots(num_categories, 1, figsize=(8, 6 * num_categories))

    for i, category_column in enumerate(category_columns):
        ax = axes[i]
        unique_categories = data[category_column].unique()
        
        churn_counts = []
        for category in unique_categories:
            subset = data[data[category_column] == category]
            churn_counts.append(subset['churn'].value_counts())

        stacked_counts = pd.DataFrame(churn_counts).fillna(0)
        stacked_counts.index = unique_categories
        
        stacked_counts = stacked_counts.sort_values(by=1, ascending=False)

        churn_totals = stacked_counts.sum(axis=1)
        churn_percentages = stacked_counts.div(churn_totals, axis=0) * 100

        stacked_counts.plot(kind='bar', stacked=True, ax=ax, color=['steelblue', 'tomato'])

        ax.set_title(f'Сравнение оттока в {category_column} с расчетом процента оттока внутри категории', fontsize=16)
        ax.set_xlabel(category_column, fontsize=12)
        ax.set_ylabel('Количество клиентов', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(['Нет оттока', 'Отток'], loc='upper right')

        for j, category in enumerate(stacked_counts.index):
            for k, percentage in enumerate(churn_percentages.loc[category][1:], 1):  
                total_customers = stacked_counts.loc[category].sum()
                ax.text(j, stacked_counts.loc[category, :k].sum(), f'{percentage:.1f}%', ha='center', va='bottom')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Разворот подписей оси x

    plt.tight_layout()
    plt.show()


# In[44]:


### костыль, для столбца продукты в Ростове (расчеты дальше по городам), так как по какой-то причине на этом столбце,
### при выводе информации по отдельным городам функция показывала, не отток, а процент оставшихся клиентов
### Почему-то именно для Ростова, потратил много времени на поиск ошибки, в итоге сделал такую заплатку

def plot_churn_comparison_single(data, category_column):
    unique_categories = data[category_column].unique()
        
    churn_counts = []
    for category in unique_categories:
        subset = data[data[category_column] == category]
        churn_counts.append(subset['churn'].value_counts())

    stacked_counts = pd.DataFrame(churn_counts).fillna(0)
    stacked_counts.index = unique_categories
        
    stacked_counts = stacked_counts.sort_values(by=1, ascending=False)

    churn_totals = stacked_counts.sum(axis=1)
    churn_percentages = stacked_counts.div(churn_totals, axis=0) * 100

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    stacked_counts.plot(kind='bar', stacked=True, ax=ax, color=['steelblue', 'tomato'])

    ax.set_title(f'Сравнение оттока в {category_column} с расчетом процента оттока внутри категории', fontsize=16)
    ax.set_xlabel(category_column, fontsize=12)
    ax.set_ylabel('Количество клиентов', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(['Нет оттока', 'Отток'], loc='upper right')

    for j, category in enumerate(stacked_counts.index):
        for k, percentage in enumerate(churn_percentages.loc[category][1:], 1):
            ax.text(j, stacked_counts.loc[category, :k].sum(), f'{percentage:.1f}%', ha='center', va='bottom')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Разворот подписей оси x

    plt.tight_layout()
    plt.show()


# In[45]:


plot_churn_comparison(data, ['city', 'gender', 'category', 'objects', 'products', 'creditcard'])


# In[46]:


categories = [7, 8, 9]

for category in categories:
    category_data = data[data['category'] == category]
    max_salary = category_data['estimated_salary'].max()
    min_salary = category_data['estimated_salary'].min()
    print(f"Категория {category}:")
    print(f"Максимальная зарплата: {max_salary}")
    print(f"Минимальная зарплата: {min_salary}")
    print()


# In[47]:


def plot_churn_by_age(data, age_interval=None):
    age_counts = data['age'].value_counts().sort_index()
    churn_counts = data[data['churn'] == 1]['age'].value_counts().sort_index()

    churn_percentages = churn_counts / age_counts * 100

    if age_interval:
        age_counts = age_counts.loc[(age_counts.index >= age_interval[0]) & (age_counts.index <= age_interval[1])]
        churn_percentages = churn_percentages.loc[(churn_percentages.index >= age_interval[0]) & (churn_percentages.index <= age_interval[1])]

    top_5_churn_percentages = churn_percentages.nlargest(5)

    plt.figure(figsize=(12, 6), dpi=100)
    ax = sns.histplot(data=data, x='age', hue='churn', bins=len(age_counts), multiple='stack',
                 palette=['steelblue', 'tomato'], alpha=0.7, kde=True, common_norm=False)

    for x, y in zip(age_counts.index, age_counts):
        if x in top_5_churn_percentages.index:
            plt.text(x, y + 25, f'{churn_percentages[x]:.1f}%', ha='center')

    plt.xlabel('Возраст')
    plt.ylabel('Количество клиентов')
    plt.title('Отток клиентов по возрасту в процентном отношении к количеству')
    plt.legend(['Отток', 'Нет оттока'])
    
    if age_interval:
        ax.set_xlim(age_interval[0], age_interval[1])

    plt.show()


# In[48]:


plot_churn_by_age(data)


# Видно плохо, но сектор внимания уже заметен 30-60 лет

# In[49]:


plot_churn_by_age(data, age_interval=(30, 60))


# ## Общие тенденции

# После проведенного исследования можено выделить следующие параметры сегмента оттока: 
# 
#     - Город: Отток клиентов из города "Ростов Великий" составляет 32,4%. Возможно, в этом городе существуют определенные факторы или проблемы, которые влияют на удержание клиентов.
# 
#     - Пол: Отток женщин составляет более 25,1%. Возможно, женщины имеют определенные предпочтения или проблемы с услугами или продуктами банка.
# 
#     - Доход: В диапазоне от 19,3% до 21,9% наблюдается отток клиентов в разных категориях дохода. Однако заметна тенденция, что состоятельные клиенты с высоким доходом (от 139435 и выше) имеют более высокий уровень оттока. Возможно, это связано с неудовлетворительным качеством предоставляемых услуг или недостаточной персонализацией под их потребности.
# 
#     - Количество объектов: Критичной разницы в показателях оттока не выявлено.
# 
#     - Количество продуктов: Клиенты с 3 и 4 продуктами имеют самый высокий уровень оттока (82,7% и 100%). Это может указывать на то, что наличие большого количества продуктов или кредитов может создавать негативное восприятие клиентами и стать причиной ухода.
# 
#     - Кредитные карты: Не выявлено критических показателей оттока в срезе наличия кредитных карт.
# 
#     - Возраст: Наибольший отток наблюдается в возрастном диапазоне 51-56 лет (55-71%). Возможно, в этой возрастной группе клиенты сталкиваются с определенными финансовыми или жизненными изменениями, которые влияют на их взаимодействие с банком.

# ### По категориям

# #### Ростов Великий

# In[50]:


data_rostov = data.query ('city == "Ростов Великий"')


# In[51]:


plot_churn_comparison(data_rostov, ['gender', 'category', 'objects', 'products', 'creditcard'])


# In[52]:


def plot_churn_comparison_single(data, category_column):
    unique_categories = data[category_column].unique()
        
    churn_counts = []
    for category in unique_categories:
        subset = data[data[category_column] == category]
        churn_counts.append(subset['churn'].value_counts())

    stacked_counts = pd.DataFrame(churn_counts).fillna(0)
    stacked_counts.index = unique_categories
        
    stacked_counts = stacked_counts.sort_values(by=1, ascending=False)

    churn_totals = stacked_counts.sum(axis=1)
    churn_percentages = stacked_counts.div(churn_totals, axis=0) * 100

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    stacked_counts.plot(kind='bar', stacked=True, ax=ax, color=['tomato','steelblue'])

    ax.set_title(f'Сравнение оттока в {category_column} с расчетом процента оттока внутри категории', fontsize=16)
    ax.set_xlabel(category_column, fontsize=12)
    ax.set_ylabel('Количество клиентов', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(['Отток','Нет оттока'], loc='upper right')

    for j, category in enumerate(stacked_counts.index):
        for k, percentage in enumerate(churn_percentages.loc[category][::1][:1], -1):  # Изменение порядка процентов
            ax.text(j, stacked_counts.loc[category, :(k+1)].sum(), f'{percentage:.1f}%', ha='center', va='bottom')  
            

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()


# In[53]:


plot_churn_comparison_single(data_rostov, 'products')
# костыль в действии, стиль сохранил, но элементы визуализации немного отличаются


# In[54]:


plot_churn_by_age(data_rostov)


# In[55]:


plot_churn_by_age(data_rostov, age_interval = (35, 70))


# Для Ростова стоит отметить следующие корректировки сегмента:
# 
#     - по категории gender - отток женщин больше 37,6% (в общей категории 32,4%)
#     - по количеству продуктов - картина немного изменилась, с 1 продуктом в общем по банку 27,7% против 42.8% в Ростове, 7.6% против 12.1% в Ростове. Можем уже предположить, что в Ростове какие-то проблемы с каналом Клиент-Банк. 
#     - в срезе возрата картина от общей изменилась не сильно, также самый высокий процент относительного оттока после 50 лет к 56 годам он достигает 83,3%, детальный параметр возраста, будем рассматривать чуть позже.   
#     

# #### Ярославль
# Для полноты картины, посмотрим на аналогичные визуализации по другим городам, возможно имеет место проблема с обслуживанием клиентов именно в Ростове.

# In[56]:


data_yaros = data.query ('city == "Ярославль"')


# In[57]:


plot_churn_comparison(data_yaros, ['gender', 'category', 'objects', 'products', 'creditcard'])


# In[58]:


plot_churn_by_age(data_yaros)


# In[59]:


plot_churn_by_age(data_yaros, age_interval = (35, 70))


# Для Ярославля больше характерны тенденции отмеченные для всего датасета, все значения менее выражены, чтобы может указывать, что обслуживание в Яровлавле лучше чем в Ростове. 

# #### Рыбинск 

# In[60]:


data_ribinsk = data.query ('city == "Рыбинск"')


# In[61]:


plot_churn_comparison(data_ribinsk, ['gender', 'category', 'objects', 'products', 'creditcard'])


# In[62]:


plot_churn_by_age(data_ribinsk)


# In[63]:


plot_churn_by_age(data_ribinsk, age_interval = (35, 70))


# Общая картина в Рыбинске схожа с Ярославлем - характерны тенденции отмеченные для всего датасета, все значения менее выражены, чтобы может указывать, что обслуживание остальных городах лучше чем в Ростове.

# ## Проверка статистических гипотез

# ### Формулировка гипотез
# 
# Первая гипотеза у нас дана в ТЗ
# 
#     1. Н0 - статистической разницы между теми клиентами, которые ушли и теми которые остались в доходах нет.
#        Н1 - Имеется статистическая разница между теми клиентами, которые ушли и теми которые остались.
#       
#     2. Н0 - Статистической разницы в уровне дохода между клиентами из Ростова и клиентами из Рыбинска нет.
#        Н1 - имеется статистическая разница в уровне дохода между клиентами из Ростова и клиентами из Рыбинска.
#        
#     3. Н0 - Статистической разницы в уровне оттока между клиентами из Ростова и Рынбинска нет.
#        Н1 - имеется статистическая разница в уровне оттока между клиентами из Ростова и Рынбинска нет.
#        

# #### Проверка гипотезы № 1 - статистической разницы между теми клиентами, которые ушли и теми которые остались в доходах нет.
# Перед тестом проверим выбранные выборки

# In[64]:


# для проверки гипотез нам потребуется сделать срез соответствующие проверки. А также проверить равенство дисперсий
churn = data.query ('churn == 1')
no_churn = data.query ('churn == 0')


# In[65]:


# тесты будем проводить несколько раз, поэмутому сделаем функцию

def compare_groups_test(group1, group2):
    # Проверка равенства дисперсий с помощью теста Левена
    levene_stat, levene_pvalue = st.levene(group1, group2)
    print(f"Значение статистики Левена: {levene_stat}")
    print(f"Значение p-value Левена: {levene_pvalue}")
    
    if levene_pvalue > 0.05:
        print("Дисперсии равны")
    else:
        print("Дисперсии не равны")
    print()
    
    # Проверка равномерности распределения с помощью теста Колмогорова-Смирнова
    ks_stat, ks_pvalue = st.ks_2samp(group1, group2)
    print(f"Значение статистики Колмогорова-Смирнова: {ks_stat}")
    print(f"Значение p-value Колмогорова-Смирнова: {ks_pvalue}")
    
    if ks_pvalue > 0.05:
        print("Выборки имеют равномерное распределение")
    else:
        print("Выборки не имеют равномерное распределение")
    
    print ()
    # Проверка статистической значимости различий с помощью t-теста Стьюдента
    t_stat, t_pvalue = st.ttest_ind(group1, group2)
    print(f"Значение t-статистики: {t_stat}")
    print(f"Значение p-value t-теста: {t_pvalue}")
    
    alpha = 0.05  
    if t_pvalue < alpha:
        print("Различия статистически значимы")
    else:
        print("Различия статистически незначимы")


# In[66]:


group1 = churn ['estimated_salary']
group2 = no_churn['estimated_salary']


# In[67]:


compare_groups_test (group1, group2)


# выборки не однородны и меют однородное распределение. 

# Учитывая результатыпроведенного теста, у нас нет статистически значимых доказательств о наличии разницы в доходах между клиентами, которые ушли и теми, которые остались. Мы не можем отвергнуть нулевую гипотезу

# #### Проверка гипотезы № 2 - Статистической разницы в уровне дохода между клиентами из Ростова и клиентами из Рыбинска нет.

# In[68]:


rost = data.query ('city == "Ростов Великий"')
ribin = data.query ('city == "Рыбинск"')

group1 = rost ['estimated_salary']
group2 = ribin ['estimated_salary']


# In[69]:


compare_groups_test (group1, group2)


# Учитывая результатыпроведенного теста, у нас нет статистически значимых доказательств о наличии разницы в доходах между клиентами из Ростова и Рыбинска. Мы не можем отвергнуть нулевую гипотезу

# #### Проверка гипотезы № 3 - Статистической разницы в уровне оттока между клиентами из Ростова и Рынбинска нет.

# In[70]:


rost = data.query ('city == "Ростов Великий"')
ribin = data.query ('city == "Рыбинск"')

group1 = rost ['churn']
group2 = ribin ['churn']


# In[71]:


compare_groups_test (group1, group2)


# Дисперсии не равны и выборки не имеют равномерного распределения, проведем тогда тест Уилкоксона-Манна-Уитни

# In[72]:


statistic, p_value = mannwhitneyu(group1, group2)

print(f"Значение статистики Манна-Уитни: {statistic}")
print(f"Значение p-value Манна-Уитни: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Различия статистически значимы")
else:
    print("Различия статистически незначимы")


# можем отвергнуть нулевую гипотезу и сделать вывод о наличии статистически значимых различий в уровне оттока между двумя группами клиентов.

# ## Выделение сегментов

# ### Общий параметр город - Ростов Великий

# #### Возраст

# In[73]:


segmet_rostov_age = data.query ('(city=="Ростов Великий")')


# In[74]:


segmet_rostov_gender_man = data.query ('(gender == "М") and (city=="Ростов Великий")')


# In[75]:


plot_churn_by_age(segmet_rostov_gender_man, age_interval = (35, 70))


# In[76]:


def plot_churn_visualization(data, name=None, total_count=None, churn_count=None):
    plt.figure(figsize=(6, 6))
    ax = data.plot(kind='bar', color=['steelblue', 'tomato'])

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.annotate(f'{height:.1f}%', (x + width / 2, y + height), ha='center', va='bottom')

    ax.set_xlabel('Отток')
    ax.set_ylabel('Процент клиентов')
    ax.set_title(f'График оттока {name} (всего {total_count}, из которых отточные {churn_count})')
    ax.set_xticklabels(['Нет оттока', 'Отток'], rotation=0)
    ax.legend().remove()

    plt.show()


# In[77]:


# Чувствую, что вызывать такой фильр как ниже и функцию буду много раз, значит опять нужна функция
def analyze_segment(data, query_filter, plot_title):
    segment = data.query(query_filter)
    total_count = segment.shape[0]
    churn_count = segment['churn'].sum()
    churn_percentages = segment['churn'].value_counts() / total_count * 100
    plot_churn_visualization(churn_percentages, name=plot_title, total_count=total_count, churn_count=churn_count)


# In[78]:


analyze_segment(data, '(gender == "М") and (city=="Ростов Великий")', 'по Ростову и гендеру (М)')


# In[79]:


analyze_segment(data, '(gender == "М") and (city=="Ростов Великий") and (age>40)', 'по Ростову, гендеру (М) и возрасту')


# In[80]:


segmet_rostov_gender_women = data.query ('(gender == "Ж") and (city=="Ростов Великий")')
segmet_rostov_gender_women.shape
plot_churn_by_age(segmet_rostov_gender_women, age_interval = (35, 70))


# In[81]:


analyze_segment(data, '(gender == "Ж") and (city=="Ростов Великий") and (age>40)', 'по Ростову, гендеру (Ж) и возрасту 40+')


# Мы помним с прошлых этапов, что для возраста 50+ в Ростове Великом картина меняется, отток среди мужчин составляет 47,7% (тут и далее в скобках  будут общебанковские показатели - 27.8%) , женщины 42,4% (37,6%), что говорит, о том, в Ростове в общем и целом отток с возрастом клиентов только увеличивается, при чем у мужчин чуть быстрее. 

# Прихожу к пониманию, что если брать связку гендер+возраст по Ростову, ключевым значением будет все таки возраст, за границу берем 39+ лет, так как в данном сегменте наблюдается активное падение продуктовой активности, как у мужчин, так и у женщин.

# In[82]:


analyze_segment(data, '(city=="Ростов Великий") and (age>39)', 'по Ростову и возрасту 39+')


# In[83]:


analyze_segment(data, 'age>39', 'по всем городам и возрасту 39+')


# In[84]:


analyze_segment(data, '(age>39) and (city != "Ростов Великий")', 'по городам без Ростова и возрасту 39+')


# #### Вывод по категории "Возраст"
# в Ростове отток составляет 49,2%, в общем датасете с такими же параметрами 35,9% - разница почти 15%, довольно много, а если исключить Ростов то получается - 30%, и разница уже  19,2%. Явно что-то не так с работой с клиентами среднего и старшего возраста в Ростове. 

# #### Количество продуктов
# 
# Самый сильный отток по продуктам на прошлых этапах был для клиентов с 1 продуктом, посмотрим теперь в срезе городов на продукты

# In[85]:


analyze_segment(data, '(products==1) and (city == "Ростов Великий")', 'по Ростову и 1 продукту')
analyze_segment(data, '(products==1) and (city != "Ростов Великий")', 'без Ростова и 1 продукту')


# In[86]:


analyze_segment(data, '(products==2) and (city == "Ростов Великий")', 'по Ростову и 2-м продуктам')
analyze_segment(data, '(products==2) and (city != "Ростов Великий")', 'без Ростова и 2-м продуктам')


# #### Вывод По количеству продуктов у клиентов.
# В Ростове по 1 продукту (1349 человек в Ростове) почти на 20% (в абсолютном выражении) отток больше - разница в 2 раза (22,2% против 42,8%), чем в остальных городах, можно предположить, что люди после получения 1 кредита сталкиваются с какими-то проблемами и пытаются уйти из банка. Схожая негативная картина и 2 продуктам (1040), а вот с 3 и далее - отток уже меньше чем в других городах. Кто выжил и добрался до 3-х продуктов, тот приобретает иммунитет или научился пользоваться банковскими каналами в Ростове?))))

# #### Комбинация "тонких мест" Ростова

# In[87]:


analyze_segment(data, '(products<=2) and (city == "Ростов Великий") and (age>34) and (age<=44)', 'по Ростову и 1/2 продуктам и возрасту от 34 до 45')
analyze_segment(data, '(products<=2) and (city != "Ростов Великий") and (age>34) and (age<=44)', 'без Ростова и 1/2 продукту и возрасту от 34 до 45')


# #### Выводы по комбинации условий
# В ходе изучения комбинаций различных фактов, а именно город+продукты+возраст, приходим к выводу, что ключевой фактор город+продукт, так как измениен параметров возраста показывает превышение 2 раза и выше оттока в Ростове, по сравнению с другими городами. Пожно выделить дополнительный параметр к названным - диапазон возраста от 33 и выше, при комбинации от 34 до 45 лет - отток в Ростове более чем в 2 раза выше чем в других городах. 
# Получается Банк теряет в двое больше клиентов самого активного кредитного возраста в Ростове (937 человек в Ростове в данной группе)

# ## Общие выводы по результатам исследования

#     Исходя из проведенного исследования и наблюдений, можно сделать следующие общие итоговые выводы для анализа оттока клиентов банка:
# 
#      - Город: Отток клиентов из города "Ростов Великий" составляет 32,4%. Это может указывать на наличие определенных факторов или проблем, которые влияют на удержание клиентов в данном городе. Рекомендуется более детально изучить причины и особенности оттока в этом конкретном городе и разработать меры по его снижению.
# 
#     - Пол: Отток женщин составляет более 25,1%. Возможно, женщины имеют определенные предпочтения или сталкиваются с проблемами взаимодействия с услугами или продуктами банка. Рекомендуется более глубоко изучить потребности и предпочтения женской аудитории и предложить персонализированные решения, чтобы улучшить их удовлетворенность и удержание.
# 
#     - Доход: Состоятельные клиенты с высоким доходом (от 139435 и выше) имеют более высокий уровень оттока. Это может указывать на неудовлетворительное качество услуг или недостаточную персонализацию под их потребности. Рекомендуется улучшить качество обслуживания и предложить дополнительные преимущества для клиентов с высоким доходом, чтобы повысить их лояльность и удержание.
# 
#     - Количество продуктов: Клиенты с 3 и 4 продуктами имеют самый высокий уровень оттока. Это может указывать на то, что наличие большого количества продуктов или кредитов может создавать негативное восприятие клиентами и стать причиной ухода. Рекомендуется более тщательно анализировать потребности клиентов и предлагать им наиболее подходящие продукты, а также обеспечивать их удовлетворенность и поддержку после покупки.
# 
#     - Возраст: Наибольший отток наблюдается в возрастной группе 51-56 лет (55-71%). Это может указывать на наличие определенных финансовых или жизненных изменений, которые влияют на взаимодействие клиентов с банком. Рекомендуется предоставлять персонализированные услуги и поддержку для клиентов в  возрастной группе от 51 и старше, а также проводить проактивную работу по удержанию, например при привлечении в качестве клиентов людей указанного возраста, проводить краткое обучение пользованию приложением, рассказывать о каналах взаимодействия с Банком, проводить инструктаж по способам погашения кредитов, можно рассмотреть возможность дополнить велкампаки краткими памятками. Все изложенное может поспособствовать сохранению лояльности среди клиентов 51+.
# 
# **Основываясь на этих выводах, рекомендуется разработать и внедрить меры по снижению оттока клиентов, такие как улучшение качества обслуживания, предоставление персонализированных предложений, обратная связь с клиентами и предоставление дополнительных преимуществ. Также важно продолжать мониторинг и анализ оттока клиентов для дальнейшей оптимизации стратегий удержания и повышения лояльности клиентов, изучить проблемы обслуживания на местах, например использовать методику - mystery shopping, и посмотреть с какими проблемами сталкивается клиент (особенно рекомендую обратить внимание на подобную работу в Ростове Великом). Могу рекомендовать пройти все этапы клиентского пути при взаимодействии с Банком**
# 
# 
#     Так как, Ростов Великий показал наихудшие статистические результаты, стоит обратить внимание на следующие особенности и корректировки в данном  сегменте оттока:
# 
#     - Пол: Отток женщин составляет более 37,6%. Это является более высоким показателем, чем общий отток женщин в других городах. Рекомендуется провести дополнительное исследование причин оттока женской аудитории в Ростове и разработать меры по удержанию и привлечению данного сегмента клиентов. Помимо выявления глубинных причин такого оттока у женщин, могу рекомендовать продумать дополнительные услуги и акции для данной категории клиентов (системы кэш-бэка (актуально для любой категории по гендеру), продумать дополнительные партнерские программы, апсейлы и т.п.).
# 
#     - Количество продуктов: В Ростове наблюдается изменение в показателях оттока в зависимости от количества продуктов. Самый высокий уровень оттока наблюдается у клиентов с 2 продуктами (87,9%) и 1 продуктом (57,2%), в то время как клиенты с 3 и более продуктами имеют значительно меньший уровень оттока (не более 10,5%). Это может указывать на проблемы взаимодействия с банком после получения кредита или использования продуктов (гипотезы - проблемы в каналах погашения кредитов, отсутствие достаточного количества точек внесения, дополнительные отделения Банка). Рекомендуется улучшить качество обслуживания и коммуникации с клиентами, особенно в отношении тех, кто имеет 1-2 продукта, чтобы снизить уровень оттока в этой категории (возможно рекомендация с вэлкампаками для людей 51+ будет актуальна и тут).
# 
#     - Возраст: оказался ключевым показателем для Ростова также наблюдается высокий уровень оттока после 39 лет, и к 56 годам этот показатель достигает 83,3%. Рекомендуется более детально изучить потребности и предпочтения клиентов данной возрастной группы и предложить им персонализированные решения, учитывающие их финансовые и жизненные изменения (рекомендации в общих тенденциях актуальны).
# 
#     
#     Исходя из проведенных тестов и проверки гипотез, можно сделать следующие итоговые выводы:
# 
#     - Проверка гипотезы № 1: Н0 - Статистической разницы в уровне дохода между клиентами, которые ушли и теми, которые остались, нет. 
#     Результаты теста показали, что дисперсии равны, выборки имеют равномерное распределение, и различия статистически незначимы. Мы не можем отвергнуть нулевую гипотезу.
# 
#     - Проверка гипотезы № 2: Н0 - Статистической разницы в уровне дохода между клиентами из города Ростова и клиентами из города Рыбинска нет. 
#     Результаты теста показали, что дисперсии равны, выборки имеют равномерное распределение, и различия статистически незначимы. Мы не можем отвергнуть нулевую гипотезу.
# 
#     - Проверка гипотезы № 3: Н0 - Статистическая разницы в уровне оттока между клиентами из города Ростова и города Рыбинска нет. 
#     Однако, дисперсии были не равны, выборки не имели равномерного распределения. В связи с этим, был проведен тест Манна-Уитни, который показал статистически значимые различия в уровне оттока между этими двумя группами клиентов. Мы можем отвергнуть нулевую гипотезу и сделать вывод о наличии статистически значимых различий в уровне оттока.
#     
#     
#     
#     Сегментное исследование:
#     
#     В Ростове отток составляет 49,2%, в то время как в общем датасете с аналогичными параметрами отток составляет 35,9%. Это означает, что разница в оттоке между Ростовом и остальными городами составляет почти 15%. Если исключить Ростов из анализа, то общий отток снижается до 30%, и разница с другими городами увеличивается до 19,2%. Эти данные указывают на проблемы с работой с клиентами среднего и старшего возраста в Ростове.
# 
#     Когда рассматривается отток по клиентам с 1 продуктом, в Ростове наблюдается значительно больший отток (42,8%) по сравнению с остальными городами (22,2%). Аналогичная тенденция наблюдается и для клиентов с 2 продуктами. Однако, отток снижается для клиентов с 3 и более продуктами в Ростове по сравнению с другими городами.
# 
#     При анализе комбинации факторов, таких как город, продукты и возраст, становится ясно, что ключевым фактором является сочетание города и продуктов. Изменение возрастных параметров показывает, что отток в Ростове превышает в два и более раза отток в других городах. Особенно выделяется диапазон возраста от 34 до 45 лет, где отток в Ростове более чем в два раза выше, чем в других городах. В данной группе банк теряет в два раза больше клиентов самого активного кредитного возраста в Ростове (937 человек в Ростове в этой группе, из которых 234 отточные).
# 
#     Эти выводы указывают на необходимость уделить особое внимание клиентам в Ростове, особенно в возрастной группе от 34 до 45 лет, и проанализировать возможные причины высокого оттока. Банку стоит обратить внимание на улучшение условий и качества обслуживания в Ростове, а также разработать меры по удержанию клиентов этой целевой группы.
# 
# **Такми образом, на основе проведенных проверок гипотез и анализа данных, мы приходим к выводу о наличии статистически значимых различий в уровне оттока клиентов банка, а именно отточные клиенты из Ростова Великого активного кредитного, почти в раза чаще уходят из Банка, чем в других городах, отмечу, что основнуя часть клиентов составляют именно люди указанного возрастного диапазона. Рекомендуется рассмотреть изложенные выше рекомендации и принять меры по улучшению удержания клиентов, а с учетом выявленных особенностей и факторов, которые могут влиять на отток, рекомендуется обратить особое внимание на Ростов Великий**

# ## Презентация

# [Презентация по результатам](https://disk.yandex.ru/i/JCAuuRarAmvSvQ)

# ## Дашборд

# [Дашборд](https://public.tableau.com/shared/X2WFF77RQ?:display_count=n&:origin=viz_share_link)

# Ниже оставил код, с помощью которого выгружал нужные файлы

# In[88]:


try:
    file_path = r'C:\Users\PC_Maks\Desktop\study\final_project\preza\bank_dataset_final.csv'
    data.to_csv(file_path, index=False)
except:
    display ('датасет сохранен локально')


# In[89]:


quantile_ranges = (
    data.groupby('category')
    ['estimated_salary']
    .quantile([i / 10 for i in range(11)])
    .unstack()
    .reset_index()
)

quantile_ranges


# In[90]:


quantiles = quantile_ranges[['category', 0.0, 1.0]].copy()
new_columns = {
    0.0: 'min',
    1.0: 'max'
}
quantiles.rename(columns=new_columns, inplace=True)
quantiles=quantiles.round(0).astype(int)


# In[91]:


try:
    csv_file_path = r'C:\Users\PC_Maks\Desktop\study\final_project\cat_range.csv'
    quantiles.to_csv(csv_file_path, index=False)
except:
    display ('датасет сохранен локально')

