#!/usr/bin/env python
# coding: utf-8

# # Проект "Тестирование замены шрифтов" 
# Вы работаете в стартапе, который продаёт продукты питания. Нужно разобраться, как ведут себя пользователи вашего мобильного приложения. 
# 
# Необходимо изучить воронку продаж. 
# 
# Узнайть, как пользователи доходят до покупки. 
# 
# Сколько пользователей доходит до покупки, а сколько — «застревает» на предыдущих шагах? На каких именно?
# 
# 
# 
# **После этого необходимо исследовать результаты A/A/B-эксперимента.** 
# 
# Дизайнеры захотели поменять шрифты во всём приложении, а менеджеры испугались, что пользователям будет непривычно. Договорились принять решение по результатам A/A/B-теста. 
# 
# Пользователей разбили на 3 группы: 2 контрольные со старыми шрифтами и одну экспериментальную — с новыми. Выясните, какой шрифт лучше.
# 
# Создание двух групп A вместо одной имеет определённые преимущества. Если две контрольные группы окажутся равны, вы можете быть уверены в точности проведенного тестирования. Если же между значениями A и A будут существенные различия, это поможет обнаружить факторы, которые привели к искажению результатов. Сравнение контрольных групп также помогает понять, сколько времени и данных потребуется для дальнейших тестов.
# 
# 
# В случае общей аналитики и A/A/B-эксперимента работайте с одними и теми же данными. В реальных проектах всегда идут эксперименты. Аналитики исследуют качество работы приложения по общим данным, не учитывая принадлежность пользователей к экспериментам.
# 
# 

# # Описание данных
# 
# Каждая запись в логе — это действие пользователя, или событие:
#     
#     - EventName — название события;
#     - DeviceIDHash — уникальный идентификатор пользователя;
#     - EventTimestamp — время события;
#     - ExpId — номер эксперимента: 246 и 247 — контрольные группы, а 248 — экспериментальная.

# <a id='выполнение_проекта'></a>
# # Выполнение проекта
# 
# 
# <a id='этап_1'></a>
# 
# 
# 
# **Этап 1.** Подготовьте данные
# 
#     - Замените названия столбцов на удобные для вас;
#     - Проверьте пропуски и типы данных. Откорректируйте, если нужно;
#     - Добавьте столбец даты и времени, а также отдельный столбец дат;
# 
# <a id='этап_2'></a>
# 
# 
# 
# 
# **Этап 2.** Изучите и проверьте данные
# 
#     - Сколько всего событий в логе?
#     - Сколько всего пользователей в логе?
#      -Сколько в среднем событий приходится на пользователя?
#     - Данными за какой период вы располагаете? Найдите максимальную и минимальную дату. Постройте гистограмму по дате и времени. Можно ли быть уверенным, что у вас одинаково полные данные за весь период? Технически в логи новых дней по некоторым пользователям могут «доезжать» события из прошлого — это может «перекашивать данные». Определите, с какого момента данные полные и отбросьте более старые. Данными за какой период времени вы располагаете на самом деле?
#     - Много ли событий и пользователей вы потеряли, отбросив старые данные?
#      -Проверьте, что у вас есть пользователи из всех трёх экспериментальных групп.
# 
# <a id='этап_3'></a>
# 
# 
# 
# 
# **Этап 3.** Изучите воронку событий
# 
#     - Посмотрите, какие события есть в логах, как часто они встречаются. Отсортируйте события по частоте.
#     - Посчитайте, сколько пользователей совершали каждое из этих событий. Отсортируйте события по числу пользователей.
#     - Посчитайте долю пользователей, которые хоть раз совершали событие.
#     - Предположите, в каком порядке происходят события. Все ли они выстраиваются в последовательную цепочку? Их не нужно учитывать при расчёте воронки.
#     - По воронке событий посчитайте, какая доля пользователей проходит на следующий шаг воронки (от числа пользователей на предыдущем). То есть для последовательности событий A → B → C посчитайте отношение числа пользователей с событием B к количеству пользователей с событием A, а также отношение числа пользователей с событием C к количеству пользователей с событием B.
#     - На каком шаге теряете больше всего пользователей?
#     - Какая доля пользователей доходит от первого события до оплаты?
# 
# <a id='этап_4'></a>
# 
# 
# 
# 
# **Этап 4.** Изучите результаты эксперимента
# 
#     - Сколько пользователей в каждой экспериментальной группе?
#     - Есть 2 контрольные группы для А/А-эксперимента, чтобы проверить корректность всех механизмов и расчётов. Проверьте, находят ли статистические критерии разницу между выборками 246 и 247.
#     - Выберите самое популярное событие. Посчитайте число пользователей, совершивших это событие в каждой из контрольных групп. Посчитайте долю пользователей, совершивших это событие. Проверьте, будет ли отличие между группами статистически достоверным. Проделайте то же самое для всех других событий (удобно обернуть проверку в отдельную функцию). Можно ли сказать, что разбиение на группы работает корректно?
#     - Аналогично поступите с группой с изменённым шрифтом. Сравните результаты с каждой из контрольных групп в отдельности по каждому событию. Сравните результаты с объединённой контрольной группой. Какие выводы из эксперимента можно сделать?
#     - Какой уровень значимости вы выбрали при проверке статистических гипотез выше? Посчитайте, сколько проверок статистических гипотез вы сделали. При уровне значимости 0.1 каждый десятый раз можно получать ложный результат. Какой уровень значимости стоит применить? Если вы хотите изменить его, проделайте предыдущие пункты и проверьте свои выводы.

# **Общий план выполнения проекта** 
# 
# Перед выполнением любой из задач, в первую очередь необходимо подготовить набор библиотек, познакомится с данными.
# 
# Сам процесс исследования будет проведен в следующем порядке (основные этапы):
# 
#     - загрузка библиотек и знакомство с данными
#     - предобработка данных (Этап 2)
#     - формирование временных таблиц, включая новые сводные (если потребуется)
#     - проведение 3 Этапа - первичная оценка данных, в части соразмерности, общих границ, аномалий, попаданий в разные группы одних и тех же пользователей (если применимо), детальное изучение наполнения всех трех групп
#     - проведение 4 Этапа, на котором построим первые воронки, оценм отток, визуализируем первые воронки
#     - проведение 5 Этапа - изучение результатов эксперемента (к этому этапу необходимо провести все первчиные корректировки и оценки)
#     - Общие выводы по результатам
# 
# *Итоговая цель анализа* - оценка результатов текущего этапа тестирования, обобщение рещультатов изучения конверсии, выявление точек внимания, рекомендации

# ## Этап. Предварительная обработка данных и предварительная обработка

# [Перейти к разделу содержания "Этап 1"](#этап_1)

# ### Загрузка необходимого набора библиотек

# In[1]:


import pandas as pd
import numpy as np
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


# ### Загрузка DS 

# In[2]:


try:
    data = pd.read_csv (r'C:\Users\PC_Maks\Desktop\study\S 14 Event_analytics\project\logs_exp.csv', sep='\s+')
except:
    data = pd.read_csv ('/datasets/logs_exp.csv', sep='\s+') 
    
    
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)


# In[3]:


data


# В столбце EventTimestamp у нас целочисленное значение, количество секунд с начала эпохи Unix  (1 января 1970), сразу добавим корректный столбец

# In[4]:


data['datetime'] = data['EventTimestamp'].apply(lambda x: datetime.fromtimestamp(int(x)))
data['date'] = data['datetime'].dt.date
data['date'] = pd.to_datetime(data['date'])

data.info()


# ### Предварительная обработка (пропуски, дубли, переименование)
# 
# Воспользуемся заготовленными ранее функциями для изучения и первичной обработки DS 

# In[5]:


def data_preprocessing(data):
    # приводим названия столбцов к нижнему регистру и заменяем пробелы на _
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    
    # переименовываем столбцы
    data = data.rename(columns=lambda x: x.lower().replace(' ', '_'))
    
    return data


# In[6]:


data = data_preprocessing (data)


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


# удалим яные дубли и еще раз проверим 

# In[9]:


duplicates = data.duplicated()
data = data.drop_duplicates()
display(f'Удалено явных дубликатов: {duplicates.sum()}')
check (data=data)


# Переименуем столбцы для более удобной работы

# In[10]:


data.columns = ['event_name', 'user_id', 'pre_time', 'group', 'date_time', 'date']


# создадим копию исходного датасета, который потребуется для вычисления общих потерь на финальном этапе предобработки

# In[11]:


data_base = data


# Изучим содержание столбцов

# In[12]:


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
        if len(data[col].unique()) > 10:
            print(f"В столбце {col} более 10 уникальных значений")
        else:
            print(f"Уникальные значения в столбце {col}:")
            print(data[col].unique())
        print('---------------------')


# In[13]:


check_unique(data)


# ### Проверка на вхождения в несколько тестовых групп
# 
# Результат проверки - вхождения пользователей сразу в несколько групп не выявлено, переименуем группы для более удобного восприятия

# In[14]:


double_entries = data.groupby('user_id').agg({'group' : 'nunique'}).query('group>1')
display (double_entries.count ())
data ['group'].nunique ()


# In[15]:


def rename_group(group):
    if group==246:
        return 'A1'
    elif group==247:
        return 'A2'
    elif group==248:
        return 'B'
    else:
        return 'UnknownGroup'

data['group'] = data['group'].apply(rename_group)


# ### Процентное отношение в группах
# 
# Группа B немного больше, пока держим это в голове, и после чистки и корректировки ДС оценим повторно

# In[16]:


temp=data.groupby('group').agg({'user_id' : 'count'}).reset_index()
display (temp)
total = temp ['user_id'].sum ()
temp ['percent'] = (temp['user_id'] / total) * 100
temp


# ## Этап. Изучение и проверка данных

# ### Количество событий в логе.

# In[17]:


display(f"Количество уникальных значений в столбце event_name: {data['event_name'].nunique()}")
display(f"Всего записей в столбце event_name: {data['event_name'].count()}")
display (f"Уникальные значения в столбце event_name: {data['event_name'].unique()}")
event_name_counts = data['event_name'].value_counts()
display(pd.DataFrame({'event_name': event_name_counts.index, 'counts': event_name_counts.values}))


# ### Сколько всего пользователей в логе

# In[18]:


display(f"Количество уникальных значений в столбце user_id: {data['user_id'].nunique()}")


# ### Среднее количество событий на пользователя. 

# In[19]:


user_event = data.groupby('user_id').agg({'event_name' : 'count'}).mean ().round (1)
user_event


# ### Построим гистограмму по дате и времени. 

# Дополним ранее полученные расчеты дат интервалом 

# In[20]:


display (f"Временной интервал : {data['date_time'].max() - data['date_time'].min()}")


# In[21]:


#определим количетсов корзин
bin_limit=data['date'].nunique ()

plt.figure(figsize=(12, 6))

ax = data['date_time'].hist(bins=bin_limit, grid=True, edgecolor='white', color='blue', alpha=0.7, align='left', rwidth=0.9)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 

plt.xticks(rotation=45)
plt.xlabel('Дата')
plt.ylabel('Количество событий')
plt.title('Распределение событий по датам')

plt.show()


# Как видно из графика, основные данные поступают в период с 2019-07-31 по 2019-08-07, данные до 31 июля и после 07 августа можно отбросить, но сначала посмотрим на данные в связке количества действий на пользователя, а после принятия решения о корректировке и изучении выбросов, оценим повторно

# In[22]:


user_event_group = data.groupby('user_id').agg(event_count=('event_name', 'count')).reset_index()

percentile_98 = np.percentile(user_event_group['event_count'], 98).astype(int)
display('Процентиль количества действий на одного пользователя (оценка 98%):', percentile_98)

plt.figure(figsize=(12, 6))
plt.hist(x=user_event_group['event_count'], bins=50, range=[136, 2500], edgecolor='white', color='blue', alpha=0.7)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Количество событий на одного пользователя')
plt.ylabel('Количество пользователей')
plt.title('Распределение количества событий на одного пользователя')
plt.axvline(x=percentile_98, color='red', label='98% percentile')
plt.legend()
plt.show()


# In[23]:


display (user_event_group ['event_count'].describe())


# **Промежуточный итог**
# 
# До перерасчетов, среднее значение количества операций на одного пользователя 32, медиана 20, в 98% укладывается активность до 136 операций на одного пользователя, прихожу к пониманию допустимости удаления пользователей, у которых более 136 операций, так я и удалю "догрузившиеся данные" и не критично потеряю в исходных данных, так как 98% исходных данных остаются. 
# 
# З.Ы. отталкиваться только от типов действий, полагаю нецелесообразным, для принятия решения о чистке, так как некоторые пользователи могли посмотреть товар, но не добавить в карзину, кто-то добавил но не оплатил, и т.п., а потом возвращаться и просматривать, доходить к разделу оплаты и в итоге опять не принять решение купить. 

# ### Оценка потерь в количестве событий и пользователей, после того, как отбросили старые данные.

# In[24]:


user_event_group_98 = user_event_group.query ('event_count<=@percentile_98')


# In[25]:


display(f"Количество уникальных значений в столбце event_name: {data['event_name'].nunique()}")
display(f"Всего записей в столбце event_name: {data['event_name'].count()}")


# In[26]:


display (f'Количество удаленных пользователей {user_event_group.shape[0]-user_event_group_98.shape[0]}')


# In[27]:


user_event_group_98 ['event_count'].describe ()


# Медиана с 20 уменьшилась до 19, среднее значение с 32 уменьшилось до 26.6, среднее всегда более чувствительно к выбросам, поэтому и значение изменилось сильнее. В типах событий не потеряли. А вот в количестве событий ушло 46590, достаточно много, но мы и предпологали, что имеем дело с выбросами из-за лага в загрузках. Идем дальше.

# In[28]:


filter = data['user_id'].isin(user_event_group_98['user_id'])
data = data[filter]
temp=data.groupby('user_id').agg({'event_name' : 'count'}).reset_index()
temp['event_name'].describe()


# Сохранили обновленный ДС

# ### Проверим, что у нас есть активность пользователей и уникальные пользователи во всех трёх экспериментальных группах.

# In[29]:


result = data.pivot_table(index='group', 
                          values='user_id', 
                          aggfunc=['count', 'nunique'])
result.columns = ['total_users', 'unique_users']
result['percent_total'] = (result['total_users'] / result['total_users'].sum()) * 100
result['percent_unique'] = (result['unique_users'] / result['unique_users'].sum()) * 100
result


# В итоговом наборе у нас остаются представители всех групп, и что также важно, теперь примерно в равных пропорциях.

# ### Итоговая оценка периода

# Повторно построим график количества действий пользователей по датам, и посмотрим изменилась ли общая картина

# In[30]:


period = [    ('2019-07-25','2019-08-08'),
          ('2019-07-31 00:00:00', '2019-08-01 23:59:59'),
          ('2019-08-07 00:00:00', '2019-08-09 23:59:59')
         ]

for start, end in period:
    plt.figure(figsize=(12, 6))
    data_per = data.query('(date_time >= @start) & (date_time <= @end)')
    ax = data_per['date_time'].hist(bins=20, grid=True, edgecolor='white',                                    color='#4C72B0', alpha=0.7, align='left', rwidth=0.9)
    ax.set_xlabel('Дата', fontsize=14)
    ax.set_ylabel('Количество событий', fontsize=14)
    ax.set_title('Распределение событий по периодам', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    ax.legend(['Количество событий'])
    plt.show()


# Координальных изменнеий не произошло после прошлых этапов чистки, мы также наблюдаем дату начала сбора статистики от 01 августа (есть и ранее значения, но их объем не существенный). Могу предположить, что первую неделю шла отладка/корректировка систем, таким образом, данные до 01 августа можно удалить. А для анализа оставить только врменной интервал с с 01 августа (00-00) до 07 включительно

# ### Финальная "чистка" и оценка "потерь"

# In[31]:


data_new = data.query('date >= "2019-08-01"')


# In[32]:


def calculate_data_loss(data, data_new):
    try:
        lost_data = data.shape[0] - data_new.shape[0]  
        percent_lost_data = round((lost_data / data.shape[0]) * 100, 2)  
    
        lost_users= data ['user_id'].nunique () - data_new ['user_id'].nunique ()
        percent_lost_users = round((lost_users / data ['user_id'].nunique ()) * 100, 2)
    
        lost_event= data ['event_name'].count () - data_new ['event_name'].count ()
        percent_lost_event = round((lost_event / data ['event_name'].count ()) * 100, 2)

        display(f'Количество потерянных строк: {lost_data} ({percent_lost_data}%)')
        display(f'Количество потерянных столбцов: {data.shape[1] - data_new.shape[1]}')
        display(f'Количество потерянных событий: {lost_event} ({percent_lost_event}%)')
        display(f'Количество потерянных пользователей: {lost_users} ({percent_lost_users}%)')
    
    
    except Exception as e:
        display (f'ERROR: {e}')


# In[33]:


calculate_data_loss (data, data_new)


# После корректировки периода выгрузки, "потери" в событиях и пользователях не критичные - 1,32% и 0,23% соответственно, но для полноты картины проверим сколько составили потери от базового датасета 

# In[34]:


calculate_data_loss (data_base, data_new)


# Как видно из результатов, если корректировка временного интервала не дала потерь более 1,32% по каждой из позиций ДС, то удаление "запоздалых" и аномальных записей увеличило количество потерь в событиях до 20,18%, в клиентах 2.2%. Но, учитывая что потери в уникальных клиентах всего 2.2%, данная чистка приемлема. Идем дальше.

# In[35]:


# сохраним скорректированный ДС как основной
data = data_new


# ## Этап. Воронка событий

# [Перейти к разделу содержания "Этап 3"](#этап_3)

# ### Посмотрим, какие события есть в логах, как часто они встречаются. Отсортеруем события по частоте.
# Main Screen Appear - главный экран
# 
# Offers Screen Appear - экран предложений
# 
# Cart Screen Appear - экран корзины
# 
# Payment Screen Successful - экран успешной оплаты
# 
# Tutorial - обучение
# 
# 

# ### Посчитаем уникального пользователя на каждом этапе

# In[36]:


event_users_uniq= data.groupby ('event_name') ['user_id'].nunique ().sort_values (ascending=False)
display (event_users_uniq)
data ['user_id'].nunique()


# При проведении прошлых этапов, мы исходили из того, что любой пользователь сначала попадает на главный экран, но сейчас мы видим, что у нас разница между уникальными пользователями попавшими на главный экран, и всеми уникальными пользователями составляет 113, надо выяснить что это за 113 пользователей, которые прошли на следующие этапы без главного экрана

# In[37]:


users_without_main_screen_appear = data.query('event_name == "MainScreenAppear"')['user_id'].unique()
ds_without_main = data.query('user_id not in @users_without_main_screen_appear')

ds_without_main ['user_id'].nunique ()


# In[38]:


ds_without_main_group = ds_without_main.groupby (['group', 'event_name']) ['user_id'].nunique()
ds_without_main_group


# Прим. странно что они есть и в тестовой группе В, при условии, что тест начался в первый день выгрузки, или какая-то ошибка технического плана, или тестовое разделение началось до периода формирования первых логов выгрузки.

# ### Посчитаем, сколько пользователей совершали каждое из этих событий. Отсортируем события по числу пользователей, а также количество уникальных пользователей на каждом этапе

# In[39]:


event_group_users = data.groupby('event_name')['user_id'].agg(['nunique', 'count']).reset_index()
event_group_users.rename(columns={'nunique': 'users_count', 'count': 'event_count'}, inplace=True)
#display(event_group_users)
event_group_users = event_group_users.query ('event_name not in "Tutorial"')

all_uniq_user = data ['user_id'].nunique ()

event_group_users ['ration_uniq_users'] = (event_group_users ['users_count']/all_uniq_user*100).round (2)

event_group_users = event_group_users.sort_values (by='users_count',ascending = False).reset_index(drop=True)
event_group_users


# Посчитаем количество событий по типам, с учетом повторов (некоторые пользователи делают одни и теже действия), а также отношение событий в каждой категории к общему количеству категорий

# In[40]:


total_users= event_group_users['event_count'].sum ()

event_group_users ['event_ration'] = (event_group_users ['event_count']/total_users*100).round (1)
event_group_users.sort_values (by='event_count', ascending = False).reset_index(drop=True)


# Расчитаем коверсию шага в событиях

# In[41]:


for i, row in enumerate(event_group_users.itertuples(), start=0):
    if i == 0:
        event_group_users.at[row.Index, 'conv_step'] = 100
    else:
        prev_row = event_group_users.iloc[i - 1]
        event_group_users.at[row.Index, 'conv_step'] = (row.users_count / prev_row.users_count * 100).round(2)


# In[42]:


event_group_users


# In[43]:


def graf_uniq(data, column_group):
    try:
        fig, ax = plt.subplots(ncols=len(column_group), figsize=(25,15))
        plt.style.use('seaborn-muted')
        
        for i, col in enumerate(column_group):
            counts = data[col]
            labels = data['event_name']
            explode = [0.05] * len(counts) 
            colors = ['#55A868', '#4C72B0', '#DD8452', '#CCB974', '#C44E52', '#8C4D3E']

            fig_size = (10, 8)
            
            ax[i].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.8, explode=explode,
                      colors=colors, textprops={'fontsize': 14})
            ax[i].set_title(f'Распределение событий {col}', fontsize=16)

            ax[i].grid(linestyle='--')
            
        

        plt.show()
        
    except Exception as e:
        display (f'ERROR: {e}')


# In[44]:


graf_uniq(event_group_users, ['users_count', 'event_count'])


# Посмотрим на распределение событий внутри групп.

# In[45]:


fig_size = (10, 8)
fig, ax = plt.subplots(figsize=fig_size)

event_order = ['MainScreenAppear', 'OffersScreenAppear', 'CartScreenAppear', 'PaymentScreenSuccessful']

grouped_data = data.groupby(['group', 'event_name'])['user_id'].count().reset_index()
groups = ['A1', 'A2', 'B']
grouped_data = grouped_data[grouped_data['group'].isin(groups)]

grouped_data.pivot(index='event_name', columns='group', values='user_id').reindex(event_order).plot.bar(ax=ax, legend='best')

ax.set_title('Сравнение количества событий по группам', fontsize=16)
ax.set_xlabel('События', fontsize=14)
ax.set_ylabel('Количество событий', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax.grid(axis='y', alpha=0.7)

plt.show()


# Больше всего пользователей заходит на главную страницу (что логично) - 56,5% активностей именно в этой категории (процентное отношение от общей суммы событий), самая не "популярное" - чтение обучения, но ее мы удалили при анализе из сводной таблицы, и из интересующих событий - меньше всего событий в категории успешной оплаты. Прим. держим в голове что это не уникальные пользователи, и любой из пользоватлей может быть в каждой из категорий, причем по нескольку раз. Прим, внутри категорий событий, группы распределены примерно одинаково.
# 
# В части конверсии всех пользователей которые зашли на сайт, если брать за 100% первый шаг - главный экран, то на второй этап - просмотр предложения переходит только 61,14% уникальных пользователей, на следующих шага, потеря в конверсии не превышает 20%. Детально рассмотрим на эатпе выводов.
# 

# ### Посчитаем долю пользователей, которые хоть раз совершали событие (вероятно имеется ввиду, что любое событие отличное от входа на главную стриницу, но проверим).

# In[46]:


# пользователей которые ничего не делали в таблице нет, хоть и проверяли на пропуски, ну лучше перепроверить
data.groupby ('user_id').agg ({'event_name':'nunique'}).query('event_name==0')


# Все пользователи в таблице так или иначе что-то совершали, как минимум заходили на первую страницу (страно если бы было иначе), поэтому посмотрим на тех пользователей которые только зашли на стриницу и больше ничего не делали

# In[47]:


# фильтруем пользователей только с событием MainScreenAppear
users_with_main_only = data.groupby('user_id')['event_name'].unique().reset_index()
users_with_main_only = users_with_main_only[users_with_main_only['event_name'].apply(lambda x: len(x) == 1 and 'MainScreenAppear' in x)]

# находим количество таких пользователей
only_main_screen_appear_users_count = len(users_with_main_only)
display(f"Количество пользователей, которые только выполнили событие MainScreenAppear: {only_main_screen_appear_users_count}")

all_uniq_users = data['user_id'].nunique()

# вычисляем количество пользователей, которые прошли любой из этапов, кроме MainScreenAppear
users_with_other_events = all_uniq_users - only_main_screen_appear_users_count
display(f"Количество пользователей, которые прошли любой из этапов, помимо MainScreenAppear: {users_with_other_events}")


# ### Выясним в каком порядке происходят события. Все ли они выстраиваются в последовательную цепочку? Какая конверсия к первому действию, конверсия от шага к шагу, и выясним где больше всего "отваливается" пользователей

# Дополним нашу таблицу event_group_users и добавим расчет конверсии к первому действию.
# 
# Берем за 100% шаг с главным экраном

# из таблицы event_group_users уже удалено  обучение (tutorial), оно не относится к расчетам конверсии 

# In[48]:


event_group_users

first_step = event_group_users.loc [0] ['users_count']

event_group_users ['conv_first_step'] = (event_group_users ['users_count']/first_step*100).round (2)
event_group_users['lost_users'] =event_group_users['users_count'].diff().fillna(0)
event_group_users['lost_users'] = abs(event_group_users['lost_users'])
event_group_users


# In[49]:


fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(event_group_users['event_name'], event_group_users['conv_first_step'], color='#008fd5')
ax.set_title('Конверсия пользователей к первому событию', fontsize=16)
ax.set_xlabel('Событие', fontsize=14)
ax.set_ylabel('Конверсия', fontsize=14)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.set_xticks(range(len(event_group_users)))
ax.set_xticklabels(event_group_users['event_name'], rotation=45, ha='right')
ax.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(event_group_users['conv_first_step']):
    ax.text(i, v, f'{v:.1f}%', ha='center', fontsize=12)

plt.show()


# Общая конверсия к первому шагу по пользователям составляет 46,7%, т.е. из числа всех уникальных пользователей в датасете, к последнему шагу так или иначе доходят 46,7%

# Посмотрим на конверсию от шага в шаге в графике

# In[50]:


fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(event_group_users['event_name'], event_group_users['conv_step'], color='#008fd5')
ax.set_title('Конверсия шага от события к событию', fontsize=16)
ax.set_xlabel('Событие', fontsize=14)
ax.set_ylabel('Конверсия пользователей по этапам', fontsize=14)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticks(range(len(event_group_users)))
ax.set_xticklabels(event_group_users['event_name'], rotation=45, ha='right')

for i, v in enumerate(event_group_users['conv_step']):
    ax.text(i, v, f'{v:.1f}%', ha='center', fontsize=12)

plt.show()


# In[51]:


labels = ['Просмотр главной страницы', 'Просмотр страницы с предложениями', 'Просмотр страницы корзины', 'Успешная оплата']

colors = ['#005b99', '#0077b5', '#0092d2', '#00ade6']

fig = go.Figure(go.Funnel(
    x = event_group_users['users_count'],
    y = event_group_users['event_name'],
    textfont = {"size": 18},
    textposition = "inside",
    text = labels,
    marker = {"color": colors}))


fig.update_layout(title = "Количество уникальных пользователей на каждом шаге",
                  xaxis_title = "Количество пользователей",
                  yaxis_title = "Шаги")


fig.show()


# In[52]:


labels = ['Просмотр главной страницы', 'Просмотр страницы с предложениями', 'Просмотр страницы корзины', 'Успешная оплата']

colors = ['#001f3f', '#0074D9', '#7FDBFF', '#39CCCC']

fig = go.Figure(go.Funnel(
    x = event_group_users['conv_first_step'],
    y = event_group_users['event_name'],
    textfont = {"size": 18},
    textposition = "inside",
    text = labels,
    marker = {"color": colors}))


fig.update_layout(title = "Конверсия пользователей по отношению к главной странице",
                  yaxis_title = "Шаги")


fig.show()


# In[53]:


labels = ['Просмотр главной страницы', 'Просмотр страницы с предложениями', 'Просмотр страницы корзины', 'Успешная оплата']

colors = ['#7FDBFF', '#001f3f', '#0092d2', '#00ade6']

fig = go.Figure(go.Funnel(
    x = event_group_users['conv_step'],
    y = event_group_users['event_name'],
    textfont = {"size": 18},
    textposition = "inside",
    text = labels,
    marker = {"color": colors}))


fig.update_layout(title = "Конверсия пользователей на каждом шаге",
                  yaxis_title = "Шаги")


fig.show()


# ### Вывод
# 
# В результате изучения воронок мы выяснили, что:
#     
#     - в логах встречается 5 типов событий (с частотой появления):
#         - Main Screen Appear - главный экран (109324, с удельной пропорцией 56,5%)
#         - Offers Screen Appear - экран предложений (37696, 19,5%)
#         - Cart Screen Appear - экран корзины (26539, 13,7%)
#         - Payment Screen Successful - экран успешной оплаты (19997, 10.3%)
#         - Tutorial - 808
#         
#     - события по типам и расчетом уникальных пользователей на каждое событие:
#         - MainScreenAppear - 7292
#         - OffersScreenAppear - 4466
#         - CartScreenAppear - 3588
#         - PaymentScreenSuccessful - 3395
#         
#         
#     - Доля уникальных пользователей, которые хоть раз совершали любое событие отличное от просмотра главного экрана -  61,14%, дальше главного экрана не ушли 38,86% (на первом шаге 7292, на втором только 4466) пользователей - довольно большая цифра, стоит взять на заметку, возможно имеет место техническая проблема.
#     
#     - логически события идут в следующем порядке: MainScreenAppear - OffersScreenAppear - CartScreenAppear - PaymentScreenSuccessful. Просмотр Tutorial необязательный этап, его мы исключили при разборе ранее.
#     
#     - Ранее мы выяснили, что среди уникальных пользователей дальше главного экрана не ушло 38,7% пользователей, в общем отношении всех операций тоже не очень радужные (видимо некоторые пользователи не столкнулись с потенциальной технической проблемой, которую мы предположили при расчете уникальных пользователей), и так:
#         - MainScreenAppear - 100% (шаг 1)
#         - на шаге OffersScreenAppear  только 61,14% от тех кто был на главной странице
#         - от предложения до корзины CartScreenAppear доходит 80,7% из шага 2 (предложение)
#         - от корзины до успешной оплаты PaymentScreenSuccessful 94,62% из шага 3 (корзина)
#         
#     - Таким, образом, безотносительно оценки уникальных пользователей, явные проблемы с конверсией от главной страницы до просмотра предложения (вопрос для передачи коллегам так как, или у нас технические проблемы, или пользователям настолько неудобно пользоваться сайтом)
#  
# 

# <a id='этап_4_1'></a>

# ## Этап. Изучение результатов эксперимента

# 
# - Есть 2 контрольные группы для А/А-эксперимента, чтобы проверить корректность всех механизмов и расчётов. Проверьте, находят ли статистические критерии разницу между выборками 246 и 247.
# - Выберите самое популярное событие. Посчитайте число пользователей, совершивших это событие в каждой из контрольных групп. Посчитайте долю пользователей, совершивших это событие. Проверьте, будет ли отличие между группами статистически достоверным. Проделайте то же самое для всех других событий (удобно обернуть проверку в отдельную функцию). Можно ли сказать, что разбиение на группы работает корректно?
# - Аналогично поступите с группой с изменённым шрифтом. Сравните результаты с каждой из контрольных групп в отдельности по каждому событию. Сравните результаты с объединённой контрольной группой. Какие выводы из эксперимента можно сделать?
# - Какой уровень значимости вы выбрали при проверке статистических гипотез выше? Посчитайте, сколько проверок статистических гипотез вы сделали. При уровне значимости 0.1 каждый десятый раз можно получать ложный результат. Какой уровень значимости стоит применить? Если вы хотите изменить его, проделайте предыдущие пункты и проверьте свои выводы.

# ### Количество пользователей в каждой группе
# 
# после всех наших "чисток" и корректировок круппы примерно равны, около 33% уникальных пользователей в каждой группе

# In[54]:


data_test = data.query ('event_name not in "Tutorial"')
group_table = data_test.pivot_table (values='user_id', index = 'group', aggfunc = ['count', 'nunique']).reset_index()
group_table.columns = ['group','total_events', 'unique_users_count']
total_user = group_table ['unique_users_count'].sum ()
group_table ['ratoin_all'] = (group_table ['unique_users_count']/total_user*100). round (2)
group_table


# ### Проверим есть ли между контрольными группами А1 и А2 (выборка А/А-эксперимента) статистическая разница.

# Перед тестированием соберем таблицу, в которую должны войти все эвенты, и разбивка на категории, для начала категориями будут выступать группы, отношение групповых пользователей к общей сумме пользователей на этапе, т.е. все ранее проводимые расчеты соберем в единую таблицу, для проведения тестов.

# In[55]:



table_for_test = data_test.pivot_table (values='user_id',
                                   columns='group',
                                    index='event_name',
                                   aggfunc='nunique').sort_values (by='A1', ascending=False).reset_index(drop=False)

table_for_test ['all_group'] = table_for_test ['A1']+ table_for_test ['A2']+ table_for_test ['B']
table_for_test ['unit_control_group'] = table_for_test ['A1']+ table_for_test ['A2']
table_for_test 


# добавим таблицу в которую войдут все уникальные клиенты по группам. 
# 
# **Вопрос для проверяющего** Нужен совет. При анализе выявил 110 уникальных клиентов, которые не проходили этап главного экрана, по логике понимаю, что все клиенты его проходят, предположил, что возможны основных 2 варианты: люди ранее были на главном экране, и потом просто спустя какое-то время вернулись к сайту и покупками через закладку отличную от главного экрана, и поэтому их нет в выборке главного экрана, и второй вариант тех.сбой при формировании логов, гипотезу, что они появились из-за моего среза дозагрузки, отверг, там удалял по пользователю, а не по эвентам, и они бы не появилась. Так вот, если я чищу от них ДС, то тесты объединенной группы (А1+А2) показывают все время просто р-значение=0, когда вернулся к датасету без чистки этих 110 пользователей, все работает. Немного запутался, логика в голове говорит - удаляй, логика тестов указывает на некорректность результатов. Отмечу, что странный результат только в тесте А1+А2/В

# In[56]:


group_test_group = data.groupby ('group') ['user_id'].nunique()
group_test_group ['unit_control_group'] = group_test_group ['A1'] + group_test_group ['A2']
group_test_group


# ### Проведение статистических тестов

# Нам предстоит сделать несколько однотипных тестов, поэтому для выдержки концепции DRY стоит написать функцию для расчетов.
# 
# Скелет теста у нас имеется из теории, нужно только добавить цикл, который будет проходить по строчкам таблицы table_for_test и выполнять заложенный тест. изначальный уровень альфа в задании выставлен в 10%, что конечно большой уровень допущения, так как потенциально каждый 10-ый тест будет с ложным результатом 

# In[57]:


def AB_test_base_without_holm(group1, group2, alpha=0.1):
    
    try:
        p_values = []
        for index in table_for_test.index:
            p1 = table_for_test[group1][index] / group_test_group[group1]
            p2 = table_for_test[group2][index] / group_test_group[group2]

            p_combined = (table_for_test[group1][index] + table_for_test[group2][index]) / (
                    group_test_group[group1] + group_test_group[group2])
            difference = p1 - p2
            z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (
                        1 / group_test_group[group1] + 1 / group_test_group[group2]))
            distr = st.norm(0, 1)
            p_value = (1 - distr.cdf(abs(z_value))) * 2

            step = table_for_test['event_name'][index]
            print(f'Этап - {step}')
            print(f'p-значение: {p_value}')

            if p_value < alpha:
                print('До приминения поправки Холма. Отвергаем нулевую гипотезу: между долями есть значимая разница')
            else:
                print('До приминения поправки Холма. Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными')

            
            p_values.append(p_value)

            print('--------------------------------')

        # вычисление поправки Холма
        reject, p_corrected, a1, a2 = smm.multipletests(p_values, alpha=alpha, method='holm')

        print(f'p-значения: {p_values}')
        print(f'p-значения после поправки Холма: {p_corrected}')
        print(f'Отвергаем нулевую гипотезу для p-значений {a1}')
        print(f'Не отвергаем нулевую гипотезу для p-значений {a2}')
        
    except Exception as e:
        print(f'ERROR: {e}')


# In[58]:


def AB_test_base_with_holm(group1, group2, alpha):
    
    try:
        
        for index in table_for_test.index:
            p1 = table_for_test[group1][index] / group_test_group[group1]
            p2 = table_for_test[group2][index] / group_test_group[group2]

            p_combined = (table_for_test[group1][index] + table_for_test[group2][index]) / (
                    group_test_group[group1] + group_test_group[group2])
            difference = p1 - p2
            z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (
                        1 / group_test_group[group1] + 1 / group_test_group[group2]))
            distr = st.norm(0, 1)
            p_value = (1 - distr.cdf(abs(z_value))) * 2

            step = table_for_test['event_name'][index]
            print(f'Этап - {step}')
            print(f'p-значение: {p_value}')

            if p_value < alpha:
                print('После приминения поправки Холма. Отвергаем нулевую гипотезу: между долями есть значимая разница')
            else:
                print('После приминения поправки Холма. Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными')

            
            print('--------------------------------')

        
    except Exception as e:
        print(f'ERROR: {e}')


# #### Тест А1/А2
# 
# **Гипотеза H0**
#     - статистической разницы между группами А1 и А2 нет
# 
# **Гипотеза H1**
#     - есть статистическая разница между группами А1 и А2
#     
# Для итогового тестирования, выбран уровень значимости 0,01, так как мы предполагаем что А1 и А2 равны, обе являются контрольными, и статистическая разница между ними должна быть крайне невысокой, а после расчета поправки Холма, запустим повторный тест с новым уровнем значимости

# In[59]:


AB_test_base_without_holm ('A1', 'A2', 0.01)


# **Из результатов можно сделать вывод, что гипотеза о равенстве долей в группах не отвергается для значений p-значения выше 0.002509430066318874 и принимается для значений p-значения ниже 0.0025.**

# In[60]:


AB_test_base_with_holm ('A1', 'A2', 0.0025)


# На основании результатов теста мы убедились, что контрольные группы с высокой долей вероятности не отличаются, разделения были сделаны корректно. Такм образом, мы проверили работоспособность системы тестирования и оценили уровнь случайных колебаний в данных. 
# 
# Вывод: мы убедились, что результаты тестирования действительно отражают реальное поведение пользователей, а не являются случайным шумом или ошибками в методике тестирования.

# #### Тест А1/B
# 
# **Гипотеза H0**
#     - статистической разницы между группами А1 и B нет
# 
# **Гипотеза H1**
#     - есть статистическая разница между группами А1 и B
#     
#     Для итогового тестирования, выбран уровень значимости 0,05, так как мы проводим тест контрольной группы и тестовой, уровень значимости 0,05 допустимое значение, а после расчета поправки Холма, запустим повторный тест с новым уровнем значимости

# In[61]:


AB_test_base_without_holm ('A1', 'B', 0.05)


# **Из результатов можно сделать вывод, что гипотеза о равенстве долей в группах не отвергается для значений p-значения выше 0.012741455098566168 и принимается для значений p-значения ниже 0.0125.**

# In[62]:


AB_test_base_with_holm ('A1', 'B', 0.0125)


# #### Тест А2/B
# 
# **Гипотеза H0**
#     - статистической разницы между группами А2 и B нет
# 
# **Гипотеза H1**
#     - есть статистическая разница между группами А2 и B
#     
#     Для итогового тестирования, выбран уровень значимости 0,05, так как мы проводим тест контрольной группы и тестовой, уровень значимости 0,05 допустимое значение, а после расчета поправки Холма, запустим повторный тест с новым уровнем значимости

# In[63]:


AB_test_base_without_holm ('A2', 'B', 0.05)


# **Из результатов можно сделать вывод, что гипотеза о равенстве долей в группах не отвергается для значений p-значения выше 0.012741455098566168 и принимается для значений p-значения ниже 0.0125.**
# Прим. расчет поправки Холма идентичен для А1/В и для А2/В

# In[64]:


AB_test_base_with_holm ('A2', 'B', 0.0125)


# #### Тест unit_control_group/B
# 
# **Гипотеза H0**
#     - статистической разницы между группами unit_control_group и B нет
# 
# **Гипотеза H1**
#     - есть статистическая разница между группами unit_control_group и B
#     
#     Для итогового тестирования, выбран уровень значимости 0,05, так как, несмотря на то, что используется в тесте объединенная контрольная группа, это не означает необходимость замены уровня значимости. 
#     проводим тест объединенной контрольной группы и тестовой, уровень значимости 0,05 допустимое значение, а после расчета поправки Холма, запустим повторный тест с новым уровнем значимости

# In[65]:


AB_test_base_without_holm ('unit_control_group', 'B', 0.05)


# **Из результатов можно сделать вывод, что гипотеза о равенстве долей в группах не отвергается для значений p-значения выше 0.012741455098566168 и принимается для значений p-значения ниже 0.0125.**
# Прим. аналогичная ситуация для расчета поправки Холма  с объединенной группой

# In[66]:


AB_test_base_with_holm ('unit_control_group', 'B', 0.0125)


# ## Итоговый вывод.

# ### По результатам исследования можем выделить несколько моментов:
# 
#     1.	В представленном датасете (далее – ДС) общий период составляет около 14 дней. Детальное изучение ДС показало, что за период с 25 июля по 31 августа включительно, данных практически нет. Однако, с 1 августа наблюдается многократный рост активности. В этой связи, проведен анализ совершенных эвентов в срезе пользователей, и выявлены 149 пользователей, которые имеют аномально высокую активность (более 136 совершенных действий за период с 1 по 8 августа). В этой связи, принято решение скорректировать в ДС 2 момента:
#     I.	Удалить гиперактивных пользователей.
#     II.	Скорректировать временной интервал анализа, а именно оставлен как корректный период с 1 по 8 августа.
# 
#     2.	В результате изучения воронок мы установили, что:
#     •	в логах встречается 5 типов событий (с частотой появления):
#         - Main Screen Appear - главный экран (109324, с удельной пропорцией 56,5%)
#         - Offers Screen Appear - экран предложений (37696, 19,5%)
#         - Cart Screen Appear - экран корзины (26539, 13,7%)
#         - Payment Screen Successful - экран успешной оплаты (19997, 10.3%)
#         - Tutorial - 808
# 
#     •	события по типам и расчетом уникальных пользователей на каждое событие:
#         - MainScreenAppear - 7292
#         - OffersScreenAppear - 4466
#         - CartScreenAppear - 3588
#         - PaymentScreenSuccessful - 3395
# 
#     •	Доля уникальных пользователей, которые хоть раз совершали любое событие отличное от просмотра главного экрана -  63,14%, дальше главного экрана не ушли 38,86% (на первом шаге 7292, на втором только 4466) пользователей - довольно большая цифра, стоит взять на заметку, возможно имеет место техническая проблема.
# 
#     •	логически события идут в следующем порядке: MainScreenAppear - OffersScreenAppear - CartScreenAppear - PaymentScreenSuccessful. Просмотр Tutorial необязательный этап, его мы исключили при разборе.
# 
#     •	Ранее мы выяснили, что среди уникальных пользователей дальше главного экрана не ушло 38,7% пользователей, в общем отношении всех операций значения также требуют изучения за рамками настоящего анализа (видимо некоторые пользователи не столкнулись с потенциальной технической проблемой, которую мы предположили при расчете уникальных пользователей), и так:
#     o	MainScreenAppear - 100% (шаг 1)
#     o	на шаге OffersScreenAppear только 61,14% от тех, кто был на главной странице
#     o	от предложения до корзины CartScreenAppear доходит 80,7% из шага 2 (предложение)
#     o	от корзины до успешной оплаты PaymentScreenSuccessful 94,62% из шага 3 (корзина)
# 
#     •	Таким, образом, безотносительно оценки уникальных пользователей, явные проблемы с конверсией от главной страницы до просмотра предложения (вопрос для передачи коллегам так как, или у нас технические проблемы, или пользователям настолько неудобно пользоваться сайтом).
#     
#     3.	Проведены статистические тесты А1/А2-тест, А1/В-тест, А2/В-тест, А1+А2/В-тест.
#         a.	Всего проведено 3 A/B теста и 1 А/А тест, в каждом по 4 этапа - итого 16 проверок гипотез на данных без использования поправки Холма, и столько же тестов с использованием поправок Холма. При проведении тестов проводилась оценка по следующим гипотезам:
#             I.	Гипотеза H0 - статистической разницы между группами нет
#             II.	Гипотеза H1 - есть статистическая разница между группами 
# 
#     На первом этапе применялись в качестве уровня значимости указанные, в результатах без поправки Холма, значения.
#     
# **Результаты тестов до приминения поправки Холма:**
#         
#         -  А1/А2-тест, уровень значимости 0,01:
#                1. Этап 'MainScreenAppear'. 'p-значение:0.5904565100159676'
#                'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'               
#                2. Этап 'OffersScreenAppear'. 'p-значение:0.36114194983637615'
#                'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'               
#                3. Этап 'CartScreenAppear'. 'p-значение:0.23446149906995117'
#                'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'               
#                4. Этап 'PaymentScreenSuccessful'.'p-значение:0.13334226878893363'
#                'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
# 
#         - А1/В-тест, уровень значимости 0,05:
#             1. 'Этап - MainScreenAppear'.'p-значение:0.32579209352565'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'            
#             2.'Этап - OffersScreenAppear'. 'p-значение:0.21848106975343873'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'            
#             3. 'Этап - CartScreenAppear'.'p-значение:0.08725015373223144'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'            
#             4. 'Этап - PaymentScreenSuccessful'.'p-значение:0.22040644524309005'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             
#         - А2/В-тест, уровень значимости 0,05:
#             1. 'Этап - MainScreenAppear'.'p-значение:0.6544821577013522'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             2. 'Этап - OffersScreenAppear'.'p-значение:0.7502650724782389'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             3.'Этап - CartScreenAppear'.'p-значение:0.6011268322884902'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             4.'Этап - PaymentScreenSuccessful'.'p-значение:0.7813716681767917'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             
#         - А1+А2/В-тест, уровень значимости 0,05:
#             1.'Этап - MainScreenAppear'.'p-значение:0.40587169814078106'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             2.'Этап - OffersScreenAppear'.'p-значение:0.3725288984802724'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             3.'Этап - CartScreenAppear'.'p-значение:0.19844602291192937'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             4.'Этап - PaymentScreenSuccessful'.'p-значение:0.5870703788523293'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
# 
# 
# **Результаты тестов после приминения поправки Холма:**
# 
# Значения уровня значимости, после расчета поправки Холма, существенно скорректировались в меньшую сторону, что уменьшает вероятность ложнопозетивного результата.
#         
#         -  А1/А2-тест, уровень значимости 0,0025:
#                1. Этап 'MainScreenAppear'. 'p-значение:0.764988759488143'
#                'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'               
#                2. Этап 'OffersScreenAppear'. 'p-значение:0.2794373773714116'
#                'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'               
#                3. Этап 'CartScreenAppear'. 'p-значение:0.26909398870001455'
#                'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'               
#                4. Этап 'PaymentScreenSuccessful'.'p-значение:0.13975791207596155'
#                'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
# 
# 
#         - А1/В-тест, уровень значимости 0,0125:
#             1. 'Этап - MainScreenAppear'.'p-значение:0.2369891935845656'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'            
#             2.'Этап - OffersScreenAppear'. 'p-значение:0.17797143659165315'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'            
#             3. 'Этап - CartScreenAppear'.'p-значение:0.06394416898469357'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'            
#             4. 'Этап - PaymentScreenSuccessful'.'p-значение:0.19927895265165052'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             
#   
#         - А2/В-тест, уровень значимости 0,0125:
#             1. 'Этап - MainScreenAppear'.'p-значение:0.37400643924274246'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             2. 'Этап - OffersScreenAppear'.'p-значение:0.790405486710803'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             3.'Этап - CartScreenAppear'.'p-значение:0.45343385549545934'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             4.'Этап - PaymentScreenSuccessful'.'p-значение:0.8454610539792848'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#      
#      
#         - А1+А2/В-тест, уровень значимости 0,0125:
#             1.'Этап - MainScreenAppear'.'p-значение:0.22315599235522665'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             2.'Этап - OffersScreenAppear'.'p-значение:0.3533640949113348'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             3.'Этап - CartScreenAppear'.'p-значение:0.13364566367079211'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             4.'Этап - PaymentScreenSuccessful'.'p-значение:0.5321457255242299'
#             'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
#             
# 
# 
# Таким образом, по результатам вышеописанного исследования значимой разницы между контрольными группами и тестовой не выявлено, корректировка шрифта существенного эффекта не оказала. Опасения менеджеров не подтвердились, тестирование предлагается признать успешным.
# 
