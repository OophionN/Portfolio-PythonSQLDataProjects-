#!/usr/bin/env python
# coding: utf-8

# # Проект анализа продаж в интернет-магазине "Стримчик" 

# # Общее описание проекта
# 
# Перед вами данные до 2016 года. Представим, что сейчас декабрь 2016 г., и вы планируете кампанию на 2017-й. Нужно отработать принцип работы с данными. Неважно, прогнозируете ли вы продажи на 2017 год по данным 2016-го или же 2027-й — по данным 2026 года.
# 
# Описание данных
# 
# Name — название игры
# 
# Platform — платформа
# 
# Year_of_Release — год выпуска
# 
# Genre — жанр игры
# 
# NA_sales — продажи в Северной Америке (миллионы проданных копий)
# 
# EU_sales — продажи в Европе (миллионы проданных копий)
# 
# JP_sales — продажи в Японии (миллионы проданных копий)
# 
# Other_sales — продажи в других странах (миллионы проданных копий)
# 
# Critic_Score — оценка критиков (максимум 100)
# 
# User_Score — оценка пользователей (максимум 10)
# 
# Rating — рейтинг от организации ESRB (англ. Entertainment Software Rating Board). Эта ассоциация определяет рейтинг компьютерных игр и присваивает им подходящую возрастную категорию.
# 
# Данные за 2016 год могут быть неполными.

# ## Подготовка к анализу, загрузка библиотек

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
get_ipython().system('pip install tabulate')
from tabulate import tabulate
from scipy import stats as st


# ### Знакомство в сетом

# Общий размер сета 16714 строк, имеются пропуски в разделах: Год релиза, жанр, название, оценки критиков и игроков, а также в разделе рейтинга

# вне проекта, оставил ссылку чтобы не переписывать каждый раз, работаю с проектом локально
# 
# ('/datasets/games.csv', sep=',')
# 
# (r'C:\Users\PC_Maks\Desktop\study\prefabricated_project\games.csv', sep=',')

# In[2]:


try:
    data = pd.read_csv ('/datasets/games.csv', sep=',')
except:
    data = pd.read_csv ('https://code.s3.yandex.net/datasets/games.csv', sep=',')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
display (data.shape)
display (data.head ())
display (data.info ())


# ##  Подготовка данных
# 
# 1. Замена названия столбцов (приведите к нижнему регистру);
# 2. Преобразование данных в нужные типы, с описанием в каких столбцах провели замену и почему;
# 3. Обработка пропусков (при необходимости):
#     3.1. Объяснить, почему заполнили пропуски определённым образом или почему не стали это делать;
#     3.2. Описать причины, которые могли привести к пропускам;
# 4. Изучить аббревиатуру 'tbd' в столбце с оценкой пользователей. Отдельно разобрать это значение и описать, как его обработать;
# 5. Посчитать суммарные продажи во всех регионах и записать их в отдельный столбец.

# **Сразу можем сказать, что потребуется замена типов данных в столбце 'Year_of_Release' на int, а также в столбце 'User_Score' на float64 

# ### Замена названия столбцов**

# In[3]:


data.columns = [x.lower() for x in data.columns]


# #### Проверка на дубликаты
# Сразу проверим на дубликаты - дубликат только 1 - Madden NFL 13, было еще переиздание NFS, его оставляем. 

# In[4]:


print (f' Явные дублика во всех строках и столбцах {data.duplicated ().sum ()}')

duplicates = data.duplicated(subset={'name', 'platform', 'year_of_release'})
duplicate_rows = data.loc[duplicates]
display (duplicate_rows.info())
display (duplicate_rows)


# In[5]:


data.drop_duplicates(subset=['name', 'platform', 'year_of_release'], inplace=True)


# проверим что получилось

# In[6]:


duplicates = data.duplicated(subset={'name', 'platform'})
duplicate_rows = data.loc[duplicates]
print(duplicate_rows.info())
print(duplicate_rows)


# ### Обработка типов данных, пропусков
# Преобразование данных в нужные типы, с описанием в каких столбцах провели замену и почему
# 
# Перед заменой типов данных посмотрим на содержание 

# In[7]:


display (data.isna ().sum ()/ len(data)* 100)
display (data.isna ().sum ())


# Пропуски в разделе года выпуска - 1,6%, 2 пропуска в названии и жанре - 0,01%, и более 50%/40% пропуски в разделах рейтингов (критики/пользователи) и возрастного рейтинга. Будем разбираться по порядку. 

# In[8]:


def pass_value_barh_1(df):
    try:
        (
            (df.isna().mean()*100)
            .to_frame()
            .rename(columns = {0:'space'})
            .query('space > 0')
            .sort_values(by = 'space', ascending = True)
            .plot(kind = 'barh', figsize = (19,6), legend = False, fontsize = 14, grid=True)
            .set_title('Пример' + "\n", fontsize = 24, color = 'SteelBlue')
            
        );
        plt.xlabel('% пропусков',  fontsize = 18)
        plt.ylabel('Столбец с пропусками',  fontsize = 18)
        
    except:
        print('пропусков не осталось :) или произошла ошибка в первой части функции ')


# In[9]:


pass_value_barh_1(data)


# #### *Пропуски в названиях игр*
# 
# тип данных корректен, можем приступить к отработки пропусков 

# In[10]:


display (data ['name'].unique ())
df1 = data [data ['name'].isna () == True]
df1


# Пропуск в разделе название только 2 позициях, игры 1993 года, для платформы Gen, при это обращаем внимание, что пропуск не только в названиях, но и в категории жанр. Учитывая незначительное количество (менее 0.01 процента) - удаляем. И сразу проверим удалились ли пропуски в столбце названия игр и жанр

# In[11]:


data.head()


# In[12]:


#data = data.dropna (subset = 'name').reset_index (drop=True)
data = data.dropna (subset=['name']).reset_index (drop=True)
data.isna ().sum ()


# #### Обработка типов данных (год)
# Столбец - год выпуска, обработка типа данных и пропусков
# 
# Тип данных необходимо заменить на int, но предварительно отработаем пропуск. Учитывая текущие этап анализа, пропуски данного столбца заменяем на заглушку 1970 (игры из списка начинаются с 1980)

# In[13]:


data ['year_of_release'] = data ['year_of_release'].fillna (1970).astype (int)


# *Посмотрим на игры у которых пропущен год выпуска*

# In[14]:


df_year = data.query ('year_of_release == 1970').reset_index (drop=True)
display (df_year['platform'].value_counts ())


# *Теоритечески можно найти даты выходов каждой из игра, или взять дату экватора периода жизни консоли, но это трудоемкая процедура, и учитывая количество пропусков около 1,6%  - прихожу к выводу, что допустимо их удалить*

# In[15]:


data = data.query ('year_of_release != 1970'). reset_index (drop=True)
data.isna ().sum ()


# #### Анализ пропусков (рейтингы и оценки)
# Переходим к анализу и обработке пропусков в разделах рейтинга критиков, пользователей и возрастного рейтинга

# Для следующих шагов мне потребуется добавить столбец - общее количество продаж. Пропусков в продажах по регионам у нас нет, поэтому можем споконой посчитать значения для нового столбца

# In[16]:


data ['total_sales'] = data['na_sales'] + data['eu_sales']+data['jp_sales']+data ['other_sales']   
data.head ()


# ##### Пропуски в оценках критиков 
# *Сначала посмотри на отсутствующие значения в столбце рейтинг критиков*
# 
# оценка критиков в диапазоне от 0 до 100, значения вещественные

# In[17]:


display (data ['critic_score'].unique ())


# Посмтрим на различные оценки критиков в зависимости от жанров и платформ, пропуски пока уберем

# In[18]:


platform_critic=data.pivot_table(index='platform',columns='genre',values='critic_score')
platform_critic=platform_critic.dropna ()
platform_critic


# In[19]:


plt.figure(figsize=(10,7))
sns.heatmap(platform_critic)


# Видимо Wii не очень получается делать адвенчуры и симуляторы. А из данных можно сделать вывод, что оценки критиков более менее равноеро распределеяются по жанрам. Можем заменить пропуски средними значениями по жанру (разбросы не большие, диапазон фиксированный) 

# In[20]:


data ['critic_score'] = data ['critic_score'].fillna (111)


# ##### *Пропуски в оценках игроков*
# 
# оценка игроков диапазоне от 0 до 10, значения вещественные. Повторим процедуру анализа как и с оценкой критиков, но сначала разберемся с некоторыми значениями

# In[21]:


display (data ['user_score'].unique ())


# В данном столбце встречается значение TBD - аббревиатура от английского To Be Determined (будет определено)
# или To Be Decided (будет решено)
# С высокой долей вероятсности можно предположить, что отзывов пока нет, либо их количество крайне мало.
# 
# Для дальнейшей работы заменим "tbd" на nan

# In[22]:


data['user_score'] = data['user_score'].replace('tbd', np.NaN).astype (float) 
display (data ['user_score'].unique ())
data['user_score'].isna ().sum ()


# сгруппируем рейтинг пользователей, создадим таблицу со средними значениями

# используем заглушку и проверим нашу теорию с жанрами, годами - результат проверки - подавляющее большенство игр выпущены до 2000 годов, только 102 игры за период с 2000 по 2001 год остались без рейтинга пользоватлей после предварительной обработки данных - учитывая результаты пологаю можно заполнить пропуски средним значением

# In[23]:


data['user_score'].unique()
data['user_score'] = data['user_score'].fillna(111)
df_with_nan_years =data.query ('user_score==111').sort_values (by='year_of_release') 
display (df_with_nan_years)
display (df_with_nan_years ['year_of_release'].value_counts ())
df_with_nan_years ['platform'].value_counts ()


# In[24]:


#df_without_111 = data.query ('user_score!=111')
#user_score_mean_without_111 = df_without_111 ['user_score'].mean ()
#data['user_score'] = data['user_score'].replace(111,user_score_mean_without_111)

data['user_score'].isna ().sum ()


# In[25]:


# check
data['user_score'].value_counts()


# ##### *Пропуски возрастного рейтинга**
# 
# Тип корректен, заменим пропуски на undefined, так как вероятно оценка рейтинга не давалась, а также рейтинг K-A ("Kids to Adults") заменим на рейтинг E («Everyone» — «Для всех»), ранее рейтинг E обозначался как K-A

# In[26]:


data ['rating'] = data ['rating'].fillna ('undefined')
data['rating'] = data['rating'].replace('K-A','E')
display (data ['rating'].unique ())


# In[27]:


# check
data ['critic_score'] = data ['critic_score'].astype (int)
data.info()


# **Первичная обработка завершена. Замену пропусков, где поставили пока заглушки считаю нецелесообразной, так как это сильно исказит картину, при дальнейшем анализе примем решение оставить их как есть с заглушками и будем отсекать при исследовании, или найдем решение по их замещении какими-либо значениями**

# ### check

# ## Исследовательский анализ данных**
# 
# **3.1.** Посмотрите, сколько игр выпускалось в разные годы. Важны ли данные за все периоды?
# 
# **3.2.** Посмотрите, как менялись продажи по платформам. Выберите платформы с наибольшими суммарными продажами и постройте распределение по годам. За какой характерный срок появляются новые и исчезают старые платформы?
# 
# **3.3.** Возьмите данные за соответствующий актуальный период. Актуальный период определите самостоятельно в результате исследования предыдущих вопросов. Основной фактор — эти данные помогут построить прогноз на 2017 год.
# Не учитывайте в работе данные за предыдущие годы.
# 
# **3.4.** Какие платформы лидируют по продажам, растут или падают? Выберите несколько потенциально прибыльных платформ.
# 
# **3.5.** Постройте график «ящик с усами» по глобальным продажам игр в разбивке по платформам. Опишите результат.
# 
# **3.6.** Посмотрите, как влияют на продажи внутри одной популярной платформы отзывы пользователей и критиков. Постройте диаграмму рассеяния и посчитайте корреляцию между отзывами и продажами. Сформулируйте выводы.
# 
# **3.7.** Соотнесите выводы с продажами игр на других платформах.
# 
# **3.8.** Посмотрите на общее распределение игр по жанрам. Что можно сказать о самых прибыльных жанрах? Выделяются ли жанры с высокими и низкими продажами?

# **----------------------------------------------------------------------**
# 
# Основная часть игр вышла в период с 2003 по 2010 гг.

# Для удобства сделаем функцию, которая будет выводить результаты в виде графика, так как мы будем обращаться несколько раз к одному и тому же типу расчету по разным столбцам

# In[28]:


def stat_bar(column_group, task, name='name'):
    try:
        plt.style.use('seaborn-muted')
        draw_plot = data.groupby(column_group)[name]        
        if task == 'count':
            draw_plot_calculated = draw_plot.count()
            plot = draw_plot_calculated.plot(kind='bar', y=name, grid=True, figsize=(10,5))
            plt.title('Количество выпускаемых игр по годам', fontsize=16, fontweight='bold' )
            plt.xlabel('Годы', fontsize=18)
            plt.ylabel('Количество игр', fontsize=18)
            plt.show ()
            plt.rcParams['font.family'] = 'Arial'
            plt.show()
            
        elif task == 'sum':
            draw_plot_calculated = draw_plot.sum().sort_values()
            plot = draw_plot_calculated.plot(kind='barh', y=name, grid=True, figsize=(10,10))
            plt.title('Количество выпускаемых игр в срезе платформ', fontsize=16, fontweight='bold' )
            plt.xlabel('Количество выпускаемых игр', fontsize=18)
            plt.ylabel('Платформа', fontsize=18)
            plt.show ()
            plt.rcParams['font.family'] = 'Arial'
            plt.show()

       
    except:
        return 'Проверь данные'
                


# ###  Анализ выхода игр
# *Посмотрите, сколько игр выпускалось в разные годы. Важны ли данные за все периоды?*
# 
# На графике мы видим, что начиная с 1994 года начинается рост количества выпускаемых игр, что можно объяснить развитием технологий и в целом индустрии видеоигр (прим. наращиваются мощности консолей, развивается рынок домашних ПК). 
# 
# **Вывод:** сведения за периоды до 1994 года, можно отбросить при исследовании, но сделаем это после исследования, так как какая-то из консолей могла быть выпущена в 1993 году и существать долгое время. Важно понимать тенденцию развития консолей, те приставки которые активно продавались в 90/00-х на 2016 год в общем и целом потеряли свою актуальность к 2016. Учитывая это пологаю, что для анализа на 2017 год, будет корректнее взять за основу последний период в 10 лет

# In[29]:


stat_bar('year_of_release', 'count')


# Уберем сразу период до 2001

# In[30]:


data = data.query ('year_of_release>=2001')
data.info ()


# ### Уровень продаж платформ
# *Посмотрим на общий уровень продаж платформ.*

# In[31]:


stat_bar('platform', 'sum', 'total_sales')


# **Вывод** 
# 
# Можем отметить, что на рынке, по состянию на 2016 год, присуствуют явные лидеры. Но важно помнить, что этот показатель за весь период жизни консоли, и те платформы, которые вышли на рынок не так давно, еще не нарастили обороты. Прим. консоль GBA в расчет не берем, так как на момент выгрузке приставке 11 лет, с момента последней модификации прошло более 5 лет, 32-х разрядная приставка морально и технически устарела. PS4 хоть и по продажам почти равна гембою, но она только поступила в продажи

# ### Срок жизни платформ
# *За какой характерный срок появляются новые и исчезают старые платформы?*

# *соберем лидеров в отдельный список*

# In[32]:


platforms_leaders =data.groupby('platform')['total_sales'].sum().sort_values()[-15:].index.tolist()
platforms_leaders


# *создадим сет только с лидерами*

# In[33]:


data_leaders = data.query ('platform== @platforms_leaders')
data_leaders.head () 


# Построим графики продаж по годам выбранных лидеров, для начала соберем наших лидеров в отдельную таблицу

# In[34]:


platform_leaders=data_leaders.pivot_table(index='platform',values='total_sales', aggfunc='sum')
platform_leaders.sort_values(by='total_sales',ascending=False).plot(kind='bar',figsize=(10,7), grid=True)

plt.title('Общие продажи', fontsize=16, fontweight='bold')
plt.xlabel('Платформы', fontsize=14)
plt.ylabel('Количество продаж', fontsize=14)
plt.rcParams['font.family'] = 'Arial'
plt.show()


# Теперь сведем даные по продажам 

# In[35]:


platform_leaders_years=data_leaders.pivot_table(index="platform",columns='year_of_release',values='total_sales',aggfunc=('sum'))
plt.figure(figsize=(12,8))
platform_leaders_years.plot(kind='barh', cmap='coolwarm')
plt.grid(axis='x')
plt.title('Общие продажи', fontsize=16)
plt.xlabel('количество продаж', fontsize=14)
plt.ylabel('платформы', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# Как видно визуализация не очень удачная, посмтроить тепловую карту

# In[36]:


plt.figure(figsize=(9,9))
sns.heatmap(platform_leaders_years, cmap='Spectral')


# Такой график становится более понятным. Можем утверждать что в компания, которые выпускают несколько поколений консолей, наблюдается тенденция к циклу в 10 лет жизни каждого поколения устройств, особено хорошо видно это на приставках от компании сони (PS-PS2_PS3-PS4) и майкрософт (XB-XB360-XBOne), из этого описания выбивается PC. В основном стартуют все платформы хорошо, с высокими продажами, к середине цикла прадажи наращиваются. Из этого описания выбивает приставка Wii, которая стартовала очень хорошо, и стала резко терять популярность в 2011-2012 гг. Уже сейчас на текущих данных, можно сделать обоснованное предположение на 2017 год, что продажи будут только расти ближайшие годы для недавно вышедших платформ - PS4 и XBOne

# 

# ### Уровень продаж и срок жизни платформ срок 3 года 
# *Проведем аналогичное исследование но на более короткой дистанции - 3 года. (замена 5 на 3 года)*

# In[37]:


data_3 = data.query ('year_of_release>=2014')
platforms_leaders_3 =data_3.groupby('platform')['total_sales'].sum().sort_values()[-7:].index.tolist()
data_leaders_3 = data_3.query ('platform== @platforms_leaders_3')


# In[38]:


platform_leaders_3 = data_leaders_3.pivot_table(index='platform', values='total_sales', aggfunc='sum')
platform_leaders_3.sort_values(by='total_sales', ascending=False).plot(kind='bar', figsize=(10, 7), grid=True)

plt.title('Общие продажи', fontsize=16, fontweight='bold')
plt.xlabel('Платформы', fontsize=14)
plt.ylabel('Количество продаж', fontsize=14)
plt.rcParams['font.family'] = 'Arial'
plt.show()


# In[39]:


platform_leaders_years_3=data_leaders_3.pivot_table(index="platform",columns='year_of_release',values='total_sales',                                                    aggfunc=('sum'))


# In[40]:



plt.figure(figsize=(12,8))
platform_leaders_years_3.plot(kind='barh', cmap='coolwarm')
plt.grid(axis='x')
plt.title('Общие продажи', fontsize=16)
plt.xlabel('количество продаж', fontsize=14)
plt.ylabel('платформы', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[41]:



plt.figure(figsize=(12,10))
sns.heatmap(platform_leaders_years_3, annot=True, cmap='coolwarm')
plt.title('Общие продажи', fontsize=16)
plt.xlabel('Платформы', fontsize=14)
plt.ylabel('Количество продаж', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# **Вывод**
# 
# В общем и целом картина не поменялась, лидеры консоли от майкрософт и сони. Сони выглядит значительно преспективнее. Но про приставки от компании Nindendo, не стоит забывать

# ### "Коробка с усами" 
# Построим график «ящик с усами» по глобальным продажам игр в разбивке по платформам. Опишем результат.
# 
# Для начала соберем сводной таблицей все продажи по платформам

# In[42]:


platform_sales_3=data_leaders_3.pivot_table(index="platform",values='total_sales',aggfunc=('sum')).reset_index ()
platform_sales_3


# *построим графики*

# In[43]:


plt.figure(figsize=(15,10))
plt.ylim (0,2.5)
plt.grid(axis='y', alpha=1)

sns.boxplot(x='platform', y='total_sales', data=data_leaders_3, palette='Spectral')
plt.title('Диаграмма размаха. Продажи с 2001 г.', fontsize=16)
plt.xlabel('Платформы', fontsize=14)
plt.ylabel('Количество продаж', fontsize=14)
plt.show()


# **Вывод**
# 
# На графике можем обратить внимание, что платформы последних покалений от сони и майкрософт (PS3/PS4 и XB360/XBOne), имеют практически одинаковые медианы и квартили, с достаточно выраженной скошенностью вправо. Сначала предположил, что консоли одного поколения, безотносительно производителя, имеюет схожие разбросы, проверил принадлежность к тому или иному покалению консоли из списка - гипотеза не подтвердилась, просто визуальное совпадение. Вероятно, схожая картина в части общего спроса, но хотя гранийы "коробки" почти совпадают, у них разнятся скосы медиан. 
# Война консолей в графике))))

# In[44]:


plt.figure(figsize=(15,10))
plt.ylim (0,15)
plt.grid(axis='y', alpha=1)

sns.boxplot(x='platform', y='total_sales', data=data_leaders_3, palette='Spectral')
plt.title('Диаграмма размаха. Продажи с 2001 г.', fontsize=16)
plt.xlabel('Платформы', fontsize=14)
plt.ylabel('Количество продаж', fontsize=14)
plt.show()


# ### Расчет корреляции
# Посмотрим, как влияют на продажи внутри одной популярной платформы отзывы пользователей и критиков. Построим диаграмму рассеяния и посчитаем корреляцию между отзывами и продажами. 
# За популярную консоль возьмем PS3 (я за сони))) )
# 
# **Прим** Для отработки отсеем наши заглушки по категориям
# 
# #### Расчет для одной платформы 
# Для начала сделаем выборку по платформе

# In[45]:


ps3_data_3_crit = data_leaders_3.query ('platform=="PS3" and critic_score<=100')
ps3_data_3_user = data_leaders_3.query ('platform=="PS3" & user_score<=10')
display (ps3_data_3_crit.describe ())
ps3_data_3_user.describe ()


# In[46]:


display (ps3_data_3_crit ['total_sales'].corr(ps3_data_3_crit['critic_score']))
display (ps3_data_3_user ['total_sales'].corr(ps3_data_3_user['user_score']))


# Коррелиция низкая, отзывы критиков чуть лучше коррелируются с продажами, а вот отзывы игроков, почти не связаны с продажами. 

# In[47]:


plt.figure(figsize=(7, 5))
plt.ylim (0,5)
plt.xlim (0,100)
plt.scatter(x=ps3_data_3_crit['critic_score'], y=ps3_data_3_crit['total_sales'], c='orange', alpha=0.7)
plt.grid(True)
plt.xlabel('Отзывы критиков')
plt.ylabel('Сумма продаж')
plt.title('Диаграмма разброса для игр на PS3')
plt.legend(['Игры на PS3'])
plt.show()


# In[48]:


plt.figure(figsize=(7, 5))
plt.ylim (0,4)
plt.scatter(x=ps3_data_3_user['user_score'], y=ps3_data_3_user['total_sales'], c='orange', alpha=0.7)
plt.grid(True)
plt.xlabel('Отзывы игроков')
plt.ylabel('Сумма продаж')
plt.title('Диаграмма разброса для игр на PS3')
plt.legend(['Игры на PS3'])
plt.show()


# #### Корреляция для лидеров рынка
# Посмотрим на продажи игр на других платформах
# 
# у нас уже есть выборка лидеров продаж data_leaders (дс с лидерами) и platforms_leaders (список лидеров)

# In[49]:


platforms_leaders_3


# In[50]:


for index in platforms_leaders_3:
    df= data_leaders_3.query (f'platform == "{index}" and critic_score<=100')
    correlation = df['total_sales'].corr(df['critic_score'])
    display ( f'Корреляция {index} : {correlation:.4f}')
    df.plot(y='total_sales', x='critic_score', kind='scatter', grid=True, figsize=(3,3))
    plt.grid(True)
    plt.title(index)
    plt.ylabel('Сумма продаж')
    plt.xlabel('Отзывы критиков')
    plt.show()
    
    


# *Сделаем тоже самое для оценки игроков*

# In[51]:


for index in platforms_leaders_3:
    df = data_leaders_3.query (f'platform == "{index}" and user_score<=10')
    correlation = df['total_sales'].corr(df['user_score'])
    display ( f'Корреляция {index} : {correlation:.4f}')
    df.plot(y='total_sales', x='user_score', kind='scatter', grid=True, figsize=(3,3))
    plt.grid(True)
    plt.title(index)
    plt.ylabel('Сумма продаж')
    plt.xlabel('Отзывы игроков')
    plt.show()


# В общем и целом картина схожа с той что получили с PS3, однако есть и примеры отрицательной коррелиции с оценками пользователей у пользоватлей ПК, PS4, XBOne.
# Это может быть вызвано несколькими причинами. Во-первых, оценки игроков могут быть неоднородными и не всегда соответствовать реальному качеству игры, так как они могут быть субъективными и зависеть от индивидуальных предпочтений каждого игрока. Во-вторых, игры с высокими оценками игроков не обязательно являются коммерческими успехами, так как продажи могут зависеть от других факторов, таких как маркетинг, реклама, доступность игры, цена и т.д.
# 
# Также следует отметить, что корреляция не обязательно означает причинно-следственную связь между оценками игроков и продажами игр. Это может быть просто случайность или результат влияния других факторов на продажи игр.

# ### Оценка жанров игр по продажам
# Посмотрим на общее распределение игр по жанрам. Попробуем ответить на вопрос - Что можно сказать о самых прибыльных жанрах? Выделяются ли жанры с высокими и низкими продажами?

# #### Самые продаваемые: Экшен, спорт, шутеры, хуже всего - стратегии и пазл. З.Ы. явно не мой личный рейтинг

# In[52]:


data_genre = data_3.pivot_table (index='genre', values='total_sales', aggfunc ='count').sort_values (by='total_sales',                                                                                                 ascending=False)
plt.figure(figsize=(12,8))
data_genre.plot(kind='barh', cmap='coolwarm')
plt.grid(axis='x')
plt.title('Игры по жанрам', fontsize=16)
plt.xlabel('Количество выпущенных игр', fontsize=14)
plt.ylabel('Жанры', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[53]:


plt.figure(figsize=(15,10))
plt.ylim (0,4)
plt.grid(axis='y', alpha=1)

sns.boxplot(x='genre', y='total_sales', data=data_3, palette='Spectral')
plt.title('Диаграмма размаха. Продажи с 2014 г.', fontsize=16)
plt.xlabel('Жанр', fontsize=14)
plt.ylabel('Количество продаж', fontsize=14)
plt.show()


# In[54]:


plt.figure(figsize=(15,10))
plt.ylim (0,15)
plt.grid(axis='y', alpha=1)

sns.boxplot(x='genre', y='total_sales', data=data_3, palette='Spectral')
plt.title('Диаграмма размаха. Продажи с 2014 г.', fontsize=16)
plt.xlabel('Жанр', fontsize=14)
plt.ylabel('Количество продаж', fontsize=14)
plt.show()


# ## Анализ рынков регионов и пользователей
# ### Составьте портрет пользователя каждого региона**
# 
# Определите для пользователя каждого региона (NA, EU, JP):
# 
#     1.1. Самые популярные платформы (топ-5). Опишите различия в долях продаж.
#     1.2. Самые популярные жанры (топ-5). Поясните разницу.
#     1.3. Влияет ли рейтинг ESRB на продажи в отдельном регионе?

# In[55]:



data_3


# ####  Топ-5 платформ по регионам
# Начнем с платформ. Сгруппируем дс по платформам и продажам в каждом из регионов, выберем 5 лидеров для каждого региона

# Нам предстоят однотипные вычисления, сделаем для них функцию

# In[56]:


def region_sales(data, task, n=5):
    region = ['na', 'eu', 'jp']
    sales_column = ['na_sales','eu_sales', 'jp_sales']
    fig, axs = plt.subplots(nrows=1, ncols=len(sales_column), figsize=(15,5))    
    
    if task==1:
        
        
        for i, r in enumerate(region):
            for j, s in enumerate(sales_column):
                if r in s:
                    region_sales = data.pivot_table(index='platform', values=s, aggfunc='sum').sort_values(by=s)[-n:]
                    axs[j].barh(region_sales.index, region_sales[s])
                    axs[j].set_xlabel('Количество продаж в {}'.format(r))
                    axs[j].set_ylabel('Платформы')
                    axs[j].set_title('Топ {} платформ по продажам в {}'.format(n, r))
        
    
   
        plt.tight_layout()
        plt.show()
        
        
    if task==2:
        for i, r in enumerate(region):
            for j, s in enumerate(sales_column):
                if r in s:
                    region_sales = data.pivot_table(index='genre', values=s, aggfunc='sum').sort_values(by=s)[-n:]
                    axs[j].barh(region_sales.index, region_sales[s])
                    axs[j].set_xlabel('Количество продаж в {}'.format(r))
                    axs[j].set_ylabel('Жанр')
                    axs[j].set_title('Топ {} жанров по продажам в {}'.format(n, r))
        
    
   
        plt.tight_layout()
        plt.show()
        
        
    if task==3:
        for i, r in enumerate(region):
            for j, s in enumerate(sales_column):
                if r in s:
                    region_sales = data.pivot_table(index='rating', values=s, aggfunc='sum').sort_values(by=s)[-n:]
                    axs[j].barh(region_sales.index, region_sales[s])
                    axs[j].set_xlabel('Количество продаж в {}'.format(r))
                    axs[j].set_ylabel('Рейтинг')
                    axs[j].set_title('Топ {} рейтингов по продажам в {}'.format(n, r))
        
        plt.tight_layout()
        plt.show()
        


# Как можем видеть, для североамериканского региона самые популярными являются приставки X360, PS2, Wii, PS3, DS. Но мы помним, что есть две приставки, которые вышли позже всех, и они все еще набирают обороты продаж, и как показывает анализ, апогей продаж еще у них впереди.

# In[57]:


region_sales(data_3, task=1)


# **Вывод**
# 
# В североамериканском регионе лидирует консоль от Microsoft, что ожидаемо, так как там основной рынок данной компании. Для рынка Японии лидер DS - портативная приставка, родом так же из Японии, учитывая специфику страны в части плотности населения, размера квартир, зачастую соместного проживания большой семьей в небольшом помещении, ритм жизни (жизнь в дороге), японцам удобнее играть на портативных приставках. А вот рынок Европы уже больше склоняется к полноразмерной игровой приставке, но таких ограничений нет как в Японии, а распостранение компании Microsoft слабее, скорее всегоименно за пределами родных стран идет основная борьба между Microsoft и Sony

# #### Самые популярные жанры (топ-5). 
# 
# По аналогии с обработкой сета по платформам, сделаем обработку по жанрам

# In[58]:


region_sales(data_3, task=2)


# **Вывод** 
# 
# Распределение лидеров в Америке и Европе практически совпали, лидеры Экшен и спорт, в данных регионах популярны приключения и спортивные игры (фифа, нба и т.п.), а в Японии лидируют ролевые игры, что тоже можно объяснить тем, что Японимая является родиной ролевых игр.

# #### Топ-5 жанров по регионам
# Разберем влияет ли рейтинг ESRB на продажи в отдельном регионе.

# In[59]:


region_sales(data_3, task=3)


# **Вывод**
# 
# Картины в целом схожи, топ-5 позиций занимают одинаковые значения: M («Mature» — «Для взрослых»), E ((«Everyone») — «Для всех»), T («Teen» — «Подросткам»), E10+ («Everyone 10 and older» — «Для всех от 10 лет и старше»).
# Стоит отметить, что система рейтинга ESRB - это американская система, разработная компание RockStar (серия игр GTA их детище). Соответвенно и распостранение система получала изначально на своем рынке. 
# 
# А вот в Японии, как мы видим, много игр без рейтинга, при условии допущения, что ошибок в выгрузке в части рейтинга нет, нужно помнить, что в Японии игровая индустрия имеет давние традиции и культурные особенности, которые могут объяснить большое количество игр без возрастного рейтинга.
# 
# Во-первых, в Японии есть традиция разработки игр, которые могут нацелены на различные возрастные группы и не требуют обязательного ограничения по возрасту. Такие игры могут быть более простыми и легкими, и могут содержать более мягкую насилие или другие контент, который в других странах может быть ограничен.
# 
# Во-вторых, в Японии существует система рейтинговой оценки игр, которая аналогична ESRB в США и PEGI в Европе, называется CERO. Однако, это необязательная система рейтинговой оценки, и разработчики не обязаны проходить ее процедуру оценки, что может приводить к большему количеству игр без рейтинга.
# 
# Также в Японии существуют более лояльные к насилию и отклонениям в поведении стандарты, поэтому многие игры, которые могут быть запрещены или ограничены в других странах, могут быть свободно выпущены в Японии.
# 
# Наконец, стоит отметить, что существует некоторое количество игр в Японии, которые содержат откровенно сексуальный контент, но эти игры не могут быть проданы в обычных магазинах, а распространяются через специальные магазины и интернет-сайты, которые специализируются на этом типе контента.
# 
# **Прим.*** учитывая что для этого анализа взял более короткий период а не весь период игростроя, чтож, могу утверждать, что люди стали более злыми)))) теперь в топе рейтинг для взрослых, где много крови, расчлененка и жестокости (это не тот контент 18+)))) )))

# В подтверждение гипотезы визуализируем распределение рейтинга между платформами в Японии. Как видно из результатов лидируют портативные платформы и линейка PS. Присутствие последней в этом топе тоже можно объяснить - родина приставки PS Япония, и как мы видели ранее она занимает второе место по продажам в Японии. Учитывая все вышесказанное в части системы рейтинга, гипотеза об о причинах отсуствия рейтинга в играх японского рынка нала свое подтверждение.

# In[60]:


rating_genre=data_3.pivot_table (index='rating', columns='platform', values='jp_sales', aggfunc = 'count')
rating_genre_filt=rating_genre.query ('rating=="undefined"')
df = pd.DataFrame(rating_genre_filt)

df_transposed = df.T
df_transposed.plot(kind='pie', subplots=True, figsize=(10, 10), legend=False)
plt.show()


# ## Проверьте гипотезы**
# 
# Средние пользовательские рейтинги платформ Xbox One и PC одинаковые;
# Средние пользовательские рейтинги жанров Action (англ. «действие», экшен-игры) и Sports (англ. «спортивные соревнования») разные.

# ### **Первая гипотеза**
# 
# Н0-Средние пользовательские рейтинги платформ Xbox One и PC равны,
# 
# Н1-Средние пользовательские рейтинги платформ Xbox One и PC не равны

# Сразу проведем подготовку, сделаем необходимые выборки, расчеты средних, дисперсий, стандарного отклонения, а также построим графики распределения

# Для проверки гипотез нам нужна выборка без заглушек

# In[61]:


#датасет без оценкок-заглушек игроков
without_111_data = data_3.query ('user_score<=10')


# In[62]:


xbox_data = without_111_data.query ('platform=="XOne"')
pc_data = without_111_data.query ('platform=="PC"')
mean_xbox_user_score = xbox_data['user_score'].mean ()
mean_pc_user_score = pc_data['user_score'].mean ()

variance_xbox = np.var (xbox_data['user_score'], ddof=1)
display (f'Коэффициент дисперсии для Xbox: {variance_xbox:.4f}') 

variance_pc = np.var (pc_data['user_score'], ddof=1)
display (f'Коэффициент дисперсии для PC: {variance_pc:.4f}')  

stand_devia_xbox = np.sqrt (variance_xbox)
display (f'Стандарное отклонение для Xbox: {stand_devia_xbox:.4f}') 

stand_devia_pc = np.sqrt (variance_pc)
display (f'Стандарное отклонение для PC: {stand_devia_pc:.4f}') 


# In[63]:


xbox_data


# In[64]:


sns.histplot(xbox_data['user_score'])


# In[65]:


sns.histplot(pc_data['user_score'])


# В нашем случае выборка оценок небольшая, берем ее всю

# In[66]:


sample_1 = without_111_data[without_111_data['platform']=="XOne"]['user_score']
sample_2 = without_111_data[without_111_data['platform']=="PC"]['user_score']
result =st.ttest_ind (sample_1, sample_2, equal_var=False) #дисперсии разные, поэтому ставим False
display (f'р-значение:{result.pvalue} ')

alpha=0.05

if (result.pvalue < alpha): 
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу") 


# Нулевая гипотеза не отвергнута. Н0- Средние значения двух генеральных выборок пользовательских рейтингов платформы XOne и PC равны. Прим. в оценке участовали игры только за 3 года

# ### **Вторая гипотеза**
# 
# Средние пользовательские рейтинги жанров Action (англ. «действие», экшен-игры) и Sports (англ. «спортивные соревнования») разные.

# H0-Средние пользовательские рейтинги жанров Action и Sports равны,
# 
# Н1-Средние пользовательские рейтинги жанров Action и Sports не равны,
# 

# In[67]:


action_data = without_111_data.query ('genre=="Action"')
spotrs_data = without_111_data.query ('genre=="Sports"')
mean_action_user_score = action_data['user_score'].mean ()
mean_spotrs_user_score = spotrs_data['user_score'].mean ()

variance_action = np.var (action_data['user_score'], ddof=1)
display (f'Коэффициент дисперсии для action: {variance_action:.4f}') 

variance_spotrs = np.var (spotrs_data['user_score'], ddof=1)
display (f'Коэффициент дисперсии для spotrs: {variance_spotrs:.4f}')  

stand_devia_action = np.sqrt (variance_action)
display (f'Стандарное отклонение для action: {stand_devia_action:.4f}') 

stand_devia_spotrs = np.sqrt (variance_spotrs)
display (f'Стандарное отклонение для spotrs: {stand_devia_spotrs:.4f}') 


# In[68]:


sns.histplot(action_data['user_score'])


# In[69]:


sns.histplot(spotrs_data['user_score'])


# In[70]:


sample_action = without_111_data[without_111_data['genre']=="Action"]['user_score']
sample_sports = without_111_data[without_111_data['genre']=="Sports"]['user_score']
result_2 =st.ttest_ind (sample_action, sample_sports, equal_var=False) #дисперсии разные, поэтому ставим False
display (f'р-значение:{result_2.pvalue} ')

alpha=0.05

if (result_2.pvalue < alpha): 
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу") 


# Нулевая гипотеза отвергнута. Н0 - Средние значения двух генеральных выборок пользовательских рейтингов жанров Action и Sports равны

# Прим. офф-топ, проверку гипотез понял слабо, писал по записям в тетради и конспектам от практикума, к сожалению почти без понимания расчетов. буду признателен если получится прислать интересные материалы для изучения факультативно. Пока сам читаю в основном хабр (там статья по р-значению более менее понятная нашлась)

# ## **Резюме**
# В результате исследования можем отметить несколько ключевых моментов:
#     
#     - основной рост продаж игр начинается с 1994 года, пик наблюдается в период 2008-2009 гг.
#     - установлено, что в основном платформы “живут” около 10 лет.
#     - на текущий момент потенциальные лидеры продаж, с учетом периода жизни - PS4, XBox One и 3DS, а учитывая длительный период нахождения в целом в списках платформ, то и PC
#     - при определении акцентов по платформам важно учесть и географию, так как в разных регионах свои лидеры. Так:
#         - для Североамериканского региона лидер платформы от Microsoft (серия XBox)
#         - для европейской части стран - серия платформ от Sony (Play Station)
#         - для японского рынка платформы от Nintendo - в данный момент 3DS.
#     - стоит учитывать популярность определенных жанров, для североамериканского региона и Европы - Экшен, спортивные игры и шутеры, а для японского рынка - ролевые игры, экшен и misc.
#     - лидерами продаж по рейтингам можно отметить игры доступные для всех, с рейтингом М. Но японский рынок и тут выделяется, для данного рынка свойственно не использовать рейтинг, так как культура покупки игры там отличается от принятой на западе. 
# 
# В ходе исследования проверены 2 гипотезы:
# 
# Средние пользовательские рейтинги платформ Xbox One и PC одинаковые. Гипотеза подтвердилась.
# 
# Средние пользовательские рейтинги жанров Action и Sports одинаковые. Гипотеза не подтвердилась.
