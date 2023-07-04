#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Общие-вводные" data-toc-modified-id="Общие-вводные-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Общие вводные</a></span><ul class="toc-item"><li><span><a href="#Описание-данных" data-toc-modified-id="Описание-данных-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Описание данных</a></span></li><li><span><a href="#Задание" data-toc-modified-id="Задание-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Задание</a></span></li><li><span><a href="#Общие-описание-действий-при-выполнении-задания" data-toc-modified-id="Общие-описание-действий-при-выполнении-задания-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Общие описание действий при выполнении задания</a></span></li></ul></li><li><span><a href="#Подготовка-инструментов-для-выполнения-задания" data-toc-modified-id="Подготовка-инструментов-для-выполнения-задания-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Подготовка инструментов для выполнения задания</a></span></li><li><span><a href="#Знакомство-с-таблицами" data-toc-modified-id="Знакомство-с-таблицами-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Знакомство с таблицами</a></span></li><li><span><a href="#Выполнение-заданий" data-toc-modified-id="Выполнение-заданий-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Выполнение заданий</a></span><ul class="toc-item"><li><span><a href="#Сколько-книг-вышло-после-1-января-2000-года" data-toc-modified-id="Сколько-книг-вышло-после-1-января-2000-года-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Сколько книг вышло после 1 января 2000 года</a></span></li><li><span><a href="#Для-каждой-книги-посчитаем-количество-обзоров-и-среднюю-оценку" data-toc-modified-id="Для-каждой-книги-посчитаем-количество-обзоров-и-среднюю-оценку-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Для каждой книги посчитаем количество обзоров и среднюю оценку</a></span></li><li><span><a href="#Определим-издательство,-которое-выпустило-наибольшее-число-книг-толще-50-страниц;" data-toc-modified-id="Определим-издательство,-которое-выпустило-наибольшее-число-книг-толще-50-страниц;-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Определим издательство, которое выпустило наибольшее число книг толще 50 страниц;</a></span></li><li><span><a href="#Определим-автора-с-самой-высокой-средней-оценкой-книг-—-учитывайте-только-книги-с-50-и-более-оценками." data-toc-modified-id="Определим-автора-с-самой-высокой-средней-оценкой-книг-—-учитывайте-только-книги-с-50-и-более-оценками.-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Определим автора с самой высокой средней оценкой книг — учитывайте только книги с 50 и более оценками.</a></span></li><li><span><a href="#Посчитаем-среднее-количество-обзоров-от-пользователей,-которые-поставили-больше-48-оценок." data-toc-modified-id="Посчитаем-среднее-количество-обзоров-от-пользователей,-которые-поставили-больше-48-оценок.-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Посчитаем среднее количество обзоров от пользователей, которые поставили больше 48 оценок.</a></span></li></ul></li><li><span><a href="#Вывод." data-toc-modified-id="Вывод.-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Вывод.</a></span></li></ul></div>

# # Общие вводные

# Коронавирус застал мир врасплох, изменив привычный порядок вещей. В свободное время жители городов больше не выходят на улицу, не посещают кафе и торговые центры. Зато стало больше времени для книг. Это заметили стартаперы — и бросились создавать приложения для тех, кто любит читать.
# 
# 
# Ваша компания решила быть на волне и купила крупный сервис для чтения книг по подписке. Ваша первая задача как аналитика — проанализировать базу данных.
# В ней — информация о книгах, издательствах, авторах, а также пользовательские обзоры книг. Эти данные помогут сформулировать ценностное предложение для нового продукта.

# ## Описание данных

# Таблица books (Содержит данные о книгах):
# 
#     - book_id — идентификатор книги;
#     - author_id — идентификатор автора;
#     - title — название книги;
#     - num_pages — количество страниц;
#     - publication_date — дата публикации книги;
#     - publisher_id — идентификатор издателя.
# 
# Таблица authors (Содержит данные об авторах):
# 
#     - author_id — идентификатор автора;
#     - author — имя автора.
# 
# Таблица publishers (Содержит данные об издательствах):
# 
#     - publisher_id — идентификатор издательства;
#     - publisher — название издательства;
#     
# Таблица ratings (Содержит данные о пользовательских оценках книг):
# 
#     - rating_id — идентификатор оценки;
#     - book_id — идентификатор книги;
#     - username — имя пользователя, оставившего оценку;
#     - rating — оценка книги.
#     
# Таблица reviews (Содержит данные о пользовательских обзорах на книги):
# 
#     - review_id — идентификатор обзора;
#     - book_id — идентификатор книги;
#     - username — имя пользователя, написавшего обзор;
#     - text — текст обзора.

# <a id="задачи"></a>

# ## Задание 
# 
# 
#     1. Посчитайте, сколько книг вышло после 1 января 2000 года;
#     2. Для каждой книги посчитайте количество обзоров и среднюю оценку;
#     3. Определите издательство, которое выпустило наибольшее число книг толще 50 страниц — так вы исключите из анализа брошюры;
#     3. Определите автора с самой высокой средней оценкой книг — учитывайте только книги с 50 и более оценками;
#     4. Посчитайте среднее количество обзоров от пользователей, которые поставили больше 48 оценок.

# ## Общие описание действий при выполнении задания
# 
#     1. Описание исследования;
#     2. Исследование таблицы (нужно выввести первые строки);
#     3. 1 задание - 1 запрос;
#     4. Выводы по каждой из решённых задач.

# # Подготовка инструментов для выполнения задания

# In[1]:


import pandas as pd
from sqlalchemy import text, create_engine
# библиотеки для выведения ER-диаграммы
from IPython.display import Image, display 
import requests

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)


# <a id="ER_диаграмма"></a>

# In[2]:


try:
    image_path = r"C:\Users\PC_Maks\Desktop\1.jpg"
    display(Image(filename=image_path))
except:
    image_url = "https://i.postimg.cc/yYDBx1qK/1.jpg"
    response = requests.get(image_url)
    image_data = response.content
    display(Image(image_data))


# In[3]:


# устанавливаем параметры
db_config = {'user': 'praktikum_student', # имя пользователя
'pwd': 'Sdf4$2;d-d30pp', # пароль
'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
'port': 6432, # порт подключения
'db': 'data-analyst-final-project-db'} # название базы данных
connection_string = 'postgresql://{user}:{pwd}@{host}:{port}/{db}'.format(**db_config)


# In[4]:


# сохраняем коннектор
engine = create_engine(connection_string, connect_args={'sslmode':'require'})


# # Знакомство с таблицами
# Посмотрим на перые строки

# [Перейти к диаграмме](#ER_диаграмма)
# 
# [Перейти к задачам](#задачи)

# In[5]:


list_table = ['books', 'authors', 'publishers', 'ratings', 'reviews']


# In[6]:


for name_table in list_table:
    query = '''SELECT * FROM {} LIMIT 5'''.format(name_table)
    con=engine.connect()
    df = pd.io.sql.read_sql(sql=text(query), con=con)
    con.close()
    print(f'НАЗВАНИЕ ТАБЛИЦЫ: {name_table}', '\n')
    print(df, '\n')


# # Выполнение заданий

# ## Сколько книг вышло после 1 января 2000 года

# [Перейти к диаграмме](#ER_диаграмма)
# 
# [Перейти к задачам](#задачи)

# In[7]:


query = '''SELECT count (title)
FROM books 
WHERE publication_date > '2000-01-01'
'''
con=engine.connect()
pd.io.sql.read_sql(sql=text(query), con = con)


# Всего после 1 января 2000 года вышло 819 книг 

# ## Для каждой книги посчитаем количество обзоров и среднюю оценку

# [Перейти к диаграмме](#ER_диаграмма)

# In[8]:


query = '''SELECT b.book_id, b.title, ROUND(AVG(r.rating), 2) as rating, COUNT(DISTINCT rev.review_id) as count_review
FROM books AS b
LEFT JOIN ratings AS r ON b.book_id = r.book_id
LEFT JOIN reviews AS rev ON b.book_id = rev.book_id
GROUP BY b.book_id, b.title
ORDER BY count_review DESC;

'''

df_rating = pd.io.sql.read_sql(sql=text(query), con = con)
df_rating


# Оценки распределились в диапазоне от 1,5 до 5.0, а обзоры от 0 до 7

# 

# ## Определим издательство, которое выпустило наибольшее число книг толще 50 страниц;

# [Перейти к диаграмме](#ER_диаграмма)

# In[9]:


query = '''SELECT pub.publisher, count (distinct b.book_id) as count_books
FROM books AS b
JOIN publishers AS pub ON pub.publisher_id = b.publisher_id
WHERE b.num_pages > 50
GROUP BY pub.publisher
ORDER BY count_books DESC
limit 1
'''
df_big_publ = pd.io.sql.read_sql(sql=text(query), con = con)
df_big_publ


# 

# ## Определим автора с самой высокой средней оценкой книг — учитывайте только книги с 50 и более оценками.

# [Перейти к диаграмме](#ER_диаграмма)

# In[18]:


query = '''WITH top_books AS (
    SELECT b.book_id, b.title, b.author_id, COUNT(r.rating) AS rating_count, AVG(r.rating) AS rating_avg
    FROM books AS b
    JOIN ratings AS r ON b.book_id = r.book_id
    GROUP BY b.book_id, b.title, b.author_id
    HAVING COUNT(r.rating) >= 50
), top_authors AS (
    SELECT a.author_id, AVG(tb.rating_avg) AS avg_rating
    FROM top_books AS tb
    JOIN authors AS a ON tb.author_id = a.author_id
    GROUP BY a.author_id
)
SELECT a.author, ta.avg_rating AS average_rating
FROM top_authors AS ta
JOIN authors AS a ON ta.author_id = a.author_id
ORDER BY ta.avg_rating DESC
LIMIT 1;

'''
df_great_author = pd.io.sql.read_sql(sql=text(query), con = con)
df_great_author


# Самая высокая средняя оценка книг, у которых от 50 и более оценок, среди всех авторов - у J.K. Rowling/Mary GrandPré

# ## Посчитаем среднее количество обзоров от пользователей, которые поставили больше 48 оценок.

# [Перейти к диаграмме](#ER_диаграмма)

# In[11]:


query = '''WITH rated_users AS (
    SELECT username
    FROM ratings
    GROUP BY username
    HAVING COUNT(rating) > 48
)
SELECT AVG(review_count) AS average_reviews
FROM (
    SELECT COUNT(review_id) AS review_count
    FROM reviews
    WHERE username IN (
        SELECT username
        FROM rated_users
    )
    GROUP BY username
) subquery


'''
df_most_active_users = pd.io.sql.read_sql(sql=text(query), con = con)
df_most_active_users


# В среднем активные пользователи пишут по 24 обзора. Учитывая наш фильр в 48 оценок, получается, что каждый активный пользователь пишет обзор на каждую вторую книгу, которой поставил оценку.

# # Вывод. 

#     1. Всего после 1 января 2000 года было выпущено 818 книг. Эта информация помогает нам понять объем книжного контента, доступного для анализа.
# 
#     2. У всех книг в исходнных данных до 7 обзоров, что свидетельствует о разнообразии мнений пользователей. Средняя оценка книг варьируется от 1.5 до 5.0, что позволяет нам оценить качество и популярность книг в базе данных.
# 
#     3. При анализе издательств, которые выпустили книги толще 50 страниц, наибольшее количество таких книг было выпущено издательством Penguin Books — 42 книги. Это указывает на значительную активность издательства в данной области.
# 
#     4. При рассмотрении авторов с самыми высокими средними оценками книг (с учетом только книг с 50 и более оценками), мы обнаружили, что J.K. Rowling/Mary GrandPré имеет самую высокую среднюю оценку книг - 4.28. Это подчеркивает популярность и качество их работы среди читателей.
# 
#     5. В среднем, активные пользователи пишут около 24 обзоров. Если мы учитываем только пользователей, которые поставили более 48 оценок, можно сделать вывод, что каждый активный пользователь пишет обзор на каждую вторую книгу, которой он поставил оценку. Это указывает на высокую активность и заинтересованность этой группы пользователей в оценивании и обсуждении книг.
