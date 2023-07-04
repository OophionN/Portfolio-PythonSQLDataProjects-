#!/usr/bin/env python
# coding: utf-8

# # Исследование объявлений о продаже квартир
# 
# В вашем распоряжении данные сервиса Яндекс.Недвижимость — архив объявлений о продаже квартир в Санкт-Петербурге и соседних населённых пунктов за несколько лет. Нужно научиться определять рыночную стоимость объектов недвижимости. Ваша задача — установить параметры. Это позволит построить автоматизированную систему: она отследит аномалии и мошенническую деятельность. 
# 
# По каждой квартире на продажу доступны два вида данных. Первые вписаны пользователем, вторые — получены автоматически на основе картографических данных. Например, расстояние до центра, аэропорта, ближайшего парка и водоёма. 

# ## Откройте файл с данными и изучите общую информацию. 

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


try:
    realty_df = pd.read_csv ('/datasets/real_estate_data.csv', sep = '\t', parse_dates=['first_day_exposition'])
except:
    realty_df = pd.read_csv('https://code.s3.yandex.net/datasets/real_estate_data.csv',                            sep = '\t', parse_dates=['first_day_exposition'])

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
print (realty_df.head (10))
print (realty_df.info ())
print (realty_df.hist(figsize=(15, 20)))


# ## Предобработка данных

# In[3]:


realty_df.isna ().sum ()


# посчитали сколько пропусков и в каких столбцах. Сразу можно отметить пропуски в столбцах с указанием объектов в округе и расстояние до них (аэропорт, парки, водоемы) значение практически совпали. отсутствие значений в столбце высота потолков, явно невозможно (будем рассматривать и решать, чем заполнить). Аналогичная ситуация с названием н.п., количеством этажей, жилой площадью. Столбец апартаменты - необходимо посмотреть в каких строках пропуски, предварительное предположение, что пропуск в квартирах без статуса апарты. будем разбивать выполнение задание на куски, начнем с очевидных пропусков и будем проводить обработку данных.

# ### Адреса.
# начнем с адреса. для начала посмотрим на объекты с пропущенными адресами. всего пропущено 49 адресов, учитывая, что в данном случае обогатить адреса в моменте не представляется возможным, а прямой связи не видно в части пропусков адресов,  принимаю решение удалить 49 строк из ДФ, а также проводим корректировку названий, встречается разное написание (поселок/посёлок)

# In[4]:


realty_df ['locality_name'].isnull ().sum ()

realty_df = realty_df.dropna (subset = ['locality_name']).reset_index(drop=True)

realty_df ['locality_name'] = realty_df ['locality_name'].str.lower ()

def uniq_name (data):
    row = data ['locality_name']
    if 'посёлок станции' in row:
        return row.replace ('посёлок станции ', '')
    elif 'поселок станции' in row:
        return row.replace ('поселок станции ', '')
    elif 'посёлок городского типа' in row:
        return row.replace ('посёлок городского типа ', '')
    elif 'поселок городского типа' in row:
        return row.replace ('поселок городского типа ', '')
    elif 'деревня' in row:
        return row.replace ('деревня ', '')
    elif 'городской посёлок' in row:
        return row.replace ('городской посёлок ', '')
    elif 'городской поселок' in row:
        return row.replace ('городской поселок ', '')
    elif 'пок городского типа' in row:
        return row.replace ('пок городского типа ', '')
    elif 'садоводческое некоммерческое товарищество ' in row:
        return row.replace ('садоводческое некоммерческое товарищество ', '')
    elif 'коттеджный посёлок' in row:
        return row.replace ('коттеджный посёлок ', '')
    elif 'коттеджный поселок' in row:
        return row.replace ('коттеджный поселок ', '')
    elif 'коттеджный ' in row:
        return row.replace ('коттеджный ', '')
    elif 'садовое товарищество' in row:
        return row.replace ('садовое товарищество ', '')
    elif 'посёлок при железнодорожной станции' in row:
        return row.replace ('посёлок при железнодорожной станции ', '')
    elif 'поселок ' in row:
        return row.replace ('поселок ', '')
    elif 'посёлок' in row:
        return row.replace ('посёлок ', '')
    elif 'пок' in row:
        return row.replace ('пок ', '')
    elif 'село' in row:
        if 'Красное село' in row:
            return row
        else:
            return row.replace ('село ', '')
    else:
        return row
    
realty_df ['locality_name'] = realty_df.apply (uniq_name, axis=1)


realty_df ['locality_name'].unique ()


# ### Аппартаменты.
# Посмотрим на значение апартаменты. данное поле является категоричным, бул.значения - или да или нет. Значения пропущены в 20890 ячейках. Так как апартаменты являются коммерческой недвижимостью, и по своей сущности не могут иметь жилой площади, необходимо провести исследование, в котором сопоставим значения из столбца жилой площади и апартаментов. Так как в ячейках жилой площади нет значений 0, но есть пропуски, берем за допущение что все значения с пропусками это 0. (прим. пропусков в разделе общей площади нет). После этого у нас останется часть объектов не апартов с жилой площадью 0, к ним вернемся после оценки раздела с кухнями.
# справочный материал по апартаментам https://journal.tinkoff.ru/opasnosti-apartamentov/
# 

# In[5]:


print (realty_df ['is_apartment'].value_counts (dropna=False))

realty_df ['is_apartment'] = realty_df ['is_apartment'].fillna(True)
print (realty_df ['is_apartment'].value_counts (dropna=False))

realty_df ['living_area'] = realty_df ['living_area'].fillna (0)
print (realty_df ['living_area'].value_counts (dropna=False))

def match (row):
    value1= row ['living_area']
    value2= row ['is_apartment']
    try:
        if value1==0 and value2 == True:
            return True
        else:
            return False
    except:
        return 'Error'

realty_df ['analysis_apart'] = realty_df.apply (match, axis=1)
realty_df['analysis_apart'].value_counts ()


# In[6]:


realty_df['locality_name'].replace('d', 'fr')


# ### Балконы.
# 
# Теперь разберёмся с объектами, в столбце "жилая площадь" мы проставили 0, но они не попали под категорию апартаментов. Пропусков у нас больше нет, но 1898 объектов с значением 0. Но для начала нужно разобраться с балконами, так каких площадь входит в общую площадь (примем за допущение коэффициент расчета отношения площади балкона/лоджии равный 0 - в общую входит, в жилую нет (Прим. в законодательстве предусмотрен расчет: лоджии 0.5, балконы 0.3 от общей площади последних относится к жилой). Если количество балконов не указано - принимаем что их нет. также сразу заменим тип столбца на целое
# Замена типа данных обусловлена целочисленным значением объекта "Балкон"
# 

# In[7]:


realty_df ['balcony'].value_counts (dropna=False)
realty_df ['balcony'] = realty_df ['balcony'].fillna (0).astype (int)
realty_df ['balcony'].value_counts (dropna=False)


# ### Квартиры с большим количеством балконов.
# В таблице есть объекты, в которых больше 3 балконов, что выглядит странным. посмотрим на эти квартиры. Было бы странно, если в квартире на 40 кв.м. 5 балконов. Аномальных квартир менее 50-ти. Считаю возможным исключить из ДФ объекты в которых более 3-х балконов, даже при допущении такой ситуации, их количество не должно сильно повлиять на итоговые данные.

# In[8]:


realty_df = realty_df.query ('balcony <= 3').reset_index(drop=True)
print (realty_df.info())


# ### Высота потолков.
# 9195 пропусков. Значение обязательно должны быть. разберемся со значениями выбросами в высоте потолков. Исходя из понимания, что согласно нормам в НПА, высота потолка в самых теплых помещениях составляет не менее 2,7 метра, а в остальных помещениях — не менее 2,5 метра, учитывая распространение навесных потолков (-15/20 см от общей высоты), все что менее 2.3  метров (норма до 2.5m минус 15/20 см на подвесной потолок), считаю выбросом, что подтверждается границами 1 квартили по высоте потолка, что выше 4 м аномалия. посмотрим сколько таких квартир в ДФ. Всего 75 квартир с высотой потолков выше 4 метров, несмотря на возможность потолков 4+, но оставшиеся значения или выглядят как аномалия (высота в 100 метров) или возможно квартиры старого фонта (постройки 50-х годов), что тоже маловероятно для региона и н.п.. Принимаю решение допустимости исключения данных объектов из итогового ДФ

# Значение медианы выглядит логично - 2,65.  Прихожу к пониманию, что пропуски можно заменить медианой. 

# In[9]:


#print (realty_df ['ceiling_height'].describe())
#print (realty_df.plot (y='ceiling_height', kind='hist', range=(0, 5), bins=50))

height_median = realty_df ['ceiling_height'].median()
realty_df ['ceiling_height_stat'] = realty_df ['ceiling_height'].fillna (height_median)

# пропусков в разделе высоты потолков больше нет - столбец ceiling_height_stat         
realty_df.isna ().sum()


# In[10]:


realty_df = realty_df.query ('2.3 <= ceiling_height_stat <= 4.0').reset_index(drop=True)
realty_df.info()


# ### Кухня.
# 
# в столбце кухня пропуски в 2269 ячейках, исходя из понимания, что в студиях кухни как отдельной зоны нет, и могут соответственно не указывать, так как зона кухни входит в общую зону, заменим значением 0 все ячейки для студий. Перед манипуляциями сохраним медиану в отдельную переменную (предварительно просмотрели- медиана 9.1 кв.м. что близко к жизни). Сейчас  149 студий в списке. После замены пропусков заглушкой 9999, проверили, что ДФ не было студий с параметром размера кухни, и все пропуски студия+значение размера кухни стали 9999. 
# Теперь можем заменить заглушку в студиях размер кухни на 0, а остальных помещениях на медиану. Создав новый столбец kitchen_stat_norm. Изучив данные после замен, один объект выбивается на мой взгляд, № 504 - кухня 50 кв.м.+жилая 13, а общая площадь 69. Примем за допущение (возможно евродвушка, где 1 жилая комната, а второе помещение - кухня совмещенная со столовой)
# 

# In[11]:


#print (realty_df.sort_values(by='kitchen_area', ascending = False))
#print (realty_df ['kitchen_area'].describe())
kitchen_median = realty_df ['kitchen_area'].median ()
realty_df ['kitchen_area'] = realty_df ['kitchen_area'].fillna (9999)

realty_df.query ('studio==True & kitchen_area !=9999').value_counts()

realty_df ['kitchen_stat_norm'] = realty_df ['kitchen_area'].where (realty_df['studio'] == False, 0)
realty_df ['kitchen_stat_norm'] = realty_df ['kitchen_stat_norm'].where (realty_df ['kitchen_stat_norm']!=9999, kitchen_median)

realty_df ['kitchen_stat_norm'].value_counts()
realty_df.query ('kitchen_stat_norm >45').groupby ('total_area').head ()


# ### Комнаты. 
# 
# Квартиры, в которых количество комнат более 6-ти, всего 86, в целом данные квартиры можно отнести к категории - аномалии. Удаляем

# In[12]:


realty_df = realty_df.query ('rooms<=6').reset_index (drop=True)


# ### Студии.
# 
# Исходя из понимания, что в студиях не может быть комнаты, поэтому в разделе комнаты у студии должно быть указано 0. студий у нас в списке 144, объектов с указанием что комнат 0 - 190, нужно посмотреть, входят ли все студии в список объектов без комнат, и если - да, посмотреть какие объекты не студии, но без комнат. По указанному сочетанию получили 133 объекта, оставшиеся студии указаны как однокомнатные, с общей площадью не более 32.5 кв.м., с высокой долей вероятности эти 11 студий, по какой-то причине указали с комнатами - исправим это. Также обращаем внимание, что в список попала апартаменты с площадью 370 кв.м. без указания количества комнат, учитывая единичный случай, их можно удалить из ДФ
# 
# В оставшемся ДФ, 56 объектов с общей площадью менее 45 кв.м. без указания комнат, учитывая площадь, могу с высокой долей вероятности предположить, что это квартиры студии, проведем корректировку статуса студий. Итого в ДФ стало 200 студий
# 

# In[13]:


def room_studio (data, task):
    try:
        room=data ['rooms']
        studio = data ['studio']
        if task==0:
            if studio == True:
                        
                return 0
            else:
                return room
        if task==1:
            if room == 0:
                studio = True
                return studio
            else:
                studio = False
                return studio
        
    except:
        return 'Error'
            

realty_df ['rooms'] = realty_df.apply (room_studio, task=0, axis=1)
realty_df ['rooms'] = realty_df ['rooms'].astype (int)
realty_df=realty_df.query ('total_area !=371.00')
realty_df ['studio'] = realty_df.apply (room_studio, task=1, axis=1)
zero_not_studio = realty_df.query ('rooms==0 & studio == True')
realty_df ['rooms']. value_counts ()


# ### Жилая площадь
# отсутствующие значения жилой площади (забендины 0), исключив апарты и студии из расчета, можно заполнить средним размером комнат к жилой площади уже нам известной (возьмем сумму всей жилой площади (0 не добавят объема), без апартов и студий, и разделим на общее количество комнат в ДФ)
# 
# Мы получили коэффициент в размере 16.4 кв.м. на 1 комнату в средней статистике, заполним таблицу исходя из этих данных и количества комнат
# 
# 

# In[14]:


live_area_filltred = realty_df.query ('(analysis_apart!=True) & (studio!=True)')
summa_room = live_area_filltred ['rooms'].sum ()
summa_living_area = live_area_filltred ['living_area'].sum ()
ratoin_living_area = round (summa_living_area/summa_room, 1)


def add_living_area (data, ratio):
    try:
        area = data ['living_area']
        rooms = data ['rooms']
        
        if area > 0:
            return area
        else:
            area = rooms*ratoin_living_area
            return area
        
    except:
        return 'Error'

realty_df ['living_area'] = realty_df.apply (add_living_area, ratio=ratoin_living_area,  axis=1)


# ### Этажи.
# 
# в столбце этажи 85 пропусков. Можно предположить, что пропуски связаны с тем, что это частные дома, и там нет этажей, но для начала проверим какие значения есть в разделе этажей. Необходимо не забывать, что если указано что этаж размещения объекта выше 1, то с высокой долей вероятности пропуск значения в разделе этажности - ошибка.
# Медиана = 9-ти этажные здания, но есть и 30+ этажность, в санкт-петербурге есть здание в котором 87 этажей, но средняя застройка в высотках редко бывает выше 28 этажей, посмотрим, что за объекты с этажами выше 28 и где они расположены. Итого 2 квартиры выше 28-го этажа, что менее 0,001% - прихожу к выводу, что можно удалить. (статья https://lenobldoma.ru/doma-po-etazham) 
# 
# Заменим тип данных на целое. Прим (сейчас вещественное, поменяем на целое, этажи в домах в подавляющем большинстве целое значение)
# 

# In[15]:


realty_df ['floors_total'].value_counts (dropna=False, ascending=False)
realty_df ['floors_total'].describe()
print (realty_df.query('floors_total > 28').value_counts().sum ())
realty_df = realty_df.query('floors_total <= 28').reset_index(drop=True)

realty_df ['floors_total'] = realty_df ['floors_total'].astype (int)


#проверим на пропуски в этажности оставшиеся данные ДФ - пропусков больше нет.
print (realty_df.isnull().sum())


# ### Удалим явные дубли.
#     

# In[16]:


realty_df = realty_df.drop_duplicates().reset_index(drop=True)
realty_df.isnull().sum()


# ### Инфраструктура 
# попробуем разобраться с инфраструктурой. одним из решений вижу возможность пропуски дозаполнить средними значениями для отдельно взятого н.п. для начала нам нужна сводная таблица средних расстояний для каждого н.п. (Прим. заменим типы данных на целые). Примененные способы группировки и использования значений заглушек подвели меня к пониманию, что данное значение можем оставить без заполнения пропусков. Обоснование: во-первых, данные предоставлены сервисом Яндекс.Недвижимость, соответственно использование сведений от Яндекс.Карты не вызывало бы сложностей при формирование реестра (ошибки в пропусках маловероятны), во-вторых, сведений о расстоянии и количестве объектов может не быть, потому что их просто нет рядом (в списке различные СТН, деревни и т.п.) Учитывая все изыскания, и указанное выше, использовав в качестве заглушки -77, и сменив формат на целочисленное (для расстояния не критично опустить метры, а прудов и парков не может быть не целое число), идем дальше.

# In[17]:


# сохраним в переменную название всех городов
sity_name = realty_df ['locality_name'].unique ()
realty_df[['parks_around3000','airports_nearest','parks_nearest','ponds_around3000',           'ponds_nearest']] = realty_df[['parks_around3000','airports_nearest','parks_nearest',                                          'ponds_around3000','ponds_nearest']].fillna(-77)
# сгруппируем по названию н.п. таблицу с параметрами инфраструктуры
realty_group_sity = realty_df.groupby ('locality_name') [['parks_around3000','airports_nearest',                                                          'parks_nearest','ponds_around3000','ponds_nearest']].agg ('median')
realty_group_sity.head (25)
realty_df = realty_df.astype({'parks_around3000': np.int64, 'airports_nearest': np.int64,                              'parks_nearest': np.int64,'ponds_around3000': np.int64,'ponds_nearest': np.int64})
realty_df.info()


# ### Размещение объявление в днях
# Пропуски в количестве дней размещений не критичны, так как возможно объект еще не снят с продажи. Идем дальше

# ### Тип этажа квартиры 
# («первый», «последний», «другой»). Создадим столбец категорий этажности и переберем все квартиры. Функции расчетов схожие, сделаем все в 1-ой

# In[18]:


def setting (data, task):
    try:
        # отфильтровка блока который необходимо выполнить
########## этажи по категориям ###############
        if task==0:
            total = data['floors_total']
            floor = data['floor']
            if floor == 1:
                return 'первый'
            elif floor == total:
                return 'последний'
            else:
                return 'другой'

################ цена за 1 кв.м. ##########        
        elif task==5:
            price = data ['last_price']
            area = data ['total_area']
            ration = price/area
            return round(ration, 1)
  ##########день недели ###############      
        elif task==1:
            day = data ['first_day_exposition'].weekday ()
            if day == 0:
                return 'Понедельник'
            elif day == 1:
                return 'Вторник'
            elif day == 2:
                return 'Среда'
            elif day == 3:
                return 'Четверг'
            elif day == 4:
                return 'Пятница'
            elif day == 5:
                return 'Суббота'
            else:
                return 'Воскресенье'
 ############# месяц ########         
        elif task==2:
            date_full = data ['first_day_exposition']
            month = date_full.month
            list_month = [0, 'янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
            
            try:
                for index in range (13):
                    if index == month:
                        return list_month [index]
            except:
                return 'Error.month'
            
########  год ########            
        elif task==3:
            date_full = data ['first_day_exposition']
            year = date_full.year
            return year
            
        
    except:
        return 'Error. Total'
        
        
                
                
                        
realty_df ['floor_type'] = realty_df.apply (setting, task=0, axis=1)
realty_df ['floor_type'].value_counts()


# 

# ## Посчитайте и добавьте в таблицу новые столбцы

# ### цена 1 кв.м.
# У нас есть подготовленный столбец с ценной на объект недвижимости и его общая площадь, ценная 1 кв.м. будет в их отношении

# In[19]:



realty_df ['ration_price_area'] = realty_df.apply (setting, task=5, axis=1)
realty_df.head()


# ### расчет и добавление дня недели создания объявления
# у нас есть столбец first_day_exposition, применим метод weekday

# In[20]:



realty_df ['first_weekday'] = realty_df.apply (setting, task=1, axis=1)
realty_df.head()


# ### расчет и добавление месяца публикации объявления
# Также используем first_day_exposition

# In[21]:



realty_df ['month'] = realty_df.apply (setting, task=2, axis=1)
realty_df.head ()


# ### расчет и добавление года публикации объявления
# Также используем first_day_exposition

# In[22]:


realty_df ['year'] = realty_df.apply (setting, task=3, axis=1)
realty_df.head ()


# ### Центр
# Сгруппируем и посмотрим на медиативные значения расстояний до центра. всего пропусков 5376, медиану смогли расчитать только для 24 городов, по остальным н.п. нет данных
# Остальные пропуски заполним заглушками (-7777)

# 

# In[23]:


display (len(realty_df[realty_df['cityCenters_nearest'].isna()]))

group_sity_center = realty_df.pivot_table (index = 'locality_name',  values = 'cityCenters_nearest', aggfunc='median').astype (int).reset_index()
display (group_sity_center)
group_sity_center.info()


# In[24]:


for t in realty_df['locality_name'].unique():
    realty_df.loc[(realty_df['locality_name'] == t) & (realty_df['cityCenters_nearest'].isna()), 'cityCenters_nearest'] =     realty_df.loc[(realty_df['locality_name'] == t), 'cityCenters_nearest'].median()


# смогли заполнить только 66 пропусков медиативным значением по городу, учитывая что в ДФ много небольших населенных пунктов, скорее всего в них нет понятия расстояния до центра.

# In[25]:


realty_df['cityCenters_nearest'] = realty_df['cityCenters_nearest'].fillna (-7777)
realty_df['cityCenters_nearest'].isna().sum()


# Добавим столбец среднего расстояния в км, округлим его и заменим формат на целочисленый

# In[26]:


realty_df ['cityCenters_averag_km'] = round (realty_df ['cityCenters_nearest']/1000).astype (int)


# ## Проведите исследовательский анализ данных

# ### Гистограммы для обработанных параметров
# 

# пересоберем итоговую таблицу

# In[27]:


realty_df_norm = realty_df.drop (columns=['ceiling_height', 'is_apartment', 'kitchen_area'], axis=1).reset_index (drop=True)
realty_df_norm.info()
realty_df_norm.isnull().sum ()


# ### общая площадь. 
# Как из гистограммы так и из характеристик столбца можем сделать вывод, что распределение площадей квартир соответствуют корреляции Пиросна, со медиативным значением около 51 кв.м. Есть квартиры очень малогабаритные от 12 кв.м., среднее значение немного смещено в сторону более больших по площади квартир, сред.площадь около 59 кв.м.

# In[28]:


print (realty_df_norm['total_area'].describe())

realty_df_norm.plot (y='total_area', kind='hist', ylim=(0,7000), xlim = (0,500), figsize=(5,5), bins=40, legend = False)
plt.title('Оценка общей площади')
plt.xlabel('общая площадь')
plt.ylabel('количество объявлений')
plt.show()


# ### Жилая площадь.
# 
# Найден 21 странный объект, у которого жилая площадь менее 10 кв.м. из более чем 25 квадратных метров, но взяв во внимание общий тренд на квартиры с большими общими площадями и минимальными жилыми зонами, оставляем в ДФ эти квартиры. Построим гистограмму по данной категории. Средние значения в области 33,5 кв.м. медиана около 30. имеется пики в районе 18-20 и 30 кв.м.,  в ДФ около 30% объектов - однокомнатные и студии, в которых жилая площадь редко превышает 20 кв.м., следующим шагом идут 2-х комнатные квартиры, в которых по статистике средняя жилая площадь около 30-33 кв.м., а так как 1-х и 2-х основная часть квартир в реестре, поэтому мы и наблюдаем 2 пика в соответствии с распределением комнатности.
# 

# In[29]:


print (realty_df_norm ['living_area'].describe())
live = realty_df_norm.query ('living_area!=0.0 & analysis_apart!=True')
live.query ('living_area<10 & total_area>25').head (25)


print (realty_df_norm.plot (y='living_area', kind='hist', ylim=(0,8000), xlim = (0,150), figsize=(5,5), bins=70, legend = False))
plt.title('Оценка жилой площади Общая')
plt.xlabel('жилая площадь')
plt.ylabel('количество объявлений')
print (live.plot (y='living_area', kind='hist', ylim=(0,8000), xlim = (0,150), figsize=(5,5), bins=70, legend = False))
plt.title('Оценка жилой площади после перебора')
plt.xlabel('жилая площадь')
plt.ylabel('количество объявлений')
plt.show()


# ### площадь кухни.
# Все объекты с площадью кухни равной 0 - студии, это объясняется форматом квартир-студий, они не имеют четкого зонирования с разделением кухонь от остальной площади жилого помещения. Пики в районе 6 и 9 кв.м. могу объяснить спецификой строительства в стране, так как до конца 2000-х по основным типовым объектам строительства (включая период СССР), кухни были или 6\9\12 кв.м.,  но мы видим на графике, что тенденция немного сдвигается в большую сторону, что видно по линии угасания графика после 11 кв.м., пока квартиры с кухнями более 12 кв.м. относительная редкость в общей массе, у верен что данный показатель в новостройках последних 5-7 лет в МСК будет смещен еще существеннее. Медиана в районе 9 кв.м. соответствуют объективным обстоятельствам на рынке.

# In[30]:


print (realty_df_norm ['kitchen_stat_norm']. describe ())
realty_df_norm.plot (y='kitchen_stat_norm', kind='hist', ylim=(0,8000), xlim = (0,50), figsize=(5,5), bins=50, legend=False)
plt.title('Оценка площади кухни')
plt.xlabel('площадь кухни')
plt.ylabel('количество объявлений')
plt.show()
#zero_kitchen = realty_df_norm.query ('kitchen_stat_norm == 0 ')
#zero_kitchen


# ### цена объекта
# Изучив актуальные предложения в открытых источниках на рынке можно сделать вывод, что ценник на квартиры старого фонда начинаются от 350000. Однако квартира размером 109 квадратов с ценной 12190 явно выбивается из общего формата, как в части размер/цена, а это явно относительно новая квартиры (этажность 20+), так и соответственно в части среднего ценника за квадратный метр. Удалим эту квартиру из выборки. Квартиры стоимостью 85+ млн. как оказалось тоже не редкость, согласно открытым источникам максимальная цена в 2021 году доходила до 800+ млн. в новостройке https://www.fontanka.ru/2021/12/28/70345097/
# Но для более комфортного исследования среднестатистической квартиры, полагаю необходимо сделать выборку в пределах коробки с усами. Уберем 000 нуля, для более удобного восприятия. 
# 
# Итого, мы получаем медиативную стоимость около 4.5 млн., средняя стоимость выше - 5,9, что объясняется наличием многократно более дорогими квартирами, если ниже средней стоимость у нас минимальная планка всего на 5.5 млн. меньше, то верхняя планка на 60+ млн. выше.  
# 

# In[31]:


#print (realty_df_norm ['last_price']. describe ())
# новя переменная которая будет использоваться при работе с гистаграммами, realty_df_norm_correct_price, если
realty_stat = realty_df_norm.query ('(last_price > 350000) & (last_price < 66500000)')

realty_df_norm_correct_price_without_zero = realty_stat ['last_price']/1000
display (realty_stat[realty_stat['locality_name']=="гатчина"].count ())
print (realty_df_norm_correct_price_without_zero.describe ())
realty_df_norm_correct_price_without_zero.plot (y='last_price', kind='hist', xlim=(0,66500), figsize=(5,5), bins=100, legend=False)
plt.title('Цена объектов')
plt.xlabel('цена')
plt.ylabel('количество объявлений')
plt.show()


# ### Срез по средней стоиомсти кв.м.
# Важно оценить не только общую стоимость квартир, но и стоимость за квадратный метр, для оценки оставим те же ограничения с общей стоимости. Средняя стоимость и медиана близки около 98000 и 95000 соответственно, но есть объекты, которые выбиваются по минимуму и максиму значительно, посмотрим, что это. Объект стоимостью 28000000 и средней ценной за кв.м. около 848 тыс. размещен 2 раза, в разные дни. учитывая, что это дубль по своей сути, и то, что он попадает в выброс, его можно удалить из анализа. Объекты с стоимостью за кв.м. около 10000 расположены в поселках, часто в Сланцы, имеют площадь около 50 кв.м. а ценна около 500 тыс., Но из этой выборки выбиваются объекты в Гатчине -  1,45 млн за 138 кв.м. и 850000 за 780000 (исключим их из выборки как и дубли выше) Прим. объекты в Гатчине, согласно открытым источникам, должны стоить почти в 10 раз дороже. https://spb.cian.ru/cat.php?deal_type=sale&engine_version=2&location%5B0%5D=174563&mintarea=100&offer_type=flat
# 
# 

# In[32]:



realty_stat = realty_stat.query('last_price!=1450000 & total_area!=138')
realty_stat = realty_stat.query('ration_price_area < 800000').reset_index (drop=True)
display (realty_stat ['ration_price_area'].describe ())

realty_stat.plot (y='ration_price_area', kind='hist', xlim=(0,500000), figsize=(5,5), bins=100, legend=False)
plt.title('Стоиомсть 1 кв.м.')
plt.xlabel('стоимость')
plt.ylabel('количество объявлений')
plt.show()


# ### количество комнат
# Количество комнат, как и высота потолков, этаж квартиры, тип этажа квартиры и общее количество этажей, будут иметь схожие по структуре команды на построение гистограммы (данные в основном категорийные, кроме высоты потолков, хотя и их можно отнести к таковым, так как диапазон значений, довольно ограничен), значит лучше сделать функцию, которая будет вызывать гистограмму
# 
# В части оценки объектов по количество комнат, учитывая, что самый распространённый вариант - 1 комната, среднее значение как и медианта - 2, так как 2-х комнатных объектов также большое количество, и за счет объектов с большей комнатностью, среднее и медиана в районе 2-х.
# 

# In[33]:


print (realty_stat ['rooms'].describe ())
realty_stat ['rooms'].plot(kind='hist', range=(0,4))
plt.title('Количество комнат')
plt.xlabel('комнаты')
plt.ylabel('количество объявлений')
plt.show()


# ### высота потолков
# медиаативное значение как и среднее соответствует стандартам градостроительства и находится в районе 2.65 м.

# In[34]:


print (realty_stat ['ceiling_height_stat'].describe ())
realty_stat['ceiling_height_stat'].plot(kind='hist', range=(2,5))
plt.title('Срез по высоте потолков')
plt.xlabel('высота потолков')
plt.ylabel('количество объявлений')
plt.show()


# ### Этаж квартиры
# Средняя этажность в районе 6-го этажа, медиана по всем объектам 4-этаж, учитывая, что распространение "хрущовок" которые в основном 5-ти этажные (в районных центрах и остальных поселках), и 9/12 - ти этажные в областных центрах, тенденция понятна. Стоит отметить, что в ДФ присутствуют объекты небольших н.п., которые редко строят выше 5-ти этажей, а также распространение общей тенденции - комфорт-жилья - микрорайоны с этажностью до 9 этажей.

# In[35]:


print (realty_stat ['floor'].describe ())
realty_stat['floor'].plot(kind='hist', range=(0,27))
plt.title('Срез по этажам расположения квартир')
plt.xlabel('этаж')
plt.ylabel('количество объявлений')
plt.show()


# ### Этажность домов
# Анализ и выводы озвученные выше в части характерности застройки в 5/9 этажей находят свое подтверждение в гистаграмме.

# In[36]:


print (realty_stat ['floors_total'].describe ())
realty_stat['floors_total'].plot(kind='hist', range=(0,27))
plt.title('Срез по этажности')
plt.xlabel('этаж')
plt.ylabel('количество объявлений')
plt.show()


# ### Тип этажа
# Вывод очевиден, большинство квартир не, а первом и не на последнем этаже. На что стоит обратить внимание, чуть больше квартир на последнем этаже, чем на первом, во-первых, возможно люди чуть реже выбирают первый этаж, во-вторых, часто на первых этажах находятся организации и коммерческие объекты, и жилые квартиры начинаются со второго.

# In[37]:


realty_stat ['floor_type'].value_counts().sort_index(ascending=True).plot.bar()
plt.title('Типы этажей')

plt.ylabel('количество объявлений')
plt.show()


# ### расстояние до центра города в метрах
# Учитывая значения заглушки (-77) отсекаем при выборке нижние границы уса, и получаем, что большенство объектов находится на расстоянии 13000-14000 м. от центра города, дальше 20000 м. количество объектов падает.

# In[38]:


print (realty_stat ['cityCenters_nearest'].describe ())
distance_without_plugs = realty_stat.query ('cityCenters_nearest>0')

print (distance_without_plugs ['cityCenters_nearest'].describe ())

distance_without_plugs.plot (y='cityCenters_nearest', kind='hist', xlim=(0,40400), figsize=(7,7), bins=25, legend=False)
plt.title('Срез по удаленности объекта от центра')
plt.xlabel('метры')
plt.ylabel('количество объявлений')
plt.show()


# ### расстояние до ближайшего аэропорта.
# До аэропорта по статистике дальше, чем до ближайшего центра города, что и логично, количество аэропортов значительно меньше, далеко не в каждом городе есть аэропорт. в среднем аэропорт находится на удалении 28000 м. от объекта

# In[39]:


print (realty_stat ['airports_nearest'].describe ())
distance_without_plugs = realty_stat.query ('airports_nearest>0')

print (distance_without_plugs ['airports_nearest'].describe ())

distance_without_plugs.plot (y='airports_nearest', kind='hist', xlim=(0,90000), figsize=(5,5), bins=100, range = (0, 200000), legend=False)
plt.title('Срез по удаленности объекта от аэропорта')
plt.xlabel('метры')
plt.ylabel('количество объявлений')
plt.show()


# ### расстояние до ближайшего парка
# В отличии от центра города и аэропортов, парки находятся достаточно близко, в диапазоне 288-612 метров от объекта. Но нужно учитывать, что значение с данными по парке есть только по 30% объектов, хотя можно с уверенностью можно предположить, что такие элементы инфраструктуры достаточно распространены

# In[40]:


print (realty_stat ['parks_nearest'].describe ())
distance_without_plugs = realty_stat.query ('parks_nearest>0')
print (distance_without_plugs ['parks_nearest'].describe ())

distance_without_plugs.plot (y='parks_nearest', kind='hist', xlim=(0,3000), figsize=(5,5), bins=100, range=(0, 4000), legend=False)
plt.title('Срез по удаленности объекта от ближайшего парка')
plt.xlabel('метры')
plt.ylabel('количество объявлений')
plt.show()


# ### день и месяц публикации объявления
# Основные публикации в средине недели, меньше всего в выходные и понедельник
# 
# А выставление объектов на продажу чаще всего в начале года: февраль - апрель, и есть всплеск октябрь-ноябрь. Период февраль - после новогодних праздников, когда много выходных, люди начинают возвращаться в привычный ритм. осень - период после отпусков летних, что также подтверждается снижением количества выставления на продажу летом.
# 

# In[41]:


realty_stat.groupby ('first_weekday').agg('count').sort_values (by='total_area').plot (y='total_area', kind='bar', figsize=(8,5), legend=False)

plt.title('Срез по количеству объявлений по дням недели')
plt.xlabel('Дни недели')
plt.ylabel('Количество объявлений')
plt.show()


# In[42]:


realty_stat.groupby ('month').agg('count').sort_values (by='total_area').plot (y='total_area', kind='bar', figsize=(8,5), legend=False)

plt.title('Срез по количеству объявлений по месяцам')
plt.ylabel('Количество объявлений')
plt.show()


# ## как быстро продавались квартиры (столбец days_exposition). Этот параметр показывает, сколько дней «висело» каждое объявление.
# В основном квартиры продаются в течении 95 дней, среднее время около полугода, быстрыми можно назвать продажи в течении первых 45 дней, а если объявление находится более 7-ми месяцев активным, это уже долгая продажа.

# In[43]:


df_days_exposition = realty_stat.query ('days_exposition >=0')
df_days_exposition ['days_exposition'].hist (figsize=(8,5), bins=100, range=(0,365))
print (df_days_exposition ['days_exposition'].describe())


# ### Какие факторы больше всего влияют на общую (полную) стоимость объекта? 
# Проанализируем, какие факторы больше всего влияют на общую (полную) стоимость объекта. Построим графики, которые покажут корреляцию цены от следующих параметров: 
# общая площадь, 
# жилая площадь, 
# площадь кухни, 
# количество комнат, 
# типа этажа, 
# даты размещения (день недели, месяц, год).

# Корреляция 0,766. Взаимосвязь между стоимость и площадью имеет место быть

# In[44]:


print (realty_stat ['last_price'].corr(realty_stat['total_area']))
print (realty_stat.plot(x='last_price', y='total_area', kind='scatter'))

plt.xlabel('Стоимость')
plt.ylabel('Общая площадь')
plt.show()


# немного хуже прослеживается корреляция между жилой площадью и стоиомостью

# In[45]:


print (realty_stat ['last_price'].corr(realty_stat['living_area'])) 
print (realty_stat.plot(x='living_area', y='last_price', kind='scatter'))
plt.ylabel('Стоимость')
plt.xlabel('Жилая площадь')
plt.show()


# Слабая корреляция между стоиостью и размером кухни присутствует

# In[46]:


print (realty_stat ['last_price'].corr(realty_stat['kitchen_stat_norm']))
print (realty_stat.plot(x='kitchen_stat_norm', y='last_price', kind='scatter')) 
plt.xlabel('Площадь кухни')
plt.ylabel('Стоимость')
plt.show()


# коэффициент корреляции менее 0,5, можем сделать вывод, что она отсуствует

# In[47]:


print (realty_stat ['last_price'].corr(realty_stat['rooms']))
realty_stat.groupby('rooms')['last_price'].median().plot()
plt.xlabel('Комнаты')
plt.ylabel('Стоимость')
plt.show()


# 1-й этаж все же дешевле. Прим. значения не числовые, сделаем сравнение через plt.bar

# In[48]:


plt.bar (realty_stat['floor_type'], realty_stat ['last_price'])


# Явной зависимости нет, но в воскресенье меньше всего поступает на продажу объектов, поэтому просадка

# In[49]:


plt.figure (figsize=(8,8))
plt.bar (realty_stat['first_weekday'], realty_stat ['last_price'])
plt.title('Сравнение количества объявлений по дням публикации')
plt.xlabel('дни недели')
plt.xticks (rotation=45)
plt.show()


# Явной зависимости нет

# In[50]:


plt.bar (realty_stat['month'], realty_stat ['last_price'])


# 

# In[51]:



realty_stat.groupby('year')['last_price'].median().plot()
plt.title('количество объявлений по годам публикации')

plt.xlabel('годы')
plt.show()


# # Можно утверждать, что на стоимость квартиры больше всего влияет общая и жилая площадь 

# ## Оценка средней стоимости по населенным пунктам
# Посчитаем среднюю цену одного квадратного метра в 10 населённых пунктах с наибольшим числом объявлений. 
# Выделим населённые пункты с самой высокой и низкой стоимостью квадратного метра.

# Статистика по самым популярным по количеству объявления попали основные н.п. Ленинградской области, где самое больше количество новостроек. 

# In[52]:


avg_price_local = pd.pivot_table(realty_stat, index='locality_name', values='ration_price_area', aggfunc={'mean', 'count'})
avg_price_local = avg_price_local.sort_values(by='count', ascending=False)
avg_price_local_group = avg_price_local.head(10)
avg_price_local_group


# Самый "дорогой" кв.м. в поселке Лисий нос, а самый "дешевый" в послелке ям-тесово

# In[53]:


avg_price_local_max = avg_price_local_group.sort_values(by='mean', ascending=False)
print (avg_price_local_max.head(1))

avg_price_local_min = avg_price_local_group.sort_values(by='mean', ascending=True)
print (avg_price_local_min.head(1))


# 

# Сделаем выборку по городу, и возьмем среднюю цену каждого километра.
# Можно сделать вывод, что чем ближе, тем дороже

# In[54]:


spb_dist = realty_stat[realty_stat['locality_name'] == 'санкт-петербург'].groupby('cityCenters_averag_km')['last_price'].mean().plot(grid=True, style='o-',figsize = (25, 10))
#


# ## Общий вывод
# 
# 
# В анализируемом DF в основном 1- и 2-комнтаные квартиры, площадью около 50-59 кв.м., с характерной жилой площадью 1-комнатных– 18-20 кв.м. и 2-комнатых – 30 кв.м.
# 
# Основная часть квартир имеет типовую кухню размером 9 кв.м.. Медиативная ценна на такие квартиры в районе 4,5 млн.рублей. 
# 
# Потолки в подавляющем большинстве около 2,65 м., что соответствует нормативам, действующим в стране. Основные объявления относятся к квартирам, которые находятся на 4-6 этажах в среднем в 9-10 этажном здании. Первый и последний этаж популярностью не пользуются. 
# 
# Расстояние до центра города около 13-14 км., что объясняется застройкой спальных районов. 
# 
# До аэропорта около 28 км. 
# 
# До парков достаточно близко, большинство объектов находятся на расстоянии от 288 до 600 метров до ближайшего парка.
# 
# Самые активные периоды создания объявлений февраль-март и октябрь-ноябрь, что можно в некоторой мере обяснить новогодним периодом и периодом отпусков/дач, после которых люди возвращаются в штатный режим жизни не сразу. Схожая ситуация и с днями недели – в основном объявления загружают в середине недели (среда/четверг).
# 
# Средне время продажи – 95 дней, если квартира продается быстрее 45 дней – это очень быстрая продажа, если дольше 158 дней – долгая. 
# 
# Основное влияние на стоимость квартиры оказывает общая площадь и  жилая. А в самом Санкт-Петербурге добавляется еще параметр близости к центру города, что объясняется исторической и туристической привлекательностью квартиры (как развитая инфраструктура, объекты культуры, так и потенциал в части сдачи в аренду).
# 
# Самая большая цена за квадратный метр из Топ-10 по количеству объявлений  - Санкт-Петербург (в среднем 112 533 руб.за 1 кв.м.), а самая низкая в Выборге – 58 242 рублей за 1 кв.м.
# 

# **Чек-лист готовности проекта**
# 
# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  открыт файл
# - [ ]  файлы изучены (выведены первые строки, метод `info()`, гистограммы и т.д.)
# - [ ]  определены пропущенные значения
# - [ ]  заполнены пропущенные значения там, где это возможно
# - [ ]  есть пояснение, какие пропущенные значения обнаружены
# - [ ]  изменены типы данных
# - [ ]  есть пояснение, в каких столбцах изменены типы и почему
# - [ ]  устранены неявные дубликаты в названиях населённых пунктов
# - [ ]  устранены редкие и выбивающиеся значения (аномалии) во всех столбцах
# - [ ]  посчитано и добавлено в таблицу: цена одного квадратного метра
# - [ ]  посчитано и добавлено в таблицу: день публикации объявления (0 - понедельник, 1 - вторник и т.д.)
# - [ ]  посчитано и добавлено в таблицу: месяц публикации объявления
# - [ ]  посчитано и добавлено в таблицу: год публикации объявления
# - [ ]  посчитано и добавлено в таблицу: тип этажа квартиры (значения — «первый», «последний», «другой»)
# - [ ]  посчитано и добавлено в таблицу: расстояние в км до центра города
# - [ ]  изучены и описаны следующие параметры:
#         - общая площадь;
#         - жилая площадь;
#         - площадь кухни;
#         - цена объекта;
#         - количество комнат;
#         - высота потолков;
#         - этаж квартиры;
#         - тип этажа квартиры («первый», «последний», «другой»);
#         - общее количество этажей в доме;
#         - расстояние до центра города в метрах;
#         - расстояние до ближайшего аэропорта;
#         - расстояние до ближайшего парка;
#         - день и месяц публикации объявления
# - [ ]  построены гистограммы для каждого параметра
# - [ ]  выполнено задание: "Изучите, как быстро продавались квартиры (столбец days_exposition). Этот параметр показывает, сколько дней «висело» каждое объявление.
#     - Постройте гистограмму.
#     - Посчитайте среднее и медиану.
#     - В ячейке типа markdown опишите, сколько обычно занимает продажа. Какие продажи можно считать быстрыми, а какие — необычно долгими?"
# - [ ]  выполнено задание: "Какие факторы больше всего влияют на общую (полную) стоимость объекта? Постройте графики, которые покажут зависимость цены от указанных ниже параметров. Для подготовки данных перед визуализацией вы можете использовать сводные таблицы."
#         - общей площади;
#         - жилой площади;
#         - площади кухни;
#         - количество комнат;
#         - типа этажа, на котором расположена квартира (первый, последний, другой);
#         - даты размещения (день недели, месяц, год);
# - [ ]  выполнено задание: "Посчитайте среднюю цену одного квадратного метра в 10 населённых пунктах с наибольшим числом объявлений. Выделите населённые пункты с самой высокой и низкой стоимостью квадратного метра. Эти данные можно найти по имени в столбце `locality_name`."
# - [ ]  выполнено задание: "Ранее вы посчитали расстояние до центра в километрах. Теперь выделите квартиры в Санкт-Петербурге с помощью столбца `locality_name` и вычислите среднюю цену каждого километра. Опишите, как стоимость объектов зависит от расстояния до центра города."
# - [ ]  в каждом этапе есть промежуточные выводы
# - [ ]  есть общий вывод
