import numpy as np
import re
import json
import contextlib
import pycountry
import geonamescache

gc = geonamescache.GeonamesCache()
cities = gc.get_cities()


CITY_TO_COUNTRY = {}
for city in cities.values():
    city_name = city["name"].lower()
    country_code = city["countrycode"]
    country = pycountry.countries.get(alpha_2=country_code)
    if country:
        CITY_TO_COUNTRY [city_name] = country.name

with open("/Users/vanessaguerrier/Downloads/projet_TER_M2/data/nombre _anglais.json","r",encoding="utf-8") as f:
    DATA = json.load(f)

EVAL_FIELDS = ["year", "gpu_count", "country", "parameter_count", "training_duration"]


def normalize_predictions(predicted_words):
    output = []

    for word in predicted_words:
        w = word.strip().lower()
        with contextlib.suppress(LookupError):
            country = pycountry.countries.lookup(word)
            output.append(country.name)
            continue
        if w in CITY_TO_COUNTRY:
            output.append(CITY_TO_COUNTRY[w])
            continue
        else: 
            output.append(word)
    return np.unique(np.array(output))


def traitement_country(country):
    if len(country)==0:
        return None
    country = normalize_predictions(country)
    return (",").join(country[i] for i in range(len(country)))


def convert_params_to_int(param_str):
    if not isinstance(param_str, str):
        return param_str
    param_str = param_str.strip().lower()
    if param_str.isdigit():
        return int(param_str)
    match = re.match(r"([\d\.]+)\s*(k|m|b|thousand|million|billion)?", param_str)
    if not match:
        return param_str
    number = float(match.group(1))
    unit = match.group(2)
    multipliers = {
        "k": 10**3,
        "thousand": 10**3,
        "m": 10**6,
        "million": 10**6,
        "b": 10**9,
        "billion": 10**9,
        None: 1
    }
    return int(number * multipliers[unit])


def converte_hour(time_string):
    return DATA[time_string] if time_string in DATA.keys() else time_string


def texte_table(tab):
    tab_array = np.array(tab)
    if type(tab_array) == int:
        return np.sum(tab_array)
    val=[int(tab_array[i]) for i in range(len(tab_array)) if str(tab_array[i]).isdigit() ]
    return np.sum(np.array(val))
    
    
def convert_string_to_hours(time_string):
    if not isinstance(time_string, str):
        return time_string
    time_string = time_string.strip().lower()
    if re.fullmatch(r"\d+", time_string):
        return time_string
    if re.fullmatch(r"[a-zA-Z]+", time_string):
        return converte_hour(time_string)
    year_match  = re.search(r'(\d+)\s*years?', time_string)
    month_match = re.search(r'(\d+)\s*months?', time_string)
    week_match  = re.search(r'(\d+)\s*weeks?', time_string)
    day_match   = re.search(r'(\d+)\s*days?', time_string)

    years  = int(year_match.group(1)) if year_match else 0
    months = int(month_match.group(1)) if month_match else 0
    weeks  = int(week_match.group(1)) if week_match else 0
    days   = int(day_match.group(1)) if day_match else 0

    total_days = years * 365 + months * 30 + weeks * 7 + days

    return total_days * 24


def traitement_year(year):
    if len(year)==0:
        return None
    elif len(year)==1:
        return year[0]
    else:
        val=[int(mot) for mot in year if mot.isdigit()]
        return str(max(val))
    
def convert_gpu_count(gpu_count_str):
    if not isinstance(gpu_count_str, str):
        return gpu_count_str
    elif gpu_count_str in ["[cls]", "[sep]", "", "none"]:
        return gpu_count_str
    elif not gpu_count_str.isdigit() :
        return gpu_count_str
    else :
        return int(gpu_count_str)

def traitement_(gpu_count,fonction=convert_gpu_count):
    if len(gpu_count)==0:
        return None
    elif len(gpu_count)==1:
        return fonction(gpu_count[0])
    else:
        val= []
        for mot in gpu_count:
            conv= fonction(mot)
            if conv is not None:
                val.append(conv)
        return texte_table(val)
    
def traiter_one(article):
    dic= {}
    year= traitement_year(article["information"]["year"])
    if year is not None :
        dic["year"]= year
    gpu_count=traitement_(article["information"]["gpu_count"])
    if gpu_count is not None :
        dic["gpu_count"]= gpu_count
    country=traitement_country(article["information"]["country"])
    if country is not None :
        dic["country"]= country
    hardrew=traitement_country(article["information"]["country"])
    if hardrew is not None :
        dic["hardrew"]= hardrew
    parameter_count= traitement_(article["information"]["parameter_count"],convert_params_to_int)
    if parameter_count is not None :
        dic["parameter_count"]= parameter_count
    training_duration= traitement_(article["information"]["training_duration"],convert_string_to_hours)
    if training_duration is not None :
        dic["training_duration"]= training_duration
    return dic


def traiter_alllire_json(file_path):
    dic= {}
    with open(file_path, 'r', encoding='utf-8') as fichier:
        datas = json.load(fichier)
    for article in datas:
        id = article[ "article_id"]
        dic[id]=traiter_one(article)
    return dic

