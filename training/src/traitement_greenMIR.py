
import re
import json
EVAL_FIELDS = ["year", "gpu_count", "country", "parameter_count", "training_duration"]
def traitement_country(country):
    if len(country)==0:
        return None
    elif len(country)==1:
        return country[0]
    else:
        return (",").join(country[i] for i in range(len(country)))


def convert_params_to_int(param_str):
    if not isinstance(param_str, str):
        return None
    param_str = param_str.strip().lower()
    if param_str.isdigit():
        return int(param_str)
    match = re.match(r"([\d\.]+)\s*(k|m|b|thousand|million|billion)?", param_str)
    if not match:
        raise ValueError(f"Format invalide : {param_str}")
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


def convert_string_to_hours(time_string):
    if not isinstance(time_string, str):
        return None
    time_string = time_string.strip().lower()
    if re.fullmatch(r"\d+", time_string):
        return None
    if re.fullmatch(r"[a-zA-Z]+", time_string):
        return None
    years = 0
    months = 0
    days = 0
    year_match = re.search(r'(\d+)\s*year', time_string)
    month_match = re.search(r'(\d+)\s*month', time_string)
    day_match = re.search(r'(\d+)\s*day', time_string)
    if year_match:
        years = int(year_match.group(1))
    if month_match:
        months = int(month_match.group(1))
    if day_match:
        days = int(day_match.group(1))
    if years == 0 and months == 0 and days == 0:
        return None
    total_days = years * 365 + months * 30 + days
    total_hours = total_days * 24
    return total_hours


def traitement_year(year):
    if len(year)==0:
        return None
    elif len(year)==1:
        return year[0]
    else:
        val=[int(mot) for mot in year]
        return str(max(val))
    
def convert_gpu_count(gpu_count_str):
    if not isinstance(gpu_count_str, str):
        return None
    elif gpu_count_str in ["[cls]", "[sep]", "", "none"]:
        return None
    elif not gpu_count_str.isdigit() :
        return None
    else :
        return int(gpu_count_str)

def traitement_(gpu_count,fonction=convert_gpu_count):
    if len(gpu_count)==0:
        return None
    elif len(gpu_count)==1:
        print(gpu_count[0])
        return fonction(gpu_count[0])
    else:
        val= []
        for mot in gpu_count:
            conv= fonction(mot)
            if conv is not None:
                val.append(conv)
        return max(val, default=None)
    
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

