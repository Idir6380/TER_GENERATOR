import re 
from transformers import AutoTokenizer
import requests
from io import BytesIO
from PyPDF2 import PdfReader

def decomposition_en_phrase(text):
    text = re.sub(r'\s*,\s*', ' , ', text)
    text = re.sub(r'\s*<\s*', ' <', text)
    text = re.sub(r'\s*>\s*', '> ', text)
    text= re.sub(r'\s*([;:])\s*', r' \1 ', text)
    phrases = re.split(r'(?<!\d)[.!?]+(?!\d)', text)
    phrases = [p.strip()+" ." for p in phrases if p.strip()]
    return phrases


def decomposition_en_list_mot(text):  # sourcery skip: for-append-to-extend, inline-immediately-returned-variable, list-comprehension
    phrases=decomposition_en_phrase(text)
    list_mot=[]
    for phrase in phrases:
        list_mot.append(phrase.split())
    return list_mot

def extraire_nom_balise(tag):
    return re.sub(r'[</>]', '', tag)


def labeliser(list_paragraphe):
    features = []
    labels = []
    for phrase in list_paragraphe:
        fe ,la = [],[]
        i = 0
        n = len(phrase)

        while i < n:
            token = phrase[i]
            if token.startswith("<") and not token.startswith("</"):
                nom = extraire_nom_balise(token)
                j = i + 1
                while j < n and not phrase[j].startswith("</"):
                    j += 1
                if j == n:
                    i += 1
                    continue
                taille = j - i - 1
                for k in range(taille):
                    mot = phrase[i + 1 + k]
                    fe.append(mot)
                    if k == 0:
                        la.append(f"B-{nom}")
                    else:
                        la.append(f"I-{nom}")
                i = j + 1
            else:
                fe.append(token)
                la.append("O")
                i += 1
        features.append(fe)
        labels.append(la)
    return features, labels


def read_file_train(namefile):
    text=""
    with open(namefile, "r", encoding="utf-8") as file:
        for line in file:
            if line is not None :
                text += line+" " 
    return text


def read_all(lits_file_name):
    fe,la= [],[]
    for name in lits_file_name:
        text= read_file_train(name)
        text= decomposition_en_list_mot(text)
        features, labels= labeliser(text)
        fe+=features
        la+= labels
    return fe,la

def read_pdf_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/112.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Erreur lors du téléchargement : {response.status_code}")
    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_train_all(lits_file_name):
    fe,la= [],[]
    for name in lits_file_name:
        text= read_file_train(name)
        text= decomposition_en_list_mot(text)
        features, labels= labeliser(text)
        fe+=features
        la+= labels
    return fe,la

def remove_references(text):
    # Cherche le début après 'Abstract'
    abstract_keywords = ["Abstract", "ABSTRACT"]
    start_idx = 0
    for kw in abstract_keywords:
        idx = text.find(kw)
        if idx != -1:
            start_idx = idx + len(kw)
            break  # on prend le premier trouvé

    # Cherche la fin avant 'References' ou 'Bibliography'
    reference_keywords = ["References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY"]
    end_idx = len(text)
    for kw in reference_keywords:
        idx = text.find(kw)
        if idx != -1:
            end_idx = idx
            break
    main_text = text[start_idx:end_idx].strip()
    return main_text

def read_file_test(url):
    fe= []
    text = read_pdf_from_url(url)
    text = remove_references(text)
    text= decomposition_en_list_mot(text)
    fe+= text
    return fe