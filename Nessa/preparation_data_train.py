import re 
from transformers import AutoTokenizer


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


def labeliser(list_phrase):
    features = []
    labels = []
    for phrase in list_phrase:
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
