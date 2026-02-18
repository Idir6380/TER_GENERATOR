import re 
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from collections import Counter
import json
from tokenizer import *
from torch.utils.data import  DataLoader
from transformers import  DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split

def decomposition_en_phrase(text):
    text = re.sub(r'\s*,\s*', ' , ', text)
    text = re.sub(r'\s*<\s*', ' <', text)
    text = re.sub(r'\s*>\s*', '> ', text)
    text= re.sub(r'\s*([;:])\s*', r' \1 ', text)
    phrases = re.split(r'(?<!\d)[.!?]+(?!\d)', text)
    phrases = [p.strip()+" ." for p in phrases if p.strip()]
    return phrases


def decomposition_en_list_mot(text,idx):  # sourcery skip: for-append-to-extend, inline-immediately-returned-variable, list-comprehension
    phrases=decomposition_en_phrase(text)
    list_mot=[]
    for phrase in phrases:
        list_mot.append(phrase.split())
    id_paragraphe= [idx]*len(list_mot)
    return list_mot,id_paragraphe

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
    with open(namefile, 'r', encoding='utf-8') as fichier:
        datas = json.load(fichier)
    n= len(datas)
    n_test,n_eval= int(n*0.1),int(n*0.2)
    datas_train,datas_eval,datas_test=[],[],[]
    j=0
    while j < n:
        if len(datas_train)< n-n_eval-n_test:
            datas_train.append(datas[j])
            j+=1
        if len(datas_test)< n_test:
            datas_test.append(datas[j])
            j+=1
        if len(datas_eval)< n_eval:
            datas_eval.append(datas[j])
            j+=1
    return datas_train,datas_eval,datas_test

def decomposition_and_labelisation(data,id_article):
    text= data[id_article]["article"]
    id_paragraphe= data[id_article]["metadata"]["article_number"]
    text,id_para= decomposition_en_list_mot(text,id_paragraphe)
    features, labels= labeliser(text)
    return features, labels,id_para


def read_all(datas):
    fe,la,doc_id= [],[],[]
    for article in range (len(datas)):
        features, labels,idx=decomposition_and_labelisation(datas,article)
        fe+=features
        la+= labels
        doc_id+= idx
    return fe,la,doc_id

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


def remove_references(text):
    # sourcery skip: inline-immediately-returned-variable
    # Cherche le début après 'Abstract'
    abstract_keywords = ["Abstract", "ABSTRACT"]
    start_idx = 0
    for kw in abstract_keywords:
        idx = text.find(kw)
        if idx != -1:
            start_idx = idx + len(kw)
            break 
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


def create_vocab(labels):
    list_global= []
    i= 0
    vocab={}
    for li in labels:
        for mot in li:
            if mot not in vocab:
                vocab[mot]=i
                i+=1
    inv_vocab = {i: mot for mot, i in vocab.items()}
    return vocab,inv_vocab


def dataloader(feature,labels,doc_id,tokenizer,batch_size=32,train=True):
    dataset = NERDataset(feature, labels,doc_id, tokenizer,train=train)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)   

def split_train_eval_test(feature,label):
    fe_train, fe_test, la_train, la_test = train_test_split(feature,label,test_size=0.3,random_state=42)
    fe_eval, fe_test, la_eval, la_test = train_test_split(fe_test,la_test,test_size=0.4,random_state=42)
    return fe_train,fe_eval,fe_test,la_train,la_eval,la_eval

def data(file_name,tokenizer,batch_size=32):
    datas_train,datas_eval,datas_test= read_file_train(file_name)
    print("total corpus:",len(datas_train)+len(datas_test)+len(datas_eval) )
    fe_train,la_train,doc_ids = read_all(datas_train)
    fe_eval,la_eval,doc_id_ev=read_all( datas_eval)
    vocab,inv_vocab= create_vocab(la_train)
    labels_ids = [[vocab[l] for l in sent] for sent in la_train]
    labels_ids_e = [[vocab[l] for l in sent] for sent in la_eval]
    dataloader_train= dataloader(fe_train,labels_ids,doc_ids,tokenizer,batch_size=batch_size)
    dataloader_eval= dataloader(fe_eval,labels_ids_e,doc_id_ev,tokenizer,batch_size=10,train=False)
    return dataloader_train,dataloader_eval,vocab,inv_vocab

