import torch 
from transformers import  AutoModelForTokenClassification
from collections import Counter,defaultdict
import sys 
from metric import *
import numpy as np
from preparation_data_train import*
from train2 import *

def predict(model,tokeniser,text):
    inputs = tokeniser(
            text,
            is_split_into_words=True,
            truncation=True,
            return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return torch.argmax(logits, dim=-1),inputs


def convert (model,tokeniser,text,inv_vocab):
    predictions,inputs= predict(model,tokeniser,text)
    tokens = tokeniser.convert_ids_to_tokens(inputs["input_ids"][0])
    pred_labels = [inv_vocab[p.item()] for p in predictions[0]] 
    return tokens,pred_labels,inputs.word_ids()


def reconstruction_per_sentence(pred_labels, text, word_ids):
    word_to_pred = {}

    for pred, wid in zip(pred_labels, word_ids):
        if wid is None:
            continue
        if wid not in word_to_pred:
            word_to_pred[wid] = pred
    results = []
    for i in range(len(text)):
        if i in word_to_pred:
            results.append(word_to_pred[i])
        else:
            results.append("O")
    return results

def reconstruction(model,tokeniser,fe,inv_vocab):
    new_la= []
    for i in range(len(fe)):
        tokens,pred_labels,word_ids=convert (model,tokeniser,fe[i],inv_vocab)
        pred_la= reconstruction_per_sentence(pred_labels,fe[i],word_ids)
        new_la.append(pred_la)
    return new_la

def performance(model,tokeniser,data,inv_vocab):
    true_label,pred_label=[],[]
    for i in range(len(data)):
        fe ,la,_ = decomposition_and_labelisation(data,i)
        la_pred= reconstruction(model,tokeniser,fe,inv_vocab)
        true_label+= la
        pred_label+=la_pred
    p,r,f1= precision_recall_f1(true_label,pred_label)
    precision= p
    recall=r
    f1_score=f1
    return precision ,recall,f1_score


def join_all(fe):
    feature=[]
    for i in range (len(fe)):
        feature.extend(fe[i])
    return feature


def performance_per_article(model,tokeniser,data,inv_vocab):
    feature,label,label_pred=[],[],[]
    for i in range(len(data)):
        fe ,la,_ = decomposition_and_labelisation(data,i)
        f= join_all(fe)
        l= join_all(la)
        la_pred= reconstruction(model,tokeniser,[f],inv_vocab)
        feature.append(f)
        label.append(l)
        label_pred.append(la_pred[0])
    
    p,r,f1= precision_recall_f1(label,label_pred)
    precision= p
    recall=r
    f1_score=f1
    return precision ,recall,f1_score


def performance_per_entite(model,tokeniser,data,inv_vocab):
    true_label, pred_label= [],[]
    for i in range(len(data)):
        fe ,la,_ = decomposition_and_labelisation(data,i)
        la_pred= reconstruction(model,tokeniser,fe,inv_vocab)
        true_label+= la
        pred_label+=la_pred
    affichage(true_label, pred_label)


def initialisation_for_test(global_path,mod="model.pt",model=1):
    fichier_model= f"{global_path}{mod}"
    all_model = torch.load(fichier_model)
    print("CONTENU DU FICHIER MODEL :", all_model.keys())
    inv_vocab= all_model["inv_vocab_t"]
    vocab= all_model["vocab_t"]
    model_name= all_model["model_name"]
    tokeniser= all_model["tokeniser"]
    if model== 1:
        model = AutoModelForTokenClassification.from_pretrained(model_name,num_labels=len(vocab),id2label=inv_vocab,label2id=vocab)
    else: 
        model= BertForTokenClassificationCustom(model_name=model_name,num_labels=len(vocab))
    model.load_state_dict(all_model["model"]) 
    return model, tokeniser,inv_vocab 
    
    
if __name__ == "__main__":
    data_train,datas_eval,datas_test= read_file_train("/Users/vanessaguerrier/Downloads/projet_TER_M2/data/all_articles.json")
    model,tokeniser,inv_vocab= initialisation_for_test("/Users/vanessaguerrier/Downloads/projet_TER_M2/",mod="model1.pt",model=1)
    model.eval()
    precision_test ,recall_test,f1_test= performance(model,tokeniser,datas_test,inv_vocab)
    precision_ev ,recall_ev,f1_ev= performance(model,tokeniser,datas_eval,inv_vocab)


    print(f"train: {len(data_train)} ,test: {len(datas_test)}, eval : {len(datas_eval)}")
    print("total corpus:",len(data_train)+len(datas_test)+len(datas_eval) )
    print("------------test---------------")
    print(f"mean_preci : {np.mean(precision_test):2f} ")
    print(f",mean_rapell : {np.mean(recall_test):2f}")
    print(f",mean_f1-Score : {np.mean(f1_test):2f}")
    print()
    print("------------eval---------------")
    print(f"mean_preci : {np.mean(precision_ev):2f} ")
    print(f",mean_rapell : {np.mean(recall_ev):2f}")
    print(f",mean_f1-Score : {np.mean(f1_ev):2f}")
    print()
