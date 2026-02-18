import torch 
from transformers import  AutoModelForTokenClassification
from collections import Counter,defaultdict
import sys 
from metric import *
import numpy as np
from preparation_data_train import*

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


def reconstruction(pred_labels,text,word_ids):
    word_to_preds =defaultdict(list)
    for pred, wid in zip(pred_labels, word_ids):
        if wid is None:
            continue
        word_to_preds[wid].append(pred)
    results = []
    for i, word in enumerate(text):
        preds = word_to_preds.get(i, [])
        if not preds:
            results.append("O")
            continue

        most_common_label = Counter(preds).most_common(1)[0][0]
        results.append(most_common_label)
    return results

def recontruction_per_article(model,tokeniser,fe,inv_vocab):
    new_la= []
    for i in range(len(fe)):
        tokens,pred_labels,word_ids=convert (model,tokeniser,fe[i],inv_vocab)
        pred_la= reconstruction(pred_labels,fe[i],word_ids)
        new_la.append(pred_la)
    return new_la

def preformance(model,tokeniser,data):
    true_label,pred_label=[],[]
    for i in range(len(data)):
        fe ,la,_ = decomposition_and_labelisation(data,i)
        la_pred= recontruction_per_article(model,tokeniser,fe,inv_vocab)
        true_label+= la
        pred_label+=la_pred
    p,r,f1= precision_recall_f1(true_label,pred_label)
    precision= p
    recall=r
    f1_score=f1
    return precision ,recall,f1_score


def performance_per_entite(model,tokeniser,data):
    true_label, pred_label= [],[]
    for i in range(len(data)):
        fe ,la,_ = decomposition_and_labelisation(data,i)
        la_pred= recontruction_per_article(model,tokeniser,fe,inv_vocab)
        true_label+= la
        pred_label+=la_pred
    affichage(true_label, pred_label)

data_train,datas_eval,datas_test= read_file_train("/Users/vanessaguerrier/Downloads/projet_TER_M2/data/all_articles.json")
fichier_model= "/Users/vanessaguerrier/Downloads/projet_TER_M2/model1.pt"
all_model = torch.load(fichier_model)
print("CONTENU DU FICHIER MODEL :", all_model.keys())



inv_vocab= all_model["inv_vocab_t"]
vocab= all_model["vocab_t"]
epoch= all_model["epoch"]
model_name= all_model["model_name"]
tokeniser= all_model["tokeniser"]



model = AutoModelForTokenClassification.from_pretrained(model_name,num_labels=len(vocab),id2label=inv_vocab,label2id=vocab)

model.load_state_dict(all_model["model"])  
model.eval()
precision_test ,recall_test,f1_test= preformance(model,tokeniser,datas_test)
precision_ev ,recall_ev,f1_ev= preformance(model,tokeniser,datas_eval)


print(f"train: {len(data_train)} ,test: {len(datas_test)}, eval : {len(datas_eval)}")
print("total corpus:",len(data_train)+len(datas_test)+len(datas_eval) )
print("------------test---------------")
print(f"mean_preci : {np.mean(precision_test):2f} ")
print(f",mean_rapell : {np.mean(recall_test):2f}")
print(f",mean_f1-Score : {np.mean(f1_test):2f}")
print()
performance_per_entite(model,tokeniser,datas_test)
print()
print("------------eval---------------")
print(f"mean_preci : {np.mean(precision_ev):2f} ")
print(f",mean_rapell : {np.mean(recall_ev):2f}")
print(f",mean_f1-Score : {np.mean(f1_ev):2f}")
print()
performance_per_entite(model,tokeniser,datas_eval)