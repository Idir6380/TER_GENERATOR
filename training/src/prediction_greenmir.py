from preparation_data_train import *
from pred import *
import json

def create_path(global_path,i):
    return f"{global_path}data/GreenMIR/text_xml_nettoyer/article_{i+1}.txt"


def lire_et_nettoyage_file(file_name):
    with open(file_name,"r",encoding="utf-8")as f:
        contenu = f.read()
    return create_feature_test(contenu)


def create_dic(feature, labels, id):
    dic = {
        "article_id": id+1,
        "information": {
            "model_name": [],
            "parameter_count": [],
            "gpu_count": [],
            "hardware": [],
            "training_duration": [],
            "country": [],
            "year": []
        }
    }
    mapping = {
        "model": "model_name",
        "params": "parameter_count",
        "gpu_count": "gpu_count",
        "hardware": "hardware",
        "training": "training_duration",
        "country": "country",
        "year": "year"
    }
    current_entity = ""
    current_key = None
    for j, sentence in enumerate(labels):
        for i in range(len(sentence)):
            tag = sentence[i]
            if tag == "O":
                if current_entity and current_key:
                    dic["information"][current_key].append(current_entity.strip())
                    current_entity = ""
                    current_key = None
                continue
            pref, syn = tag.split("-", 1)
            value = feature[j][i]
            key = mapping.get(syn)
            if not key:
                continue
            if pref == "B":
                if current_entity and current_key:
                    dic["information"][current_key].append(current_entity.strip())
                current_entity = value
                current_key = key
            elif pref == "I" and current_key == key:
                current_entity += " " + value
        if current_entity and current_key:
            dic["information"][current_key].append(current_entity.strip())
            current_entity = ""
            current_key = None
    return dic

def deduplicate_information(dic):
    new_dic = {
        "article_id": dic["article_id"],
        "information": {}
    }
    for key, values in dic["information"].items():
        seen = set()
        unique_values = []
        for v in values:
            if v not in seen:
                seen.add(v)
                unique_values.append(v)
        new_dic["information"][key] = unique_values
    return new_dic

def predict_label_greenmir(global_path):
    list_valeur_predict= []
    model, tokeniser,inv_vocab= initialisation_for_test(global_path)
    model.eval()
    for i in range(113):
        file=create_path(global_path,i)
        features= lire_et_nettoyage_file(file)
        labels_pred= recontruction(model,tokeniser,features,inv_vocab)
        dic= create_dic(features, labels_pred, i)
        list_valeur_predict.append(deduplicate_information(dic))
    return list_valeur_predict

# def predict_label_greenmir_per_article(global_path):
#     list_valeur_predict= []
#     model, tokeniser,inv_vocab= initialisation_for_test(global_path,model="model1.pt")
#     model.eval()
#     for i in range(113):
#         file=create_path(global_path,i)
#         features= lire_et_nettoyage_file(file)
#         features= join_all(features)
#         labels_pred=  reconstruction_per_sentence(model,tokeniser,[features],inv_vocab)
#         labels_pred=labels_pred[0]
#         dic= create_dic(features, labels_pred, i)
#         list_valeur_predict.append(deduplicate_information(dic))
#     return list_valeur_predict
        

global_path="/Users/vanessaguerrier/Downloads/projet_TER_M2/"
valeur= predict_label_greenmir(global_path)
with open( "/Users/vanessaguerrier/Downloads/projet_TER_M2/data/GreenMIR/greenmir_pred_1.json", 'w', encoding='utf-8') as fichier:
    json.dump(valeur, fichier, ensure_ascii=False, indent=4)