from preparation_data_train import *
from gliner import GLiNER
from prediction_greenmir import deduplicate_information

def chunk_text(text, max_chars=800):
    chunks = []
    current = ""
    for p in text:
        if len(current) + len(p) <= max_chars:
            current += " " + p
        else:
            chunks.append(current.strip())
            current = p
    if current:
        chunks.append(current.strip())
    return chunks

def lecture_and_chunks(file, max_chars=800):
    with open(file,"r",encoding="utf-8")as f:
        contenu = f.read()
    text= decomposition_en_phrase(contenu)
    chunks= chunk_text(text,max_chars=max_chars) 
    return chunks

def prediction(file,model,labels, max_chars=800):
    all_entities = []
    chunks =lecture_and_chunks(file, max_chars=max_chars)
    for chunk in chunks:
        ents = model.predict_entities(chunk, labels)
        all_entities.extend(ents)
    return all_entities 

def affichage(all_entities,id):
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
    for e in all_entities:
        if e["label"] == "year":
            dic["information"]["year"].append( e["text"])
        elif e["label"] == "country":
            dic["information"]["country"].append( e["text"])
        elif e["label"] == "hardware":
            dic["information"]["hardware"].append( e["text"])
        elif e["label"] == "training duration":
            dic["information"]["training_duration"].append( e["text"])
        elif e["label"] == "gpu count":
            dic["information"]["gpu_count"].append( e["text"])
        elif e["label"] == "model name":
            dic["information"]["model_name"].append( e["text"])
        elif e["label"] == "parameter count":
            dic["information"]["parameter_count"].append( e["text"])
    dic= deduplicate_information(dic)
    return dic




if __name__ == "__main__":
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    labels= [
    "year", "gpu count", "training duration",
    "parameter count", "country", "hardware", "model name",]
    valeur =[]
    for i in range(113):
        file= f"data/GreenMIR/text_xml_nettoyer/article_{i+1}.txt"
        all_entities= prediction(file,model,labels, max_chars=100)
        dic = affichage(all_entities,i)
        print(dic)
        valeur.append(dic)
    with open( "/Users/vanessaguerrier/Downloads/projet_TER_M2/data/GreenMIR/greenmir_pred_2.json", 'w', encoding='utf-8') as fichier:
        json.dump(valeur, fichier, ensure_ascii=False, indent=4)