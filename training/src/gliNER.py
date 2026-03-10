from preparation_data_train import *
from gliner import GLiNER

def chunk_text(text, max_chars=800):
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i:i+max_chars])
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

def affichage(all_entities):
    for e in all_entities:
        print(f'{e["text"]} -> {e["label"]}')  




if __name__ == "__main__":
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    labels= [
    "year", "gpu count", "training duration",
    "parameter count", "country", "hardware", "model name",]
    
    for i in range(113):
        file= f"data/GreenMIR/text_xml_nettoyer/article_{i+1}.txt"
        all_entities= prediction(file,model,labels, max_chars=100)
        affichage(all_entities)
        