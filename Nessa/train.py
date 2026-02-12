from transformers import  AutoModelForTokenClassification,AutoTokenizer
from preparation_data_train import data
import json

model_name="bert-base-cased"
file_name = '/Users/vanessaguerrier/Downloads/M2_TER/data/all_articles.json'
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataloader_train,dataloader_eval,fe_test,la_test,vocab,inv_vocab=data(file_name,tokenizer,300,batch_size=32)
print(len(dataloader_eval),len(dataloader_train))
def fit(vocab_t,inv_vocab_t):
    model= AutoModelForTokenClassification.from_pretrained(num_labels=len(vocab_t),id2label=inv_vocab_t,label2id=vocab_t)
    
    

