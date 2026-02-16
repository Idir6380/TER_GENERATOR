from transformers import  AutoModelForTokenClassification,AutoTokenizer
from preparation_data_train import data
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time

model_name="distilbert-base-cased"
file_name = '/Users/vanessaguerrier/Downloads/projet_TER_M2/data/all_articles.json'


def initialisation(vocab_t,inv_vocab_t,model_name):
    model= AutoModelForTokenClassification.from_pretrained(model_name,num_labels=len(vocab_t),id2label=inv_vocab_t,label2id=vocab_t)
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def perf(model,dataloader_eval):
    model.eval()
    total_loss = 0
    with torch.no_grad(): 
        for batch in dataloader_eval:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader_eval)

def fit(model,dataloader_train,dataloader_eval,epoch):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
    losses_train,losses_eval =[],[]
    for ep in range(epoch):
        model.train()
        total_loss= 0
        for batch in dataloader_train:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader_train)
        loss_eval= perf(model,dataloader_eval)
        losses_train.append(avg_loss)
        losses_eval.append(losses_eval)
        print(f"Epoch {ep+1} — Loss moyenne train : {avg_loss:.4f} — Loss moyenne eval : {loss_eval:.4f}")
    return losses_train,losses_eval


def plot_(epoch, loss_train,loss_eval):
    n_ep= np.arange(epoch)+1
    plt.figure()
    plt.plot(n_ep, loss_train, label="loss_train")
    plt.plot(n_ep, loss_eval, label="loss_eval")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("fig.png")
    plt.close()


tokeniser = AutoTokenizer.from_pretrained(model_name)
dataloader_train,dataloader_eval,_,_,vocab,inv_vocab=data(file_name,tokeniser,300,batch_size=32)

epoch= 10
model= initialisation(vocab,inv_vocab,model_name)
debut= time()

losses_train,losses_eval= fit(model,dataloader_train,dataloader_eval,epoch)
print(f"fin d'entrainement {(time()-debut)/60} minutes")


all_model= {"model":model.state_dict(),"inv_vocab_t":inv_vocab, "vocab_t":vocab,"epoch":epoch,"model_name":model_name,"tokeniser":tokeniser} 
torch.save(all_model,"model.pt")

plot_(epoch,losses_train,losses_eval)

