import torch
import torch.nn as nn
from transformers import AutoModel,AutoTokenizer
from preparation_data_train import * 
from time import time

class BertForTokenClassificationCustom(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.1):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        for param in self.bert.parameters():
            param.requires_grad = False 
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):   
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )
        return logits, loss


def evaluate_loss(model, dataloader_eval):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader_eval:
            logits, loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            total_loss += loss.item()
    return total_loss / len(dataloader_eval)

def fine_tune_custom_bert( dataloader_train,dataloader_eval,num_labels,model_name="bert-base-cased",epochs=3,lr=2e-5):
    model = BertForTokenClassificationCustom(model_name, num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader_train:
            logits, loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader_train)
        eval_loss = evaluate_loss(model, dataloader_eval)
        print(f"epoch:{epoch} , training loss: {avg_loss}, eval_loss : {eval_loss }")
    return model

if __name__ == "__main__":
    model_name="bert-base-cased"
    file_name = 'data/all_articles.json'
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    dataloader_train,dataloader_eval,vocab,inv_vocab=data(file_name,tokeniser,batch_size=32,model1=False)
    num_labels= len(vocab)
    epochs= 10
    debut= time()

    model= fine_tune_custom_bert( dataloader_train,dataloader_eval,num_labels,model_name=model_name,epochs=epochs)
    print(f"fin d'entrainement {(time()-debut)/60} minutes")


    all_model= {"model":model.state_dict(),"inv_vocab_t":inv_vocab, "vocab_t":vocab,"epoch":epochs,"model_name":model_name,"tokeniser":tokeniser} 
    torch.save(all_model,"model2.pt")
