import torch 
import torch.nn as nn 

from transformers import AutoModel
from types import SimpleNamespace

class SciBERTNER(nn.Module):

    def __init__(self, n_finetune_layers, model_name = 'allenai/scibert_scivocab_cased', num_labels=10, layer_mode='L12'):

        super(SciBERTNER, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.n_finetune_layers = n_finetune_layers
        self.layer_mode = layer_mode

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        

        # Freez layers 
        for param in self.bert.parameters():
            param.requires_grad = False 
        
        # unfreez 
        if n_finetune_layers > 0:
            for layer in self.bert.encoder.layer[-self.n_finetune_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        if self.layer_mode == 'L12':
            representation = hidden_states[12]
        elif self.layer_mode == 'L10':
            representation = hidden_states[10]
        elif self.layer_mode == 'L8':
            representation = hidden_states[8]
        elif self.layer_mode == 'AVG':
            representation = torch.stack([hidden_states[i] for i in range(9, 13)], dim=0).mean(dim=0)
        else:
            raise ValueError(f"layer_mode must be L12, L10, L8 or AVG, got {self.layer_mode}")

        loss = None
        representation = self.dropout(representation)
        logits = self.classifier(representation)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return SimpleNamespace(loss=loss, logits=logits)