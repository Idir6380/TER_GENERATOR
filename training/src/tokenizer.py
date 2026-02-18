import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, sentences, labels, doc_ids, tokenizer,train= True):
        self.sentences = sentences
        self.labels = labels
        self.doc_ids = doc_ids
        self.tokenizer = tokenizer
        self.train = train
        

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):

        target_words = self.sentences[idx]
        target_labels = self.labels[idx]
        if self.train :
            if idx == 0 or self.doc_ids[idx] != self.doc_ids[idx - 1]:
                context_words = []
                context_labels = []
            else:
                context_words = self.sentences[idx - 1]
                context_labels = [-100] * len(context_words)
            
            all_words = context_words + target_words
            all_labels = context_labels + target_labels
        else: 
            all_words = target_words
            all_labels = target_labels
        
        tokenized_inputs = self.tokenizer(
            all_words,
            is_split_into_words=True,
            truncation=True,
            padding= 'max_length',
            return_tensors="pt"
        )

        
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        aligned_labels = []
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(all_labels[word_idx])

        item = {key: val.squeeze(0) for key, val in tokenized_inputs.items()}
        item['labels'] = torch.tensor(aligned_labels)

        return item