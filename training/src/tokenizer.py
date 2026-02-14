import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=16):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words = self.sentences[idx]
        labels = self.labels[idx]
        tokenized_inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        aligned_labels = []
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  
            else:
                aligned_labels.append(labels[word_idx])
        
        item = {key: val.squeeze(0) for key, val in tokenized_inputs.items()}
        item['labels'] = torch.tensor(aligned_labels)
        
        return item

