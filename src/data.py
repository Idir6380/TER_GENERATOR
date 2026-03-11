import json
import re
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split


class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        word_labels = self.labels[idx]

        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        word_ids = tokenized.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(word_labels[word_id])
            else:
                aligned_labels.append(-100)
            prev_word_id = word_id

        item = {key: val.squeeze(0) for key, val in tokenized.items()}
        item['labels'] = torch.tensor(aligned_labels)
        return item


def load_articles(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def split_into_sentences(text):
    text = re.sub(r'\s*,\s*', ' , ', text)
    text = re.sub(r'\s*<\s*', ' <', text)
    text = re.sub(r'\s*>\s*', '> ', text)
    text = re.sub(r'\s*([;:)(.])\s*', r' \1 ', text)
    phrases = re.split(r'(?<!\d)[.!?]+(?!\d)', text)
    phrases = [p.strip() + " ." for p in phrases if p.strip()]
    return phrases


def xml_to_bio(sentence):
    tokens, labels = [], []
    words = sentence.split()
    i, n = 0, len(words)

    while i < n:
        word = words[i]
        if word.startswith("<") and not word.startswith("</"):
            tag = re.sub(r'[</>]', '', word)
            j = i + 1
            while j < n and not words[j].startswith("</"):
                j += 1
            if j == n:
                i += 1
                continue
            for k in range(j - i - 1):
                tokens.append(words[i + 1 + k])
                labels.append(f"B-{tag}" if k == 0 else f"I-{tag}")
            i = j + 1
        else:
            tokens.append(word)
            labels.append("O")
            i += 1

    return tokens, labels


def build_vocab(all_labels):
    all_unique = sorted(set(label for labels in all_labels for label in labels))
    vocab = {label: i for i, label in enumerate(all_unique)}
    inv_vocab = {i: label for label, i in vocab.items()}
    return vocab, inv_vocab


def build_windows(articles_sentences, articles_labels, vocab, context_size):
    all_sentences, all_label_ids = [], []
    
    for art_sents, art_labels in zip(articles_sentences, articles_labels):
        for i in range(len(art_sents)):
            if context_size == 0:
                all_sentences.append(art_sents[i])
                all_label_ids.append([vocab[label] for label in art_labels[i]])
            else:
                start = max(0, i - context_size)
                end = min(len(art_sents), i+context_size+1)

                window_tokens, window_labels = [], []

                for j in range(start, end):
                    window_tokens.extend(art_sents[j])
                    if j == i:
                        window_labels.extend([vocab[label] for label in  art_labels[j]])
                    else :
                        window_labels.extend([-100] * len(art_sents[j]))
                all_sentences.append(window_tokens)
                all_label_ids.append(window_labels)
    return all_sentences, all_label_ids

def get_dataloaders(file_path, tokenizer, batch_size=16, test_size=0.1, eval_size=0.1, context_size=0, split_mode='article'):
    articles = load_articles(file_path)

    #all_sentences, all_labels = [], []
    """
    aritcles_sentences, contains the sentences of each article, 
    and articles_labels contains the corresponding BIO labels for each sentence.
    all_labels_strs is a list of all unique label strings found in the dataset, 
    which is used to build the vocabulary for label encoding.
    """
    articles_sentences, articles_labels, all_label_strs = [], [], []

    for article in articles:
        text = article["article"]
        
        art_sents = []
        art_labels_strs = []
        
        for sentence in split_into_sentences(text):
            tokens, labels = xml_to_bio(sentence)
            if tokens:
                art_sents.append(tokens)
                art_labels_strs.append(labels)
                all_label_strs.extend(labels)
        if art_sents:
            articles_sentences.append(art_sents)
            articles_labels.append(art_labels_strs)

    vocab, inv_vocab = build_vocab([labels for art in articles_labels for labels in art])

    if split_mode == 'sentence':
        # Sentence split (ancien comportement — data leakage possible)
        all_sentences, all_label_ids = build_windows(articles_sentences, articles_labels, vocab, context_size)
        sentences_train, sentences_test, labels_train, labels_test = train_test_split(
            all_sentences, all_label_ids, test_size=test_size, random_state=42
        )
        sentences_train, sentences_eval, labels_train, labels_eval = train_test_split(
            sentences_train, labels_train, test_size=eval_size, random_state=42
        )
    else:
        # Article split (pas de data leakage)
        art_idx = list(range(len(articles_sentences)))
        idx_train, idx_test = train_test_split(art_idx, test_size=test_size, random_state=42)
        idx_train, idx_eval = train_test_split(idx_train, test_size=eval_size, random_state=42)

        def _build(idx):
            sents = [articles_sentences[i] for i in idx]
            labs = [articles_labels[i] for i in idx]
            return build_windows(sents, labs, vocab, context_size)

        sentences_train, labels_train = _build(idx_train)
        sentences_eval, labels_eval = _build(idx_eval)
        sentences_test, labels_test = _build(idx_test)


    collator = DataCollatorForTokenClassification(tokenizer)

    train_loader = DataLoader(NERDataset(sentences_train, labels_train, tokenizer), batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader  = DataLoader(NERDataset(sentences_eval,  labels_eval,  tokenizer), batch_size=batch_size, collate_fn=collator)
    test_loader  = DataLoader(NERDataset(sentences_test,  labels_test,  tokenizer), batch_size=batch_size, collate_fn=collator)

    print(f"Train: {len(sentences_train)} phrases | Eval: {len(sentences_eval)} | Test: {len(sentences_test)}")
    print(f"Vocab size: {len(vocab)} labels → {list(vocab.keys())}")

    return train_loader, eval_loader, test_loader, vocab, inv_vocab

