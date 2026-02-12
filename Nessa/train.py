from transformers import  AutoModelForTokenClassification


def fit(vocab_t,inv_vocab_t):
    model= AutoModelForTokenClassification.from_pretrained("bert-base-cased",num_labels=len(vocab_t),id2label=inv_vocab_t,label2id=vocab_t)