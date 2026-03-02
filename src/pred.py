import torch 
from transformers import AutoTokenizer
from model import SciBERTNER


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(check_point_path):
    chkpt = torch.load(check_point_path, map_location=DEVICE, weights_only=False)

    model = SciBERTNER(
        n_finetune_layers=chkpt['n_finetune_layers'],
        model_name=chkpt['model_name'],
        num_labels=len(chkpt['vocab_t']),
        layer_mode=chkpt['layer_mode']
    ).to(DEVICE)

    model.load_state_dict(chkpt['model'])
    model.eval()

    tokenizer=AutoTokenizer.from_pretrained(chkpt['model_name'])
    inv_vocab=chkpt['inv_vocab_t']
    context_size=chkpt.get('context_size', 0)

    return model, tokenizer, inv_vocab, context_size

def predict_article(model, tokenizer, sentences, inv_vocab, context_size):
    all_labels = []

    for i in range(len(sentences)):
        start = max(0, i - context_size)
        end = min(len(sentences), i+context_size+1)

        window_words = []
        target_start = 0

        for j in range(start, end):
            if j == i:
                target_start = len(window_words)
            window_words.extend(sentences[j])
        target_end = target_start + len(sentences[i])

        tokenized = tokenizer(
            window_words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        input_ids      = tokenized['input_ids'].to(DEVICE)
        attention_mask = tokenized['attention_mask'].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        preds    = outputs.logits.argmax(dim=-1)[0]
        word_ids = tokenized.word_ids(batch_index=0)

        # premier sous-mot de chaque mot → label
        word_to_pred = {}
        prev_word_id = None
        for tok_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                word_to_pred[word_id] = inv_vocab[preds[tok_idx].item()]
            prev_word_id = word_id

        # garder uniquement les labels de la phrase cible
        sent_labels = [word_to_pred.get(w, 'O') for w in range(target_start, target_end)]
        all_labels.append(sent_labels)

    return all_labels


TAG_TO_FIELD = {
    'country':   'country',
    'gpu_count': 'gpu_count',
    'hardware':  'hardware',
    'model':     'model_name',
    'params':    'parameter_count',
    'training':  'training_hours',
    'year':      'year',
}


def bio_to_dict(sentences, labels):
    result = {field: [] for field in TAG_TO_FIELD.values()}

    for sent_words, sent_labels in zip(sentences, labels):
        current_tokens = []
        current_tag    = None

        def flush():
            if current_tokens and current_tag:
                field = TAG_TO_FIELD.get(current_tag)
                if field:
                    result[field].append(' '.join(current_tokens))

        for word, label in zip(sent_words, sent_labels):
            if label == 'O':
                flush()
                current_tokens, current_tag = [], None
            elif label.startswith('B-'):
                flush()
                current_tag    = label[2:]
                current_tokens = [word]
            elif label.startswith('I-'):
                tag = label[2:]
                if current_tokens and tag == current_tag:
                    current_tokens.append(word)
                else:
                    flush()
                    current_tag    = tag
                    current_tokens = [word]
        flush()

    return {k: list(dict.fromkeys(v)) for k, v in result.items()}


def remove_references(text):
    start_idx = 0
    for kw in ["Abstract", "ABSTRACT"]:
        idx = text.find(kw)
        if idx != -1:
            start_idx = idx + len(kw)
            break
    end_idx = len(text)
    for kw in ["References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY"]:
        idx = text.find(kw)
        if idx != -1:
            end_idx = idx
            break
    main_text = text[start_idx:end_idx]
    lines = main_text.split("\n")
    cleaned = [l for l in lines if sum(c.isdigit() for c in l) / (len(l) + 1) < 0.4]
    return "\n".join(cleaned).strip()


def preprocess_article(text):
    from data import split_into_sentences
    text = remove_references(text)
    sentences = split_into_sentences(text)
    return [s.split() for s in sentences if s.strip()]


def predict_file(model, tokenizer, inv_vocab, context_size, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = preprocess_article(text)
    if not sentences:
        return {field: [] for field in TAG_TO_FIELD.values()}
    labels = predict_article(model, tokenizer, sentences, inv_vocab, context_size)
    return bio_to_dict(sentences, labels)

