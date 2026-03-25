from gliner2 import GLiNER2
import json
import re
from data import split_into_sentences


HARDWARE_RE = re.compile(
    r'\b(NVIDIA|GeForce|Tesla|Quadro|RTX|GTX|A100|V100|H100|T4|P100|A40|A30|A10|'
    r'AMD|Radeon|MI300|MI250|TPU|DGX)\b',
    re.IGNORECASE
)

LEARNING_RATE_RE = re.compile(r'^\s*\d+(\.\d+)?[eE][-\u2212]?\d+\s*$')

PARAM_SUFFIX_RE = re.compile(
    r'\b(M|B|K|million|billion|thousand|parameters?|params?)\b',
    re.IGNORECASE
)


def chunk_text(text, max_words=350):
    sentences = split_into_sentences(text)
    sentences = [s for s in sentences if s.strip()]
    chunks = []
    current_words = []
    prev_sentence = ""

    for sent in sentences:
        words = sent.split()
        if len(current_words) + len(words) > max_words and current_words:
            chunks.append(' '.join(current_words))
            current_words = prev_sentence.split() + words
        else:
            current_words.extend(words)
        prev_sentence = sent
    if current_words:
        chunks.append(' '.join(current_words))

    return chunks


def merge_entities(all_results):
    merged = {k: [] for k in ENTITY_SCHEMA}
    seen = {k: set() for k in ENTITY_SCHEMA}
    for result in all_results:
        # Gère les deux formats possibles : {'entities': {...}} ou {field: [...]}
        entities = result.get('entities', result) if isinstance(result, dict) else result
        for field, values in entities.items():
            if field not in merged:
                continue
            for v in values:
                text = v.strip() if isinstance(v, str) else v.get('text', '').strip()
                key = text.lower()
                if key and key not in seen[field]:
                    seen[field].add(key)
                    merged[field].append(text)
    return merged


def postprocess(merged):
    """Filtres mécaniques post-prédiction."""
    # hardware : garder seulement si contient un keyword GPU/TPU connu
    merged['hardware'] = [v for v in merged['hardware'] if HARDWARE_RE.search(v)]

    # parameter_count : rejeter les learning rates (1e-4, 2.5e-5...)
    # et garder seulement si contient un suffixe de paramètres (M, B, million...)
    merged['parameter_count'] = [
        v for v in merged['parameter_count']
        if not LEARNING_RATE_RE.match(v) and PARAM_SUFFIX_RE.search(v)
    ]

    # Tous les champs : supprimer tokens trop courts ou sans lettre
    for field in merged:
        merged[field] = [
            v for v in merged[field]
            if len(v) >= 2 and any(c.isalpha() for c in v)
        ]

    return merged


# Descriptions courtes (~35 tokens) + thresholds différenciés par entité
ENTITY_SCHEMA = {
    "model_name": {
        "description": (
            "Proper name of the deep learning model proposed by the authors of this paper "
            "(e.g. 'DiffTransfer', 'MusicLM'). "
            "NOT models from related work, baselines, or generic terms like 'the proposed model'."
        ),
        "threshold": 0.65,
    },
    "hardware": {
        "description": (
            "GPU or TPU chip used for training. Must be a specific model: "
            "NVIDIA A100, RTX 3090, V100, TPU v3. NOT neural networks or algorithms."
        ),
        "threshold": 0.70,
    },
    "gpu_count": {
        "description": (
            "Number of GPUs or accelerators used for training. "
            "Examples: 'a single GPU', '8 GPUs', '8x GeForce RTX', '4 A100s', "
            "'two V100 GPUs', 'trained on 16 GPUs'. "
        ),
        "threshold": 0.70,
    },
    "year": {
        "description": (
            "4-digit year this experiment was conducted (e.g. '2022', '2023'). "
            "Not from citations or bibliography."
        ),
        "threshold": 0.35,
    },
    "training_duration": {
        "description": (
            "Time required to train the model (e.g. '12 hours', '3 days', '2 weeks'). "
            "Not inference or evaluation time."
        ),
        "threshold": 0.20,
    },
    "parameter_count": {
        "description": (
            "Number of model parameters (e.g. '80M', '7B', '175 million'). "
            "Not dataset size or vocabulary size."
        ),
        "threshold": 0.20,
    },
    "country": {
        "description": (
            "Country of the research institution (e.g. 'France', 'Japan', 'United States', 'USA', 'U.S'). "
            "Infer from university or city name if needed."
        ),
        "threshold": 0.35,
    },
}


OUTPUT_FILE = '../results/greenmir_pred_gliner2.json'
TEXT_DIR    = '../data/GreenMIR/text_xml_nettoyer'
N_ARTICLES  = 113

model = GLiNER2.from_pretrained('fastino/gliner2-large-v1')

with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
    out.write('[\n')
    for i in range(1, N_ARTICLES + 1):
        path = f'{TEXT_DIR}/article_{i}.txt'
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = chunk_text(text)
        all_results = [model.extract_entities(chunk, ENTITY_SCHEMA) for chunk in chunks]
        merged = merge_entities(all_results)
        merged = postprocess(merged)

        entry = {"article_id": i, "information": merged}
        suffix = ',\n' if i < N_ARTICLES else '\n'
        out.write(json.dumps(entry, ensure_ascii=False, indent=2) + suffix)
        out.flush()

        print(f"[{i:3d}/113] {len(chunks)} chunks — {sum(len(v) for v in merged.values())} entités")

    out.write(']\n')

print(f"\nSauvegardé : {OUTPUT_FILE}")
