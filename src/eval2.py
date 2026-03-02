

import json
import re
import pandas as pd
from rapidfuzz.distance import JaroWinkler


PRED_FILE   = "../results/greenmir_pred_scibert_F4_L10_1.json"
DATASET_CSV = "../ref/thibault/static/dataset.csv"

EVAL_FIELDS = ["year", "gpu_count", "training_duration", "parameter_count", "country", "hardware", "model_name"]



def _parse_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_year(s):
    v = _parse_float(s)
    if v is not None and 1990 <= v <= 2030:
        return int(v)
    return None


def _parse_hours(s):

    if not isinstance(s, str):
        return None
    s = s.lower().strip()
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if not m:
        return None
    val = float(m.group(1))
    if 'day' in s:
        return val * 24
    if 'week' in s:
        return val * 168
    if 'min' in s:
        return val / 60
    return val


def _parse_params(s):

    if not isinstance(s, str):
        return None
    s_low = s.lower().strip()
    m = re.search(r'(\d+(?:\.\d+)?)', s_low)
    if not m:
        return None
    val = float(m.group(1))
    if 'billion' in s_low or 'b' == s_low[-1:]:
        return val * 1e9
    if 'million' in s_low or 'm' == s_low[-1:]:
        return val * 1e6
    if 'k' in s_low:
        return val * 1e3
    return val


def reduce_year(values):
    for v in values:
        y = _parse_year(str(v))
        if y is not None:
            return y
    return None


_WORD_TO_NUM = json.load(open("../data/nombre _anglais.json", encoding="utf-8"))
_WORD_TO_NUM.update({'a': 1, 'an': 1, 'single': 1})

def reduce_gpu_count(values):
    nums = []
    for v in values:
        s = str(v).strip().lower()
        if s in _WORD_TO_NUM:
            nums.append(float(_WORD_TO_NUM[s]))
        elif _parse_float(s) is not None:
            nums.append(float(s))
    return sum(nums) if nums else None


def reduce_training_duration(values):
    for v in values:
        h = _parse_hours(str(v))
        if h is not None:
            return h
    return None


def reduce_parameter_count(values):
    for v in values:
        p = _parse_params(str(v))
        if p is not None:
            return p
    return None


def reduce_country(values):
    clean = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    return ', '.join(clean) if clean else None


def reduce_hardware(values):
    """
    Garde la mention hardware la plus spécifique :
    filtre les tokens génériques ("GPU", "a", nombres seuls),
    prend la valeur la plus longue.
    """
    GENERIC = {'gpu', 'gpus', 'tpu', 'tpus', 'cpu', 'a', 'an', 'the'}
    candidates = []
    for v in values:
        if not isinstance(v, str):
            continue
        v = v.strip()
        if not v:
            continue
        if v.lower() in GENERIC:
            continue
        if re.fullmatch(r'\d+', v):      # nombre seul → skip
            continue
        candidates.append(v)
    if not candidates:
        return None
    return max(candidates, key=len)      # plus longue = plus spécifique


def reduce_model_name(values):
    """Prend la valeur la plus longue (la plus spécifique)."""
    candidates = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if not candidates:
        return None
    return max(candidates, key=len)


REDUCERS = {
    'year':               reduce_year,
    'gpu_count':          reduce_gpu_count,
    'training_duration':  reduce_training_duration,
    'parameter_count':    reduce_parameter_count,
    'country':            reduce_country,
    'hardware':           reduce_hardware,
    'model_name':         reduce_model_name,
}

PRED_FIELD_MAP = {
    'year':               'year',
    'gpu_count':          'gpu_count',
    'training_duration':  'training_duration',
    'parameter_count':    'parameter_count',
    'country':            'country',
    'hardware':           'hardware',
    'model_name':         'model_name',
}


def reduce_predictions(info):
    """Réduit les listes brutes du JSON en scalaires prêts pour l'évaluation."""
    return {
        field: REDUCERS[field](info.get(PRED_FIELD_MAP[field], []))
        for field in EVAL_FIELDS
    }


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path):
    """
    Charge dataset.csv de Thibault.
    index 0 → article_id 1, ..., index 112 → article_id 113.
    """
    df = pd.read_csv(csv_path)
    gt = {}
    for idx, row in df.iterrows():
        article_id = idx + 1
        gt[article_id] = {
            'year':               int(row['year'])              if pd.notna(row['year'])              else None,
            'country':            str(row['country'])           if pd.notna(row['country'])           else None,
            'hardware':           str(row['hardware'])          if pd.notna(row['hardware'])          else None,
            'gpu_count':          float(row['hardware_number']) if pd.notna(row['hardware_number'])   else None,
            'training_duration':  float(row['training_time'])  if pd.notna(row['training_time'])     else None,
            'parameter_count':    float(row['parameters'])     if pd.notna(row['parameters'])        else None,
            'model_name':         str(row['model name'])       if pd.notna(row['model name'])        else None,
        }
    return gt


def load_predictions(pred_path):
    """Charge les prédictions NER et réduit les listes en scalaires."""
    with open(pred_path, encoding='utf-8') as f:
        data = json.load(f)
    return {a['article_id']: reduce_predictions(a['information']) for a in data}


# ---------------------------------------------------------------------------
# Distances  (identiques à Thibault)
# ---------------------------------------------------------------------------

def year_distance(pred, truth):
    try:
        return min(1.0, abs(int(pred) - int(truth)) / 5.0)
    except (ValueError, TypeError):
        return None


def numerical_distance(pred, truth):
    try:
        p, t = float(pred), float(truth)
        if p == 0 and t == 0:
            return 0.0
        denom = max(abs(p), abs(t))
        return min(abs(p - t) / denom, 1.0)
    except (ValueError, TypeError):
        return None


def country_distance(pred, truth):
    """
    Match multi-pays : distance 0.0 si au moins un pays de la prédiction
    est similaire à au moins un pays du ground truth (JaroWinkler ≥ 0.85).
    Sinon 1.0.  — même logique que le CountryLookup de Thibault.
    """
    if not pred or not truth:
        return None
    pred_parts  = [p.strip().lower() for p in str(pred).split(',')]
    truth_parts = [t.strip().lower() for t in str(truth).split(',')]
    for pp in pred_parts:
        for tp in truth_parts:
            if JaroWinkler.normalized_similarity(pp, tp) >= 0.85:
                return 0.0
    return 1.0


def hardware_distance(pred, truth):
    """
    Similarité Jaro-Winkler sur le nom complet.
    Seuil 0.6 (même que HardwareLookup de Thibault).
    """
    if not pred or not truth:
        return None
    score = JaroWinkler.normalized_similarity(str(pred).lower(), str(truth).lower())
    return 0.0 if score >= 0.6 else 1.0


def model_name_distance(pred, truth):
    """
    Similarité Jaro-Winkler sur le nom du modèle.
    Seuil 0.85 : gère les variantes de formatage ("GPT-3.5" vs "GPT 3.5")
    tout en restant strict sur les noms distincts.
    """
    if not pred or not truth:
        return None
    score = JaroWinkler.normalized_similarity(str(pred).lower(), str(truth).lower())
    return 0.0 if score >= 0.85 else 1.0


def compute_distance(pred, truth, field):
    if field == 'year':
        return year_distance(pred, truth)
    elif field == 'country':
        return country_distance(pred, truth)
    elif field == 'hardware':
        return hardware_distance(pred, truth)
    elif field == 'model_name':
        return model_name_distance(pred, truth)
    else:
        return numerical_distance(pred, truth)


# ---------------------------------------------------------------------------
# Métriques F1  (identiques à Thibault / Vanessa)
# ---------------------------------------------------------------------------

def compute_f1_stats(distances, n_total):
    """
    Pour chaque champ :
      precision = 1 - mean(distances)
      recall    = n_predicted / n_total_gt
      F1        = harmonic mean
    """
    results = []
    for field in EVAL_FIELDS:
        vals = distances[field]
        if not vals:
            results.append({
                'field': field, 'precision': 0.0, 'recall': 0.0,
                'f1': 0.0, 'n_predicted': 0, 'n_total': n_total[field]
            })
            continue
        n          = len(vals)
        mean_dist  = sum(vals) / n
        precision  = 1.0 - mean_dist
        recall     = n / n_total[field] if n_total[field] > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        results.append({
            'field':       field,
            'precision':   round(precision, 3),
            'recall':      round(recall, 3),
            'f1':          round(f1, 3),
            'n_predicted': n,
            'n_total':     n_total[field],
        })
    return results


# ---------------------------------------------------------------------------
# Évaluation principale
# ---------------------------------------------------------------------------

def evaluate(predictions, ground_truth):
    distances = {f: [] for f in EVAL_FIELDS}
    n_total   = {f: 0  for f in EVAL_FIELDS}

    for article_id, gt in ground_truth.items():
        pred = predictions.get(article_id, {})
        for field in EVAL_FIELDS:
            gt_val = gt.get(field)
            if gt_val is None:
                continue
            n_total[field] += 1

            pred_val = pred.get(field)
            if pred_val is None:
                continue

            dist = compute_distance(pred_val, gt_val, field)
            if dist is not None:
                distances[field].append(dist)

    results = compute_f1_stats(distances, n_total)

    print(f"\n{'Field':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'N_pred':>8} {'N_total':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['field']:<20} {r['precision']:>10.3f} {r['recall']:>10.3f} "
              f"{r['f1']:>10.3f} {r['n_predicted']:>8} {r['n_total']:>8}")

    # Macro-average (champs avec au moins 1 prédiction)
    active = [r for r in results if r['n_predicted'] > 0]
    if active:
        macro_p  = sum(r['precision'] for r in active) / len(active)
        macro_r  = sum(r['recall']    for r in active) / len(active)
        macro_f1 = sum(r['f1']        for r in active) / len(active)
        print("-" * 70)
        print(f"{'macro avg':<20} {macro_p:>10.3f} {macro_r:>10.3f} {macro_f1:>10.3f}")

    return results


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading ground truth from dataset.csv...")
    gt = load_ground_truth(DATASET_CSV)

    print("Loading predictions from JSON...")
    preds = load_predictions(PRED_FILE)

    print(f"  Ground truth : {len(gt)} articles")
    print(f"  Predictions  : {len(preds)} articles")
    print(f"  Articles with ≥1 non-empty field : "
          f"{sum(1 for p in preds.values() if any(v is not None for v in p.values()))}")

    evaluate(preds, gt)
