"""
eval3.py — Évaluation avec les métriques de Thibault (mode strict uniquement).

Deux niveaux de métriques (identiques aux Tables 2–6 du rapport Thibault) :

  1. Métriques catégorielles  (présence/absence)
       TP : prédit ET dans le ground truth
       FP : prédit MAIS absent du ground truth  (hallucination NER)
       FN : non prédit MAIS présent dans le ground truth (oubli)
       P  = TP / (TP + FP)
       R  = TP / (TP + FN)
       F1 = harmonique de P et R

  2. Métriques souples  (qualité de la valeur extraite)
       S_i = similarité entre valeur prédite et référence (∈ [0,1])
       P_a  (précision alignée)  = ΣS_i / TP
       P_g  (précision globale)  = ΣS_i / (TP + FP)   ← pénalise les hallucinations
       R_s  (rappel souple)      = ΣS_i / (TP + FN)
       F1_s = 2 × P_g × R_s / (P_g + R_s)

Similarités par champ :
  - year              : max(0, 1 − |pred−gt| / 5)          [continu 0–1]
  - gpu_count         : 1 − erreur relative symétrique      [continu 0–1]
  - training_duration : 1 − erreur relative symétrique      [continu 0–1]
  - parameter_count   : 1 − erreur relative symétrique      [continu 0–1]
  - country           : JaroWinkler ≥ 0.85 → 1.0 sinon 0.0 [binaire]
  - hardware          : JaroWinkler ≥ 0.60 → 1.0 sinon 0.0 [binaire]
  - model_name        : token_set_ratio / 100               [continu 0–1]

Comparaison équitable : NER ↔ Thibault mode strict
  (le NER n'extrait que ce qui est explicitement écrit, tout comme le mode strict)
"""

import json
import re
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz as rf_fuzz


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRED_FILE = "../results/greenmir_pred_scibert_F4_L12_1.json" 
DATASET_CSV = "../ref/thibault/static/dataset.csv"

EVAL_FIELDS = [
    "year", "gpu_count", "training_duration",
    "parameter_count", "country", "hardware", "model_name",
]


# ---------------------------------------------------------------------------
# Parsers (identiques à eval2.py)
# ---------------------------------------------------------------------------

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
    if 'week' in s:
        return val * 168
    if 'day' in s:
        return val * 24
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
    if 'billion' in s_low or s_low.endswith('b'):
        return val * 1e9
    if 'million' in s_low or s_low.endswith('m'):
        return val * 1e6
    if 'k' in s_low:
        return val * 1e3
    return val


# ---------------------------------------------------------------------------
# Réducteurs (listes brutes → scalaires)
# ---------------------------------------------------------------------------

_WORD_TO_NUM = json.load(open("../data/nombre_anglais.json", encoding="utf-8"))
_WORD_TO_NUM.update({'a': 1, 'an': 1, 'single': 1})


def reduce_year(values):
    for v in values:
        y = _parse_year(str(v))
        if y is not None:
            return y
    return None


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
    GENERIC = {'gpu', 'gpus', 'tpu', 'tpus', 'cpu', 'a', 'an', 'the'}
    candidates = []
    for v in values:
        if not isinstance(v, str):
            continue
        v = v.strip()
        if not v or v.lower() in GENERIC:
            continue
        if re.fullmatch(r'\d+', v):
            continue
        candidates.append(v)
    return max(candidates, key=len) if candidates else None


def reduce_model_name(values):
    candidates = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    return max(candidates, key=len) if candidates else None


REDUCERS = {
    'year':               reduce_year,
    'gpu_count':          reduce_gpu_count,
    'training_duration':  reduce_training_duration,
    'parameter_count':    reduce_parameter_count,
    'country':            reduce_country,
    'hardware':           reduce_hardware,
    'model_name':         reduce_model_name,
}


def reduce_predictions(info):
    # 'training' tag maps to 'training_hours' in pred.py but eval uses 'training_duration'
    merged = dict(info)
    if 'training_hours' in merged and 'training_duration' not in merged:
        merged['training_duration'] = merged['training_hours']
    return {
        field: REDUCERS[field](merged.get(field, []))
        for field in EVAL_FIELDS
    }


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path):
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
    with open(pred_path, encoding='utf-8') as f:
        data = json.load(f)
    return {a['article_id']: reduce_predictions(a['information']) for a in data}


# ---------------------------------------------------------------------------
# Similarités par champ  (S_i ∈ [0, 1])
# ---------------------------------------------------------------------------

def _normalize_model_name(name):
    """Normalisation identique à matching.py de Thibault."""
    if not name:
        return ""
    s = str(name).lower().strip()
    for h in ['\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2212']:
        s = s.replace(h, '-')
    for noise in ['(', ')', '[', ']', '"', "'", ',', '.', ':']:
        s = s.replace(noise, ' ')
    return ' '.join(s.replace('-', ' ').replace('_', ' ').split())


def similarity(pred, truth, field):
    """
    Retourne S_i ∈ [0, 1] (1 = parfait, 0 = totalement faux).
    Retourne None si l'un des deux est manquant.
    """
    if pred is None or truth is None:
        return None

    if field == 'year':
        try:
            return max(0.0, 1.0 - abs(int(pred) - int(truth)) / 5.0)
        except (ValueError, TypeError):
            return None

    elif field == 'country':
        if not pred or not truth:
            return None
        pred_parts  = [p.strip().lower() for p in str(pred).split(',')]
        truth_parts = [t.strip().lower() for t in str(truth).split(',')]
        for pp in pred_parts:
            for tp in truth_parts:
                if JaroWinkler.normalized_similarity(pp, tp) >= 0.85:
                    return 1.0
        return 0.0

    elif field == 'hardware':
        if not pred or not truth:
            return None
        score = JaroWinkler.normalized_similarity(str(pred).lower(), str(truth).lower())
        return 1.0 if score >= 0.6 else 0.0

    elif field == 'model_name':
        if not pred or not truth:
            return None
        n1 = _normalize_model_name(str(pred))
        n2 = _normalize_model_name(str(truth))
        if not n1 or not n2:
            return None
        return rf_fuzz.token_set_ratio(n1, n2) / 100.0

    else:
        # Champs numériques : erreur relative symétrique (même formule que Thibault)
        try:
            p, t = float(pred), float(truth)
            if p == 0 and t == 0:
                return 1.0
            denom = max(abs(p), abs(t))
            rel_err = min(abs(p - t) / denom, 1.0)
            return 1.0 - rel_err
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Calcul des métriques
# ---------------------------------------------------------------------------

def evaluate(predictions, ground_truth):
    """
    Pour chaque champ, calcule :
      - Métriques catégorielles : TP, FP, FN, P, R, F1
      - Métriques souples       : P_a, P_g, R_s, F1_s
    """
    # Accumulateurs
    TP  = {f: 0   for f in EVAL_FIELDS}
    FP  = {f: 0   for f in EVAL_FIELDS}
    FN  = {f: 0   for f in EVAL_FIELDS}
    SUM = {f: 0.0 for f in EVAL_FIELDS}   # ΣS_i sur les TP

    for article_id, gt in ground_truth.items():
        pred = predictions.get(article_id, {})

        for field in EVAL_FIELDS:
            gt_val   = gt.get(field)
            pred_val = pred.get(field)

            gt_present   = gt_val   is not None
            pred_present = pred_val is not None

            if gt_present and pred_present:
                TP[field] += 1
                s = similarity(pred_val, gt_val, field)
                SUM[field] += (s if s is not None else 0.0)
            elif pred_present and not gt_present:
                FP[field] += 1
            elif gt_present and not pred_present:
                FN[field] += 1
            # TN (les deux absents) : ignoré

    # Affichage métriques catégorielles
    print("\n── Métriques catégorielles (présence / absence) ──────────────────────────")
    print(f"{'Champ':<22} {'TP':>5} {'FP':>5} {'FN':>5}  {'P':>7} {'R':>7} {'F1':>7}")
    print("─" * 62)
    cat_results = []
    for field in EVAL_FIELDS:
        tp, fp, fn = TP[field], FP[field], FN[field]
        P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        cat_results.append({'field': field, 'TP': tp, 'FP': fp, 'FN': fn,
                             'P': P, 'R': R, 'F1': F1})
        print(f"{field:<22} {tp:>5} {fp:>5} {fn:>5}  {P:>7.3f} {R:>7.3f} {F1:>7.3f}")

    # Macro catégoriel (champs avec au moins 1 annotation GT)
    active_cat = [r for r in cat_results if (r['TP'] + r['FN']) > 0]
    if active_cat:
        print("─" * 62)
        macro_P  = sum(r['P']  for r in active_cat) / len(active_cat)
        macro_R  = sum(r['R']  for r in active_cat) / len(active_cat)
        macro_F1 = sum(r['F1'] for r in active_cat) / len(active_cat)
        print(f"{'macro avg':<22} {'':>5} {'':>5} {'':>5}  "
              f"{macro_P:>7.3f} {macro_R:>7.3f} {macro_F1:>7.3f}")

    # Affichage métriques souples
    print("\n── Métriques souples (qualité de la valeur extraite) ─────────────────────")
    print(f"{'Champ':<22} {'P_a':>7} {'P_g':>7} {'R_s':>7} {'F1_s':>7}  "
          f"{'n(TP)':>6} {'n(GT)':>6}")
    print("─" * 70)
    soft_results = []
    for field in EVAL_FIELDS:
        tp, fp, fn = TP[field], FP[field], FN[field]
        s_sum = SUM[field]

        P_a  = s_sum / tp          if tp > 0          else 0.0
        P_g  = s_sum / (tp + fp)   if (tp + fp) > 0   else 0.0
        R_s  = s_sum / (tp + fn)   if (tp + fn) > 0   else 0.0
        F1_s = (2 * P_g * R_s / (P_g + R_s)) if (P_g + R_s) > 0 else 0.0

        soft_results.append({'field': field, 'P_a': P_a, 'P_g': P_g,
                              'R_s': R_s, 'F1_s': F1_s})
        n_gt = tp + fn
        print(f"{field:<22} {P_a:>7.3f} {P_g:>7.3f} {R_s:>7.3f} {F1_s:>7.3f}  "
              f"{tp:>6} {n_gt:>6}")

    # Macro souple (champs avec au moins 1 TP)
    active_soft = [r for r in soft_results if r['P_g'] > 0 or r['R_s'] > 0]
    if active_soft:
        print("─" * 70)
        macro_Pa   = sum(r['P_a']  for r in active_soft) / len(active_soft)
        macro_Pg   = sum(r['P_g']  for r in active_soft) / len(active_soft)
        macro_Rs   = sum(r['R_s']  for r in active_soft) / len(active_soft)
        macro_F1s  = sum(r['F1_s'] for r in active_soft) / len(active_soft)
        print(f"{'macro avg':<22} {macro_Pa:>7.3f} {macro_Pg:>7.3f} "
              f"{macro_Rs:>7.3f} {macro_F1s:>7.3f}")

    return cat_results, soft_results


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
    n_nonempty = sum(1 for p in preds.values() if any(v is not None for v in p.values()))
    print(f"  Articles avec ≥1 champ prédit : {n_nonempty}")

    evaluate(preds, gt)
