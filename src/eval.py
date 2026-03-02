import pandas as pd
from rapidfuzz.distance import JaroWinkler

EVAL_FIELDS = ["year", "gpu_count", "country", "parameter_count", "training_hours"]

#Distance for numerical fiels
def numerical_distance(pred, truth):
    if pred is None or truth is None:
        return None
    try:
        p, t = float(pred), float(truth)

    except (ValueError, TypeError):
        return None
    
    if p == 0 and t == 0:
        return 0.
    denom = max(abs(p), abs(t))
    return min(abs(p - t) / denom, 1.0)

# Yaer distance
def year_distance(pred, truth):
    if pred is None or truth is None:
        return None
    try:
        return min(1.0, abs(int(pred) - int(truth)) / 5.)
    except (ValueError, TypeError):
        return None
    
# sidrance for text fields
def text_distance(pred, truth):
    if not pred or not truth:
        return None
    score = JaroWinkler.normalized_similarity(str(pred).lower(), str(truth).lower())
    return 1 - score

def compute_distance(pred, truth, field):
    if field == 'year':
        return year_distance(pred, truth)
    elif field in ('country', 'hardware'):
        if not pred or not truth:
            return None
        return 0.0 if str(pred).lower() == str(truth).lower() else 1.0
    else:
        return numerical_distance(pred, truth)

def compute_f1_stats(distances, n_total):
    """
    distances: dict[field, list[float]] - distances for each field (only non-None)
    n_total: dict[field, int] - total number of articles with ground truth for each field
    """
    results = []
    for field, vals in distances.items():
        if not vals:
            continue
        n = len(vals)
        mean_dist = sum(vals) / n
        precision = 1.0 - mean_dist
        recall = n / n_total[field] if n_total[field] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({
            "field": field,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "n": n,
        })
    return results


def evaluate(predictions, ground_truth):
    """
    predictions: dict[article_id, dict[field, value]]
    ground_truth: dict[article_id, dict[field, value]] (from load_ground_truth)
    """
    distances = {f: [] for f in EVAL_FIELDS}
    n_total = {f: 0 for f in EVAL_FIELDS}

    for article_id, gt in ground_truth.items():
        pred = predictions.get(article_id, {})

        for field in EVAL_FIELDS:
            gt_val = gt.get(field)
            # skip if no ground truth for this field
            if gt_val is None or (isinstance(gt_val, float) and pd.isna(gt_val)):
                continue
            n_total[field] += 1

            pred_val = pred.get(field)
            dist = compute_distance(pred_val, gt_val, field)
            if dist is not None:
                distances[field].append(dist)

    results = compute_f1_stats(distances, n_total)

    print(f"\n{'Field':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'N':>5}")
    print("-" * 60)
    for r in results:
        print(f"{r['field']:<20} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f} {r['n']:>5}")

    return results


def test():
    from utils import load_ground_truth
    gt = load_ground_truth()

    # Fake predictions for articles that have ground truth data
    fake_predictions = {
        # Article 11: RTX 3090, 1 GPU, 2022, Taiwan/USA, 500k params, 63h
        11: {"year": 2022, "gpu_count": 1, "country": "Taiwan, USA", "parameter_count": 500000, "training_hours": 63},
        # Article 13: V100, 1 GPU, 2022, Taiwan, 48h — partially wrong
        13: {"year": 2023, "gpu_count": 1, "country": "Taiwan", "training_hours": 40},
        # Article 14: V100, 1 GPU, 2022, China, 11.5h — missing some fields
        14: {"year": 2022, "gpu_count": 2, "country": "China"},
    }

    print("=== Test with fake predictions ===")
    evaluate(fake_predictions, gt)


if __name__ == "__main__":
    test()
