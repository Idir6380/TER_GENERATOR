from collections import Counter,defaultdict

def group_bio_entities(labels):
    # sourcery skip: merge-duplicate-blocks, remove-redundant-if
    entities = []
    current = []
    current_type = None
    start = None
    for i, label in enumerate(labels):
        if label == "O":
            if current:
                entities.append((current_type, start, i - 1))
                current = []
                current_type = None
            continue
        prefix, typ = label.split("-")
        if prefix == "B":
            if current:
                entities.append((current_type, start, i - 1))
            current = [label]
            current_type = typ
            start = i
        elif prefix == "I":
            if current and typ == current_type:
                current.append(label)
            else:
                if current:
                    entities.append((current_type, start, i - 1))
                current = [label]
                current_type = typ
                start = i
    if current:
        entities.append((current_type, start, len(labels) - 1))
    return set(entities)

def fuzzy_match(true_ent, pred_ent, seuil=0.5):
    t_type, t_start, t_end = true_ent
    p_type, p_start, p_end = pred_ent
    
    if t_type != p_type:
        return False
    
    inter_start = max(t_start, p_start)
    inter_end = min(t_end, p_end)
    intersection = max(0, inter_end - inter_start + 1)
    union = max(t_end, p_end) - min(t_start, p_start) + 1
    return (intersection / union) >= seuil

def tp_fp_fn(true_labels, pred_labels,seuil= 0.5):
    tp = fp = fn = 0
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        true_ents = group_bio_entities(true_seq)
        pred_ents = group_bio_entities(pred_seq)
        matched_true = set()
        matched_pred = set()
        for i, pred_ent in enumerate(pred_ents):
            for j, true_ent in enumerate(true_ents):
                if j in matched_true:
                    continue
                if fuzzy_match(true_ent, pred_ent, seuil= seuil):
                    tp += 1
                    matched_true.add(j)
                    matched_pred.add(i)
                    break
        fp += len(pred_ents) - len(matched_pred)
        fn += len(true_ents) - len(matched_true)
    return tp, fp, fn
    

def precision_recall_f1 (la_true,la_pred):
    tp_total, fp_total, fn_total = tp_fp_fn(la_true, la_pred)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall,f1

def precision_recall_f1_per_label(y_trues, y_preds):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for y_true, y_pred in zip(y_trues, y_preds):
        for true_label, pred_label in zip(y_true, y_pred):
            if pred_label == true_label:
                tp[pred_label] += 1
            else:
                fp[pred_label] += 1
                fn[true_label] += 1
                
    metrics = {}
    all_labels = set(list(tp.keys()) + list(fp.keys()) + list(fn.keys()))
    for label in all_labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[label] = {"precision": precision, "recall" :recall, "f1-score":f1}
    return metrics


def affichage(y_trues, y_preds):
    metric= precision_recall_f1_per_label(y_trues, y_preds)
    mean_precision,mean_recall,mean_f1= 0,0,0
    print( " Label    precision   recall   f1-score")
    for key in metric.keys():
        mean_precision += metric[key]["precision"]
        mean_recall += metric[key]["recall"]
        mean_f1 += metric[key]["f1-score"]
        print(f"{key}:    {metric[key]["precision"]:2f}   {metric[key]["recall"]:2f}   {metric[key]["f1-score"]:2f}")
    print(f"macro-avg:   {(mean_precision/len(metric)):2f}   {(mean_recall/len(metric)):2f}   {(mean_f1/len(metric)):2f}")
