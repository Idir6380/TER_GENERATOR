def group_bio_entities(labels):
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


def tp_fp_fn(true_labels, pred_labels):
    tp = fp = fn = 0
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        true_ents = group_bio_entities(true_seq)
        pred_ents = group_bio_entities(pred_seq)

        tp += len(true_ents & pred_ents)
        fp += len(pred_ents - true_ents)
        fn += len(true_ents - pred_ents)
    return tp, fp , fn

def precision_recall_perArticle (la_true,la_pred):
    tp_total, fp_total, fn_total = tp_fp_fn(la_true, la_pred)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0

    return precision, recall