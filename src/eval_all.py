import os
import sys
import glob
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(__file__))

from eval3 import load_ground_truth, load_predictions, evaluate

DATASET_CSV = "../ref/thibault/static/dataset.csv"
RESULTS_DIR = "../results/"
OUTPUT_TXT  = "../results/eval_all_results.txt"

if __name__ == "__main__":
    gt = load_ground_truth(DATASET_CSV)

    pred_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "greenmir_pred_scibert*.json")))

    if not pred_files:
        print("Aucun fichier de prédictions trouvé.")
        sys.exit(1)

    print(f"{len(pred_files)} fichiers trouvés.\n")

    with open(OUTPUT_TXT, 'w', encoding='utf-8') as out:
        for pred_file in pred_files:
            model_id = os.path.basename(pred_file).replace(".json", "")
            header = f"\n{'='*60}\n{model_id}\n{'='*60}\n"
            print(header)
            out.write(header)

            preds = load_predictions(pred_file)
            n_nonempty = sum(1 for p in preds.values() if any(v is not None for v in p.values()))
            info = f"  Ground truth : {len(gt)} articles\n  Predictions  : {len(preds)} articles\n  Articles avec ≥1 champ prédit : {n_nonempty}\n"
            print(info)
            out.write(info)

            # Capturer la sortie de evaluate()
            buf = io.StringIO()
            with redirect_stdout(buf):
                evaluate(preds, gt)
            result_str = buf.getvalue()

            print(result_str)
            out.write(result_str)

    print(f"\nRésultats sauvegardés dans {OUTPUT_TXT}")
