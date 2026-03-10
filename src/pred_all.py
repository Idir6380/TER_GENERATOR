import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from pred import load_model, predict_file

TEXT_DIR   = "../data/GreenMIR/text_xml_nettoyer"
N_ARTICLES = 113
MODELS_DIRS = {
    "../models/scibert/":          "article",
    "../models/scibert_sentence/": "sentence",
}


def predict_greenmir(checkpoint, text_dir, output_file):
    print(f"\nLoading model: {checkpoint}")
    model, tokenizer, inv_vocab, context_size = load_model(checkpoint)

    predictions = []
    for i in range(1, N_ARTICLES + 1):
        file_path = os.path.join(text_dir, f"article_{i}.txt")
        result = predict_file(model, tokenizer, inv_vocab, context_size, file_path)
        predictions.append({"article_id": i, "information": result})
        print(f"  Article {i}/113 done")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {output_file}")


if __name__ == "__main__":
    for models_dir, split_tag in MODELS_DIRS.items():
        if not os.path.exists(models_dir):
            print(f"Dossier non trouvé, skip : {models_dir}")
            continue

        checkpoints = sorted([f for f in os.listdir(models_dir) if f.endswith(".pt")])
        if not checkpoints:
            print(f"Aucun modèle dans {models_dir}, skip.")
            continue

        print(f"\n=== Split: {split_tag} — {len(checkpoints)} modèle(s) ===")
        for ckpt_name in checkpoints:
            checkpoint = os.path.join(models_dir, ckpt_name)
            model_id = ckpt_name.replace(".pt", "")
            output_file = f"../results/greenmir_pred_scibert_{model_id}_{split_tag}.json"
            predict_greenmir(checkpoint, TEXT_DIR, output_file)

    print("\nToutes les prédictions terminées.")
