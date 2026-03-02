import json
import os 
import sys
sys.path.insert(0, os.path.dirname(__file__))

from pred import load_model, predict_file

CHECKPOINT  = "models/F2_L12_1.pt"                                          
TEXT_DIR    = "data/GreenMIR/text_xml_nettoyer"
OUTPUT_FILE = "results/greenmir_pred_scibert_L12.json"
N_ARTICLES  = 113


def predict_greenmir(checkpoint, text_dir, output_file):
    model, tokenizer, inv_vocab, context_size = load_model(checkpoint)

    predictions = []

    for i in range(1, N_ARTICLES+1):
        file_path = os.path.join(text_dir, f"article_{i}.txt")
        result = predict_file(model, tokenizer, inv_vocab, context_size, file_path)

        predictions.append({"article_id": i, "information": result})
        print(f"Article {i}/113 done")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        print(f"Saved to {output_file}")

if __name__ == "__main__":
    predict_greenmir(CHECKPOINT, TEXT_DIR, OUTPUT_FILE)        