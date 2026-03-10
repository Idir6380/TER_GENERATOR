import torch 
import numpy as np
import matplotlib.pyplot as plt

from model import SciBERTNER
from data import get_dataloaders
from transformers import AutoTokenizer
from time import time 
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
import pandas as pd
import os

MODEL_NAME = "allenai/scibert_scivocab_cased"
DATA_FILE = '../data/all_articles_augmented.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINETUNE_CONFIGS = {'F0': 0, 'F1': 1, 'F2': 2, 'F4': 4}                       
LAYER_CONFIGS = ['L8', 'L10', 'L12', 'AVG']               
CONTEXT_CONFIGS = range(3)


def evaluate(model, eval_loader, inv_vocab):
    model.eval()
    total_loss = 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask, labels)
            total_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1)

            for pred_seq, true_seq in zip(preds, labels):
                pred_labels, true_labels = [], []
                for p, t in zip(pred_seq, true_seq):
                    if t.item() == -100:
                        continue
                    pred_labels.append(inv_vocab[p.item()])
                    true_labels.append(inv_vocab[t.item()])
                all_pred.append(pred_labels)
                all_true.append(true_labels)

    avg_loss = total_loss / len(eval_loader)
    f1 = f1_score(all_true, all_pred)
    return avg_loss, f1

def train_one_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)



def train(model, train_loader, eval_loader, inv_vocab, epochs, lr_bert=2e-5, lr_classifier=1e-3, patience=5):
    train_losses, eval_losses, f1_scores = [], [], []
    best_f1 = 0.
    best_state = None
    patience_counter = 0

    optimizer = torch.optim.AdamW([                                               
          {"params": model.bert.parameters(), "lr": lr_bert},
          {"params": model.classifier.parameters(), "lr": lr_classifier}
      ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer)
        eval_loss, f1 = evaluate(model, eval_loader, inv_vocab)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)        
        f1_scores.append(f1)

        scheduler.step(f1)

        pbar.set_postfix({"train": f"{train_loss:.4f}", "eval": f"{eval_loss:.4f}", "f1": f"{f1:.4f}"})

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, eval_losses, f1_scores

def test(model, test_loader, inv_vocab):
    model.eval()
    all_true, all_pred = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask, labels)
            preds = outputs.logits.argmax(dim=-1)

            for pred_seq, true_seq in zip(preds, labels):
                pred_labels, true_labels = [], []
                for p, t in zip(pred_seq, true_seq):
                    if t.item() == -100:
                        continue
                    pred_labels.append(inv_vocab[p.item()])
                    true_labels.append(inv_vocab[t.item()])
                all_pred.append(pred_labels)
                all_true.append(true_labels)

    return classification_report(all_true, all_pred, output_dict=True)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    os.makedirs("../models/scibert/", exist_ok=True)
    os.makedirs('../plots/', exist_ok=True)
    results = {}

    for finetune_name, n_layers in FINETUNE_CONFIGS.items():
        for layer_mode in LAYER_CONFIGS:
            for ctxt in CONTEXT_CONFIGS:
                exp_name = f"{finetune_name}_{layer_mode}_{ctxt}"
                layer_to_max = {'L8': 1, 'L10': 2, 'L12': 4, 'AVG': 4} 
                if n_layers > layer_to_max[layer_mode]:
                    continue
                train_loader, eval_loader, test_loader, vocab, inv_vocab = get_dataloaders(DATA_FILE, tokenizer, batch_size=64, context_size=ctxt)

                print(f"\n{'='*50}")
                print(f"Experiment: {exp_name}")
                print(f"{'='*50}")

                model = SciBERTNER(
                    n_finetune_layers=n_layers,
                    model_name=MODEL_NAME,
                    num_labels=len(vocab),
                    layer_mode=layer_mode
                ).to(DEVICE)
                t_start = time()
                train_losses, eval_losses, f1_scores = train(
                    model, train_loader, eval_loader, inv_vocab, epochs=300, patience=5
                )
                train_time = time() - t_start

                results[exp_name] = {
                    "eval_loss": min(eval_losses),
                    "eval_f1": max(f1_scores),
                    "train_losses": train_losses,
                    "eval_losses": eval_losses,
                    "f1_scores": f1_scores,
                    "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                    "vocab_t": vocab,
                    "inv_vocab_t": inv_vocab,
                    "n_finetune_layers": n_layers,
                    "layer_mode": layer_mode,
                    "context_size": ctxt,
                    'train_time': train_time
                }
                print(f"{exp_name} → eval_f1: {max(f1_scores):.4f}")

    # Top-3 modèles selon eval_f1
    top3_exps = sorted(results, key=lambda x: results[x]["eval_f1"], reverse=True)[:3]
    test_rows = []

    for rank, exp_name in enumerate(top3_exps):
        res = results[exp_name]
        _, _, test_loader, _, _ = get_dataloaders(DATA_FILE, tokenizer, batch_size=64, context_size=res["context_size"])
        m = SciBERTNER(
            n_finetune_layers=res["n_finetune_layers"],
            model_name=MODEL_NAME,
            num_labels=len(res["vocab_t"]),
            layer_mode=res["layer_mode"]
        ).to(DEVICE)
        m.load_state_dict(res["state_dict"])
        test_report = test(m, test_loader, res["inv_vocab_t"])
        torch.save({
            "model": res["state_dict"],
            "vocab_t": res["vocab_t"],
            "inv_vocab_t": res["inv_vocab_t"],
            "model_name": MODEL_NAME,
            "n_finetune_layers": res["n_finetune_layers"],
            "layer_mode": res["layer_mode"],
            "context_size": res["context_size"]
        }, f"../models/scibert/{exp_name}.pt")
        print(f"[Top-{rank+1}] {exp_name} → test_micro: {test_report['micro avg']['f1-score']:.4f} | test_macro: {test_report['macro avg']['f1-score']:.4f} | saved")
        test_rows.append({
            "rank": rank + 1,
            "config": exp_name,
            "finetune": exp_name.split("_")[0],
            "layer": exp_name.split("_")[1],
            "context": exp_name.split("_")[2],
            "eval_f1": res["eval_f1"],
            "test_f1_micro": test_report["micro avg"]["f1-score"],
            "test_f1_macro": test_report["macro avg"]["f1-score"]
        })

    # Courbes pour le top-1 uniquement
    best_exp = top3_exps[0]
    best = results[best_exp]
    epochs_range = range(1, len(best["train_losses"]) + 1)
    plt.figure()
    plt.plot(epochs_range, best["train_losses"], label="train_loss")
    plt.plot(epochs_range, best["eval_losses"], label="eval_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Learning curves — {best_exp}")
    plt.legend()
    plt.savefig("../plots/best_model_loss_scibert_curves.png")
    plt.close()

    f1_range = range(1, len(best['f1_scores'])+1)
    plt.figure()
    plt.plot(f1_range, best["f1_scores"], label='eval_f1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title(f'F1 Score {best_exp}')
    plt.legend()
    plt.savefig('../plots/best_model_f1_scibert_curves.png')
    plt.close()
    print(f"\nTop-1: {best_exp} — curves saved to ../plots")

    # CSV train
    df_train = pd.DataFrame([
        {"config": exp, "finetune": exp.split("_")[0], "layer": exp.split("_")[1], "context": exp.split("_")[2],
         "eval_loss": res["eval_loss"], "eval_f1": res["eval_f1"], 'train_time':res['train_time']}
        for exp, res in results.items()
    ])
    df_train = df_train.sort_values("eval_f1", ascending=False)
    df_train.to_csv("../models/train_results.csv", index=False)
    print(f"Train results saved to ../models/train_results.csv")

    # CSV test (top-3)
    df_test = pd.DataFrame(test_rows)
    df_test.to_csv("../models/test_results.csv", index=False)
    print(f"Test results saved to ../models/test_results.csv")

if __name__ == "__main__":
    main()





    

