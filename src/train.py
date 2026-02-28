import torch 
import numpy as np
import matplotlib.pyplot as plt

from model import SciBERTNER
from data import get_dataloaders
from transformers import AutoTokenizer
from time import time 
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score

MODEL_NAME = "allenai/scibert_scivocab_cased"
DATA_FILE = '../data/all_articles_augmented.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINETUNE_CONFIGS = {'F0': 0, 'F1': 1, 'F2': 2, 'F4': 4}                       
LAYER_CONFIGS = ['L8', 'L10', 'L12', 'AVG']               



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

def train(model, train_loader, eval_loader, inv_vocab, epochs, lr, patience=3):
    train_losses, eval_losses, f1_scores = [], [], []
    best_eval_loss = float('inf')
    best_state = None
    patience_counter = 0

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer)
        eval_loss, f1 = evaluate(model, eval_loader, inv_vocab)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        f1_scores.append(f1)

        pbar.set_postfix({"train": f"{train_loss:.4f}", "eval": f"{eval_loss:.4f}", "f1": f"{f1:.4f}"})

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Recharge le meilleur modèle
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, eval_losses, f1_scores

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader, test_loader, vocab, inv_vocab = get_dataloaders(DATA_FILE, tokenizer, batch_size=32)

    import os
    os.makedirs("../models", exist_ok=True)

    results = {}

    for finetune_name, n_layers in FINETUNE_CONFIGS.items():
        for layer_mode in LAYER_CONFIGS:
            exp_name = f"{finetune_name}_{layer_mode}"
            print(f"\n{'='*50}")
            print(f"Experiment: {exp_name}")
            print(f"{'='*50}")

            model = SciBERTNER(
                n_finetune_layers=n_layers,
                model_name=MODEL_NAME,
                num_labels=len(vocab),
                layer_mode=layer_mode
            ).to(DEVICE)

            train_losses, eval_losses, f1_scores = train(
                model, train_loader, eval_loader, inv_vocab, epochs=20, lr=2e-5, patience=3
            )

            torch.save({
                "model": model.state_dict(),
                "vocab_t": vocab,
                "inv_vocab_t": inv_vocab,
                "model_name": MODEL_NAME,
                "n_finetune_layers": n_layers,
                "layer_mode": layer_mode
            }, f"../models/{exp_name}.pt")

            results[exp_name] = {
                "eval_loss": min(eval_losses),
                "f1": max(f1_scores)
            }
            print(f"{exp_name} → best eval_loss: {min(eval_losses):.4f} | best f1: {max(f1_scores):.4f}")

    print("\n=== Final Results (sorted by F1) ===")
    for exp, res in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{exp:<15} f1: {res['f1']:.4f} | eval_loss: {res['eval_loss']:.4f}")

if __name__ == "__main__":
    main()





    

