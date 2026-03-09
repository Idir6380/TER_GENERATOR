# BERT Fine-tuning Strategy for BIO Tagging

## Task Analysis

Our task is BIO tagging of experimental variables from scientific articles (parameter_count, hardware, country, year, gpu_count, training_duration, model_name).

- **Syntactic** for structured fields: `parameter_count`, `gpu_count`, `year` (NUMBER + UNIT patterns)
- **Semantic** for contextual fields: `hardware`, `country`, `training_duration` (require contextual understanding)

→ Mixed task, closer to NER.

## Theoretical Justification

### Jawahar et al. (2019) — ACL
- Layers 1-4: surface features (morphology, word order)
- Layers 5-8: syntactic features (parsing, POS)
- **Layers 9-12: semantic features (NER, coreference)** ← our target

### Devlin et al. (2019) — Section 5.3
- Concat/avg of last 4 layers = 92.2 F1 on CoNLL-2003 NER
- Better than using layer 12 alone

### Bellili, Ghrissi, Guellili (2026) — PSTAL TP
- Best config: **F2-AVG** = fine-tune 2 last layers + avg of layers 9-12 → 88.06% accuracy
- Fine-tuning 4+ layers causes catastrophic forgetting
- L12 alone is systematically the worst extraction strategy

## Model Choice

### Why not DistilBERT?
DistilBERT has only **6 layers** instead of 12. According to Jawahar et al. (2019), layers 9-12 are the ones encoding semantic features (NER, coreference). By using DistilBERT we lose exactly the layers we need for our task.

### Why SciBERT?
SciBERT (Beltagy et al., 2019) has the **same architecture as BERT-base** (12 layers, 768 dim) but is pre-trained on **1.14M scientific papers** from Semantic Scholar. It has a dedicated scientific vocabulary (scivocab, 30K tokens) that better handles technical terms like `"NVIDIA A100"`, `"hyperparameter"`, `"fine-tuning"`, `"transformer"` — exactly the vocabulary of our corpus.

→ `allenai/scibert_scivocab_cased` is our preferred model.

### SciBERT Tokenizer: WordPiece (not BPE)

Both BERT and SciBERT use **WordPiece** tokenization, but with different vocabularies:

| | BERT-base | SciBERT |
|---|---|---|
| Tokenizer | WordPiece | WordPiece |
| Vocab built on | Wikipedia + BooksCorpus | 1.14M scientific papers |
| Vocab size | 30,522 tokens | 30,522 tokens |
| Scientific terms | rare → split | frequent → kept whole |

**BPE vs WordPiece:**
- BPE (used by GPT, RoBERTa): merges the most **frequent pair** at each step
- WordPiece (used by BERT, SciBERT): merges the pair that **maximizes corpus likelihood** → more linguistically motivated tokens

### SciBERT NER Approach (Beltagy et al. 2019)
1. Tokenize input → SciBERT encoder → token representation (last layer)
2. For subword tokens → take **first subword only** as word representation
3. Linear layer (768 → num_labels) on top
4. Full model fine-tuning
5. Standard BIO scheme

### SciBERT NER Benchmarks
| Dataset | Domain | BERT-base F1 | SciBERT F1 |
|---|---|---|---|
| BC5CDR | Biomedical | 86.9 | **90.0** |
| JNLPBA | Biology | 76.2 | **77.3** |
| SciERC | CS/ML | 64.2 | **67.6** |

SciERC is the most relevant for us — NER on ML/CS papers, same domain as our corpus.

### TODO: Verify tokenization difference
```python
from transformers import BertTokenizer, AutoTokenizer
bert = BertTokenizer.from_pretrained("bert-base-cased")
scibert = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
print("BERT-base:", bert.tokenize("NVIDIA A100 80GB"))
print("SciBERT  :", scibert.tokenize("NVIDIA A100 80GB"))
```

## Plan d'implémentation (2026-02-28)

### Fichiers à créer/modifier (dans l'ordre)

**Étape 1 — `training/src/model.py`** (nouveau fichier)
- Classe `SciBERTNER(nn.Module)`
- `AutoModel.from_pretrained("allenai/scibert_scivocab_cased")`
- Geler tous les params, puis dégeler `encoder.layer[10]` et `encoder.layer[11]`
- `forward` : `output_hidden_states=True` → moyenner `hidden_states[9:13]` → `nn.Linear(768, num_labels)`
- Retourner un objet avec `.loss` et `.logits`

**Étape 2 — `training/src/train.py`** (réécriture)
- Importer `SciBERTNER` depuis `model.py`
- Device : `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- LR : `2e-5` (AdamW), plus `5e-3`
- `.to(device)` sur le modèle ET sur chaque batch
- Fix bug : `losses_eval.append(losses_eval)` → `losses_eval.append(loss_eval)`
- Data : `../../data/all_articles_augmented.json`
- Sauvegarde : `{"model": state_dict, "vocab_t": ..., "inv_vocab_t": ..., "model_name": MODEL_NAME}` — pas l'objet tokenizer

**Étape 3 — `training/src/pred.py`** (adapter le chargement)
- Importer `SciBERTNER` depuis `model.py`
- `initialisation_for_test` : charger checkpoint → recréer `SciBERTNER` → `AutoTokenizer.from_pretrained(model_name)`

### Points de vigilance
- `hidden_states` = tuple de **13 éléments** : `[embedding, layer1, ..., layer12]`
- Layers 9-12 (1-indexé) = `hidden_states[9:13]` en code
- Couches encodeur **0-indexées** : `encoder.layer[10]` et `encoder.layer[11]` = les 2 dernières
- Ne pas sauvegarder l'objet tokenizer dans le `.pt`, juste son nom (string)

### Architecture visée

```
allenai/scibert_scivocab_cased
        ↓ (frozen: layers 1-10)
        ↓ (fine-tuned: layers 11-12)
   hidden_states[9:13]  →  mean  →  [batch, seq, 768]
                                          ↓
                               nn.Linear(768, num_labels)
                                          ↓
                                  logits → CrossEntropyLoss
```

### Configurations à tester (après F2-AVG)

| Fine-tuning | Layer extraction | Attendu |
|---|---|---|
| F0 (frozen) | L12 | baseline |
| F0 (frozen) | AVG (9-12) | meilleure baseline |
| F2 (2 layers) | L12 | bon |
| **F2 (2 layers)** | **AVG (9-12)** | **meilleur attendu** |
| F4 (4 layers) | AVG (9-12) | risque catastrophic forgetting |

### Métriques
- F1 par champ (strict, via `metric.py` existant)
- F1 macro global
- Évaluation extrinsèque sur GreenMIR (après entraînement)

### Structure dossier modèles
- `models/` — un checkpoint par config
- `models/f0_l12.pt`, `models/f0_avg.pt`, `models/f2_l12.pt`, `models/f2_avg.pt`, `models/f4_avg.pt`

---

## Contexte inter-phrases : Sliding Window pour SciBERT

### Sentence-level vs Document-level NER

**Sentence-level NER** : le modèle voit une phrase à la fois, indépendamment du reste. C'est l'approche actuelle. Suffisant quand l'entité est contenue dans une seule phrase.

**Document-level NER** : le modèle exploite le contexte global du document. Nécessaire quand l'information est dispersée ou nécessite de la coréférence (ex: "the model was trained on the aforementioned cluster").

### Sliding Window (fenêtre glissante)

Approche intermédiaire pour rester dans les 512 tokens de SciBERT tout en ajoutant du contexte :
- Input : `[phrase_N-1 + phrase_N + phrase_N+1]`
- Labels : uniquement `phrase_N` labellisée ; phrases contexte → `-100` (ignorées dans la loss et le F1)
- Contrainte : fenêtre intra-article uniquement (pas de contexte inter-articles)

### Références

**Luoma & Pyysalo (2020)** — "Exploring Cross-sentence Contexts for Named Entity Recognition with BERT" — COLING 2020
- Papier de référence pour le sliding window sur BERT NER
- Montrent que ±1 phrase de contexte améliore le F1 de +0.5 à +2 points sur CoNLL-2003 et OntoNotes
- Testent différentes tailles de fenêtre : `context_size ∈ {0, 1, 2, 3}`
- Conclusion : `context_size=1` est le meilleur compromis coût/gain
- ArXiv : https://arxiv.org/abs/2006.01400

**Devlin et al. (2019)** — BERT original — déjà cité
- Le modèle BERT de base traite des paires de phrases ([CLS] A [SEP] B [SEP]) → forme native de contexte inter-phrases

**Strakova et al. (2019)** — "Neural Architectures for Nested NER through Linearization" — ACL 2019
- Utilisent du contexte document-level pour le NER imbriqué
- Montrent que le contexte global améliore la détection d'entités longues et ambiguës

### Plan expérimental

Tester les **48 configs** (16 × context_size∈{0,1,2}) :

| Dimension | Valeurs |
|---|---|
| Fine-tuning | F0, F1, F2, F4 |
| Layer extraction | L8, L10, L12, AVG |
| Context size | 0 (sentence-level), 1 (±1 phrase), 2 (±2 phrases) |

**Total : 48 configs**

Permet une ablation study complète sur trois axes : fine-tuning, extraction de couches, et taille de contexte.

### Analyse du code de Nessa (training/src/)

Nessa a deux modes dans `preparation_data_train.py` :
- `data()` : sentence-level pur, pas de contexte — identique à notre approche
- `data_per_article()` : concatène **toutes** les phrases d'un article en une seule séquence → pseudo document-level, mais problématique car un article dépasse largement 512 tokens → SciBERT tronque → perte de la majorité du contenu

→ **Le sliding window propre (intra-article, fenêtre ±1) n'est implémenté nulle part.** C'est notre contribution originale.

### Perspective future : Contextual Majority Voting (CMV)

Luoma & Pyysalo (2020) proposent une extension du sliding window appelée **CMV** : au lieu de prédire une phrase avec un seul contexte fixe, on la prédit **plusieurs fois** avec des contextes différents (seule, avec contexte gauche, contexte droit, les deux), puis on combine par **vote majoritaire** token par token.

**Exemple :** pour le token `"cluster"` ambigu sans contexte :

| Input BERT | Prédiction |
|---|---|
| [N] seul | O |
| [N-1, N] | B-hardware |
| [N, N+1] | B-hardware |
| [N-1, N, N+1] | B-hardware |

Vote majoritaire → **B-hardware** ✅

**Pourquoi c'est une perspective et pas une priorité :** le CMV multiplie le temps d'inférence par le nombre de contextes testés. Pour notre corpus synthétique où l'info est locale, le gain sera marginal. À envisager pour l'évaluation sur GreenMIR (vrais articles avec entités ambiguës).

---

## Longformer — Alternative pour longs documents

### Pourquoi ?
SciBERT est limité à **512 tokens**. Un vrai article scientifique = 2000-5000 tokens. Solution : Longformer.

### Architecture
Longformer (AllenAI, 2020) utilise une **attention sparse** au lieu de l'attention dense de BERT :
- **Attention locale** : chaque token regarde ses voisins dans une fenêtre glissante
- **Attention dilatée** : fenêtre avec trous pour couvrir plus loin
- **Attention globale** : tokens spéciaux ([CLS]) regardent tout le document
- Complexité : O(n) au lieu de O(n²) → supporte jusqu'à **4096 tokens**

### Comparaison

| | SciBERT | Longformer |
|--|--|--|
| Max tokens | 512 | 4096 |
| Attention | Dense O(n²) | Sparse O(n) |
| Pré-entraînement | 1.14M papers scientifiques | Wikipedia + books |
| Vocabulaire scientifique | Oui (scivocab) | Non |
| NER longs docs | Difficile (découpage) | Natif |

### Modèles HuggingFace
- `allenai/longformer-base-4096`
- `allenai/longformer-large-4096`

### Statut
- SciBERT + phrase par phrase → **pipeline actuel**
- Longformer → **config future à tester** pour les vrais articles GreenMIR

---

## Avancement implémentation (2026-02-28)

### Fichiers créés dans `src/`

**`model.py`** ✅
- Classe `SciBERTNER(nn.Module)` avec `n_finetune_layers` et `use_avg`
- Gel/dégel des couches via `encoder.layer[-n:]`
- `forward` : `output_hidden_states=True` → AVG(layers 9-12) ou L12 → `nn.Linear(768, num_labels)`
- Retourne `SimpleNamespace(loss=..., logits=...)`

**`data.py`** ✅
- `NERDataset` : tokenisation + alignement labels BIO (sous-mots → -100)
- `load_articles`, `split_into_sentences`, `xml_to_bio`, `build_vocab`
- `get_dataloaders` : split 80/10/10, `DataCollatorForTokenClassification` (padding dynamique)

**`train.py`** ✅
- `evaluate()`, `train_one_epoch()`, `train()` avec tqdm
- `main()` : tokenizer → dataloaders → SciBERTNER → train → sauvegarde
- Config actuelle : F2-AVG, 5 epochs, lr=2e-5, batch=16
- Sauvegarde : `../models/f2_avg.pt`

### Choix techniques
- Padding dynamique au niveau batch (plus efficace que max_length fixe)
- Labels sous-mots = -100 (seul le premier sous-mot est labelisé)
- `SimpleNamespace` pour les sorties du modèle (API proche HuggingFace)

### Choix techniques mis à jour (2026-02-28)
- `layer_mode` : supporte L8, L10, L12, AVG
- `context_size` : 0 (sentence-level), 1 (±1 phrase), 2 (±2 phrases) — intra-article uniquement
- **48 configs** testées automatiquement : F0/F1/F2/F4 × L8/L10/L12/AVG × context{0,1,2}
- Early stopping patience=5 sur **F1** (plus pertinent que eval_loss)
- **Dropout(0.1)** entre représentation BERT et classifieur linéaire
- **ReduceLROnPlateau** : factor=0.5, patience=2, mode='max' (surveille F1)
- **LR différencié** : `lr_bert=2e-5` (couches BERT), `lr_classifier=1e-3` (classifieur)
- F1 calculé avec `seqeval` (entity-level, standard NER)
- Meilleur modèle rechargé avant sauvegarde (best_state)
- **Deux CSV** : `train_results.csv` (eval_loss, eval_f1) + `test_results.csv` (test_f1_micro, test_f1_macro)
- **Courbes d'apprentissage** du meilleur modèle → `best_model_curves.png`

### Lancement sur Vast.ai
- GPU loué sur Vast.ai (~$0.10-0.30/h, RTX 3090/4090)
- Image : `pytorch/pytorch:latest`
- SSD recommandé : 50 GB (48 checkpoints × ~400MB ≈ 20 GB)
- Commandes :
```bash
git clone https://github.com/Idir6380/TER_GENERATOR.git
cd TER_GENERATOR
pip install -q seqeval transformers accelerate scikit-learn pandas tqdm
cd src && python train.py
```

### Prochaines étapes
1. ~~**Analyser `test_results.csv`**~~ — FAIT : F4_L10_1 = meilleure config (macro F1=0.9582)
2. ~~**Évaluation extrinsèque sur GreenMIR**~~ — FAIT : eval2.py + eval3.py, résultats dans CLAUDE.md
3. **Longformer** — perspective future pour documents longs (>512 tokens)
