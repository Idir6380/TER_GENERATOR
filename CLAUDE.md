# CLAUDE.md — PFE : Extraction de variables expérimentales

## Contexte général

**Sujet** : Extraction automatique d'informations depuis des articles scientifiques IA (nom du modèle, hardware, paramètres, durée, pays, année, nombre de GPUs) pour estimer l'empreinte carbone de la recherche en IA.

**Encadrants** : Constance Douwes, Carlos Ramisch, Alexis Nasr
**Étudiants** : Idir Bellili + Vanessa Guerrier
**Langue de communication** : français

**Pipeline visé** :
```
Articles IA (PDF) → Extraction NER → Calcul énergie (TDP × GPUs × temps) → Estimation CO2
```

---

## Structure du dépôt

```
PFE/
├── src/                          ← Code Idir (SciBERT)
│   ├── model.py                  ← Classe SciBERTNER (configurable)
│   ├── data.py                   ← Pipeline XML→BIO + sliding window
│   ├── train.py                  ← Boucle entraînement 48 configs
│   ├── pred.py                   ← Inférence + BIO→dictionnaire
│   ├── perd_green_mir.py         ← Application sur GreenMIR
│   ├── eval.py                   ← Métriques de base (F1, distances) — ground truth depuis Excel
│   ├── eval2.py                  ← Évaluation extrinsèque GreenMIR (dataset.csv), métriques P/R/F1 style Thibault
│   └── eval3.py                  ← Évaluation avancée : métriques catégorielles (TP/FP/FN) + souples (S_i), comparable Thibault exact
├── training/src/                 ← Code Vanessa (BERT-base)
│   ├── preparation_data_train.py ← XML→BIO, data loaders
│   ├── train.py                  ← Fine-tuning BERT-base
│   ├── pred.py                   ← Inférence + reconstruction
│   ├── prediction_greenmir.py    ← Application sur GreenMIR
│   ├── metric.py                 ← Métriques (fuzzy match, P/R/F1)
│   └── tokenizer.py              ← Dataset PyTorch
├── data/
│   ├── all_articles.json         ← 419 articles synthétiques bruts
│   ├── all_articles_augmented.json ← 478 articles (version finale entraînement)
│   ├── nombre _anglais.json      ← Dict {"one":1, ..., "one hundred":100} (Vanessa)
│   ├── ml_hardware.csv           ← Base Epoch AI (172 accélérateurs, TDP)
│   └── GreenMIR/
│       ├── corpus1.xlsx          ← Ground truth ISMIR corpus 1 (80 articles)
│       ├── corpus2.xlsx          ← Ground truth ISMIR corpus 2 (33 articles)
│       ├── pdfs/                 ← 113 PDFs GreenMIR
│       ├── text_xml_nettoyer/    ← Texte extrait via GROBID + nettoyé (113 .txt)
│       ├── greenmir_pred_0.json  ← Prédictions Vanessa sans contexte
│       └── greenmir_pred_1.json  ← Prédictions Vanessa avec contexte
├── results/
│   ├── test_results.csv          ← F1 test des 48 configs SciBERT
│   ├── train_results.csv         ← F1 eval des 48 configs SciBERT
│   ├── greenmir_pred_scibert_F4_L10_1.json ← Prédictions GreenMIR meilleur modèle
│   └── greenmir_pred_scibert_L12.json      ← Prédictions GreenMIR F2_L12_1
├── models/                       ← Checkpoints .pt (48 configs, ~400MB chacun)
├── ref/
│   ├── thibault/                 ← Code de l'étudiant précédent (LLM zero-shot)
│   │   └── static/dataset.csv   ← Ground truth GreenMIR parsé proprement (113 lignes)
│   └── greenmir/                 ← Paper ISMIR 2024 + données
├── docs/
│   ├── Rapport_Thibault_Scheebeerger (1).pdf
│   └── 03_sujet_iaaa_2526_extraction_variables.pdf
├── main.tex                      ← Rapport LaTeX (Vanessa + Idir)
├── model_bert.md                 ← Notes techniques architecture SciBERT
├── notes_projet.md               ← Notes exhaustives du projet
├── plan_eval.md                  ← Plan d'évaluation extrinsèque
├── planif.md                     ← TODO list courte
├── prompt.md                     ← Prompt de génération des données synthétiques
├── config.py                     ← Configuration génération
├── src/utils.py                  ← Utilitaires génération données (GeminiKeyRotator, build_prompt, load_ground_truth depuis Excel, GROBID clear_article)
├── analysis_improved.ipynb       ← Analyse des données générées
└── CLAUDE.md                     ← Ce fichier
```

---

## Étape 1 — Génération de données synthétiques (FAIT)

### Données générées
- **478 articles** dans `data/all_articles_augmented.json`
- Générés par plusieurs LLMs avec balises XML inline

| LLM | Articles | Taux |
|-----|---------|------|
| Claude Sonnet (Anthropic) | 100 | 100% |
| Kimi (Groq) | 99 | 99% |
| Qwen 3-32B (Groq) | 92 | 92% |
| Gemini 2.5-flash | 64 | 64% |
| Gemini 3-flash-preview | 64 | 64% |

### 7 entités annotées (balises XML)
| Balise XML | Champ JSON | Exemple |
|-----------|-----------|---------|
| `<model>` | `model_name` | GPT-3.5 |
| `<params>` | `parameter_count` | 175 billion parameters |
| `<gpu_count>` | `gpu_count` | 8 (ou "a", "single" → 1) |
| `<hardware>` | `hardware` | NVIDIA A100 GPUs |
| `<training>` | `training_duration` | 3 weeks |
| `<country>` | `country` | United States |
| `<year>` | `year` | 2023 |

### Mécanismes importants
- **Omissions aléatoires** : 0-5 champs omis par article (min 2 présents)
- `gpu_count` : "a"/"an"/"single" dans le texte → converti en 1 dans le JSON
- **Diversité** : noms de modèles déjà générés injectés dans le prompt pour éviter les doublons
- **Biais identifiés** : Claude surreprésente Singapour, Qwen surreprésente UK

---

## Étape 2 — Modèle NER SciBERT (Idir) — FAIT

### Architecture SciBERTNER (`src/model.py`)

```
allenai/scibert_scivocab_cased
    ↓ (12 couches transformer, 768 dim)
    ↓ Fine-tuning des n dernières couches (F0/F1/F2/F4)
Extraction couche : L8 / L10 / L12 / AVG(layers 9-12)
    ↓
Dropout(0.1)
    ↓
Linear(768 → num_labels)
    ↓
CrossEntropyLoss (ignore_index=-100)
```

**Justifications théoriques** :
- Jawahar et al. (2019) : couches 9-12 encodent les features sémantiques (NER)
- Devlin et al. (2019) : AVG des 4 dernières couches optimal pour NER
- SciBERT vs BERT-base : vocabulaire scientifique (scivocab), SciERC F1 67.6 vs 64.2

### Pipeline données (`src/data.py`)
1. JSON → texte avec balises XML
2. `split_into_sentences` : découpage regex
3. `xml_to_bio` : conversion XML → labels BIO
4. Tokenisation WordPiece : premier sous-mot → label, autres → -100
5. `build_windows` : **sliding window intra-article** (contribution originale)
6. `DataCollatorForTokenClassification` : padding dynamique
7. Split 80/10/10 (random_state=42)

### Sliding Window (`context_size` ∈ {0, 1, 2})
- `context_size=0` : phrase seule (sentence-level)
- `context_size=1` : [phrase_{i-1}, phrase_i, phrase_{i+1}], labels -100 pour le contexte
- Implémenté dans `data.py:build_windows` et `pred.py:predict_article`
- **Contribution originale** : absent du code de Vanessa

### Hyperparamètres d'entraînement
- Optimiseur : AdamW, `lr_bert=2e-5`, `lr_classifier=1e-3` (différencié)
- Scheduler : ReduceLROnPlateau (mode='max', factor=0.5, patience=2) sur F1
- Early stopping : patience=5 sur F1 seqeval (entity-level)
- Max epochs : 300, batch_size=32
- Métrique : seqeval F1 (entity-level, standard NER)

### 48 configurations testées
| Dimension | Valeurs |
|-----------|---------|
| Fine-tuning (`n_finetune_layers`) | F0 (gelé), F1, F2, F4 |
| Extraction couche (`layer_mode`) | L8, L10, L12, AVG(9-12) |
| Contexte (`context_size`) | 0, 1, 2 |

---

## Résultats sur données synthétiques (test set)

**Top 5 configurations (test_f1_macro)** :

| Config | Finetune | Layer | Context | Micro F1 | Macro F1 |
|--------|----------|-------|---------|----------|----------|
| **F4\_L10\_1** | F4 | L10 | 1 | 0.9558 | **0.9582** |
| F4\_AVG\_0 | F4 | AVG | 0 | 0.9511 | 0.9544 |
| F4\_AVG\_1 | F4 | AVG | 1 | 0.9471 | 0.9496 |
| F4\_L12\_0 | F4 | L12 | 0 | 0.9469 | 0.9478 |
| F4\_AVG\_2 | F4 | AVG | 2 | 0.9446 | 0.9472 |
| F0\_L12\_0 (baseline) | F0 | L12 | 0 | 0.8933 | 0.8938 |

**Gain total : +6.4 points F1-macro** (baseline → meilleur)

**Observations clés** :
- F4 > F2 malgré la théorie (Bellili et al.) → domain match SciBERT/données synthétiques
- L10 optimal (intermédiaire syntaxe/sémantique)
- Sliding window context=1 améliore marginalement (+0.004)

---

## Évaluation extrinsèque sur GreenMIR (`src/eval2.py`)

### Ground truth utilisé
`ref/thibault/static/dataset.csv` — 113 articles, valeurs numériques déjà parsées :
- `year` (int), `hardware_number` (float), `training_time` (float, heures)
- `parameters` (float), `hardware` (texte), `country` (texte)
- index 0 → article_id 1, ..., index 112 → article_id 113

### Métriques (même protocole que Thibault)
- **Precision** = 1 - mean(distances)
- **Recall** = n_prédit / n_total_gt
- **F1** = harmonique
- `year` : distance normalisée sur 5 ans
- `gpu_count`, `training_hours`, `parameter_count` : erreur relative
- `country` : Jaro-Winkler ≥ 0.85 (multi-pays)
- `hardware` : Jaro-Winkler ≥ 0.6

### Réduction des listes → scalaires (`src/eval2.py`)
- `year` : premier token valide (1990–2030)
- `gpu_count` : somme des numériques + `nombre _anglais.json` + {"a","an","single"}→1
- `training_hours` : premier token parsable avec unités (days×24, min/60)
- `parameter_count` : premier token avec suffixe (B×1e9, M×1e6, k×1e3)
- `country` : concaténation de tous les tokens
- `hardware` : valeur la plus longue (filtre les génériques : "GPU", "a", nombres seuls)

### Résultats finaux GreenMIR

**Idir — F4\_L10\_1** (meilleur modèle synthétique) :

| Champ | Precision | Recall | F1 | N_pred | N_total |
|-------|-----------|--------|----|--------|---------|
| year | 0.992 | 0.425 | 0.595 | 48 | 113 |
| gpu_count | 1.000 | 0.300 | 0.462 | 9 | 30 |
| training_hours | 0.858 | 0.269 | 0.410 | 7 | 26 |
| parameter_count | 0.198 | 0.308 | 0.241 | 4 | 13 |
| country | 0.400 | 0.062 | 0.108 | 5 | 80 |
| hardware | 0.680 | 0.581 | 0.627 | 25 | 43 |
| **macro avg** | **0.688** | **0.324** | **0.407** | | |

**Vanessa — avec contexte** (meilleur modèle Vanessa) :

| Champ | Precision | Recall | F1 | N_pred | N_total |
|-------|-----------|--------|----|--------|---------|
| year | 0.952 | 0.770 | 0.851 | 87 | 113 |
| gpu_count | 0.714 | 0.300 | 0.422 | 9 | 30 |
| training_hours | 0.469 | 0.615 | 0.532 | 16 | 26 |
| parameter_count | 0.500 | 0.154 | 0.235 | 2 | 13 |
| country | 0.065 | 0.775 | 0.119 | 62 | 80 |
| hardware | 0.900 | 0.465 | 0.613 | 20 | 43 |
| **macro avg** | **0.600** | **0.513** | **0.462** | | |

**Analyse** :
- Idir : **précision haute, recall faible** → conservateur, peu de faux positifs
- Vanessa : **recall haut, précision faible** → libéral, beaucoup de faux positifs
- Idir meilleur sur `hardware` (0.627 vs 0.613) et `gpu_count` (0.462 vs 0.422)
- Vanessa meilleure sur `year`, `training_hours`, macro F1 (0.462 vs 0.407)
- Cause : domain shift données synthétiques → vrais articles ISMIR

---

## Modèle NER Vanessa (BERT-base) — `training/src/`

- Modèle : `bert-base-cased` via `AutoModelForTokenClassification`
- **2 configs** : sans contexte (sentence-level) et avec contexte (2 phrases consécutives)
- Données entraînement : mêmes 478 articles synthétiques
- Reconstruction : **vote majoritaire** sur les sous-mots (différent d'Idir qui prend le premier)
- Extraction texte GreenMIR : GROBID → XML → BeautifulSoup → texte propre
- Résultats sur données synthétiques : F1 ≈ 0.89 (sans contexte) vs 0.81 (avec contexte)

---

## Travail de Thibault (ref/) — LLM zero-shot

- Approche : Gemini 2.5 Flash, zero-shot, pas d'entraînement
- Pipeline : PDF → SQLite → énumération modèles → extraction par champ
- Évaluation : fuzzy matching Jaro-Winkler + lookups country/hardware (SQLite)
- Ground truth nettoyé : `ref/thibault/static/dataset.csv` ← **à utiliser pour l'évaluation**
- Son framework d'évaluation est dans `ref/thibault/src/benchmarks/`

---

## Ce qui reste à faire

### En cours — Refactoring split article-level (`src/data.py`)

**Problème identifié** : le split train/eval/test était fait au niveau **phrase** (après `build_windows`), ce qui introduit du **data leakage** — des phrases du même article pouvaient se retrouver dans train ET test, notamment avec `context_size>0` (fenêtres chevauchantes).

**Comparaison avec Vanessa** : elle fait déjà un split **article-level** dans `read_file_train` (interleaving déterministe, ratio 70/20/10). Idir utilisait `train_test_split` sur les phrases (ratio 80/10/10, random_state=42).

**Fix prévu dans `src/data.py`** (`get_dataloaders`) :
1. Splitter les **indices d'articles** avant `build_windows`
2. Appliquer `build_windows` séparément sur chaque split
3. Garder ratio 80/10/10 et random_state=42
4. Re-entraîner les 48 configs → nouveaux résultats dans `results/`

**Étapes** (step by step) :
- [ ] **Step 1** : modifier `get_dataloaders` dans `src/data.py`
- [ ] **Step 2** : vérifier que les tailles de splits sont cohérentes
- [ ] **Step 3** : relancer `train.py` sur GPU (Vast.ai)
- [ ] **Step 4** : comparer nouveaux résultats vs anciens (impact du leakage)
- [ ] **Step 5** : mettre à jour `results/test_results.csv` et `results/train_results.csv`
- [ ] **Step 6** : relancer `perd_green_mir.py` + `eval3.py` avec le nouveau meilleur modèle

---

### Priorité haute
- [ ] **Mettre à jour le tableau GreenMIR dans main.tex** avec les résultats finaux (F4_L10_1 + comparaison F2_L12_1)
- [ ] **Compiler main.tex** et vérifier le rendu PDF (figures manquantes à vérifier)
- [ ] **Évaluation F2\_L12\_1 avec eval3.py** : relancer avec métriques catégorielles + souples pour comparaison équitable avec Thibault

### Priorité moyenne
- [ ] **Aléatoire pour B/M/K** (planif.md point 1) : améliorer `_parse_params` dans `eval2.py` pour gérer "7B" → 7×10⁹ dans les prédictions (actuellement partiel)
- [ ] **Dataset de Thibault** (planif.md point 6) : utiliser ses prédictions Gemini comme baseline comparative via `eval3.py`
- [ ] **Entraînement sur l'entièreté du texte** (planif.md point 3) : Vanessa a `data_per_article()` mais dépasse 512 tokens → implémenter Longformer

### Priorité basse / perspectives
- [ ] **Longformer** (`allenai/longformer-base-4096`) : pour les vrais articles longs (>512 tokens)
- [ ] **GROBID pour Idir** : améliorer l'extraction texte dans `perd_green_mir.py` (actuellement PyMuPDF), Vanessa utilise déjà GROBID
- [ ] **Bayesian optimisation** (planif.md point 4) : optimiser les hyperparamètres au lieu de la grille exhaustive
- [ ] **Contextual Majority Voting (CMV)** : extension du sliding window (Luoma & Pyysalo 2020)
- [ ] **Fine-tuning sur quelques vrais articles annotés** : réduire le domain shift

---

## Références clés

| Référence | Pertinence |
|-----------|-----------|
| Beltagy et al. 2019 — SciBERT (EMNLP) | Architecture principale |
| Devlin et al. 2019 — BERT (NAACL) | Base théorique |
| Jawahar et al. 2019 — Probing BERT (ACL) | Justification choix couches 9-12 |
| Luoma & Pyysalo 2020 — Cross-sentence NER (COLING) | Sliding window |
| Bellili, Ghrissi, Guellili 2026 — PSTAL | Stratégies fine-tuning F2-AVG |
| Holzapfel et al. 2024 — GreenMIR (ISMIR) | Dataset d'évaluation |
| Beltagy et al. 2020 — Longformer | Perspective future |

---

## Commandes utiles

```bash
# Entraînement des 48 configs (depuis src/)
cd src && python3 train.py

# Prédiction sur GreenMIR
cd src && python3 perd_green_mir.py

# Évaluation extrinsèque
cd src && python3 eval2.py

# Compilation rapport
pdflatex main.tex

# Lancer sur GPU cloud (Vast.ai)
git clone https://github.com/Idir6380/TER_GENERATOR.git
pip install -q seqeval transformers accelerate scikit-learn pandas tqdm rapidfuzz
cd src && python3 train.py
```

---

## Fichiers importants à ne pas perdre

| Fichier | Importance |
|---------|-----------|
| `data/all_articles_augmented.json` | Données d'entraînement (478 articles) |
| `data/nombre _anglais.json` | Conversion mots anglais → chiffres |
| `ref/thibault/static/dataset.csv` | Ground truth GreenMIR (référence évaluation) |
| `data/GreenMIR/text_xml_nettoyer/` | Textes GreenMIR extraits via GROBID (113 .txt) |
| `results/test_results.csv` | Résultats des 48 configs SciBERT |
| `results/greenmir_pred_scibert_F4_L10_1.json` | Meilleures prédictions GreenMIR |
| `main.tex` | Rapport final |
| `models/F4_L10_1.pt` | Meilleur checkpoint SciBERT |
