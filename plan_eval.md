# Plan d'évaluation extrinsèque du NER

## Principe

On entraîne le modèle NER sur des données synthétiques (478 articles) et on l'évalue sur des vrais articles scientifiques (GreenMIR). L'objectif est de mesurer la capacité de généralisation du modèle.

## Pipeline

```
Vrais articles (GreenMIR PDFs)
        ↓
    Extraction texte (GROBID → XML → BeautifulSoup → .txt)
    [Vanessa utilise GROBID ; Idir utilise PyMuPDF dans perd_green_mir.py]
        ↓
    NER (notre BERT fine-tuné)
        ↓
    Tags BIO → entités extraites (model, hardware, country...)
        ↓
    Comparaison vs Ground Truth GreenMIR
        ↓
    Métriques par champ
```

## Métriques par champ

| Champ          | Type      | Métrique                        | Justification                                              |
|----------------|-----------|----------------------------------|------------------------------------------------------------|
| model          | texte     | Jaro-Winkler                    | Tolère les variations de nommage ("LLaMA-2" vs "Llama 2") |
| year           | numérique | Erreur relative (échelle 5 ans) | 1 an d'écart ≠ 5 ans d'écart                              |
| params         | numérique | Erreur relative                 | Ordres de grandeur variables (7B vs 175B)                  |
| gpu_count      | numérique | Erreur relative                 | Idem                                                       |
| hardware       | ID        | Exact match après résolution    | "A100 80GB" et "NVIDIA A100" = même GPU                   |
| country        | ID        | Exact match après résolution    | "USA" et "United States" = même pays                       |
| training       | numérique | Erreur relative (normalisé en h)| "3 weeks" vs "21 days" = même valeur                       |

### Pourquoi pas un simple exact match ?

- "NVIDIA A100 80GB GPUs" vs "A100" → même hardware, exact match dirait faux
- "United States" vs "USA" → même pays, exact match dirait faux
- "2023" vs "2024" → 1 an d'écart, pas la même gravité que 5 ans

D'où l'utilisation de lookups (résolution d'entités) et de distances graduées.

## Métriques globales

Pour chaque champ :
- **Precision** = `1 - mean_distance` (qualité des extractions)
- **Recall** = `n_extraits / n_total` (couverture)
- **F1** = `2 * P * R / (P + R)` (harmonique)

Plus un **macro-average** sur tous les champs.

## Réutilisation du code de Thibault (ref/thibault/)

### Ce qu'on réutilise
- `src/benchmarks/core/lookups.py` : résolution country (alias + Jaro-Winkler seuil 0.8) et hardware (tokenization + matching bidirectionnel seuil 0.6)
- `src/benchmarks/core/metrics.py` : distances par type (ID→binaire, numérique→erreur relative, texte→1-JW)
- `src/benchmarks/core/matching.py` : matching de noms de modèles (normalisation + Jaro-Winkler token-set, seuil 0.4)

### Ce qu'on construit
1. **Conversion NER → valeurs structurées** : tags BIO → spans de texte → champs (model_name, hardware, etc.)
2. **Chargement du ground truth GreenMIR** : depuis les fichiers Excel/CSV
3. **Script d'évaluation** : prédictions NER + ground truth → métriques par champ → P/R/F1

## Seuils (issus de Thibault)

| Composant         | Seuil | Raison                              |
|-------------------|-------|-------------------------------------|
| Matching modèles  | 0.4   | Balance faux positifs/négatifs      |
| Country lookup    | 0.8   | Haute confiance pour noms de pays   |
| Hardware lookup   | 0.6   | Plus permissif (nommage complexe)   |
| Year distance max | 5 ans | 5 ans d'écart = distance maximale   |
