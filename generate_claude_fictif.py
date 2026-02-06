import json
import os
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

from config import PROMPT, MODELS_ANTHROPIC, NUM_ARTICLES, OUTPUT_DIR

load_dotenv()


def generate_article(client, model_id: str) -> dict:
    response = client.messages.create(
        model=model_id,
        max_tokens=2000,
        messages=[{"role": "user", "content": PROMPT}]
    )

    content = response.content[0].text

    # Nettoyer le JSON si nécessaire
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    return json.loads(content)


def main():
    client = Anthropic()

    print("=" * 60)
    print("Génération d'articles scientifiques fictifs")
    print("=" * 60)

    for model_name, model_id in MODELS_ANTHROPIC.items():
        print(f"\n Modèle: {model_name} ({model_id})")
        print("-" * 40)

        # Créer le dossier du modèle
        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Liste pour stocker tous les articles
        all_articles = []

        for i in range(1, NUM_ARTICLES + 1):
            try:
                print(f"  Génération article {i}/{NUM_ARTICLES}...", end=" ", flush=True)

                article_data = generate_article(client, model_id)

                # Ajouter les métadonnées
                article_data["metadata"] = {
                    "generator_model": model_name,
                    "generated_at": datetime.now().isoformat(),
                    "article_number": i
                }

                # Ajouter à la liste globale
                all_articles.append(article_data)

                # Sauvegarder le texte dans un fichier .txt
                txt_filepath = os.path.join(model_dir, f"article_{i}.txt")
                with open(txt_filepath, "w", encoding="utf-8") as f:
                    f.write(article_data["article"])

                print(f"OK -> {txt_filepath}")

            except json.JSONDecodeError as e:
                print(f"Erreur JSON: {e}")
            except Exception as e:
                print(f"Erreur: {e}")

        # Sauvegarder tous les articles dans un seul JSON
        json_filepath = os.path.join(model_dir, "articles.json")
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        print(f"\n  JSON global sauvegardé: {json_filepath}")

    print("\n" + "=" * 60)
    print("Génération terminée!")
    print("=" * 60)


if __name__ == "__main__":
    main()
