import json
import os
import re
import time
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

from config import PROMPT, MODELS_GROQ, NUM_ARTICLES, OUTPUT_DIR

load_dotenv()


def get_client():
    return Groq()


def clean_json_string(content: str) -> str:
    if not content:
        return ""

    content = content.strip()

    # Supprimer les balises <think>...</think> (utilisées par certains modèles)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = content.strip()

    # Supprimer les blocs de code markdown
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Extraire le JSON s'il est entouré de texte
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        content = json_match.group()

    return content


def generate_article(client, model_id: str) -> dict:
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
    )

    raw_content = response.choices[0].message.content
    content = clean_json_string(raw_content)

    if not content:
        print(f"\n[DEBUG] Réponse brute vide ou invalide: {raw_content[:500] if raw_content else 'None'}")
        raise ValueError("Réponse vide après nettoyage")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"\n[DEBUG] Contenu nettoyé: {content[:500]}")
        raise e


def main():
    print("=" * 60)
    print("Génération d'articles - Groq")
    print("=" * 60)

    client = get_client()

    for model_name, model_id in MODELS_GROQ.items():
        print(f"\n Modèle: {model_name} ({model_id})")
        print("-" * 40)

        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        all_articles = []

        for i in range(1, NUM_ARTICLES + 1):
            try:
                print(f"  Génération article {i}/{NUM_ARTICLES}...", end=" ", flush=True)

                article_data = generate_article(client, model_id)

                article_data["metadata"] = {
                    "generator_model": model_name,
                    "generated_at": datetime.now().isoformat(),
                    "article_number": i
                }

                all_articles.append(article_data)

                txt_filepath = os.path.join(model_dir, f"article_{i}.txt")
                with open(txt_filepath, "w", encoding="utf-8") as f:
                    f.write(article_data["article"])

                print(f"OK -> {txt_filepath}")

                if i < NUM_ARTICLES:
                    time.sleep(3)

            except json.JSONDecodeError as e:
                print(f"Erreur JSON: {e}")
            except Exception as e:
                print(f"Erreur: {e}")

        json_filepath = os.path.join(model_dir, "articles.json")
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        print(f"\n  JSON global sauvegardé: {json_filepath}")

    print("\n" + "=" * 60)
    print("Génération terminée!")
    print("=" * 60)


if __name__ == "__main__":
    main()
