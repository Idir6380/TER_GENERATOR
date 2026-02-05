import json
import os
import re
from datetime import datetime
from openai import OpenAI

from config import PROMPT, MODELS_OLLAMA, NUM_ARTICLES, OUTPUT_DIR


def get_client():
    return OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1"
    )


def clean_json_string(content: str) -> str:
    if not content:
        return ""

    content = content.strip()

    # Supprimer les balises <think>...</think>
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = content.strip()

    # Supprimer les blocs markdown
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Extraire le JSON
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        content = json_match.group()

    return content


def generate_article(client, model_id: str, debug: bool = False) -> dict:
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": PROMPT}]
    )

    content = response.choices[0].message.content

    if debug:
        print(f"\n[DEBUG] Réponse brute:\n{content[:500]}...")

    content = clean_json_string(content)

    if not content:
        raise ValueError("Réponse vide après nettoyage")

    return json.loads(content)


def main():
    print("=" * 60)
    print("Génération d'articles - Ollama (local)")
    print("=" * 60)

    client = get_client()

    for model_name, model_id in MODELS_OLLAMA.items():
        print(f"\n Modèle: {model_name} ({model_id})")
        print("-" * 40)

        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        all_articles = []

        for i in range(1, NUM_ARTICLES + 1):
            try:
                print(f"  Génération article {i}/{NUM_ARTICLES}...", end=" ", flush=True)

                article_data = generate_article(client, model_id, debug=(i == 1))

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
