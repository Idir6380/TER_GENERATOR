import json
import os
import re
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from config import PROMPT, MODELS_GROQ, NUM_ARTICLES, OUTPUT_DIR

load_dotenv()


def get_client():
    return OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )


def clean_json_string(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    content = re.sub(r'\\n\s*"', '\n"', content)
    content = re.sub(r'\\n\s*}', '\n}', content)
    content = re.sub(r'{\s*\\n', '{\n', content)

    return content


def generate_article(client, model_id: str) -> dict:
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=2000,
        messages=[{"role": "user", "content": PROMPT}]
    )

    content = response.choices[0].message.content
    content = clean_json_string(content)

    return json.loads(content)


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
