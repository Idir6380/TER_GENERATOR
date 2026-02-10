import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from tqdm import tqdm

from anthropic import Anthropic
from groq import Groq
from openai import OpenAI
import google.generativeai as genai

from config import (
    PROMPT_IMPROVED,
    MODELS_ANTHROPIC,
    MODELS_GROQ,
    MODELS_GEMINI,
    NUM_ARTICLES,
    OUTPUT_DIR_IMPROVED, 
)

from utils import parse_response, save_articles, build_prompt, validate_omissions

load_dotenv()

def generate_anthropic(client, model_id: str, prompt: str) -> dict:

    response = client.messages.create(
        model=model_id, 
        max_tokens=2000,
        temperature=0.8,
        messages = [
            {
                "role": "user",
                "content": prompt
             }]
    )

    return parse_response(response.content[0].text)

def generate_groq(client, model_id: str, prompt: str) -> dict:

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.8,
        max_completion_tokens=4096,
        top_p=0.95
        
    )

    return parse_response(response.choices[0].message.content)

def generate_gemini(client, model_id: str, prompt: str) -> dict:
    model = genai.GenerativeModel(model_id)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.8)
        )
    return parse_response(response.text)

#==============================================================================
def run_generation(provider_name, client, models, generate_fn):

    for model_name, model_id in models.items():
        
        articles = []
        used_models = []

        MAX_RETRIES = 3

        pbar = tqdm(range(1, NUM_ARTICLES + 1), desc=f"{provider_name}/{model_name}", unit="article")
        for i in pbar:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    prompt, omitted = build_prompt(PROMPT_IMPROVED, used_models)
                    article_data = generate_fn(client, model_id, prompt)

                    if not validate_omissions(article_data, omitted):
                        pbar.set_postfix(status=f"retry {attempt}/{MAX_RETRIES}")
                        continue

                    generated_model = article_data.get("information", {}).get("model_name", "")
                    if generated_model and generated_model != "Not specified":
                        used_models.append(generated_model)

                    article_data['metadata'] = {
                        'generator_model': model_name,
                        'provider': provider_name,
                        'generated_at': datetime.now().isoformat(),
                        'article_number': i
                    }
                    articles.append(article_data)
                    pbar.set_postfix(status="ok")
                    break

                except Exception as e:
                    pbar.set_postfix(status=f"error {attempt}/{MAX_RETRIES}")
                    if attempt == MAX_RETRIES:
                        tqdm.write(f"Skipping article {i}: {e}")

        save_articles(articles, model_name, OUTPUT_DIR_IMPROVED)

def main():
    print("="*60)
    print("Starting improved article generation...")
    print("="*60)

    genai.configure(api_key=os.environ["GEMINI_API"])

    tasks = [
        ('anthropic', Anthropic(), MODELS_ANTHROPIC, generate_anthropic),
        ('groq', Groq(), MODELS_GROQ, generate_groq),
        ('gemini', None, MODELS_GEMINI, generate_gemini),
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for provider, client, models, fn in tasks:
            futures.append(executor.submit(run_generation, provider, client, models, fn))

        for f in futures:
            f.result()

    print("\n" + "="*60)
    print("Génération terminée!")
    print("="*60)

if __name__ == "__main__":
    main()
