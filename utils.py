import json 
from operator import gt
import os 
import re 
import random
from datetime import datetime 
import google.generativeai as genai
import pandas as pd
import requests
import fitz
from config import (
    OUTPUT_DIR_IMPROVED,
    TEXT_DIR,
    PDF_DIR,
    CORPUS1_PATH,
    CORPUS2_PATH
    )
ALL_FIELDS = ["model_name", "parameter_count", "gpu_count",
             "hardware", "training_duration", "country", "year"]


class GeminiKeyRotator:
    def __init__(self):
        self.keys = [                                                         
              os.environ["GEMINI_API"],
              os.environ["GEMINI_API_2"],
              os.environ["GEMINI_API_3"],
              os.environ["GEMINI_API_4"],
              os.environ["GEMINI_API_5"],
              os.environ["GEMINI_API_VANESSA"],

          ]
        self.index = 0
        self.configure_initial()

    def configure_initial(self):
        genai.configure(api_key=self.keys[self.index])
    
    def rotate(self):
        self.index = (self.index + 1) % len(self.keys)
        genai.configure(api_key=self.keys[self.index])
        

def sample_omissions():
    n_ommit = random.choice(
        [0, 1, 2, 3, 4, 5],
    )

    to_omit = set(random.sample(ALL_FIELDS, min(n_ommit, 5)))

    while len(ALL_FIELDS) - len(to_omit) < 2:
        to_omit.pop()
    
    return to_omit

def build_prompt(base_prompt: str, used_models: list = None) -> tuple[str, set]:
    omitted = sample_omissions()
    included = [f for f in ALL_FIELDS if f not in omitted]
    injection = ""

    if omitted:
        injection += f"""
**MANDATORY OMISSION INSTRUCTION (follow strictly):**
- INCLUDE these fields (with XML tags): {', '.join(included)}
- OMIT these fields (NO mention, NO XML tags): {', '.join(omitted)}
- For omitted fields, use "Not specified" in the JSON output.
"""

    if used_models:
        injection += f"""
**DIVERSITY INSTRUCTION (follow strictly):**
Do NOT use any of these model names (already generated): {', '.join(used_models)}
Choose a completely different model name.
"""

    return base_prompt + injection, omitted

def validate_omissions(result: dict, omitted: set) -> bool:
    info = result.get("information", {})
    for field in omitted:
        if info.get(field, "Not specified") != "Not specified":
            return False
    return True


def clean_json_string(content: str) -> str:
    
    if not content:
        return ''
    
    content = content.strip()

    # Suppression de tags indesirables
    content = re.sub(r'<tool_call>.*?</tool_call>|<think>.*?</think>', '', content, flags=re.DOTALL) 
    content = content.strip()

    # Supprimer les blocs markdown
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Extract the JSON 
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        content = json_match.group()
    return content

def parse_response(raw_content: str)->dict:
    #cleanaing and parsing 
    content = clean_json_string(raw_content)
    if not content:
        raise ValueError("No content to parse after cleaning.")
    return json.loads(content)

def save_articles(articles: list, model_name:str, output_dir: str):
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    for i, article in enumerate(articles, 1):
        txt_path = os.path.join(model_dir, f'article_{i}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(article['article'])
    
    json_path = os.path.join(model_dir, 'articles.json')
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(articles)} articles for model '{model_name}' in '{model_dir}'.")

def clean_article_text(text: str) -> str:
    text = text.replace('\n', ' ').replace("\t", ' ')
    text = re.sub(r' {2,}', ' ', text) 
    return text.strip()

def merge_articles(output_dir: str, data_dir:str):
    os.makedirs(output_dir, exist_ok=True)
    all_articles = []

    for model_name in os.listdir(data_dir):
        json_path = os.path.join(data_dir, model_name, 'articles.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)

                for article in articles:
                    article['article'] = clean_article_text(article.get('article', ''))
                all_articles.extend(articles)
    
    out_path = os.path.join(output_dir, 'all_articles.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(all_articles)} articles into '{out_path}'.")
    return all_articles


def load_links():
    articles = []
    for path, corps_name in [(CORPUS1_PATH, "corpus1"), (CORPUS2_PATH, "corpus2")]:
        df = pd.read_excel(path)
        df = df.iloc[1:]
        for _, row in df.iterrows():
            idx = row['Global_idx']
            link = row['Link']
            if pd.notna(link) and str(link).startswith("http"):
                articles.append({
                    "idx": int(idx),
                    "link": str(link).strip(),
                    "corpus": corps_name
                    })
    return articles


def download_pdfs(articles):
    os.makedirs(PDF_DIR, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    for a in articles:
        pdf_path = os.path.join(PDF_DIR, f'article_{a["idx"]}.pdf')
        if os.path.exists(pdf_path):
            print(f"PDF for article {a['idx']} already exists. Skipping download.")
            continue
        try:
            response = requests.get(a['link'], timeout=30, headers=headers)
            response.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded PDF for article {a['idx']}.")
        except Exception as e:
            print(f"Failed to download PDF for article {a['idx']} from {a['link']}: {e}")
def extract_text_from_pdf():
    os.makedirs(TEXT_DIR, exist_ok=True)
    for pdf_file in os.listdir(PDF_DIR):
        if not pdf_file.endswith('.pdf'):
            continue
        name = pdf_file.replace('.pdf', '')
        text_path = os.path.join(TEXT_DIR, f'{name}.txt')
        if os.path.exists(text_path):
            print(f"Text for {name} already exists. Skipping extraction.")
            continue
        try:
            doc = fitz.open(os.path.join(PDF_DIR, pdf_file))
            text = '\n'.join(page.get_text() for page in doc)
            doc.close()
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Extracted text for {name}.")
        except Exception as e:
            print(f"Failed to extract text from {pdf_file}: {e}")


def load_ground_truth():
    ground_truth = {}
    for path, corpus in [(CORPUS1_PATH, "corpus1"), (CORPUS2_PATH, "corpus2")]:
        df = pd.read_excel(path)
        df = df.iloc[1:]
        for _, row in df.iterrows():
            idx = int(row['Global_idx'])
            gt = {
                "year": row.get('Year'),
                "gpu_count": row.get('Number of GPUs'),
                "tdp": row.get('TDP/W'),
                "training_hours": row.get('Training time.1'),
                "energy_kwh": row.get('Energy cost'),
                "corpus": corpus,
            }
            if corpus == "corpus1":
                gt["country"] = row.get('Countries')
                gt["hardware_text"] = row.get('Resources')
                gt["parameter_count"] = row.get('Number of parameters')
            else:
                gt["country"] = None
                gt["hardware_text"] = row.get('GPU*')
                gt["parameter_count"] = None

            ground_truth[idx] = gt
    return ground_truth

def main():
    #merge_articles('data', OUTPUT_DIR_IMPROVED)
    articles = load_links()

    download_pdfs(articles)
    extract_text_from_pdf()
    #gt = load_ground_truth()                                                      
    #print(f"Loaded {len(gt)} articles")
    #print(gt[11])
if __name__ == "__main__":
    main()