import json 
import os 
import re 
import random
from datetime import datetime 

ALL_FIELDS = ["model_name", "parameter_count", "gpu_count",
             "hardware", "training_duration", "country", "year"]



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
