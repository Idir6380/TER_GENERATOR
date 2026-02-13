with open('prompt.md', 'r') as f:
    PROMPT_IMPROVED  = f.read()


PROMPT = """Generate a fictional scientific research paper excerpt (academic publication) in the field of artificial intelligence, consisting of 3 to 4 coherent paragraphs.

Write in academic style.

The paragraph must naturally integrate the following information:
- Model name (fictional)
- Number of parameters
- Hardware used (GPU/TPU)
- Training duration
- Country of development
- Release year

**Important**: All information must be temporally and logically coherent. For example, a GPU released in 2026 cannot be cited in a paper published in 2024. Ensure that the hardware, technologies, and methodologies mentioned are consistent with the publication year.

In the text, wrap each piece of information with the corresponding XML tags:
- <model></model> for the model name
- <params></params> for the number of parameters
- <hardware></hardware> for the hardware
- <training></training> for the training duration
- <country></country> for the country
- <year></year> for the year

Provide the complete result in the following JSON format:
{
  "article": "Full paragraph text with integrated XML tags",
  "information": {
    "model_name": "value",
    "parameter_count": "value",
    "hardware": "value",
    "training_duration": "value",
    "country": "value",
    "year": "value"
  }
}

IMPORTANT: Return ONLY the JSON, no additional text before or after."""

PROMPT_REAL = """Based on real scientific research papers in artificial intelligence, generate a realistic research paper excerpt, consisting of 3 to 4 coherent paragraphs.

Use your knowledge of real AI papers to ensure realistic and coherent information (model architectures, parameter counts, hardware, training times, etc.).

Write in academic style.

The paragraph must naturally integrate the following information:
- Model name (realistic)
- Number of parameters
- Hardware used (GPU/TPU)
- Training duration
- Country of development
- Release year

**Important**: All information must be temporally and logically coherent. For example, a GPU released in 2026 cannot be cited in a paper published in 2024. Ensure that the hardware, technologies, and methodologies mentioned are consistent with the publication year.

In the text, wrap each piece of information with the corresponding XML tags:
- <model></model> for the model name
- <params></params> for the number of parameters
- <hardware></hardware> for the hardware
- <training></training> for the training duration
- <country></country> for the country
- <year></year> for the year

Provide the complete result in the following JSON format:
{
  "article": "Full paragraph text with integrated XML tags",
  "information": {
    "model_name": "value",
    "parameter_count": "value",
    "hardware": "value",
    "training_duration": "value",
    "country": "value",
    "year": "value"
  }
}

IMPORTANT: Return ONLY the JSON, no additional text before or after."""

MODELS_ANTHROPIC = {
    "claude-sonnet": "claude-sonnet-4-20250514",
}

MODELS_GROQ = {
    #"kimi": "moonshotai/kimi-k2-instruct-0905",
    "qwen": "qwen/qwen3-32b",
}

MODELS_GEMINI = {
    #"gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
}

MODELS_OLLAMA = {
      "qwen3": "qwen/qwen3-8b:free",
      "gemma3": "google/gemma-3-4b-it:free",
      "mistral": "mistralai/mistral-7b-instruct:free"
}

NUM_ARTICLES = 100

OUTPUT_DIR = "output"
OUTPUT_DIR_IMPROVED = "output_improved"
