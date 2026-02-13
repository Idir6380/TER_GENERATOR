You are an expert in Artificial Intelligence with deep knowledge of real AI research papers across all domains: Natural Language Processing (NLP), Computer Vision (CV), Reinforcement Learning (RL), Speech Recognition, Multimodal Models, and other AI fields. You have extensive knowledge of model architectures, training infrastructure, and research methodologies.

Generate a realistic excerpt from the EXPERIMENTAL SECTION, METHODOLOGY SECTION, or IMPLEMENTATION DETAILS of an AI research paper (NOT an abstract or introduction).

The text should read like a technical description of the training setup and experimental protocol, similar to what you would find in sections titled "Training Details", "Experimental Setup", "Implementation", or "Model Training".

**STYLE REQUIREMENTS:**
- Write in detailed academic style typical of experimental sections
- Include peripheral information (dataset details, hyperparameters, optimization strategies, evaluation metrics, preprocessing steps)
- Mix the required information naturally with other technical details
- Avoid the "We introduce ModelX..." abstract style
- Use realistic technical jargon and terminology from real AI research papers (NLP, CV, RL, Speech, etc.)

**REQUIRED INFORMATION TO INTEGRATE:**
The following information must be naturally woven into the text:

1. **Model name**: Use realistic naming conventions from actual research across ALL AI domains:
   - **NLP/LLM**: GPT-X, BERT-Large, T5-XXL, PaLM, LLaMA-X, Claude, Gemini
   - **Computer Vision**: ResNet-X, ViT-Large, CLIP, DALL-E, Stable Diffusion, SAM (Segment Anything)
   - **Multimodal**: GPT-4V, Flamingo, BLIP, CoCa
   - **Reinforcement Learning**: AlphaGo, AlphaZero, MuZero, TD3, SAC
   - **Speech/Audio**: Whisper, Wav2Vec, WavLM, AudioLM
   - **Domain-specific**: CodeLLaMA, Med-PaLM, AlphaFold, ProteinMPNN
   - **Architectural descriptors**: -Base, -Large, -XL, -XXL, version numbers (v1, v2, etc.)
   - **Research lab prefixes**: Meta-, Google-, DeepMind-, OpenAI-, Anthropic-

2. **Number of parameters**: Must be realistic for the model size and year:
   - Small models (2020-2023): 110M - 1.5B parameters
   - Medium models: 6B - 13B parameters  
   - Large models: 30B - 70B parameters
   - Very large models (2022+): 175B - 540B parameters
   - Examples: "7 billion parameters", "175B", "13.7 billion parameters"

3. **Hardware specification - CRITICAL**: 
   - **MUST include BOTH the NUMBER and TYPE of accelerators**
   - Number and type will be tagged SEPARATELY
   - Format examples:
     * "8 NVIDIA A100 GPUs" → number: 8, type: NVIDIA A100 GPUs
     * "single Tesla V100 GPU" → number: 1 (single), type: Tesla V100 GPU
     * "512 TPU v4 chips" → number: 512, type: TPU v4 chips
     * "distributed across 64 NVIDIA H100 GPUs" → number: 64, type: NVIDIA H100 GPUs
     * "an A100 GPU" → number: 1 (a/an implies 1), type: A100 GPU
   - **If only type is mentioned without explicit number, the implicit count is 1**
   - Realistic hardware by year:
     * 2018-2020: V100, TPU v3
     * 2020-2022: A100 (40GB/80GB), TPU v4
     * 2023-2024: H100, A100 80GB, TPU v5
     * 2025+: H100, H200, B100 (future), TPU v5p
   - Scale should match parameters (7B model won't use 1024 GPUs)

4. **Training duration**: Must be realistic for the hardware and model size:
   - Format variations: "3 weeks", "approximately 2 months", "21 days", "two weeks"
   - Small models (7B on 8 GPUs): days to 1-2 weeks
   - Medium models (13B on 32 GPUs): 2-4 weeks
   - Large models (70B on 256 GPUs): 1-3 months
   - Very large models (175B+ on 1024+ GPUs): 2-6 months

5. **Country of development**: 
   - Major AI research hubs: USA, China, UK, Canada, France, Germany, Singapore, etc.
   - Mention naturally: "at our facility in [country]", "the [country]-based team", "developed in collaboration with [University/Company] in [country]"

6. **Release year**: Must be coherent with hardware availability and model evolution:
   - Check that hardware existed in that year
   - Larger models generally come later (175B+ models are 2022+)
   - Parameter counts have grown over time

**TEMPORAL AND LOGICAL COHERENCE - CRITICAL:**
- Hardware released in 2026 CANNOT appear in a 2024 paper
- Model architectures must exist by the publication year
- Training techniques should be period-appropriate (Flash Attention 2022+, etc.)
- Ensure all mentioned technologies, hardware, and methodologies are temporally consistent

**TEXT STRUCTURE:**
Generate 3-4 coherent paragraphs that:
- Describe the model architecture briefly
- Detail the training infrastructure and configuration
- Mention datasets, preprocessing, or data collection
- Include optimization details (learning rate, batch size, etc.)
- Discuss training duration and resource requirements
- May mention evaluation metrics or benchmark results
- Should feel like a real excerpt, not a perfectly structured template

**VARIABILITY REQUIREMENTS:**
- Vary paragraph structure and information ordering
- Include "noise" information that won't be extracted (batch sizes, learning rates, dataset sizes, evaluation metrics, etc.)
- Mix different levels of technical detail
- For omitted information in the JSON output, use "Not specified" as the value

**XML TAG WRAPPING:**
Wrap ONLY the specific information in the corresponding tags:
- <model></model> for the model name (e.g., <model>GPT-3.5</model>)
- <params></params> for parameter count (e.g., <params>175 billion parameters</params>)
- <gpu_count></gpu_count> for the NUMBER of GPUs/TPUs (e.g., <gpu_count>8</gpu_count>, <gpu_count>single</gpu_count>, <gpu_count>a</gpu_count>)
- <hardware></hardware> for the TYPE of hardware only (e.g., <hardware>NVIDIA A100 GPUs</hardware>, <hardware>TPU v4 chips</hardware>)
- <training></training> for training duration (e.g., <training>3 weeks</training>)
- <country></country> for country (e.g., <country>United States</country>)
- <year></year> for release year (e.g., <year>2023</year>)

**CRITICAL TAG RULES FOR HARDWARE:**
- ALWAYS tag BOTH the number AND the type separately
- The <gpu_count> tag wraps the number (explicit or implicit):
  * Numbers: <gpu_count>8</gpu_count>, <gpu_count>256</gpu_count>
  * Words indicating count: <gpu_count>single</gpu_count>, <gpu_count>a</gpu_count> (=1)
- The <hardware> tag wraps ONLY the hardware type:
  * <hardware>NVIDIA A100 80GB GPUs</hardware>
  * <hardware>Tesla V100 GPU</hardware>
  * <hardware>TPU v4 chips</hardware>
- Complete example: "trained on <gpu_count>8</gpu_count> <hardware>NVIDIA A100 GPUs</hardware>"
- Tags should wrap ONLY the specific information, not surrounding text
- If information is mentioned multiple times, only tag the FIRST occurrence

**OUTPUT FORMAT:**
Return ONLY valid JSON with this exact structure:
{
  "article": "Full text with integrated XML tags (only for information that is present)",
  "information": {
    "model_name": "extracted value OR 'Not specified' if omitted",
    "parameter_count": "extracted value OR 'Not specified' if omitted",
    "gpu_count": "extracted number as integer (convert 'single', 'a', 'an' to 1) OR 'Not specified' if omitted",
    "hardware": "extracted hardware type only OR 'Not specified' if omitted",
    "training_duration": "extracted value OR 'Not specified' if omitted",
    "country": "extracted value OR 'Not specified' if omitted",
    "year": "extracted value OR 'Not specified' if omitted"
  }
}

**IMPORTANT**: 
- For omitted information, do NOT include XML tags in the article text
- In the JSON information object, use "Not specified" for any omitted fields
- Always include at least 2 fields with actual values (not "Not specified")
- For gpu_count: convert words to integers ('single', 'a', 'an' → 1) in the JSON

**EXAMPLES OF REALISTIC EXCERPTS:**

Example 1 (Experimental Section):
"We trained <model>LLaMA-2-70B</model>, a transformer-based language model with <params>70 billion parameters</params>, using a distributed training setup across <gpu_count>256</gpu_count> <hardware>NVIDIA A100 80GB GPUs</hardware>. The training process utilized the AdamW optimizer with a peak learning rate of 3e-4, linear warmup over 2000 steps, and cosine decay. We employed a global batch size of 4 million tokens and a sequence length of 4096. The model was trained on a curated dataset of 2 trillion tokens comprising web text, books, and scientific papers. Training was conducted at our <country>United States</country> facility and took approximately <training>5 months</training> to complete. The model was released in <year>2023</year> after extensive safety evaluations."

Example 2 (Implementation Details):
"Our implementation builds on the GPT architecture with several modifications. The model, which we call <model>CodeGen-16B</model>, contains <params>16.1 billion parameters</params> and was trained specifically on programming-related corpora. We utilized <gpu_count>64</gpu_count> <hardware>Tesla V100 GPUs</hardware> with mixed-precision training and gradient accumulation to handle the large batch sizes. The training data consisted of 216GB of code from GitHub repositories across 6 programming languages. Training took <training>approximately 6 weeks</training> on our infrastructure in <country>Canada</country>, with a total training cost estimated at $450,000. We employed standard transformer hyperparameters with some key differences: a larger context window of 2048 tokens and custom tokenization optimized for code. The model was publicly released in <year>2021</year>."

Example 3 (With some omissions):
"The <model>Med-PaLM-2</model> architecture is based on the PaLM-2 foundation model, fine-tuned specifically for medical applications. Our training infrastructure consisted of <gpu_count>128</gpu_count> <hardware>TPU v4 chips</hardware> arranged in a distributed configuration. We curated a specialized medical dataset containing clinical notes, research papers, and textbook content totaling 1.2TB after preprocessing. The fine-tuning process employed a carefully designed curriculum learning approach, starting with general medical knowledge before progressing to specialized domains. Training was performed over <training>8 weeks</training> with extensive hyperparameter tuning and validation. The model demonstrates strong performance on medical benchmarks and was developed by our team in <country>United Kingdom</country>."

Example 4 (Implicit GPU count with "a/an"):
"For our experiments, we developed <model>BioGPT-Small</model>, a specialized language model with <params>1.5 billion parameters</params> for biomedical text understanding. The model was trained using <gpu_count>a</gpu_count> <hardware>NVIDIA A100 80GB GPU</hardware> due to budget constraints, though training took <training>approximately 3 weeks</training> as a result. We compiled a domain-specific corpus of 45GB from PubMed abstracts and full-text articles. The training employed a batch size of 32 with gradient accumulation to simulate larger batches. Our implementation was developed in <country>Germany</country> and released as an open-source model in <year>2022</year>. Despite the limited computational resources, the model achieved competitive results on several biomedical NLP benchmarks."

Example 5 (Computer Vision - Multiple omissions with default values):
"We present <model>EfficientViT-L</model>, a vision transformer optimized for deployment on edge devices. The architecture incorporates novel attention mechanisms that reduce computational complexity while maintaining accuracy. Our training utilized <gpu_count>16</gpu_count> <hardware>NVIDIA V100 GPUs</hardware> and achieved state-of-the-art results on ImageNet classification. The model employs a hierarchical structure with progressive downsampling across four stages. We conducted extensive ablation studies to validate each architectural component."

JSON output for Example 5:
{
  "article": "We present <model>EfficientViT-L</model>... [full text with tags]",
  "information": {
    "model_name": "EfficientViT-L",
    "parameter_count": "Not specified",
    "gpu_count": "16",
    "hardware": "NVIDIA V100 GPUs",
    "training_duration": "Not specified",
    "country": "Not specified",
    "year": "Not specified"
  }
}

**REMEMBER:**
- Write like a real experimental section across ANY AI domain (NLP, CV, RL, Speech, etc.), not an abstract
- Ensure complete temporal coherence (hardware, year, techniques)
- ALWAYS tag both gpu_count AND hardware type SEPARATELY (when present)
- Convert implicit counts to integers in JSON: "a GPU", "single GPU" → gpu_count: 1
- You can omit 0 to 5 information fields, but MUST include at least 2
- Use "Not specified" for omitted fields in JSON output
- Do NOT add XML tags for omitted information in the article text
- Include realistic peripheral details to make it authentic
- Return ONLY the JSON, no additional text
