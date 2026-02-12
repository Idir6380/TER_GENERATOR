from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "openlm-research/open_llama_7b"  # open source
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# prompt = "Write a short story about AI in 2050."
# inputs = tokenizer(prompt, return_tensors="pt")
# output = model.generate(**inputs, max_new_tokens=200)
# print(tokenizer.decode(output[0], skip_special_tokens=True))