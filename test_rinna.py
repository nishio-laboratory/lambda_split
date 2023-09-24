from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/bilingual-gpt-neox-4b-instruction-sft")
model = AutoModelForCausalLM.from_pretrained("rinna/bilingual-gpt-neox-4b-instruction-sft")

print(model)