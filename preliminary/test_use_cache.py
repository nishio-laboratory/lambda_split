import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# モデルとトークナイザーの初期化
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print(model)

# 初期のテキスト
initial_text = "Once upon a time"
input_ids = tokenizer.encode(initial_text, return_tensors="pt")

# キャッシュを使わない場合
print("Without cache:")
start_time = time.time()
for i in range(500):
    outputs = model(input_ids)
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
print(f"Iteration {i+1}, time taken: {time.time() - start_time:.4f} seconds")

# キャッシュを使う場合
print("\nWith cache:")
input_ids = tokenizer.encode(initial_text, return_tensors="pt")
past_key_values = None
start_time = time.time()
for i in range(500):
    outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
print(f"Iteration {i+1}, time taken: {time.time() - start_time:.4f} seconds")
