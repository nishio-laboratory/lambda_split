import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
from tqdm import tqdm

# モデルとトークナイザーの初期化
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                           torch_dtype=torch.float16,
                                           device_map={"": "cuda"})
print(model)

# 初期のテキスト
initial_text = '''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is the difference between AI, ML and DL?

### Response:
'''

# input_ids = tokenizer.encode(initial_text, return_tensors="pt").cuda()

# # キャッシュを使わない場合
# print("Without cache:")
# start_time = time.time()
# for i in tqdm(range(300)):
#     outputs = model(input_ids)
#     next_token_logits = outputs.logits[:, -1, :]
#     next_token = torch.argmax(next_token_logits, dim=1)
#     input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

#     if i % 50 == 0:
#         torch.cuda.empty_cache()

# print(f"Iteration {i+1}, time taken: {time.time() - start_time:.4f} seconds")
# print(tokenizer.decode(input_ids[0]))

# torch.cuda.empty_cache()

# キャッシュを使う場合
print("\nWith cache:")
input_ids = tokenizer.encode(initial_text, return_tensors="pt").cuda()
past_key_values = None
start_time = time.time()
for i in tqdm(range(300)):
    try:
        outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    except Exception as e:
        torch.cuda.empty_cache()
        print('torch.cuda.OutOfMemoryError')
    
print(f"Iteration {i+1}, time taken: {time.time() - start_time:.4f} seconds")
print(tokenizer.decode(input_ids[0]))
