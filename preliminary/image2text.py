import torch
from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor
from PIL import Image
import requests

# helper function to format input prompts
def build_prompt(prompt="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    user_query = "与えられた画像について、詳細に述べてください。"
    msgs = [": \n" + user_query, ": "]
    if prompt:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + prompt)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

# load model
model = AutoModelForVision2Seq.from_pretrained("stabilityai/japanese-instructblip-alpha", trust_remote_code=True)
processor = BlipImageProcessor.from_pretrained("stabilityai/japanese-instructblip-alpha")
tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# prepare inputs
url = "https://images.unsplash.com/photo-1582538885592-e70a5d7ab3d3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
prompt = "" # input empty string for image captioning. You can also input questions as prompts 
prompt = build_prompt(prompt)
inputs = processor(images=image, return_tensors="pt")
text_encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
text_encoding["qformer_input_ids"] = text_encoding["input_ids"].clone()
text_encoding["qformer_attention_mask"] = text_encoding["attention_mask"].clone()
inputs.update(text_encoding)

# generate
outputs = model.generate(
    **inputs.to(device, dtype=model.dtype),
    num_beams=5,
    max_new_tokens=32,
    min_length=1,
)
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
# 桜と東京スカイツリー
