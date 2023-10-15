# client.py
import requests

url = "http://192.168.1.221:7860/infer_cloud/"
headers = {"Content-Type": "text/plain"}

input_text = 'What is the difference between AI, ML and DL?'

response = requests.post(url, data=input_text, headers=headers, stream=True)

for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    print(chunk, end='', flush=True)

print()