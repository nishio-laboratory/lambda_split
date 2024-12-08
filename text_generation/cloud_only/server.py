# main.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import time
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token_id = 0 # unk


output_text = """Artificial intelligence (AI), machine learning (ML), and deep learning (DL) are related but distinct concepts in the field of computer science. 
Here's a brief overview of each:
Artificial Intelligence (AI):
AI refers to the broader field of research and development aimed at creating machines that can perform tasks that typically require human intelligence, such as understanding language, recognizing images, making decisions, and solving problems. AI involves a wide range of techniques, including rule-based systems, decision trees, and expert systems.
Machine Learning (ML):
ML is a subset of AI that focuses specifically on developing algorithms and statistical models that enable machines to learn from data, without being explicitly programmed. ML involves training machines to recognize patterns in data, make predictions or decisions, and improve their performance over time. Common ML techniques include supervised and unsupervised learning, reinforcement learning, and deep learning.
Deep Learning (DL):
DL is a subset of ML that focuses on developing neural networks with multiple layers, inspired by the structure and function of the human brain. DL algorithms are capable of learning and improving on their own by automatically adjusting the connections between layers, allowing them to learn and represent complex patterns in data. DL has been particularly successful in areas such as computer vision, natural language processing, and speech recognition.
In summary, AI is the broader field that encompasses both ML and DL, while ML is a subset of AI that focuses on developing algorithms that enable machines to learn from data, and DL is a subset of ML that focuses on developing neural networks with multiple layers to learn and represent complex patterns in data."""

ids = tokenizer(output_text, return_tensors="pt")["input_ids"]
text_length = 0

send_text_list = []

for i in range(300):
    text = tokenizer.decode(ids[0, :i])
    send_text = text[text_length:]
    send_text_list.append(send_text)
    text_length = len(text)


def pseudo_generate():
    for i in range(300):
        yield send_text_list[i]


app = FastAPI()
port = 7860

@app.post("/infer_cloud/")
async def pseudo_infer(request: Request):
    data = await request.body()

    return StreamingResponse(pseudo_generate(), media_type="text/plain")


uvicorn.run(app, host="0.0.0.0", port=port)