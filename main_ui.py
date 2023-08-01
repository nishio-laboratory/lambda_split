import torch
import torchinfo
from tqdm import tqdm
import gradio as gr

from src.cloud import Cloud
from src.edge import Edge
from src.util import Prompter


first_split_layer_indices = {0, 1, 2, 3, 4, 5}
second_split_layer_indices = {32, 31, 30, 29, 28, 27}

cloud = Cloud(first_split_layer_indices, second_split_layer_indices)
edge = Edge(first_split_layer_indices, second_split_layer_indices)


def main(message, history):
    instruction = message
    input = None
    prompter = Prompter('')
    prompt = prompter.generate_prompt(instruction, input)
    max_new_tokens = 1000

    inputs = edge.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(edge.device)

    for idx in tqdm(range(max_new_tokens)):
        print(idx)

        # Triadic split computing : edge -> cloud -> edge
        ## First model
        first_feature_vector = edge.infer_first_model(input_ids)

        ## Second model
        second_feature_vector = cloud.infer_second_model(first_feature_vector)

        ## Third model
        output = edge.infer_third_model(second_feature_vector)


        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, descending=True, dim=-1)
        
        if sorted_ids[0] == edge.tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
        
        yield prompter.get_response(edge.tokenizer.decode(input_ids[0]))


if __name__ == '__main__':
    demo = gr.ChatInterface(
        fn=main,
        title='Demo : Triadic Split Computing for LLM'
    ).queue()
    demo.launch(ssl_verify=False, server_name='0.0.0.0')