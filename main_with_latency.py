import io
import time

import numpy as np
import torch
import torchinfo
from tqdm import tqdm

from src.cloud import Cloud
from src.edge import Edge
from src.util import Prompter


BANDWIDTH = 500 * (1024 ** 2) # 100 Mbit/s


# テンソルのシリアル化サイズを測定する関数
def measure_tensor_size_in_memory(tensor: torch.Tensor) -> int:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    size = buffer.tell()
    return size * 8


def main(first_split_layer_indices, second_split_layer_indices):
    cloud = Cloud(first_split_layer_indices, second_split_layer_indices)
    edge = Edge(first_split_layer_indices, second_split_layer_indices)

    instruction = "Tell me about Japan."
    input = None
    prompter = Prompter('')
    prompt = prompter.generate_prompt(instruction, input)
    max_new_tokens = 100

    inputs = edge.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(edge.device)

    for idx in tqdm(range(max_new_tokens)):
        print(idx)

        # Triadic split computing : edge -> cloud -> edge
        ## First model
        first_feature_vector = edge.infer_first_model(input_ids)
        tensor_size_bit = measure_tensor_size_in_memory(first_feature_vector)
        latency = tensor_size_bit / BANDWIDTH
        print(f'{tensor_size_bit=} bit, {latency=} s')
        time.sleep(latency)

        ## Second model
        second_feature_vector = cloud.infer_second_model(first_feature_vector)
        tensor_size_bit = measure_tensor_size_in_memory(second_feature_vector)
        latency = tensor_size_bit / BANDWIDTH
        print(f'{tensor_size_bit=} bit, {latency=} s')
        time.sleep(latency)

        ## Third model
        output = edge.infer_third_model(second_feature_vector)
        

        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, descending=True, dim=-1)
        
        if sorted_ids[0] == edge.tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)

    print(edge.tokenizer.decode(input_ids[0]))


if __name__ == '__main__':
    # first, second = {0}, {0} or {32}, {32} or {0}, {32} の場合、decoder layersは分割されない
    # first == second の場合、2分割になる
    # first != second の場合、3分割になる
    first_split_layer_indices = {0, 1, 2, 3, 4, 5}
    second_split_layer_indices = {32, 31, 30, 29, 28, 27}

    main(first_split_layer_indices, second_split_layer_indices)