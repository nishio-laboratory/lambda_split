import torch
import torchinfo
from tqdm import tqdm

from src.cloud import Cloud
from src.edge import Edge
from src.util import Prompter


def main(first_split_layer_indices, second_split_layer_indices):
    # cloud = Cloud(first_split_layer_indices, second_split_layer_indices)
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
        first_feature_vector = edge.infer_first_model(input_ids)

        # TODO : 量子化
        # second_feature_vector = cloud.infer_second_model(first_feature_vector)

        # TODO : 量子化
        output = edge.infer_third_model(first_feature_vector)
        

        # if idx == 0:
        #     with open(f'torchinfo_summary_log/first_{first_split_layer_indices}_{second_split_layer_indices}.txt', 'w') as f:
        #         f.write(repr(torchinfo.summary(edge.first_model, input_data=input_ids, depth=10, col_width=50)))
        #     with open(f'torchinfo_summary_log/second_{first_split_layer_indices}_{second_split_layer_indices}.txt', 'w') as f:
        #         f.write(repr(torchinfo.summary(cloud.second_model, input_data=first_feature_vector, depth=10, col_width=50)))
        #     with open(f'torchinfo_summary_log/third_{first_split_layer_indices}_{second_split_layer_indices}.txt', 'w') as f:
        #         f.write(repr(torchinfo.summary(edge.third_model, input_data=second_feature_vector, depth=10, col_width=50)))


        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, descending=True, dim=-1)
        
        if sorted_ids[0] == edge.tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)

    print(edge.tokenizer.decode(input_ids[0]))


if __name__ == '__main__':
    # first == second の場合、2分割になる
    first_split_layer_indices = {24}
    second_split_layer_indices = {24}

    main(first_split_layer_indices, second_split_layer_indices)