import io

import numpy as np
import torch
import torchinfo
from tqdm import tqdm

from src.cloud_save_bandwidth import Cloud
from src.edge_save_bandwidth import Edge
from src.util import Prompter


def main(first_split_layer_indices, second_split_layer_indices, random_seed=42):
    cloud = Cloud(first_split_layer_indices, second_split_layer_indices)
    edge = Edge(first_split_layer_indices, second_split_layer_indices)

    first_split_layer_indices = sorted(list(first_split_layer_indices))
    second_split_layer_indices = sorted(list(second_split_layer_indices))

    num_first_split_layer_indices = len(first_split_layer_indices)
    num_second_split_layer_indices = len(second_split_layer_indices)

    # 乱数生成器
    rng = np.random.default_rng(random_seed)

    instruction = "Tell me about Japan."
    input = None
    prompter = Prompter('')
    prompt = prompter.generate_prompt(instruction, input)
    max_new_tokens = 100

    inputs = edge.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(edge.device)

    total_bit = 0
    total_bit_save_bandwidth = 0

    for idx in tqdm(range(max_new_tokens)):
        print(idx)

        # Triadic split computing : edge -> cloud -> edge
        ## First model
        ### 分割するレイヤ番号を乱数で決める
        split_first_layer_relative_index = rng.integers(0, num_first_split_layer_indices)
        split_first_layer_index = first_split_layer_indices[split_first_layer_relative_index]

        ### 0 から split_first_layer_index の層まで推論する
        first_feature_vector = edge.infer_first_model(input_ids, split_first_layer_index)
        first_feature_vector_for_send = edge.get_first_feature_vector_for_send(first_feature_vector, split_first_layer_index)

        total_bit += measure_tensor_size_in_memory(first_feature_vector)
        total_bit_save_bandwidth += measure_tensor_size_in_memory(first_feature_vector_for_send)
        print(first_feature_vector.shape, first_feature_vector_for_send.shape)


        ## Second model
        ### 分割するレイヤ番号を乱数で決める
        split_second_layer_relative_index = rng.integers(0, num_second_split_layer_indices)
        split_second_layer_index = second_split_layer_indices[split_second_layer_relative_index]

        ### split_first_layer_index から split_second_layer_index の層まで推論する
        second_feature_vector = cloud.infer_second_model(first_feature_vector_for_send, split_first_layer_index, split_second_layer_index)
        second_feature_vector_for_send = cloud.get_second_feature_vector_for_send(second_feature_vector, split_second_layer_index)

        total_bit += measure_tensor_size_in_memory(second_feature_vector)
        total_bit_save_bandwidth += measure_tensor_size_in_memory(second_feature_vector_for_send)
        print(second_feature_vector.shape, second_feature_vector_for_send.shape)
        

        ## Third model
        ### split_second_layer_index から 最後の層 (self.num_decoder_layers) まで推論する
        output = edge.infer_third_model(second_feature_vector_for_send, split_second_layer_index)

        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, descending=True, dim=-1)
        
        if sorted_ids[0] == edge.tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)

        '''
        with open(f'torchinfo_summary_log/first_{first_split_layer_indices}_{second_split_layer_indices}.txt', 'w') as f:
            f.write(repr(torchinfo.summary(edge.first_model, input_data=input_ids, depth=10, col_width=50)))
        with open(f'torchinfo_summary_log/second_{first_split_layer_indices}_{second_split_layer_indices}.txt', 'w') as f:
            f.write(repr(torchinfo.summary(cloud.second_model, input_data=first_feature_vector, depth=10, col_width=50)))
        with open(f'torchinfo_summary_log/third_{first_split_layer_indices}_{second_split_layer_indices}.txt', 'w') as f:
            f.write(repr(torchinfo.summary(edge.third_model, input_data=second_feature_vector, depth=10, col_width=50)))
        '''

    print(edge.tokenizer.decode(input_ids[0]))
    total_bit /= 1024 ** 2
    total_bit_save_bandwidth /= 1024 ** 2
    print(f'{total_bit=} Mbit, {total_bit_save_bandwidth=} Mbit')


# テンソルのシリアル化サイズ(bit)を測定する関数
def measure_tensor_size_in_memory(
        tensor: torch.Tensor,
        library: str = 'numpy'
    ) -> int:
    buffer = io.BytesIO()

    if library == 'numpy':
        tensor = tensor.to('cpu').detach().numpy().copy()
        np.save(buffer, tensor, allow_pickle=False)
        
    elif library == 'torch':
        torch.save(tensor, buffer)

    byte_size = len(buffer.getvalue())
    bit_size = byte_size * 8
    return bit_size


if __name__ == '__main__':
    # first, second = {0}, {0} or {32}, {32} or {0}, {32} の場合、decoder layersは分割されない
    # first == second の場合、2分割になる
    # first != second の場合、3分割になる
    first_split_layer_indices = {0, 1, 2, 3, 4, 5}
    second_split_layer_indices = {32, 31, 30, 29, 28, 27}

    main(first_split_layer_indices, second_split_layer_indices)