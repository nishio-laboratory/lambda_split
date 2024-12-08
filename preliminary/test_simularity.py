import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import os

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

input_ids = tokenizer.encode(initial_text, return_tensors="pt").cuda()

# キャッシュを使わない場合
print("Without cache:")
start_time = time.time()
for i in tqdm(range(300)):
    outputs = model(input_ids, output_hidden_states=True)
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    if next_token == tokenizer.eos_token_id:
        break

print(f"Iteration {i+1}, time taken: {time.time() - start_time:.4f} seconds")
print(tokenizer.decode(input_ids[0]))


# 隠れ層の状態を取得
hidden_states = outputs.hidden_states

# 各トークンに対する隠れ層ベクトルを取得
# hidden_states: [num_layers, batch_size, sequence_length, hidden_size]
num_layers = len(hidden_states)
batch_size = hidden_states[0].shape[0]
sequence_length = hidden_states[0].shape[1]

# 全ての隠れ層に対してlogitsを計算
logits = [[None for _ in range(sequence_length)] for _ in range(num_layers)]
for i, hidden_state in enumerate(hidden_states):
    for token_idx in range(sequence_length):
        # 線形変換（全結合層）を適用
        score = model.lm_head(hidden_state[:, token_idx:token_idx+1, :])
        logit = F.softmax(score, dim=-1)
        logits[i][token_idx] = logit


# logitのtop-k一致率を計算し、ヒートマップを作成
for k in [1, 2, 3, 4, 5]:
    os.makedirs(f'preliminary/heatmap_logit_top{k}', exist_ok=True)
    for token_idx in range(sequence_length):
        matrix = np.zeros((num_layers, num_layers))
        print(f"Token index: {token_idx}")
        for layer_i in range(num_layers):
            for layer_j in range(layer_i, num_layers):
                vec1_topk = torch.topk(logits[layer_i][token_idx], k, dim=-1)[1].detach().cpu().numpy()
                vec2_topk = torch.topk(logits[layer_j][token_idx], k, dim=-1)[1].detach().cpu().numpy()
                similarity = np.sum(np.isin(vec1_topk, vec2_topk)) / k
            
                matrix[layer_i][layer_j] = similarity
                matrix[layer_j][layer_i] = similarity

        # ヒートマップを作成
        fig, ax = plt.subplots(figsize=(20, 15))
        sns.heatmap(matrix, annot=True, ax=ax)
        ax.set_xlabel('Layer index')
        ax.set_ylabel('Layer index')
        ax.invert_yaxis()
        # カラーバーの表示
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0', '0.5', '1'])

        fig.tight_layout()
        fig.savefig(f'preliminary/heatmap_logit_top{k}/{str(token_idx).zfill(3)}.png')
        plt.close()
        del fig, ax

input('Press enter to continue...')

# logitどうしのコサイン類似度を計算し、ヒートマップを作成
os.makedirs(f'preliminary/heatmap_logit_cossim', exist_ok=True)
for token_idx in range(sequence_length):
    matrix = np.zeros((num_layers, num_layers))
    print(f"Token index: {token_idx}")
    for layer_i in range(num_layers):
        for layer_j in range(layer_i, num_layers):
            vec1 = logits[layer_i][token_idx].reshape(1, -1)
            vec2 = logits[layer_j][token_idx].reshape(1, -1)
            similarity = cosine_similarity(vec1.detach().cpu().numpy(), vec2.detach().cpu().numpy())

 
            matrix[layer_i][layer_j] = similarity[0][0]
            matrix[layer_j][layer_i] = similarity[0][0]

    # ヒートマップを作成
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(matrix, annot=True, ax=ax)
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Layer index')
    ax.invert_yaxis()
    # カラーバーの表示
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['0', '0.5', '1'])

    fig.tight_layout()
    fig.savefig(f'preliminary/heatmap_logit_cossim/{str(token_idx).zfill(3)}.png')
    plt.close()
    del fig, ax

'''
# 各トークン、各レイヤー間でのコサイン類似度を計算
for token_idx in range(sequence_length):
    print(f"Token index: {token_idx}")
    for layer_idx in range(num_layers - 1):
        vec1 = hidden_states[layer_idx][0][token_idx].reshape(1, -1)
        vec2 = hidden_states[layer_idx + 1][0][token_idx].reshape(1, -1)
        similarity = cosine_similarity(vec1.detach().cpu().numpy(), vec2.detach().cpu().numpy())
        print(f"Cosine similarity between layer {layer_idx} and layer {layer_idx + 1}: {similarity[0][0]:.4f}")

    print(f'Cossim between layer 0 and layer {num_layers - 1}: {cosine_similarity(hidden_states[0][0][token_idx].reshape(1, -1).detach().cpu().numpy(), hidden_states[num_layers - 1][0][token_idx].reshape(1, -1).detach().cpu().numpy())[0][0]:.4f}')

input('Press enter to continue...')
'''

# 各トークン、各レイヤー間でのコサイン類似度を計算し、ヒートマップを作成
os.makedirs(f'preliminary/heatmap_hiddenstate_cossim', exist_ok=True)
for token_idx in range(sequence_length):
    matrix = np.zeros((num_layers, num_layers))
    print(f"Token index: {token_idx}")
    for layer_i in range(num_layers):
        for layer_j in range(layer_i, num_layers):
            vec1 = hidden_states[layer_i][0][token_idx].reshape(1, -1)
            vec2 = hidden_states[layer_j][0][token_idx].reshape(1, -1)
            similarity = cosine_similarity(vec1.detach().cpu().numpy(), vec2.detach().cpu().numpy())
 
            matrix[layer_i][layer_j] = similarity[0][0]
            matrix[layer_j][layer_i] = similarity[0][0]

    # ヒートマップを作成
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(matrix, annot=True, ax=ax)
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Layer index')
    ax.invert_yaxis()
    # カラーバーの表示
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['0', '0.5', '1'])

    fig.tight_layout()
    fig.savefig(f'preliminary/heatmap_hiddenstate_cossim/{str(token_idx).zfill(3)}.png')
    plt.close()
    del fig, ax

input('Press enter to continue...')

# 各レイヤー、各トークン間でのコサイン類似度を計算
for layer_idx in range(num_layers):
    print(f"Layer index: {layer_idx}")
    for token_idx in range(sequence_length - 1):
        vec1 = hidden_states[layer_idx][0][token_idx].reshape(1, -1)
        vec2 = hidden_states[layer_idx][0][token_idx + 1].reshape(1, -1)
        similarity = cosine_similarity(vec1.detach().cpu().numpy(), vec2.detach().cpu().numpy())
        print(f"Cosine similarity between token {token_idx} and token {token_idx + 1}: {similarity[0][0]:.4f}")