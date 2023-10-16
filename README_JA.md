# Λ-Split: プライバシに配慮した生成AIサービスに向けた三分割 Split Computing フレームワーク

Local -> Cloud -> Local で分割する3分割 (Triadic) Split Computing for LLM


## 技術的要点

- 3分割することで、通信はすべて中間層出力の特徴ベクトルで行われるため、通信の安全性が高まる
- 生成AIモデルはサイズが大きいので、Splitしてサイズを小さくして多くのデバイスでロード可能にする
- LLM実装は、MetaのOpen Source LLMであるLLaMa-2 または LLaMa を使用
- Diffusion model実装には、Stable Diffusion XLを使用



## 具体的な分割の操作
1. 推論時の推論レイヤを正しく分割するための、モデルのforwardメソッドのoverride（`src/models.py` の `FirstLlamaModel` などの `forward` メソッド内でコメントアウトすることで実装）
2. メモリ使用量削減のため、不要なレイヤを Identity レイヤで置き換える（`src/models.py` の `FirstLlamaModel` などの `replace_unused_layers_with_identity` メソッドを実装）


## ファイルの説明

- `main.py` : メインプログラム
- `src/cloud.py` : クラウドクラス（first modelとthird modelを推論）
- `src/edge.py` : エッジクラス（second modelを推論）
- `src/base.py` : クラウドサーバ・エッジサーバの継承元クラス
- `src/split_models.py` : 分割用のLLMクラスである`FirstLlamaModel`・`FirstLlamaForCausalLM`・`SecondLlamaModel`・`SecondLlamaForCausalLM`・`ThirdLlamaModel`・`ThirdLlamaForCausalLM`が定義されている
- `src/utils.py` : 推論のためのutils
- `torchinfo_summary_log/` : 分割したLLMの `torchinfo.summary` の結果


## 実行方法

`main.py` の first split layer の index の集合 `first_split_layer_indices` と second split layer の index の集合 `second_split_layer_indices` を変更して、

```bash
python3 main.py
```

初回時はPre-trainedモデルのダウンロードが必要。
LLaMa-2 では、https://note.com/npaka/n/n79eebc29366d の3.1の利用申請と3.2の `huggingface-cli login` をする必要がある。
