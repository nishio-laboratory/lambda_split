# Triadic Split Computing

Edge -> Cloud -> Edge で分割する3分割 (Triadic) Split Computing for LLM


## 技術的要点

- Triadic にすることで、通信はすべて中間層出力の特徴ベクトルで行われるため、安全性が高い
- Transformerベースの言語モデルがDecoderを複数重ねていることを利用し、分割するレイヤを毎回変えることで、特徴ベクトル形状は同じものの、異なるレイヤの中間層出力を送っているため、復元が難しい
- LLMはサイズが大きいので、Splitしてサイズを小さくして多くのデバイスでロード可能にする
- (特徴ベクトルはサイズが大きいため、Int8量子化 [2] をして送信する or Dropoutする)
- LLM実装は、MetaのOpen Source LLMであるLLaMa-2 または LLaMa を使用

[1] https://speakerdeck.com/joisino/shen-ceng-moderunogao-su-hua?slide=7



## 具体的な分割の操作
1. 推論時の推論レイヤを正しく分割するための、モデルのforwardメソッドのoverride（`src/models.py` の `FirstLlamaModel` などの `forward` メソッド内でコメントアウトすることで実装）
2. メモリ使用量削減のため、不要なレイヤを Identity レイヤで置き換える（`src/models.py` の `FirstLlamaModel` などの `replace_unused_layers_with_identity` メソッドを実装）


## ファイルの説明

- `main.py` : メインプログラム
- `src/cloud.py` : クラウドクラス（first modelとthird modelを推論）
- `src/edge.py` : エッジクラス（second modelを推論）
- `src/base.py` : クラウドサーバ・エッジサーバの継承元クラス
- `src/models.py` : 分割用のLLMクラスである`FirstLlamaModel`・`FirstLlamaForCausalLM`・`SecondLlamaModel`・`SecondLlamaForCausalLM`・`ThirdLlamaModel`・`ThirdLlamaForCausalLM`が定義されている
- `src/utils.py` : LLaMa推論のためのutilを公式から持ってきている
- `torchinfo_summary_log/` : 分割したLLMの `torchinfo.summary` の結果


## 実行方法

`main.py` の first split layer の index の集合 `first_split_layer_indices` と second split layer の index の集合 `second_split_layer_indices` を変更して、

```bash
python3 main.py
```

初回時はPre-trainedモデルのダウンロードが必要。
LLaMa-2 では、https://note.com/npaka/n/n79eebc29366d の3.1の利用申請と3.2の `huggingface-cli login` をする必要がある。