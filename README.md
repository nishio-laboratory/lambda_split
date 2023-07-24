# Triadic Split Computing

Edge -> Cloud -> Edge で分割する3分割 (Triadic) Split Computing for LLM


## 技術的要点

- LLM実装は、MetaのOpen Source LLMであるLLaMaを、Stanford Alpacaのデータセットを用いてLow-rank adaptation（LoRA）でファインチューニングした Alpaca-lora [1] を使用
- Triadic にすることで、通信はすべて中間層出力の特徴ベクトルで行われるため、安全性が高い
- Transformerベースの言語モデルがDecoderを複数重ねていることを利用し、分割するレイヤを毎回変えることで、特徴ベクトル形状は同じものの、異なるレイヤの中間層出力を送っているため、復元が難しい
- 特徴ベクトルはサイズが大きいため、Int8量子化 [2] をして送信する（未実装）

[1] https://github.com/tloen/alpaca-lora

[2] https://speakerdeck.com/joisino/shen-ceng-moderunogao-su-hua?slide=31



## 具体的な分割の操作
1. 推論時の推論レイヤを正しく分割するための、モデルのforwardメソッドのoverride (`src/models.py` の `FirstLlamaModel` の `forward` メソッド内でコメントアウトすることで実装)
2. メモリ使用量削減のため、不要なレイヤを Identity レイヤで置き換える (`src/models.py` の `FirstLlamaModel` の `replace_unused_layers_with_identity` メソッドを実装)


## ファイルの説明

- `main_3_split.py` : 3分割するメインプログラム
- `main_2_split.py` : 2分割するメインプログラム（bug fixのための実験用）
- `main_1_split.py` : 分割しないメインプログラム（bug fixのための実験用）
- `src/cloud.py` : クラウドクラス（first modelとthird modelを推論）
- `src/edge.py` : エッジクラス（second modelを推論）
- `src/base.py` : クラウドサーバ・エッジサーバの継承元クラス
- `src/models.py` : 分割用のLLMクラスである`FirstLlamaModel`・`FirstLlamaForCausalLM`・`SecondLlamaModel`・`SecondLlamaForCausalLM`・`ThirdLlamaModel`・`ThirdLlamaForCausalLM`が定義されている
- `src/utils.py` : LLaMa推論のためのutilを公式から持ってきている
- `torchinfo_summary_log/?_1_31` : `main_3_split.py` で `first_split_layer_indices = {1}` と `second_split_layer_indices = {31}` を指定した場合の `torchinfo.summary` の結果


## 実行方法

`main_?_split.py` の first split layer の index の集合 `first_split_layer_indices` と second split layer の index の集合 `second_split_layer_indices` を変更して、

```bash
python3 main_?_split.py
```

開始時にモデルのローディングで1分くらいかかる。


## 発生しているバグ（乱数による仕様の可能性もある）

まずは、分割レイヤのランダム性を排除して実験しているが、正しく分割推論できていないことがある。

また、Greedy推論しかしていないのでランダム性がないはずだが、結果が一致しない。

**`first_split_layer_indices` と `second_split_layer_indices` を変化させたときの `instruction = 'Tell me about Japan.'` のレスポンス一覧**


| Number of split | File            | First | Second | Response                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------|-----------------|-------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1      | main_1_split.py | {32}  | {32}   | Japan is an island nation in East Asia located off the coast of the Asian continent. It is the world's third-largest island country, with a population of over 126 million people. Japan is known for its unique culture, which is heavily influenced by its long history and its unique geography. It is also known for its advanced technology, its vibrant cities, and its beautiful natural landscapes. |
| 1      | main_1_split.py | {0}   | {0}    | Â               Â ) Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â                                                                                                                                                                                                                                                                                                                                                   |
| 1      | main_3_split.py | {0}   | {32}   | Â               Â ) Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â                                                                                                                                                                                                                                                                                                                                                   |
| 2      | main_2_split.py | {8}   | {8}    | Japan is an island nation in East Asia. It is made up of four main islands, Honshu, Hokkaido, Kyushu, and Shikoku. The country is known for its rich culture, cuisine, and history. It is also known for its unique and vibrant cities.                                                                                                                                                                     |
| 2      | main_2_split.py | {16}  | {16}   | Japan is an island nation in East Asia, located off the coast of mainland Asia. It is home to over 127 million people, making it the world's third-largest country by population. Japan is known for its rich culture, history, and cuisine. It is also known for its advanced technology and its unique language, Japanese.                                                                                |
| 2      | main_2_split.py | {24}  | {24}   | Japan is an island nation in East Asia located off the coast of the Asian mainland. It is home to over 126 million people and is known for its rich culture, history, and cuisine. Japan is a highly developed country with a strong economy and a vibrant society. It is also known for its unique and distinctive art, literature, and music.                                                             |
| 3      | main_3_split.py | {1}   | {31}   | I am                                                                                                                                                                                                                                                                                                                                                                                                        |
| 3      | main_3_split.py | {4}   | {28}   | Japan is an island nation in the Pacific Ocean. It is a constitutional monarchy with a population of 127 million. The capital is Tokyo. The official language is Japanese. The currency is the yen. The currency is the yen. The currency is the yen. The currency is the yen. The currency is the yen. The currency is the yen. The currency is the yen. The currency is the yen. The currency is the y    |
| 3      | main_3_split.py | {8}   | {24}   | Japan is an island nation in East Asia. It is made up of four main islands, Honshu, Hokkaido, Kyushu, and Shikoku. The country is known for its rich culture, cuisine, and history. It is also known for its unique and vibrant cities.                                                                                                                                                                     |
| 3      | main_3_split.py | {12}  | {20}   | Japan is an island nation in East Asia, located off the coast of mainland Asia. It is home to over 127 million people, making it the world's third largest population. The country is known for its rich culture, history, and cuisine. It is also known for its advanced technology and innovation.                                                                                                        |
| 3      | main_3_split.py | {16}  | {20}   | Japan is an island nation in East Asia, located off the coast of mainland Asia. It is home to over 127 million people, making it the world's third-largest country by population. Japan is known for its rich culture, history, and cuisine. It is also known for its advanced technology and its unique language, Japanese.                                                                                |
| 3      | main_3_split.py | {24}  | {28}   | Japan is an island nation in East Asia located off the coast of the Asian mainland. It is home to over 126 million people and is known for its rich culture, history, and cuisine. Japan is a highly developed country with a strong economy and a vibrant society. It is also known for its unique and distinctive art, literature, and music.                                                             |