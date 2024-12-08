# Λ-Split

This repository provides demonstration programs that apply the Λ-Split to LLMs, including [Llama 2](https://huggingface.co/meta-llama), and diffusion models, including [Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).


## Videos

### Text generation using Llama 2 with GUI



https://github.com/nishio-laboratory/lambda_split/assets/63134290/e07ce6e0-df56-48aa-981f-544b46a23d03



### Text generation using Llama 2 with HTTP communication



https://github.com/nishio-laboratory/lambda_split/assets/63134290/91f91313-cccf-45e6-a04b-3ee26fc1eae9




### Image generation using SDXL with GUI




https://github.com/nishio-laboratory/lambda_split/assets/63134290/93832dcd-d547-4675-bbc4-498771728e61




## Usage

Python version : 3.8 or later

```bash
python3 -m pip install -r requirements.txt
```

### Text generation using Llama 2

1. You must agree to Meta's license as stated on the [Huggingface page](https://huggingface.co/meta-llama).

2. Execute the following command
```bash
cd text_generation
python3 main.py
```


### Text generation using Llama 2 with HTTP communication

1. You must agree to Meta's license as stated on the [Huggingface page](https://huggingface.co/meta-llama).


2. Prepare 2 computers for cloud server and local device.


3. Execute the following command on each computer

Cloud
```bash
cd text_generation
python3 cloud_main.py
```

Local
```bash
cd text_generation
python3 edge_main.py
```


### Image generation using SDXL

```bash
cd image_generation
python3 main.py
```


## Directory tree

```
lambda_split/
│
├─ text_generation/
│  ├─ main.py
│  ├─ cloud_main.py : For HTTP communication
│  ├─ edge_main.py : For HTTP communication
│  └─ src/
│     ├─ base.py
│     ├─ cloud.py
│     ├─ edge.py
│     ├─ split_models.py : Definition of split sub-models.
│     └─ utils.py
│
├─ image_generation/
│  ├─ main.py
│  ├─ evaluation.py
│  └─ src/
│     ├─ quantizers.py : For quantization
│     ├─ split_pipelines.py : Definition of split sub-models.
│     └─ utils.py
│
└─ requirements.txt
```



## Overview of split implementation
1. override forward method of models to correctly split inference layers at inference time (implemented by commenting out in `forward` method of `FirstLlamaModel` etc. in `src/models.py`)
2. replace unused layers with identity layers to reduce memory usage (implemented by `replace_unused_layers_with_identity` method in `src/models.py` `FirstLlamaModel` etc.)


