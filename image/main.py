import random
import time

import numpy as np
import torch
from tqdm import tqdm
import gradio as gr
from dataclasses import asdict

from src.split_pipelines import EdgeStableDiffusionXLPipeline, CloudStableDiffusionXLPipeline


def main(
        edge_device: str,
        cloud_device: str,
        show_ui: bool
    ):
    # Edge と Cloud の推論パイプライン
    edge = EdgeStableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
    # del edge.unet
    edge.to(edge_device)

    cloud = CloudStableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
    # del cloud.tokenizer
    # del cloud.text_encoder
    # del cloud.vae
    cloud.to(cloud_device)


    def infer_each_request(prompt, num_inference_steps, random_seed):
        torch_fix_seed(round(random_seed))

        # Triadic split computing : edge -> cloud -> edge
        inference_start_time = time.perf_counter()

        # Edge での first_pipeline の推論
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, initial_latents \
            = edge.infer_first_pipeline(prompt=prompt, num_inference_steps=num_inference_steps)
        
        # Cloud が first_hidden_layer_outputs を受け取って、denoising に必要な準備する
        cloud.receive_hidden_layer_outputs_and_prepare_for_denoising(
            prompt_embeds=prompt_embeds.to(cloud_device),
            negative_prompt_embeds=negative_prompt_embeds.to(cloud_device),
            pooled_prompt_embeds=pooled_prompt_embeds.to(cloud_device),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(cloud_device),
            latents=initial_latents.to(cloud_device),
            num_inference_steps=num_inference_steps)

        for idx in tqdm(range(num_inference_steps)):
            print(idx)

            ## Second pipeline on cloud
            predicted_noise = cloud.infer_second_pipeline(idx)
            second_model_inference_time = time.perf_counter()
            
            ## Third pipeline on edge
            image = edge.infer_third_pipeline(idx, predicted_noise.to(edge_device))
            third_model_inference_time = time.perf_counter()

            yield image


    if show_ui:
        with gr.Blocks() as demo:
            gr.Markdown(f"<h1><center>Demo : Triadic Split Computing for Stable Diffusion XL</center></h1>")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"<center>Image Generation Config</center>")

                    prompt = gr.Textbox(value="An astronaut riding a green horse")
                    num_inference_steps = gr.Slider(value=50, minimum=1, maximum=200, step=1, label="Number of inference steps")
                    random_seed = gr.Number(value=42, label="Random seed")

                    button = gr.Button(label="Generate Image")

                with gr.Column(scale=2):
                    gr.Markdown(f"<center>Generated Image</center>")
                    button.click(infer_each_request, inputs=[prompt, num_inference_steps, random_seed], outputs=gr.Image(type="pil"))

        demo.queue().launch()
        # demo.queue().launch(ssl_verify=False, server_name='0.0.0.0')

    else:
        prompt = 'An astronaut riding a green horse' # input('Prompt : ')
        num_inference_steps = 50 # int(input('Number of inference steps : '))
        random_seed = 42 # int(input('Random seed : '))

        for response in infer_each_request(prompt, num_inference_steps, random_seed):
            pass


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


if __name__ == '__main__':
    edge_device = 'cpu'
    cloud_device = 'cpu'
    show_ui = False
    
    main(edge_device, cloud_device, show_ui)