import random
import time
import gc

import numpy as np
import torch
from tqdm import tqdm
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from src.split_pipelines import EdgeStableDiffusionXLPipeline, CloudStableDiffusionXLPipeline
from src.custom_float import BasicCustomFloat, OptimizedCustomFloat


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


    def infer_each_request(prompt, num_inference_steps, random_seed, freq):
        torch_fix_seed(round(random_seed))
        torch.cuda.empty_cache()
        gc.collect()

        # Triadic split computing : edge -> cloud -> edge
        inference_start_time = time.perf_counter()

        # Edge での first_pipeline の推論
        prompt_embeds, add_text_embeds, add_time_ids, initial_latents = \
            edge.infer_first_pipeline(prompt=prompt, num_inference_steps=num_inference_steps)
        
        # Cloud が first_hidden_layer_outputs を受け取って、denoising に必要な準備する
        cloud.receive_hidden_layer_outputs_and_prepare_for_denoising(
            prompt_embeds=prompt_embeds,
            add_text_embeds=add_text_embeds,
            add_time_ids=add_time_ids,
            latents=initial_latents,
            num_inference_steps=num_inference_steps
        )

        for idx in tqdm(range(num_inference_steps)):
            print(idx)

            ## Second pipeline on cloud
            predicted_noise = cloud.infer_second_pipeline(idx)
            second_model_inference_time = time.perf_counter()
            print(f"Second model inference time : {second_model_inference_time - inference_start_time}")
            print(f'Predicted noise shape : {predicted_noise.shape}')

            predicted_noise_npy = predicted_noise[0].detach().cpu().numpy().transpose(1, 2, 0)

            custom_float = BasicCustomFloat(4, 3, True)
            predicted_noise_npy_bytes = custom_float.quantize_ndarray(predicted_noise_npy)
            predicted_noise_npy = custom_float.dequantize_ndarray(predicted_noise_npy_bytes, predicted_noise_npy.shape)

            predicted_noise_pil = []
            for i in range(4):
                predicted_noise_pil.append(predicted_noise_npy[:, :, i])
                if i < 3:
                    # 3ピクセルの間隔を追加。形状を(128, 128)に合わせる
                    predicted_noise_pil.append(np.zeros((predicted_noise_npy.shape[0], 3)))

            predicted_noise_pil = np.hstack(predicted_noise_pil)

            # 2値化
            predicted_noise_pil = np.where(predicted_noise_pil > 0, 255, 0).astype(np.uint8)
            
            print(f'Predicted noise pil shape : {predicted_noise_pil.shape}')
            predicted_noise_pil = Image.fromarray(predicted_noise_pil, mode='L')

            ## Third pipeline on edge
            decode_image = True if idx % freq == 0 or idx == num_inference_steps - 1 else False
            image = edge.infer_third_pipeline(idx, predicted_noise, decode_image)

            third_model_inference_time = time.perf_counter()
            print(f"Third model inference time : {third_model_inference_time - inference_start_time}")
            
            next_decode_image = True if (idx + 1) % freq == 0 or idx == num_inference_steps - 2 else False
            yield_str = f'{idx + 1} / {num_inference_steps}'
            if idx == num_inference_steps - 1:
                yield_str += '  (Image generation completed)'
            elif next_decode_image:
                yield_str += '  (Decoding image...)'

            yield yield_str, predicted_noise_pil, image


    if show_ui:
        # with gr.Blocks() as demo:
        #     gr.Markdown(f"<h1><center>Demo : Triadic Split Computing for Stable Diffusion XL</center></h1>")

        #     with gr.Row():
        #         with gr.Column(scale=1):
        #             gr.Markdown(f"<center>Image Generation Config</center>")

        #             prompt = gr.Textbox(value="An astronaut riding a green horse")
        #             num_inference_steps = gr.Slider(value=50, minimum=1, maximum=200, step=1, label="Number of inference steps")
        #             random_seed = gr.Number(value=42, label="Random seed")

        #             button = gr.Button(label="Generate Image")

        #         with gr.Column(scale=1):
        #             gr.Markdown(f"<center>Generated Image</center>")
        #             button.click(infer_each_request, inputs=[prompt, num_inference_steps, random_seed], outputs=gr.Image(type="pil"))

        prompt = gr.Textbox(value="An astronaut riding a green horse", label="Prompt")
        num_inference_steps = gr.Slider(value=50, minimum=1, maximum=200, step=1, label="Number of inference steps")
        random_seed = gr.Number(value=42, label="Random seed")
        freq = gr.Slider(value=10, minimum=1, maximum=50, step=1, label="Generated image update frequency (step)")

        current_step = gr.Textbox(label="Current generation step")
        predicted_noise = gr.Image(type="pil", label="Binarized image of transmission data (Latent vector of predicted noise, shape = (1, 4, 128, 128))")
        image = gr.Image(type="pil", label="Generated image")

        examples = [
            ["A robot painted as graffiti on a brick wall. a sidewalk is in front of the wall, and grass is growing out of cracks in the concrete."],
            ["Panda mad scientist mixing sparkling chemicals, artstation."],
            ["A close-up of a fire spitting dragon, cinematic shot."],
            ["A capybara made of lego sitting in a realistic, natural field."],
            ["Epic long distance cityscape photo of New York City flooded by the ocean and overgrown buildings and jungle ruins in rainforest, at sunset, cinematic shot, highly detailed, 8k, golden light"]
        ]

        demo = gr.Interface(
            infer_each_request, 
            inputs=[prompt, num_inference_steps, random_seed, freq], 
            examples=examples,
            outputs=[current_step, predicted_noise, image],
            title="Demo : Triadic Split Computing for Stable Diffusion XL"
        )

        demo.queue().launch(share=True)
        # demo.queue().launch(ssl_verify=False, server_name='0.0.0.0')

    else:
        prompt = 'An astronaut riding a green horse' # input('Prompt : ')
        num_inference_steps = 50 # int(input('Number of inference steps : '))
        random_seed = 42 # int(input('Random seed : '))
        freq = 10 # int(input('Generated image update frequency : '))

        for response in infer_each_request(prompt, num_inference_steps, random_seed, freq):
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
    cloud_device = 'cuda'
    show_ui = True
    
    main(edge_device, cloud_device, show_ui)