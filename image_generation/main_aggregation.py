import time
import argparse

import numpy as np
import torch
from tqdm import tqdm
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from src.split_pipelines import EdgeStableDiffusionXLPipeline, CloudStableDiffusionXLPipeline
from src.quantizers import quantize_wrapper
from src.utils import SplitComputingLoggerForLdm, fix_seed_and_free_memory


def main(
        edge_device: str,
        cloud_device: str,
        quantize_methods: list,
        show_ui: bool
    ):
    # Edge と Cloud の推論パイプライン
    edge = EdgeStableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
    # edge.unet
    edge.to(edge_device)

    cloud = CloudStableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
    # del cloud.tokenizer
    # del cloud.text_encoder
    # del cloud.vae
    cloud.to(cloud_device)


    def infer_each_request(prompt, num_inference_steps, random_seed, freq, quantize, cloud_quantize, num_aggregation_steps=1):
        fix_seed_and_free_memory(round(random_seed))

        # Triadic split computing : edge -> cloud -> edge
        inference_start_time = time.perf_counter()

        # Edge での first subpipeline の推論
        prompt_embeds, add_text_embeds, add_time_ids, initial_latents = \
            edge.infer_first_pipeline(prompt=prompt, num_inference_steps=num_inference_steps)
        
        initial_image = edge.decode_image()
        timesteps = edge.timesteps
        
        # Cloud が first_hidden_layer_outputs を受け取って、denoising に必要な準備する
        cloud.receive_hidden_layer_outputs_and_prepare_for_denoising(
            prompt_embeds=prompt_embeds,
            add_text_embeds=add_text_embeds,
            add_time_ids=add_time_ids,
            latents=initial_latents,
            num_inference_steps=num_inference_steps
        )

        log_dir_name = f'{quantize}_{cloud_quantize}'
        split_computing_logger = SplitComputingLoggerForLdm()
        split_computing_logger.save_initial_data(
            prompt,
            prompt_embeds,
            add_text_embeds,
            add_time_ids,
            initial_latents,
            initial_image,
            num_inference_steps,
            timesteps
        )

        yield f'0 / {num_inference_steps} (Head sub-model inference completed)', None, initial_image

        image = initial_image

        aggregation_noise_list = []

        # Denoising
        for idx in tqdm(range(num_inference_steps)):
            print(idx)

            ## Second subpipeline on cloud
            predicted_noise = cloud.infer_second_pipeline(idx, cloud_quantize, quantize)
            second_model_inference_time = time.perf_counter()
            print(f"Second model inference time : {second_model_inference_time - inference_start_time}")
            print(f'Predicted noise shape : {predicted_noise.shape}')


            # 送信データの量子化
            predicted_noise_npy = predicted_noise.detach().cpu().numpy()
            quantized_noise_npy, quantizer = quantize_wrapper(predicted_noise_npy, quantize)

            # 量子化されたノイズを再度Float32に変換
            predicted_noise = quantized_noise_npy.astype(np.float32)
            predicted_noise = torch.from_numpy(predicted_noise).to(edge_device)


            # ノイズ画像化(RGBA)
            predicted_noise_img = predicted_noise_npy[0].transpose(1, 2, 0)
            # 標準正規分布は 99.7% が [-3, 3] の範囲に入る
            threshold = 3
            # -3より小さい値は-3に、3より大きい値は3にする
            predicted_noise_npy[predicted_noise_npy < -threshold] = -threshold
            predicted_noise_npy[predicted_noise_npy > threshold] = threshold
            
            predicted_noise_img = predicted_noise_img + threshold
            predicted_noise_img = predicted_noise_img / (2 * threshold) * 255
            predicted_noise_img = Image.fromarray(predicted_noise_img.astype(np.uint8), mode='RGBA')
            predicted_noise_img = predicted_noise_img.resize((512, 512))


            # シャッフルする
            # noise_shape = predicted_noise.shape
            # predicted_noise = predicted_noise.flatten()
            # predicted_noise = predicted_noise[torch.randperm(predicted_noise.numel())]
            # predicted_noise = predicted_noise.view(noise_shape)

            # Third subpipeline on edge
            idx += 1
            aggregation_noise_list.append(predicted_noise)
            denoise_image = True if idx % num_aggregation_steps == num_aggregation_steps or idx == num_inference_steps else False
            if denoise_image:
                aggregated_noise = torch.sum(torch.stack(aggregation_noise_list, dim=0), dim=0)
                edge.infer_third_pipeline(idx, aggregated_noise)
                aggregation_noise_list = []

            # Post processing
            decode_image = True if idx % freq == 0 or idx == num_inference_steps else False
            if decode_image:
                image = edge.decode_image()

            third_model_inference_time = time.perf_counter()
            print(f"Third model inference time : {third_model_inference_time - inference_start_time}")
            
            next_decode_image = True if (idx + 1) % freq == 0 or (idx + 1) == num_inference_steps else False
            yield_str = f'{idx} / {num_inference_steps}'
            if idx == num_inference_steps:
                yield_str += '  (Image generation completed)'
            elif next_decode_image:
                yield_str += '  (Decoding image...)'

            split_computing_logger.save_data(
                idx,
                predicted_noise_npy,
                predicted_noise_img,
                quantizer=quantizer,
                decoded_image=image if decode_image else None
            )

            yield yield_str, predicted_noise_img, image


    if show_ui:
        with gr.Blocks() as demo:
            gr.Markdown(f"<h1><center>Demo : Lambda-Split for stabilityai/stable-diffusion-xl-base-1.0</center></h1>")
            gr.Markdown(f"<p><center>GitHub: <a href=https://github.com/nishio-laboratory/lambda_split>https://github.com/nishio-laboratory/lambda_split</a></center></p>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"<center>Image Generation Config</center>")

                    prompt = gr.Textbox(value="An astronaut riding a green horse", label="Prompt")
                    num_inference_steps = gr.Slider(value=50, minimum=1, maximum=200, step=1, label="Number of inference steps")
                    random_seed = gr.Number(value=42, label="Random seed")
                    freq = gr.Slider(value=10, minimum=1, maximum=50, step=1, label="Image decode frequency (step)")
                    quantize = gr.Radio(quantize_methods, label="Transmitted noise quantization method", value="FP32")
                    cloud_quantize = gr.Checkbox(label="Quantize on cloud", default=True)

                    button = gr.Button(label="Generate Image")

                with gr.Column(scale=2):
                    gr.Markdown(f"<center>Generated Image</center>")

                    current_step = gr.Textbox(label="Current generation step")
                    with gr.Row(scale=1):
                        predicted_noise = gr.Image(type="pil", label="Visualized transmission data : predicted Gaussian noise")
                        image = gr.Image(type="pil", label="Generated image")

                    button.click(infer_each_request, inputs=[prompt, num_inference_steps, random_seed, freq, quantize, cloud_quantize], outputs=[current_step, predicted_noise, image])

        # prompt = gr.Textbox(value="An astronaut riding a green horse", label="Prompt")
        # num_inference_steps = gr.Slider(value=50, minimum=1, maximum=200, step=1, label="Number of inference steps")
        # random_seed = gr.Number(value=42, label="Random seed")
        # freq = gr.Slider(value=10, minimum=1, maximum=50, step=1, label="Image decode frequency (step)")
        # quantize = gr.Radio(quantize_methods, label="Transmitted noise quantization method", value="FP32")

        # current_step = gr.Textbox(label="Current generation step")
        # predicted_noise = gr.Image(type="pil", label="Visualized transmission data : predicted Gaussian noise", width=360)
        # image = gr.Image(type="pil", label="Generated image", width=360)

        # examples = [
        #     ["A robot painted as graffiti on a brick wall. a sidewalk is in front of the wall, and grass is growing out of cracks in the concrete."],
        #     ["Panda mad scientist mixing sparkling chemicals, artstation."],
        #     ["A close-up of a fire spitting dragon, cinematic shot."],
        #     ["A capybara made of lego sitting in a realistic, natural field."],
        #     ["Epic long distance cityscape photo of New York City flooded by the ocean and overgrown buildings and jungle ruins in rainforest, at sunset, cinematic shot, highly detailed, 8k, golden light"]
        # ]

        # demo = gr.Interface(
        #     infer_each_request, 
        #     inputs=[prompt, num_inference_steps, random_seed, freq, quantize], 
        #     examples=examples,
        #     outputs=[current_step, predicted_noise, image],
        #     title="Demo : Lambda-Split for stabilityai/stable-diffusion-xl-base-1.0"
        # )

        demo.queue().launch(share=True)
        # demo.queue().launch(ssl_verify=False, server_name='0.0.0.0')

    else:
        prompt = 'An astronaut riding a green horse' # input('Prompt : ')
        num_inference_steps = 50 # int(input('Number of inference steps : '))
        random_seed = 42 # int(input('Random seed : '))
        freq = 10 # int(input('Generated image update frequency : '))
        num_aggregation_steps = 2

        for cloud_quantize in [True]:
            for quantize in quantize_methods:
                for yield_str, predicted_noise_pil, image in infer_each_request(prompt, num_inference_steps, random_seed, freq, quantize, cloud_quantize, num_aggregation_steps):
                    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge_device', type=str, default='cpu', help='cuda or mps or cpu')
    parser.add_argument('--cloud_device', type=str, default='cuda', help='cuda or mps or cpu')
    parser.add_argument('--quantize_methods', nargs='+', type=str, default=["FP32", "FP16", 'FP8(E4M3)', 'FP8(E5M2)', "INT8", "INT7", "INT6", "INT5", "INT4", "INT3", "INT2", "BOOL"], help='Quantization methods, you can add INTn (n > 8)')
    parser.add_argument('--no_gui', action='store_true', help='Disable Gradio GUI')
    args = parser.parse_args()
    print(args)

    edge_device = args.edge_device
    cloud_device = args.cloud_device
    quantize_methods = args.quantize_methods
    show_ui = not args.no_gui
    
    main(edge_device, cloud_device, quantize_methods, show_ui)