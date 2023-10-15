import datetime
import os
import random
import gc

import numpy as np
from PIL import Image
import torch


class SplitComputingLoggerForLdm:
    def __init__(self):
        cur_dt_str = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        self.log_dir = f'log/{cur_dt_str}'

        os.makedirs(self.log_dir)
        os.makedirs(f'{self.log_dir}/predicted_noise_img')
        os.makedirs(f'{self.log_dir}/decoded_image')

    def save_initial_data(
        self,
        prompt,
        prompt_embeds,
        add_text_embeds,
        add_time_ids,
        latents,
        image,
        num_inference_steps,
        timesteps
    ):
        prompt_embeds = prompt_embeds.detach().cpu().numpy()
        add_text_embeds = add_text_embeds.detach().cpu().numpy()
        add_time_ids = add_time_ids.detach().cpu().numpy()
        latents = latents.detach().cpu().numpy()
        timesteps = timesteps.detach().cpu().numpy()

        with open(f'{self.log_dir}/main.txt', 'w') as f:
            print(f'{prompt=}', file=f)
            print(f'shape of prompt_embeds : {prompt_embeds.shape}, dtype of prompt_embeds : {prompt_embeds.dtype}', file=f)
            print(f'shape of add_text_embeds : {add_text_embeds.shape}, dtype of add_text_embeds : {add_text_embeds.dtype}', file=f)
            print(f'shape of add_time_ids : {add_time_ids.shape}, dtype of add_time_ids : {add_time_ids.dtype}', file=f)
            print(f'shape of latents : {latents.shape}, dtype of latents : {latents.dtype}', file=f)
            print(f'{num_inference_steps=}', file=f)
            print(f'{timesteps=}', file=f)

            print(file=f)

        np.save(f'{self.log_dir}/prompt_embeds.npy', prompt_embeds)
        np.save(f'{self.log_dir}/add_text_embeds.npy', add_text_embeds)
        np.save(f'{self.log_dir}/add_time_ids.npy', add_time_ids)
        np.save(f'{self.log_dir}/initial_latents.npy', latents)
        image.save(f'{self.log_dir}/decoded_image/000.png')

    def save_data(
        self,
        idx,
        predicted_noise_npy,
        predicted_noise_img,
        quantizer,
        decoded_image=None
    ):

        mean, std = predicted_noise_npy.mean(), predicted_noise_npy.std()

        with open(f'{self.log_dir}/main.txt', 'a') as f:
            print(f'{idx=}', file=f)
            print(f'shape of predicted_noise : {predicted_noise_npy.shape}, dtype of predicted_noise : {predicted_noise_npy.dtype}', file=f) 
            print(f'{mean=}, {std=}', file=f)
            print(f'{quantizer=}', file=f)
            print(file=f)

        predicted_noise_img.save(f'{self.log_dir}/predicted_noise_img/{str(idx).zfill(3)}.png')

        if decoded_image is not None:
            decoded_image.save(f'{self.log_dir}/decoded_image/{str(idx).zfill(3)}.png')



def fix_seed_and_free_memory(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    # free memory
    torch.cuda.empty_cache()
    gc.collect()