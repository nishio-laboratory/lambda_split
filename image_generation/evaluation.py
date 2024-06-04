'''
Calculate the PSNR and SSIM of the generated images.
'''

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from src.vif.vif_utils import vif
from PIL import Image


log_dir = 'log/lion_42'


if __name__ == '__main__':
    method = []
    for i in ["FP32", "FP16", "INT8", "INT7", "INT6", "INT5", "INT4", "INT3", "INT2", "INT1"]:
        method.append(i + '_False')
        method.append(i + '_True')

    img_list = []
    img_fp32 = np.array(Image.open(f'{log_dir}/FP32_False/decoded_image/050.png'))
    img_list.append(img_fp32)
    img_tmp = np.array(Image.open(f'{log_dir}/FP32_True/decoded_image/050.png'))
    img_list.append(img_tmp)

    img_tmp = np.array(Image.open(f'{log_dir}/FP16_False/decoded_image/050.png'))
    img_list.append(img_tmp)
    img_tmp = np.array(Image.open(f'{log_dir}/FP16_True/decoded_image/050.png'))
    img_list.append(img_tmp)
    
    for i in range(8, 0, -1):
        img_tmp = np.array(Image.open(f'{log_dir}/INT{i}_False/decoded_image/050.png'))
        img_list.append(img_tmp)
        img_tmp = np.array(Image.open(f'{log_dir}/INT{i}_True/decoded_image/050.png'))
        img_list.append(img_tmp)

    for i, img_quantized in enumerate(img_list):
        print(f'Quantization method : {method[i]}')
        print(f'PSNR : {psnr(img_fp32, img_quantized)}')
        print(f'SSIM : {ssim(img_fp32, img_quantized, channel_axis=2)}')
        print()


    img_list = []
    img_fp32 = np.array(Image.open(f'{log_dir}/FP32_False/decoded_image/050.png').convert('L')).astype(float)
    img_list.append(img_fp32)
    img_fp32 = np.array(Image.open(f'{log_dir}/FP32_True/decoded_image/050.png').convert('L')).astype(float)
    img_list.append(img_fp32)

    img_tmp = np.array(Image.open(f'{log_dir}/FP16_False/decoded_image/050.png').convert('L')).astype(float)
    img_list.append(img_tmp)
    img_tmp = np.array(Image.open(f'{log_dir}/FP16_True/decoded_image/050.png').convert('L')).astype(float)
    img_list.append(img_tmp)
    
    for i in range(8, 0, -1):
        img_tmp = np.array(Image.open(f'{log_dir}/INT{i}_False/decoded_image/050.png').convert('L')).astype(float)
        img_list.append(img_tmp)
        img_tmp = np.array(Image.open(f'{log_dir}/INT{i}_True/decoded_image/050.png').convert('L')).astype(float)
        img_list.append(img_tmp)

    for i, img_quantized in enumerate(img_list):
        print(f'Quantization method : {method[i]}')
        print(f'VIF : {vif(img_fp32, img_quantized)}')
        print()