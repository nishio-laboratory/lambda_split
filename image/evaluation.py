'''
Calculate the PSNR and SSIM of the generated images.
'''

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image

if __name__ == '__main__':
    method = ["FP32", "FP16", "INT8", "INT7", "INT6", "INT5", "INT4", "INT3", "INT2", "INT1"]

    img_list = []
    img_fp32 = np.array(Image.open('log/FP32/decoded_image/050.png'))
    img_fp16 = np.array(Image.open('log/FP16/decoded_image/050.png'))

    img_list.append(img_fp32)
    img_list.append(img_fp16)
    
    for i in range(8, 0, -1):
        img = np.array(Image.open(f'log/INT{i}/decoded_image/050.png'))
        img_list.append(img)

    for i, img_quantized in enumerate(img_list):
        print(f'Quantization method : {method[i]}')
        print(f'PSNR : {psnr(img_fp32, img_quantized)}')
        print(f'SSIM : {ssim(img_fp32, img_quantized, channel_axis=2)}')
        print()