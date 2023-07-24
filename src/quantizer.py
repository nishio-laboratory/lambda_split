import torch
from typing import Tuple, Type


class Quantizer:
    def __init__(
            self, 
            method: str,
            dtype: torch.dtype = torch.uint8
    ) -> None:
        assert method in ['minmax', 'standard']
        if method == 'minmax':
            assert dtype in [torch.uint8, torch.bool]
        elif method == 'standard':
            assert dtype in [torch.int8, torch.bool]

        self.method = method
        self.dtype = dtype

        if self.method == 'minmax':
            if self.dtype is torch.uint8:
                self.scale = 255.0
            elif self.dtype is torch.bool:
                self.scale = 1.0
        elif self.method == 'standard':
            if self.dtype is torch.int8:
                self.scale = 127.0 / 2
            elif self.dtype is torch.bool:
                self.scale = 1.0

    def quantize(
            self, 
            tensor: torch.HalfTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.ByteTensor]:
        if self.method == 'minmax':
            return self.minmax_quantize(tensor)
        elif self.method == 'standard':
            return self.standard_quantize(tensor)
        
    def dequantize(
            self, 
            coef_1: torch.Tensor, 
            coef_2: torch.Tensor, 
            quantized_tensor: torch.ByteTensor
    ) -> torch.HalfTensor:
        if self.method == 'minmax':
            return self.minmax_dequantize(coef_1, coef_2, quantized_tensor)
        elif self.method == 'standard':
            return self.standard_dequantize(coef_1, coef_2, quantized_tensor)

    def minmax_quantize(
            self, 
            tensor: torch.HalfTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.ByteTensor]:
        

        # スケーリング因子を計算
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)

        print(f'{min_val=}, {max_val=}')

        # 正規化
        scaled_tensor = (tensor - min_val) / (max_val - min_val)

        # 量子化
        quantized_tensor = torch.round(scaled_tensor * self.scale).to(self.dtype)

        return min_val, max_val, quantized_tensor

    def minmax_dequantize(
            self, 
            min_val: torch.Tensor, 
            max_val: torch.Tensor, 
            quantized_tensor: torch.ByteTensor
    ) -> torch.HalfTensor:
        # デスケーリング
        scaled_tensor = quantized_tensor.float() / self.scale

        # 元の範囲に戻す
        tensor = scaled_tensor * (max_val - min_val) + min_val

        return tensor.half()
    
    def standard_quantize(
            self, 
            tensor: torch.HalfTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.ByteTensor]:
        # スケーリング因子を計算
        mean_val = torch.mean(tensor)
        std_val = torch.std(tensor)

        print(f'{mean_val=}, {std_val=}')

        # 正規化
        scaled_tensor = (tensor - mean_val) / std_val

        # 量子化
        if self.dtype is torch.int8:
            quantized_tensor = torch.round(scaled_tensor * self.scale).to(self.dtype)
        elif self.dtype is torch.bool:
            quantized_tensor = torch.where(scaled_tensor * self.scale >= 0, 1, 0).to(self.dtype)

        return mean_val, std_val, quantized_tensor

    def standard_dequantize(
            self, 
            mean_val: torch.Tensor, 
            std_val: torch.Tensor, 
            quantized_tensor: torch.ByteTensor
    ) -> torch.HalfTensor:
        # デスケーリング
        scaled_tensor = quantized_tensor.float() / self.scale

        # 元の範囲に戻す
        tensor = scaled_tensor * std_val + mean_val

        return tensor.half()