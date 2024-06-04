import numpy as np
from tqdm import tqdm
import io
import gzip

import matplotlib.pyplot as plt
import seaborn as sns


def quantize_wrapper(predicted_noise_npy, quantize):
    if quantize == 'FP32':
        quantizer = quantize

    elif quantize == 'FP16':
        quantizer = quantize
        predicted_noise_npy = predicted_noise_npy.astype(np.float16)

    elif 'FP8' in quantize:
        if quantize == 'FP8(E4M3)':
            quantizer = BasicCustomFloatQuantizer(4, 3, False)
        elif quantize == 'FP8(E5M2)':
            quantizer = BasicCustomFloatQuantizer(5, 2, False)

        predicted_noise_npy_bytes = quantizer.quantize_ndarray(predicted_noise_npy)
        predicted_noise_npy = quantizer.dequantize_ndarray(predicted_noise_npy_bytes, predicted_noise_npy.shape)

    elif 'INT' in quantize or quantize == 'BOOL':
        if quantize == 'BOOL':
            bit = 1
        else:
            bit = int(quantize.split('INT')[1])
        quantizer = AffineQuantizer(bit)
        predicted_noise_npy_quantized = quantizer.quantize_ndarray(predicted_noise_npy)
        predicted_noise_npy = quantizer.dequantize_ndarray(predicted_noise_npy_quantized)
    else:
        raise ValueError('Invalid quantize value')
    
    return predicted_noise_npy, quantizer


class AffineQuantizer:
    def __init__(self, bit) -> None:
        self.b = bit

    def clip(self, x, l, u):
        if x < l:
            return l
        elif l <= x <= u:
            return x
        else:
            return u

    def quantize_ndarray(self, ndarray):
        beta = ndarray.flatten().min()
        alpha = ndarray.flatten().max()

        self.s = (2 ** self.b - 1) / (alpha - beta)
        self.z = -round(beta * self.s) - 2 ** (self.b - 1)

        quantized_ndarray = [self.clip(round(self.s * x + self.z), -2 ** (self.b - 1), 2 ** (self.b - 1) - 1) for x in ndarray.flatten()]
        quantized_ndarray = np.array(quantized_ndarray).reshape(ndarray.shape)

        return quantized_ndarray
    
    def dequantize_ndarray(self, quantized_ndarray):
        dequantized_ndarray = (quantized_ndarray - self.z) / self.s

        return dequantized_ndarray
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bit={self.b})'


class CustomUnsignedIntQuantizer:
    def __init__(self, bit) -> None:
        self.bit = bit
        self.scale = 2 ** bit - 1

    def quantize_ndarray(self, ndarray):
        self.min_val = ndarray.flatten().min()
        self.max_val = ndarray.flatten().max()

        quantized_ndarray = np.round((ndarray - self.min_val) / (self.max_val - self.min_val) * self.scale)

        return quantized_ndarray

    def dequantize_ndarray(self, quantized_ndarray):
        dequantize_ndarray = quantized_ndarray.astype(np.float32) / self.scale * (self.max_val - self.min_val) + self.min_val

        return dequantize_ndarray
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bit={self.bit})'


class CustomFloatQuantizer:
    def __init__(self, exponent_bits=4, mantissa_bits=3, do_zip_compression=False):
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.do_zip_compression = do_zip_compression

        self.total_bits = exponent_bits + mantissa_bits + 1  # 1 for sign bit
        self.bias = (1 << (exponent_bits - 1)) - 1
        self.max_exponent = (1 << exponent_bits) - 1

    def zip_compression(self, ndarray_bytes):
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
            f.write(ndarray_bytes)

        compressed_data = buffer.getvalue()

        return compressed_data

    def zip_decompression(self, compressed_data):
        buffer = io.BytesIO(compressed_data)
        with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
            decompressed_data = f.read()

        return decompressed_data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(exponent_bits={self.exponent_bits}, mantissa_bits={self.mantissa_bits})'


class BasicCustomFloatQuantizer(CustomFloatQuantizer):
    '''
    q-bit 量子化で、q % 8 != 0 の場合、バイト数が切り上げになる
    Optimizedより高速
    '''
    def quantize_value(self, value: float) -> bytes:
        if value == 0:
            return bytes([0] * ((self.total_bits + 7) // 8))
        
        if value < 0:
            sign = 1
            value = -value
        else:
            sign = 0
        
        # Handling underflow and overflow
        exponent = 0
        while value < 1.0 and exponent > -self.bias:
            value *= 2.0
            exponent -= 1
        while value >= 2.0 and exponent < self.max_exponent - self.bias:
            value /= 2.0
            exponent += 1
        
        exponent = max(0, min(self.max_exponent, exponent + self.bias))
        mantissa = int((value - 1) * (1 << self.mantissa_bits))  # value is in the form 1.fraction
        mantissa = max(0, mantissa)  # Ensure non-negative mantissa

        simple_float = (sign << (self.total_bits - 1)) | (exponent << self.mantissa_bits) | mantissa
        simple_float_bytes = simple_float.to_bytes((self.total_bits + 7) // 8, 'big') # ビット数を8で割ることでバイト数を得ることができますが、ビット数が8で割り切れない場合もあるので、その際には1バイト余分に確保する必要があります。

        return simple_float_bytes

    def dequantize_value(self, simple_float_bytes: bytes) -> float:
        simple_float = int.from_bytes(simple_float_bytes, 'big')
        
        sign = (simple_float >> (self.total_bits - 1)) & 1
        exponent = (simple_float >> self.mantissa_bits) & ((1 << self.exponent_bits) - 1)
        mantissa = simple_float & ((1 << self.mantissa_bits) - 1)

        if exponent == 0:  # Treat as zero
            value = 0.0
        else:
            value = 1 + mantissa / (1 << self.mantissa_bits)  # 1.fraction form
        
        value *= 2 ** (exponent - self.bias)
        if sign == 1:
            value = -value
        
        return value
    
    def quantize_ndarray(self, ndarray):
        ndarray = ndarray.flatten()
        ndarray_bytes = b''.join([self.quantize_value(value) for value in ndarray])

        if self.do_zip_compression:
            ndarray_bytes = self.zip_compression(ndarray_bytes)

        return ndarray_bytes
    
    def dequantize_ndarray(self, ndarray_bytes, shape):
        if self.do_zip_compression:
            ndarray_bytes = self.zip_decompression(ndarray_bytes)

        ndarray = np.zeros(shape)
        for i in range(0, len(ndarray_bytes), (self.total_bits + 7) // 8):
            value_bytes = ndarray_bytes[i:i + (self.total_bits + 7) // 8]
            ndarray.flat[i] = self.dequantize_value(value_bytes)
        return ndarray        


class OptimizedCustomFloatQuantizer(CustomFloatQuantizer):
    '''
    q-bit 量子化で、q % 8 != 0 の場合でも、バイト数が切り上げにならない
    Pythonのint型がオーバーフローしないことを利用
    Basicより低速
    '''
    def quantize_value(self, value: float) -> int:
        if value == 0:
            return 0  # Return integer 0
        
        if value < 0:
            sign = 1
            value = -value
        else:
            sign = 0
        
        exponent = 0
        while value < 1.0 and exponent > -self.bias:
            value *= 2.0
            exponent -= 1
        while value >= 2.0 and exponent < self.max_exponent - self.bias:
            value /= 2.0
            exponent += 1
        
        exponent = max(0, min(self.max_exponent, exponent + self.bias))
        mantissa = int((value - 1) * (1 << self.mantissa_bits))
        mantissa = max(0, mantissa)

        return (sign << (self.total_bits - 1)) | (exponent << self.mantissa_bits) | mantissa

    def dequantize_value(self, simple_float: int) -> float:
        sign = (simple_float >> (self.total_bits - 1)) & 1
        exponent = (simple_float >> self.mantissa_bits) & ((1 << self.exponent_bits) - 1)
        mantissa = simple_float & ((1 << self.mantissa_bits) - 1)

        if exponent == 0:
            value = 0.0
        else:
            value = 1 + mantissa / (1 << self.mantissa_bits)
        
        value *= 2 ** (exponent - self.bias)
        if sign == 1:
            value = -value
        return value
    
    def quantize_ndarray(self, ndarray: np.ndarray) -> bytes:
        ndarray = ndarray.flatten()
        total_int = 0
        for value in ndarray:
            value_int = self.quantize_value(value)
            total_int = (total_int << self.total_bits) | value_int

        ndarray_bytes = total_int.to_bytes((len(ndarray) * self.total_bits + 7) // 8, 'big')

        if self.do_zip_compression:
            ndarray_bytes = self.zip_compression(ndarray_bytes)

        return ndarray_bytes
    
    def dequantize_ndarray(self, ndarray_bytes: bytes, shape: tuple) -> np.ndarray:
        if self.do_zip_compression:
            ndarray_bytes = self.zip_decompression(ndarray_bytes)

        total_int = int.from_bytes(ndarray_bytes, 'big')
        total_elements = np.prod(shape)
        mask = (1 << self.total_bits) - 1
        
        ndarray = np.zeros(shape)
        for i in range(total_elements - 1, -1, -1):
            value_int = total_int & mask
            ndarray.flat[i] = self.dequantize_value(value_int)
            total_int >>= self.total_bits
        return ndarray



if __name__ == '__main__':
    # Experimenting with the optimized class
    float_exp = True
    int_exp = True

    if float_exp:
        maes = []
        lengths = []
        optimized_custom_float = CustomFloatQuantizer(4, 4)

        # heatmapを作るためのコード
        max_e = 8
        max_m = 8
        maes = np.zeros((max_e, max_m))
        sizes = np.zeros((max_e, max_m))
        zip_sizes = np.zeros((max_e, max_m))

        for e in tqdm(range(1, max_e + 1)):
            for m in range(1, max_m + 1):
                custom_float = OptimizedCustomFloatQuantizer(e, m)
                
                n = 10000
                arr = np.random.randn(n)
                arr_bytes = custom_float.quantize_ndarray(arr)
                arr_dequantized = custom_float.dequantize_ndarray(arr_bytes, arr.shape)
                mae = np.abs(arr - arr_dequantized).mean()
                maes[e - 1, m - 1] = mae

                # 量子化後のデータのサイズを計算する
                sizes[e - 1, m - 1] = len(arr_bytes) * 8 / n

                # ZIP圧縮を行う
                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
                    f.write(arr_bytes)

                compressed_data = buffer.getvalue()

                buffer = io.BytesIO(compressed_data)
                with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
                    decompressed_data = f.read()

                assert arr_bytes == decompressed_data

                zip_sizes[e - 1, m - 1] = len(compressed_data) * 8 / n


        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
        sns.heatmap(maes, annot=True, ax=ax[0])
        sns.heatmap(sizes, annot=True, ax=ax[1])
        sns.heatmap(zip_sizes, annot=True, ax=ax[2])

        ax[0].set_title('MAE')
        ax[1].set_title('Size')
        ax[2].set_title('ZIP size')
        ax[0].set_xlabel('Mantissa bits')
        ax[1].set_xlabel('Mantissa bits')
        ax[2].set_xlabel('Mantissa bits')
        ax[0].set_ylabel('Exponent bits')
        ax[1].set_ylabel('Exponent bits')
        ax[2].set_ylabel('Exponent bits')

        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        ax[2].invert_yaxis()

        ax[0].set_xticklabels([str(i + 1) for i in range(max_m)])
        ax[0].set_yticklabels([str(i + 1) for i in range(max_e)])
        ax[1].set_xticklabels([str(i + 1) for i in range(max_m)])
        ax[1].set_yticklabels([str(i + 1) for i in range(max_e)])
        ax[2].set_xticklabels([str(i + 1) for i in range(max_m)])
        ax[2].set_yticklabels([str(i + 1) for i in range(max_e)])

        fig.tight_layout()
        fig.savefig('custom_float_heatmap.png', dpi=500)

        min_val = [1 for _ in range(max_e + max_m + 2)]
        for e in range(1, max_e + 1):
            for m in range(1, max_m + 1):
                min_val[e + m + 1] = min(min_val[e + m + 1], maes[e - 1, m - 1])
                


    if int_exp:
        maes = []
        for custon_int in [AffineQuantizer, CustomUnsignedIntQuantizer]:
            print(custon_int)
            for b in range(1, 17):
                n = 100000
                arr = np.random.randn(n)
                custom_int = custon_int(b)
                arr_quantized = custom_int.quantize_ndarray(arr)
                
                arr_dequantized = custom_int.dequantize_ndarray(arr_quantized)
                mae = np.abs(arr - arr_dequantized).mean()
                maes.append(mae)

                print(f'{b=}, {mae=}')
                # print(arr_quantized.flatten().min(), arr_quantized.flatten().max())

            print()

        maes = np.array(maes).reshape(2, 16)

        fig, ax = plt.subplots()
        ax.plot(list(range(1, 17)), maes[0], label='Integer quantization')
        ax.plot(min_val, label='Floating point quantization')
        ax.legend()
        ax.set_xlim([0, 16])
        ax.set_xlabel('Bit')
        ax.set_ylabel('MAE')
        ax.set_yscale('log')
        ax.set_title('Quantization MAE for standard normal distribution')
        fig.tight_layout()
        fig.savefig('custom_int.png', dpi=500)