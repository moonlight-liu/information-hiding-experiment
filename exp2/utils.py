import cv2
import numpy as np

def generate_random_bits(
    length: int, 
    seed: int = 2025
) -> str:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length)
    return ''.join(map(str, bits))

def calculate_accuracy(embed: str, extract: str) -> float:
    if len(embed) != len(extract):
        raise ValueError('String lengths do not match!')
    matches = sum(c1 == c2 for c1, c2 in zip(embed, extract))
    return matches / len(embed)

def load_gray_image(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # 转换为灰度图像
    if img is None:
        raise ValueError(f'Input image is None!')
    return img