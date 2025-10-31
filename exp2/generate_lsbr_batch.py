#!/usr/bin/env python3
"""
批量生成不同嵌入率的 LSBR stego 图像。

输出目录：./LSBR
输出文件名格式：lsbr_stego_{ratio:.1f}.png （例如 lsbr_stego_0.1.png）

使用：在 exp2 目录下运行 `python generate_lsbr_batch.py`
"""
import os
from pathlib import Path
import cv2

import lsbr
from utils import load_gray_image, generate_random_bits


def main():
    base_dir = Path(__file__).resolve().parent

    cover_path = base_dir / 'pics' / 'lena_gray.png'
    if not cover_path.exists():
        raise FileNotFoundError(f'Cover image not found: {cover_path}')

    out_dir = base_dir / 'LSBR'
    out_dir.mkdir(parents=True, exist_ok=True)

    cover = load_gray_image(str(cover_path))
    H, W = cover.shape

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seed = 2025

    for r in ratios:
        bit_length = int(H * W * r)
        # 生成随机消息位（使用固定 seed 可复现）
        embed_bits = generate_random_bits(bit_length, seed=seed)

        # 调用 lsbr 的嵌入函数
        stego = lsbr.lsbr_embed(embed_bits, cover.copy(), seed=seed)

        out_name = f'lsbr_stego_{r:.1f}.png'
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), stego)
        print(f'Saved {out_path} (ratio={r}, bits={bit_length})')


if __name__ == '__main__':
    main()
