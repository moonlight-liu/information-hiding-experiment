#!/usr/bin/env python3
"""
Refactored F4-like JPEG-domain embedding example (prototypical, works on gray images).

Changes vs original f4.py:
- import numpy as np
- pad image to multiple of 8 blocks
- compute available embedding slots from non-zero AC coefficients and use ratio over those
- avoid top-level side effects; provide functions and CLI via argparse
- provide extraction from stego image (independent of in-memory coeffs)
- robust integer handling for AC parity checks
"""
import argparse
import os
from typing import List, Tuple

import numpy as np
import cv2

from utils import load_gray_image, generate_random_bits, calculate_accuracy


def zigzag_scan(block: np.ndarray) -> np.ndarray:
    """
    将 8x8 块按 JPEG zig-zag 顺序展平为长度 64 的一维数组。

    输入：
    - block: 8x8 的 numpy 数组（通常为量化系数块或 DCT 系数块）。

    返回：长度为 64 的 numpy 数组，按照 JPEG zig-zag 顺序排列，便于将 DC 放在索引 0，后续为 AC 系数。
    """
    zigzag_index = [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
    ]
    return np.array([block[i,j] for i,j in zigzag_index])


def inverse_zigzag(arr: np.ndarray) -> np.ndarray:
    """
    将按 zig-zag 顺序的一维数组重构回 8x8 块。

    输入：长度为 64 的一维数组（arr）。
    返回：8x8 的 numpy 数组，值类型为 float32（用于后续反量化和 IDCT）。
    """
    zigzag_index = [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
    ]
    block = np.zeros((8,8), dtype=np.float32)
    for idx, (i,j) in enumerate(zigzag_index):
        block[i,j] = arr[idx]
    return block


# JPEG-like quantization table (same as original)
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]], dtype=np.float32)


def pad_to_block(image: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    将图像边缘按反射填充，使其高度和宽度为 8 的倍数，便于按 8x8 分块 DCT。

    输入：二维灰度图像数组。
    返回：
    - padded: 填充后的图像数组（若无需填充则为副本）
    - H, W: 原始图像的高度与宽度（用于重构时裁剪回原始大小）。
    """
    H, W = image.shape
    pad_h = (8 - (H % 8)) % 8
    pad_w = (8 - (W % 8)) % 8
    if pad_h == 0 and pad_w == 0:
        return image.copy(), H, W
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    return padded, H, W


def get_ac_dc_coeffs(image: np.ndarray) -> Tuple[List[np.ndarray], List[float], int, int]:
    """
    对图像按 8x8 块计算 DCT 并量化，返回每块的 DC 与 AC 列表（AC 为长度 63 的数组）。

    输入：二维灰度图像数组（uint8）。
    返回：
    - ac_list: 每个块的 AC 系数（列表，元素为长度 63 的 numpy 数组，顺序为 zigzag 后去掉第 0 项）。
    - dc_list: 每个块的 DC 系数（列表，元素为单个数值，对应 zigzag 后索引 0）。
    - Hp, Wp: 填充后图像的高度与宽度（用于后续重构并裁剪回原始大小）。

    说明：函数内部调用了 pad_to_block、cv2.dct、以及使用预定义量化表 Q 做四舍五入量化。
    """
    # pad image to multiples of 8
    padded, H_orig, W_orig = pad_to_block(image)
    H, W = padded.shape
    ac_list = []
    dc_list = []
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = padded[i:i+8, j:j+8].astype(np.float32)
            # optional center shift omitted; working with raw values
            dct_block = cv2.dct(block)
            quant_block = np.round(dct_block / Q)
            zz = zigzag_scan(quant_block)
            dc_coeff = zz[0]
            ac_coeffs = zz[1:]
            ac_list.append(ac_coeffs)
            dc_list.append(dc_coeff)
    return ac_list, dc_list, H, W


def count_nonzero_ac(ac_list: List[np.ndarray]) -> int:
    """
    统计所有块中非零 AC 系数的数量（可用作可嵌入位置的容量估计）。

    输入：ac_list（由 get_ac_dc_coeffs 返回的 AC 列表）。
    返回：非零 AC 系数总和（整数）。
    """
    cnt = 0
    for ac in ac_list:
        cnt += int(np.sum(ac != 0))
    return cnt


def combine_dc_ac(dc_list: List[float], ac_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    将 DC 列表和每块的 AC 列表合并为完整的长度为 64 的系数数组列表（按 zigzag 下的排列）。

    输入：dc_list, ac_list（两个列表长度应相等）。
    返回：coeff_list（列表，元素为长度为 64 的 numpy 数组，可直接用于 inverse_zigzag 并反量化）。
    """
    full_list = []
    for dc, ac in zip(dc_list, ac_list):
        full_block = np.zeros(64, dtype=np.float32)
        full_block[0] = dc
        full_block[1:] = ac
        full_list.append(full_block)
    return full_list


def reconstruct_blocks(coeff_list: List[np.ndarray], padded_shape: Tuple[int,int], orig_shape: Tuple[int,int]) -> np.ndarray:
    """
    根据给定的系数列表（按 zigzag 排列的长度 64 数组），重构整幅图像并裁剪回原始大小。

    输入：
    - coeff_list: 每块完整的 64 长度系数数组列表（通常来自 combine_dc_ac）。
    - padded_shape: (Hp, Wp) 填充后的高度宽度（与 get_ac_dc_coeffs 的返回一致）。
    - orig_shape: 原始图像的 (H, W)，用于最终裁剪回原始尺寸。

    返回：重构后的 uint8 灰度图像，已裁剪为原始尺寸。
    说明：函数对每个系数块执行 inverse_zigzag -> 反量化（*Q）-> IDCT，然后合并块并裁剪。
    """
    Hp, Wp = padded_shape
    recon_image = np.zeros((Hp, Wp), dtype=np.float32)
    idx = 0
    total = len(coeff_list)
    for i in range(0, Hp, 8):
        for j in range(0, Wp, 8):
            if idx >= total:
                # all coefficients consumed; crop and return
                recon_image = np.clip(recon_image, 0, 255).astype(np.uint8)
                return recon_image[:orig_shape[0], :orig_shape[1]]
            coeff = coeff_list[idx]
            quant_block = inverse_zigzag(coeff)
            block = quant_block * Q
            recon_block = cv2.idct(block)
            recon_image[i:i+8, j:j+8] = recon_block
            idx += 1

    recon_image = np.clip(recon_image, 0, 255).astype(np.uint8)
    return recon_image[:orig_shape[0], :orig_shape[1]]


def f4_embed(cover: np.ndarray, ratio: float, seed: int = 2025) -> Tuple[np.ndarray, str]:
    """
    基于 F4 思路在 JPEG 域（量化后的 AC 系数）中嵌入比特的实现（灰度图像示例）。

    算法要点：
    - 仅统计非零 AC 系数作为可用嵌入槽（zero coefficients 不参与）。
    - 根据 ratio（相对于非零 AC 数量）决定嵌入比特数，并用 seed 生成随机比特串。
    - 对每个非零 AC 系数按规则调整系数值以满足奇偶性映射，且避免把系数改为 0（以免改变可用槽数）。

    输入：cover（灰度 uint8 图像），ratio（相对于可用槽的嵌入率），seed（随机生成比特）。
    返回：
    - stego: 嵌入后的 uint8 灰度图像（通过量化系数修改并反变换得到）。
    - embed_bits: 嵌入的比特字符串（便于后续准确率验证）。
    """
    ac_list, dc_list, Hp, Wp = get_ac_dc_coeffs(cover)
    total_slots = count_nonzero_ac(ac_list)
    if total_slots == 0:
        raise RuntimeError('No non-zero AC coefficients available for embedding.')

    # interpret ratio over available non-zero AC slots
    bit_length = max(1, int(total_slots * ratio))
    embed_bits = generate_random_bits(bit_length, seed=seed)

    bit_idx = 0
    new_ac_list = []

    for blk_idx, blk_ac in enumerate(ac_list):
        new_blk_ac = blk_ac.copy()
        for i, ac in enumerate(new_blk_ac):
            if ac == 0:
                continue

            if bit_idx >= bit_length:
                break

            bit = int(embed_bits[bit_idx])
            # integerize coefficient to avoid float parity issues
            aci = int(np.round(ac))

            # F4 embedding must match extraction mapping:
            # extraction maps: positive odd or negative even -> '1', positive even or negative odd -> '0'
            if aci > 0:
                desired_parity = bit  # positive: parity == bit
                if (aci % 2) != desired_parity:
                    proposed = aci - 1 if aci > 0 else aci + 1
                    if proposed == 0:
                        proposed = 2
                    new_blk_ac[i] = float(proposed)
            else:
                # negative coefficients: extraction interprets '1' when abs(ac) is even
                desired_abs_parity = 0 if bit == 1 else 1
                if (abs(aci) % 2) != desired_abs_parity:
                    # move away from zero (make less negative)
                    proposed = aci + 1
                    if proposed == 0:
                        proposed = -2
                    new_blk_ac[i] = float(proposed)

            bit_idx += 1

        new_ac_list.append(new_blk_ac)
    # ensure we have AC lists for all blocks (append remaining unchanged blocks)
    if len(new_ac_list) < len(ac_list):
        new_ac_list.extend(ac_list[len(new_ac_list):])

    # combine and reconstruct
    coeff_list = combine_dc_ac(dc_list, new_ac_list)
    stego = reconstruct_blocks(coeff_list, (Hp, Wp), cover.shape)

    return stego, embed_bits


def f4_extract_from_image(stego: np.ndarray, bit_length: int) -> str:
    """
    从图像中提取通过 f4_embed 嵌入的比特（无需原始系数，只需对图像重新 DCT/量化并读取 AC 的奇偶性映射）。

    输入：stego（灰度 uint8 图像），bit_length（要提取的比特数）。
    返回：提取出的比特字符串（长度至多 bit_length）。
    说明：提取规则需与 f4_embed 中的映射一致：
      - 若 ac>0 且奇数 或 ac<0 且 abs(ac) 为偶 -> interpret '1'
      - 其他情况 interpret '0'
    """
    ac_list, _, Hp, Wp = get_ac_dc_coeffs(stego)
    bits = []
    for blk_ac in ac_list:
        for ac in blk_ac:
            if ac == 0:
                continue
            aci = int(np.round(ac))
            if (aci > 0 and aci % 2 == 1) or (aci < 0 and (abs(aci) % 2 == 0)):
                bits.append('1')
            else:
                bits.append('0')
            if len(bits) >= bit_length:
                return ''.join(bits[:bit_length])
    return ''.join(bits[:bit_length])


def main():
    parser = argparse.ArgumentParser(description='F4-like embedding demo (grayscale)')
    parser.add_argument('--cover_path', type=str, default='pics/lena_gray.png')
    parser.add_argument('--stego_path', type=str, default='res/lena_gray_f4_stego.png')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()

    cover = load_gray_image(args.cover_path)
    # 嵌入并获得 stego 图像与嵌入比特串
    stego, embed_bits = f4_embed(cover, args.ratio, seed=args.seed)

    # ensure output dir exists
    out_dir = os.path.dirname(args.stego_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(args.stego_path, stego)
    print(f'Stego written to {args.stego_path}')

    # extract and report accuracy
    extract_bits = f4_extract_from_image(stego, len(embed_bits))
    acc = calculate_accuracy(embed_bits, extract_bits) if len(embed_bits) > 0 else 1.0
    print(f'Extracted {len(extract_bits)}/{len(embed_bits)} bits. Accuracy: {acc:.4f}')


if __name__ == '__main__':
    main()
