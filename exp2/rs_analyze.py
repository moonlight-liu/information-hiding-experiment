import argparse
from typing import List, Tuple, Optional
import numpy as np

from utils import load_gray_image

def flip_pixel(pixel: int, flag: int) -> int:
    if flag == 1:
        if pixel % 2 == 0:
            pixel += 1
        else:
            pixel -= 1
    elif flag == -1:
        if pixel % 2 == 0:
            pixel -= 1
        else:
            pixel += 1
    elif flag == 0:
        return pixel
    return pixel

def flip_image(image: np.ndarray):
    H, W = image.shape
    flipped = image.copy()
    for i in range(H):
        for j in range(W):
            pixel = image[i, j]
            if pixel % 2 == 0:
                pixel += 1
            else:
                pixel -= 1
            flipped[i, j] = pixel
    return flipped

def group_correlation(group: np.ndarray):
    return np.sum(np.abs(np.diff(group)))

def count_RS(
    image: np.ndarray,
    mask: List[int],
    group_size: int) -> Tuple[int, int]:
    H, W = image.shape
    pixels = image.flatten()
    R, S = 0, 0
    for i in range(0, H * W, group_size):
        group = pixels[i: i + group_size]
        colleration_ori = group_correlation(group)
        flip_group = group.copy()
        for idx, (pixel, flag) in enumerate(zip(flip_group, mask)):
            flip_group[idx] = flip_pixel(pixel, flag)
        colleration_flip = group_correlation(flip_group)
        
        if colleration_flip > colleration_ori:
            R += 1
        elif colleration_flip < colleration_ori:
            S += 1
    return R, S

def rs_statistics(
    image: np.ndarray,
    group_size: int,
    mask: List[int]) -> Tuple[int, int]:
    neg_mask = [-x for x in mask]
    # p/2 概率下的R和S
    RM_pos, SM_pos = count_RS(
        image,
        mask,
        group_size
    )
    RM_neg, SM_neg = count_RS(
        image,
        neg_mask,
        group_size
    )
    print(f'R_M(p/2): {RM_pos}, S_M(p/2): {SM_pos}')
    print(f'R_-M(p/2): {RM_neg}, S_-M(p/2): {SM_neg}')
    
    # 1-p/2 概率下的R和S
    flipped = flip_image(image)
    RM_pos_, SM_pos_ = count_RS(
        flipped,
        mask,
        group_size
    )
    RM_neg_, SM_neg_ = count_RS(
        flipped,
        neg_mask,
        group_size
    )

    print(f'R_M(1-p/2): {RM_pos_}, S_M(1-p/2): {SM_pos_}')
    print(f'R_-M(1-p/2): {RM_neg_}, S_-M(1-p/2): {SM_neg_}')
    d0 = RM_pos - SM_pos
    d1 = RM_pos_ - SM_pos_
    d_minus0 = RM_neg - SM_neg
    d_minus1 = RM_neg_ - SM_neg_
    
    a = 2 * (d1 + d0)
    b = d_minus0 - d_minus1 - d1 - 3 * d0
    c = d0 - d_minus0
    roots = np.roots([a, b, c])
    valid_root = roots[np.argmin(np.abs(roots))]
    p = valid_root / (valid_root - 0.5)
    return p

def main(args):
    mask = [1, 0, 0, 1]
    image = load_gray_image(args.image_path)
    p = rs_statistics(image, args.group_size, mask)
    print(f'Estimated emedding rate is: {p}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSB Steganography Embed & Extract Demo')
    parser.add_argument('--image_path', type=str, default='res/lena_gray_lsbr_stego.png')
    parser.add_argument('--group_size', type=int, default=4)
    args = parser.parse_args()
    main(args)