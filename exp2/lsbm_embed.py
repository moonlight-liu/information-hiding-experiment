import argparse
import os
import numpy as np
import cv2

from utils import load_gray_image, generate_random_bits, calculate_accuracy
from lsbm import lsbm_embed, lsbm_extract, visualize_msg_positions, visualize_gray_histogram, visualize_histgram_difference


def main(args):
    cover = load_gray_image(args.cover_path)
    H, W = cover.shape
    bit_length = int(H * W * args.ratio)

    embed_bits = generate_random_bits(bit_length)
    stego = lsbm_embed(embed_bits, cover, seed=args.seed)

    os.makedirs(os.path.dirname(args.stego_path), exist_ok=True)
    cv2.imwrite(args.stego_path, stego)
    print(f'Stego has been saved to "{args.stego_path}"')

    # 验证提取正确性
    stego = load_gray_image(args.stego_path)
    extract_bits = lsbm_extract(bit_length, stego, seed=args.seed)
    acc = calculate_accuracy(embed_bits, extract_bits)
    print(f'Accuracy is {acc:.4f}')

    # 可视化
    visualize_msg_positions(cover, bit_length, seed=args.seed, save_path=args.vis_pos_path)
    visualize_gray_histogram(cover, stego, args.vis_hist_path)
    visualize_histgram_difference(cover, stego, args.vis_diff_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSBM Steganography Embed & Extract Demo')
    parser.add_argument('--cover_path', type=str, default='pics/lena_gray.png')
    parser.add_argument('--stego_path', type=str, default='res/LSBM/lsbm_stego.png')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--vis_pos_path', type=str, default='res/LSBM/lsbm_msg_pos.png')
    parser.add_argument('--vis_hist_path', type=str, default='res/LSBM/lsbm_msg_hist.png')
    parser.add_argument('--vis_diff_path', type=str, default='res/LSBM/lsbm_msg_hist_diff.png')
    args = parser.parse_args()
    main(args)


