import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *

def lsb_embed(
    msg_bits: str,
    cover: np.ndarray) -> np.ndarray:
    flatten_cover = cover.flatten()
    pixel_cnt = flatten_cover.shape[0]
    
    if len(msg_bits) > pixel_cnt:
        raise ValueError(f'Input message bits are too long to embed!')
    for idx, bit in enumerate(msg_bits):
        flatten_cover[idx] = (flatten_cover[idx] & 0b11111110) | int(bit)
    
    stego = flatten_cover.reshape(cover.shape)
    return stego

def lsb_extract(
    bit_length: int,
    stego: np.ndarray) -> str:
    flatten_stego = stego.flatten()
    pixel_cnt = flatten_stego.shape[0]
    
    msg_bits = []
    for idx in range(bit_length):
        bit = flatten_stego[idx] & 0b00000001
        msg_bits.append(str(bit))
    
    return ''.join(msg_bits)

def visualize_msg_positions(
    img: np.ndarray,
    bit_length: int,
    save_path: str
) -> None:
    assert len(img.shape) == 2, 'Only the gray image is supported!'
    H, W = img.shape
    pixel_cnt = H * W
    indices = np.arange(bit_length)
    
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ys, xs = np.unravel_index(indices, (H, W))
    for x, y in zip(xs, ys):
        cv2.circle(vis_img, (x, y), radius=1, color=(0, 0, 255), thickness=1)
    cv2.imwrite(save_path, vis_img)
    print(f'The visualization image has been saved to "{save_path}"')
    
def visualize_gray_histogram(
    cover: np.ndarray,
    stego: np.ndarray,
    save_path: str,
) -> None:
    hist_cover = cv2.calcHist([cover], [0], None, [256], [0, 256]).flatten()
    hist_stego = cv2.calcHist([stego], [0], None, [256], [0, 256]).flatten()

    hist_cover /= hist_cover.sum()
    hist_stego /= hist_stego.sum()

    bins = np.arange(256)

    plt.figure(figsize=(12, 5))

    # Subplot 1 — Cover
    plt.subplot(1, 2, 1)
    plt.bar(bins, hist_cover, color='blue', alpha=0.7)
    plt.title("Cover Image Histogram")
    plt.xlabel("Gray Level")
    plt.ylabel("Normalized Frequency")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Subplot 2 — Stego
    plt.subplot(1, 2, 2)
    plt.bar(bins, hist_stego, color='red', alpha=0.7)
    plt.title("Stego Image Histogram")
    plt.xlabel("Gray Level")
    plt.ylabel("Normalized Frequency")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Histograms have been saved to "{save_path}"')


def visualize_histgram_difference(
    cover: np.ndarray,
    stego: np.ndarray,
    save_path: str,
) -> None:
    # Compute normalized histograms
    hist_cover = cv2.calcHist([cover], [0], None, [256], [0, 256]).flatten()
    hist_stego = cv2.calcHist([stego], [0], None, [256], [0, 256]).flatten()
    hist_cover /= hist_cover.sum()
    hist_stego /= hist_stego.sum()

    # Compute |Hist[2i] - Hist[2i+1]|
    diff_bins = np.arange(0, 256, 2)
    diff_cover = np.abs(hist_cover[0::2] - hist_cover[1::2])
    diff_stego = np.abs(hist_stego[0::2] - hist_stego[1::2])

    plt.figure(figsize=(12, 5))

    # Subplot 1 — Cover difference
    plt.subplot(1, 2, 1)
    plt.plot(diff_bins, diff_cover, color='blue', label='|Hist[2i]-Hist[2i+1]|')
    plt.title("Cover |Hist[2i] - Hist[2i+1]|")
    plt.xlabel("Even Gray Level (2i)")
    plt.ylabel("Absolute Difference")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    _, y_max = plt.ylim()
    # Subplot 2 — Stego difference
    plt.subplot(1, 2, 2)
    plt.plot(diff_bins, diff_stego, color='red', label='|Hist[2i]-Hist[2i+1]|')
    plt.title("Stego |Hist[2i] - Hist[2i+1]|")
    plt.xlabel("Even Gray Level (2i)")
    plt.ylabel("Absolute Difference")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.ylim(0, y_max)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'|Hist[2i] - Hist[2i+1]| line chart has been saved to "{save_path}"')

def main(args):
    '''LSB嵌入'''
    cover = load_gray_image(args.cover_path)
    H, W = cover.shape
    bit_length = int(H * W * args.ratio) # 控制嵌入比特数量
    embed_bits = generate_random_bits(bit_length)
    stego = lsb_embed(embed_bits, cover)
    os.makedirs(os.path.dirname(args.stego_path), exist_ok=True)
    cv2.imwrite(args.stego_path, stego)
    print(f'Stego has been saved to "{args.stego_path}"')
    
    
    '''LSB提取'''
    stego = load_gray_image(args.stego_path)
    extract_bits = lsb_extract(bit_length, stego)
    acc = calculate_accuracy(embed_bits, extract_bits)
    print(f'Accuracy is {acc:.4f}')
    
    '''可视化嵌入位置和直方图'''
    visualize_msg_positions(cover, bit_length, save_path=args.vis_pos_path)
    visualize_gray_histogram(cover, stego, args.vis_hist_path)
    visualize_histgram_difference(cover, stego, args.vis_diff_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSB Steganography Embed & Extract Demo')
    parser.add_argument('--cover_path', type=str, default='pics/lena_gray.png')
    parser.add_argument('--stego_path', type=str, default='res/lena_gray_lsb_stego.png')
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--vis_pos_path', type=str, default='res/lena_gray_lsb_msg_pos.png')
    parser.add_argument('--vis_hist_path', type=str, default='res/lena_gray_lsb_msg_hist.png')
    parser.add_argument('--vis_diff_path', type=str, default='res/lena_gray_lsb_msg_hist_diff.png')
    args = parser.parse_args()
    main(args)