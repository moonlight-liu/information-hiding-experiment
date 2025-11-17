import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_gray_image
from rs_analyze import rs_statistics

def visualize_histogram(cover: np.ndarray, stego: np.ndarray, save_path: str):
    """
    绘制并保存cover与stego的灰度直方图对比。
    """
    hist_cover = cv2.calcHist([cover], [0], None, [256], [0, 256]).flatten()
    hist_stego = cv2.calcHist([stego], [0], None, [256], [0, 256]).flatten()
    hist_cover /= hist_cover.sum()
    hist_stego /= hist_stego.sum()
    bins = np.arange(256)
    plt.figure(figsize=(10,5))
    plt.plot(bins, hist_cover, label='Cover', color='blue')
    plt.plot(bins, hist_stego, label='Stego', color='red')
    plt.title('Gray Histogram Comparison')
    plt.xlabel('Gray Level')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Histogram saved to {save_path}')

def visualize_histgram_difference(cover: np.ndarray, stego: np.ndarray, save_path: str):
    """
    绘制cover与stego的偶奇灰度级差值曲线。
    """
    hist_cover = cv2.calcHist([cover], [0], None, [256], [0, 256]).flatten()
    hist_stego = cv2.calcHist([stego], [0], None, [256], [0, 256]).flatten()
    hist_cover /= hist_cover.sum()
    hist_stego /= hist_stego.sum()
    diff_bins = np.arange(0, 256, 2)
    diff_cover = np.abs(hist_cover[0::2] - hist_cover[1::2])
    diff_stego = np.abs(hist_stego[0::2] - hist_stego[1::2])
    plt.figure(figsize=(10,5))
    plt.plot(diff_bins, diff_cover, label='Cover', color='blue')
    plt.plot(diff_bins, diff_stego, label='Stego', color='red')
    plt.title('|Hist[2i] - Hist[2i+1]| Comparison')
    plt.xlabel('Even Gray Level (2i)')
    plt.ylabel('Absolute Difference')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Difference curve saved to {save_path}')

def rs_analysis(image_path: str, group_size: int = 4):
    """
    对指定图像进行RS分析，输出估计嵌入率。
    """
    mask = [1, 0, 0, 1]
    image = load_gray_image(image_path)
    p = rs_statistics(image, group_size, mask)
    print(f'Estimated embedding rate: {p}')
    return p

def main():
    import argparse
    parser = argparse.ArgumentParser(description='F5 Steganography Security Analysis')
    parser.add_argument('--cover_path', type=str, default='pics/lena_gray.png')
    parser.add_argument('--stego_path', type=str, default='res/lena_gray_f5_stego.png')
    parser.add_argument('--hist_path', type=str, default='res/lena_gray_f5_hist.png')
    parser.add_argument('--diff_path', type=str, default='res/lena_gray_f5_hist_diff.png')
    parser.add_argument('--group_size', type=int, default=4)
    args = parser.parse_args()
    cover = load_gray_image(args.cover_path)
    stego = load_gray_image(args.stego_path)
    os.makedirs(os.path.dirname(args.hist_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.diff_path), exist_ok=True)
    visualize_histogram(cover, stego, args.hist_path)
    visualize_histgram_difference(cover, stego, args.diff_path)
    print('RS analysis for cover:')
    rs_analysis(args.cover_path, args.group_size)
    print('RS analysis for stego:')
    rs_analysis(args.stego_path, args.group_size)

if __name__ == '__main__':
    main()
