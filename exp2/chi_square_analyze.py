import argparse
from typing import List
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

from utils import load_gray_image


def chi2_statistics(
    image: np.ndarray,
    ratio: int):
    H, W = image.shape
    flatten_image = image.flatten()
    partial = flatten_image[: int(H * W * ratio)]
    
    counts = np.bincount(partial, minlength=256) # 统计0-255，共256个统计量
    chi2_stat = 0.0
    for k in range(0, 256, 2):
        c0, c1 = counts[k], counts[k+1]
        if c0 + c1 == 0:
            continue
        E = (c0 + c1) / 2.0
        chi2_stat += ((c0 - E) ** 2 + (c1 - E) ** 2) / E # 累加卡方统计量
        # chi2_stat += (c0 - c1) ** 2 / (c0 + c1)
        
    p = chi2.sf(chi2_stat, df=128) # df即自由度，对于灰度图像来说自由度是256/2=128
    return chi2_stat, p

def visualize_chi2_p_values(
    p_values: List[float],
    ratios: List[float],
    save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(ratios, p_values, color='red', alpha=0.7)
    plt.xlabel('Pixel ratio')
    plt.ylabel('Chi-square p-value')
    plt.title('Chi-square p-value VS Pixel ratio')
    plt.xticks(ratios)
    plt.ylim(0, 1.01)
    plt.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Chi-square p-value VS Pixel ratio have been saved to "{save_path}"')
    
def main(args):
    image = load_gray_image(args.image_path)
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    p_list = []
    for ratio in ratios:
        chi2_stat, p = chi2_statistics(image, ratio)
        p_list.append(p)
    
    visualize_chi2_p_values(p_list, ratios, args.vis_chi2p_vs_ratio)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSB Steganography Embed & Extract Demo')
    parser.add_argument('--image_path', type=str, default='res/lena_gray_lsb_stego.png')
    parser.add_argument('--vis_chi2p_vs_ratio', type=str, default='res/chi2p_vs_ratio.png')
    args = parser.parse_args()
    main(args)
