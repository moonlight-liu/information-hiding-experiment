import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *

def do_embedding(pixel: np.uint8, bit: str, prob: float) -> np.uint8:
    """
    将单个像素嵌入一比特并返回修改后的像素值。

    说明：
    - pixel: 单个灰度像素（0-255）。
    - bit: 要嵌入的比特，'0' 或 '1'（函数内部转换为 int）。
    - prob: 随机概率，用于决定在 LSB 不一致时像素是 +1 还是 -1（prob>=0.5 则 +1，否则 -1）。

    行为：如果像素的最低有效位(LSB)已经等于目标 bit，则不修改；
    否则根据 prob 决定向上/向下微调一个单位以改变 LSB。对边界值 0 和 255 做了保护，避免越界。

    返回值：类型为 np.uint8 的修改后像素（保证在 0-255 之间）。
    """
    bit = int(bit)
    pixel = int(pixel)  # convert to Python int to avoid overflow
    lsb = pixel % 2

    # 如果 LSB 已经等于目标 bit，则无需修改
    if lsb == bit:
        return np.uint8(pixel)

    # 根据随机概率确定是 +1 还是 -1（改变像素值以翻转 LSB）
    delta = 1 if prob >= 0.5 else -1

    # 对图像边界做保护，避免越界
    if pixel == 0 and delta == -1:
        pixel += 1
    elif pixel == 255 and delta == 1:
        pixel -= 1
    else:
        pixel += delta

    return np.uint8(pixel)

def lsbm_embed(
    msg_bits: str, 
    cover: np.ndarray,
    seed: int
) -> np.ndarray:
    """
    在覆盖图像上随机位置嵌入比特串（LSBM 方法的实现）。

    输入：
    - msg_bits: 要嵌入的比特串（例如 '010101'）。
    - cover: 灰度 cover 图像的 numpy 数组（二维）。
    - seed: 用于随机选择像素位置和随机决定 +1/-1 的伪随机种子。

    行为：
    - 将 cover 展平为一维像素序列；使用给定 seed 随机选择不重复的位置来嵌入每一比特；
    - 对每个选定像素，调用 do_embedding 通过 +1/-1 调整像素以实现 LSB 的翻转或保持。

    返回：嵌入后并恢复为原始形状的 stego 图像（np.ndarray）。
    抛出：当消息位数超过像素总数时抛出 ValueError。
    """
    flatten_cover = cover.flatten()
    pixel_cnt = flatten_cover.shape[0]
    
    if len(msg_bits) > pixel_cnt:
        raise ValueError(f'Input message bits are too long to embed!')
    
    rng = np.random.default_rng(seed)
    # 随机选择嵌入位置（不重复）
    indices = rng.choice(pixel_cnt, size=len(msg_bits), replace=False)
    # 为每个嵌入位置生成一个随机数决定 +1 或 -1
    probs = rng.random(len(msg_bits))
    for idx, bit, prob in zip(indices, msg_bits, probs):
        modified_pixel = do_embedding(flatten_cover[idx], bit, prob)
        flatten_cover[idx] = modified_pixel 
    stego = flatten_cover.reshape(cover.shape)
    
    return stego

def lsbm_extract(
    bit_length: int,
    stego: np.ndarray,
    seed: int) -> str:
    """
    根据与嵌入时相同的 seed，恢复随机嵌入位置并提取 LSB 比特流。

    输入：
    - bit_length: 要提取的比特数（应与嵌入时一致）。
    - stego: 嵌入过的灰度图像（np.ndarray）。
    - seed: 与嵌入相同的随机种子，用于复现嵌入位置。

    行为：按与 lsbm_embed 相同的方式生成位置序列，从每个像素取最低有效位并按顺序拼接成比特串。
    返回：比特串（字符串，例如 '0101'）。
    """
    flatten_stego = stego.flatten()
    pixel_cnt = flatten_stego.shape[0]

    rng = np.random.default_rng(seed)
    # 使用同一 seed 生成与嵌入时相同的位置序列
    indices = rng.choice(pixel_cnt, size=bit_length, replace=False)
    msg_bits = []
    for idx in indices:
        bit = flatten_stego[idx] & 0b00000001
        msg_bits.append(str(bit))
    return ''.join(msg_bits)

def visualize_msg_positions(
    img: np.ndarray,
    bit_length: int,
    seed: int,
    save_path: str
) -> None:
    """
    在灰度图像上可视化嵌入比特的位置。

    - img: 输入灰度图（二维 numpy 数组）。
    - bit_length: 要标记的位置数（与嵌入相同）。
    - seed: 随机种子，用于复现嵌入位置。
    - save_path: 可视化图片保存路径（彩色，红点标记）。

    输出：将带有红点标注的图像写入 save_path。
    """
    assert len(img.shape) == 2, 'Only the gray image is supported!'
    H, W = img.shape
    pixel_cnt = H * W
    rng = np.random.default_rng(seed)
    indices = rng.choice(pixel_cnt, size=bit_length, replace=False)
    
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
    """
    绘制并保存覆盖图与载密图的灰度直方图（规范化后并排显示）。

    - cover / stego: 二维灰度图像数组。
    - save_path: 输出图像文件路径（PNG 等）。
    图像使用两个子图分别展示 cover 和 stego 的归一化灰度分布，方便比较嵌入前后的直方图差异。
    """
    hist_cover = cv2.calcHist([cover], [0], None, [256], [0, 256]).flatten()
    hist_stego = cv2.calcHist([stego], [0], None, [256], [0, 256]).flatten()

    # 归一化为概率分布（总和为 1）
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
    """
    计算并绘制直方图的偶数-奇数灰度级差值：|Hist[2i] - Hist[2i+1]|，用于检测 LSB 嵌入造成的偶奇偶数分布变化。

    - cover / stego: 灰度图像数组。
    - save_path: 输出图表路径。

    输出：生成两张子图，分别展示 cover 与 stego 的 |Hist[2i] - Hist[2i+1]| 曲线，便于直接比较嵌入前后的差异。
    """
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
    """
    演示流程：
    1. 读取灰度 cover 图像；
    2. 根据 ratio 计算要嵌入的比特数，生成随机比特串并嵌入；
    3. 保存 stego，并再次读取来进行提取验证（使用相同 seed）；
    4. 计算提取准确率并生成三类可视化：嵌入位置、灰度直方图、偶奇差值曲线。

    注意：本函数假定 utils.py 中包含 load_gray_image、generate_random_bits、calculate_accuracy 等辅助函数。
    """
    cover = load_gray_image(args.cover_path)
    H, W = cover.shape
    bit_length = int(H * W * args.ratio) # 控制嵌入比特数量
    embed_bits = generate_random_bits(bit_length)
    stego = lsbm_embed(embed_bits, cover, seed=args.seed)
    os.makedirs(os.path.dirname(args.stego_path), exist_ok=True)
    cv2.imwrite(args.stego_path, stego)
    print(f'Stego has been saved to "{args.stego_path}"')
    
    # 提取并计算准确率
    stego = load_gray_image(args.stego_path)
    extract_bits = lsbm_extract(bit_length, stego, seed=args.seed)
    acc = calculate_accuracy(embed_bits, extract_bits)
    print(f'Accuracy is {acc:.4f}')
    
    # 可视化：嵌入位置、直方图、偶奇差异曲线
    visualize_msg_positions(cover, bit_length, seed=args.seed, save_path=args.vis_pos_path)
    visualize_gray_histogram(cover, stego, args.vis_hist_path)
    visualize_histgram_difference(cover, stego, args.vis_diff_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSBR Steganography Embed & Extract Demo')
    parser.add_argument('--cover_path', type=str, default='pics/lena_gray.png')
    parser.add_argument('--stego_path', type=str, default='res/lena_gray_lsbm_stego.png')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--vis_pos_path', type=str, default='res/lena_gray_lsbm_msg_pos.png')
    parser.add_argument('--vis_hist_path', type=str, default='res/lena_gray_lsbm_msg_hist.png')
    parser.add_argument('--vis_diff_path', type=str, default='res/lena_gray_lsbm_msg_hist_diff.png')
    args = parser.parse_args()
    main(args)