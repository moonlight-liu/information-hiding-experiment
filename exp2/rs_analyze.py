import argparse
from typing import List, Tuple, Optional
import numpy as np

from utils import load_gray_image


"""
RS 分析工具：实现用于估计嵌入率的 RS 检测算法的辅助函数和主流程。

主要函数：
- flip_pixel: 根据 flag 翻转单个像素（用于模拟翻转操作）。
- flip_image: 对整幅图像进行 LSB 翻转（所有像素 +1/-1）。
- group_correlation: 计算组内相关性度量（相邻差的绝对和）。
- count_RS: 对图像按组统计 R 和 S 的个数（用于 RS 统计）。
- rs_statistics: 执行完整的 RS 统计并求解估计嵌入率 p。

本文件不改变图像数据类型或像素范围（假设输入为 0-255 的 uint8 灰度图）。
"""

def flip_pixel(pixel: int, flag: int) -> int:
    """
    根据 flag 翻转单个像素的 LSB 表现（模拟 F_M 或 F_-M 操作）。

    参数：
    - pixel: 单个像素值（假定为 int，0-255）。
    - flag: 翻转标志：
        1 -> 如果像素为偶数则 +1，否则 -1（即翻转 LSB）；
       -1 -> 相反的翻转方向（偶数 -1，奇数 +1）；
        0 -> 不翻转，直接返回原像素。

    返回：翻转后的像素值（int）。
    注意：调用方需保证结果仍在有效灰度范围内（本函数不做越界裁剪）。
    """
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
    """
    对整幅灰度图像执行 LSB 翻转（每个像素 +1 或 -1，以翻转其 LSB）。

    输入：二维 numpy 数组（灰度图）。
    输出：同尺寸的 numpy 数组，像素均已按 "翻转 LSB" 操作修改。
    用途：在 RS 统计中用于计算 (1 - p/2) 情况下的 R/S 值（对图像全部像素应用翻转）。
    """
    H, W = image.shape
    flipped = image.copy()
    for i in range(H):
        for j in range(W):
            pixel = image[i, j]
            # 使偶数变奇数、奇数变偶数以翻转 LSB
            if pixel % 2 == 0:
                pixel += 1
            else:
                pixel -= 1
            flipped[i, j] = pixel
    return flipped

def group_correlation(group: np.ndarray):
    """
    计算一个像素组的“相关性”度量：相邻像素差的绝对值之和。

    该度量用于比较原始组与翻转后组之间的平滑性变化，从而判定该组属于 R（相关性增加）还是 S（相关性减少）。
    """
    return np.sum(np.abs(np.diff(group)))

def count_RS(
    image: np.ndarray,
    mask: List[int],
    group_size: int) -> Tuple[int, int]:
    """
    对图像按固定大小分组，统计 R 和 S 的数量。

    参数：
    - image: 灰度图（二维 numpy 数组）。
    - mask: 翻转掩码（长度应为 group_size），掩码中元素为 1/0/-1，表示对组内每个位置应用的翻转方式。
    - group_size: 每组包含的像素数量（展开后按行优先分组）。

    行为：
    - 将图像展平后按 group_size 切片为若干组；
    - 对每组计算原始相关性与按 mask 翻转后的相关性；
    - 若翻转后相关性大于原始则计入 R，否则若更小则计入 S；
      相等的组既不计入 R 也不计入 S。

    返回：(R, S)。
    """
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
    """
    使用 RS 方法估计图像的嵌入率 p。

    过程概要：
    1. 计算 mask 和 -mask 在原图像下的 R/S：分别为 RM_pos/SM_pos 和 RM_neg/SM_neg（对应 p/2 情况）。
    2. 对整幅图像执行翻转（flip_image），计算翻转图像下的 R/S（对应 1 - p/2 情况）。
    3. 根据 RS 分析理论，构造关于 p 的二次方程并求解，返回估计的嵌入率 p。

    返回：估计的嵌入率 p（浮点数）。
    注意：函数直接打印中间 R/S 统计值以便调试和查看结果。
    """
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
    
    # 1-p/2 概率下的R和S（对整个图像先翻转再统计）
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
    
    # 根据论文推导得到关于未知量的二次方程系数 a,b,c，求根并转换为嵌入率 p
    a = 2 * (d1 + d0)
    b = d_minus0 - d_minus1 - d1 - 3 * d0
    c = d0 - d_minus0
    roots = np.roots([a, b, c])
    valid_root = roots[np.argmin(np.abs(roots))]
    p = valid_root / (valid_root - 0.5)
    return p

def main(args):
    """
    脚本主流程：加载灰度图像，使用默认 mask 调用 rs_statistics 估计嵌入率并打印结果。

    默认 mask = [1,0,0,1]（长度与默认 group_size=4 对应）。
    """
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