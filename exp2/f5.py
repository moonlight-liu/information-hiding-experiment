import numpy as np
import cv2
import os
from typing import List, Tuple
from utils import load_gray_image, generate_random_bits, calculate_accuracy
from f4Pro import get_ac_dc_coeffs, combine_dc_ac, reconstruct_blocks

def hamming_parity_matrix(r: int) -> np.ndarray:
    """
    生成 (2^r-1, r) 汉明码的校验矩阵 H。
    返回：shape=(r, n)，每列为一个码字位置的二进制表示。
    """
    n = 2 ** r - 1
    H = np.zeros((r, n), dtype=int)
    for i in range(n):
        bin_str = format(i+1, f'0{r}b')
        H[:, i] = [int(b) for b in bin_str]
    return H

def matrix_encode(lsb_group: np.ndarray, msg_bits: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    汉明码矩阵编码嵌入：
    lsb_group: 当前分组的LSB序列（长度n）
    msg_bits: 要嵌入的消息比特（长度r）
    H: 汉明码校验矩阵（r x n）
    返回：修改后的LSB序列、需修改的位置（若无需修改则返回-1）
    """
    # 计算当前LSB序列的校验位
    current_syndrome = (H @ lsb_group) % 2
    # 计算需要的校验位与当前校验位的差
    diff = (msg_bits ^ current_syndrome) % 2
    
    if np.all(diff == 0):
        return lsb_group.copy(), -1  # 无需修改
    
    # 查找需要修改的位置：找到diff对应的H矩阵列
    for i in range(H.shape[1]):
        if np.array_equal(H[:, i], diff):
            new_lsb = lsb_group.copy()
            new_lsb[i] ^= 1  # 翻转该位
            return new_lsb, i
    
    # 理论上不会到这里
    raise RuntimeError(f'Diff {diff} not found in Hamming matrix')

def f5_embed(cover: np.ndarray, msg_bits: str, r: int, seed: int = 2025) -> Tuple[np.ndarray, List[int]]:
    """
    F5嵌入主流程：
    cover: 灰度图像
    msg_bits: 要嵌入的比特串
    r: 汉明码参数
    seed: 随机种子
    返回：stego图像，嵌入位置列表
    """
    ac_list, dc_list, Hp, Wp = get_ac_dc_coeffs(cover)
    n = 2 ** r - 1
    H = hamming_parity_matrix(r)
    # 收集所有非零AC系数的位置
    ac_positions = [(blk_idx, i) for blk_idx, blk_ac in enumerate(ac_list) for i, ac in enumerate(blk_ac) if ac != 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(ac_positions)
    msg_idx = 0
    new_ac_list = [blk_ac.copy() for blk_ac in ac_list]
    modified_positions = []
    
    while msg_idx + r <= len(msg_bits):
        # 确保有足够的非零系数
        current_positions = []
        pos_idx = 0
        while len(current_positions) < n and pos_idx < len(ac_positions):
            blk, i = ac_positions[pos_idx]
            if new_ac_list[blk][i] != 0:
                current_positions.append((blk, i))
            pos_idx += 1
        
        if len(current_positions) < n:
            break
        
        # 从位置列表中移除已使用的位置
        for pos in current_positions:
            ac_positions.remove(pos)
        
        # 计算当前LSB和目标消息
        lsb_group = np.array([int(abs(new_ac_list[blk][i]) % 2) for blk, i in current_positions])
        bits = np.array([int(b) for b in msg_bits[msg_idx:msg_idx+r]])
        new_lsb, mod_idx = matrix_encode(lsb_group, bits, H)
        
        # 如果需要修改
        if mod_idx != -1:
            blk, i = current_positions[mod_idx]
            ac = new_ac_list[blk][i]
            
            # F4嵌入逻辑：修改系数使其LSB等于目标值
            current_lsb = int(abs(ac) % 2)
            target_lsb = new_lsb[mod_idx]
            
            if current_lsb != target_lsb:
                # 需要翻转LSB
                if ac > 0:
                    proposed = ac - 1 if ac % 2 == 1 else ac + 1
                else:
                    proposed = ac + 1 if abs(ac) % 2 == 1 else ac - 1
                
                # 处理收缩到0的情况 - 选择远离0的方向
                if proposed == 0:
                    if ac > 0:
                        proposed = 2
                    else:
                        proposed = -2
                
                new_ac_list[blk][i] = proposed
                modified_positions.append((blk, i))
        
        msg_idx += r
    # 重构图像
    coeff_list = combine_dc_ac(dc_list, new_ac_list)
    stego = reconstruct_blocks(coeff_list, (Hp, Wp), cover.shape)
    return stego, modified_positions

def f5_extract(stego: np.ndarray, msg_len: int, r: int, seed: int = 2025) -> str:
    """
    F5提取主流程：
    stego: 载密图像
    msg_len: 要提取的比特数
    r: 汉明码参数
    seed: 随机种子
    返回：提取出的比特串
    """
    ac_list, _, Hp, Wp = get_ac_dc_coeffs(stego)
    n = 2 ** r - 1
    H = hamming_parity_matrix(r)
    ac_positions = [(blk_idx, i) for blk_idx, blk_ac in enumerate(ac_list) for i, ac in enumerate(blk_ac) if ac != 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(ac_positions)
    bits = []
    msg_idx = 0
    while msg_idx + r <= msg_len and len(ac_positions) >= n:
        group_pos = ac_positions[:n]
        ac_positions = ac_positions[n:]
        lsb_group = np.array([int(abs(ac_list[blk][i]) % 2) for blk, i in group_pos])
        # syndrome = H @ lsb_group % 2
        bits.extend(list((H @ lsb_group % 2).astype(str)))
        msg_idx += r
    return ''.join(bits[:msg_len])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='F5 Steganography Embed & Extract Demo')
    parser.add_argument('--cover_path', type=str, default='pics/lena_gray.png')
    parser.add_argument('--stego_path', type=str, default='res/lena_gray_f5_stego.png')
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--ratio', type=float, default=0.5)
    args = parser.parse_args()
    
    print(f'F5 Steganography with Hamming({2**args.r-1}, {args.r})')
    cover = load_gray_image(args.cover_path)
    ac_list, _, _, _ = get_ac_dc_coeffs(cover)
    total_slots = sum([np.sum(np.array(ac) != 0) for ac in ac_list])
    n = 2 ** args.r - 1
    msg_len = (total_slots // n) * args.r
    
    print(f'Total non-zero AC coefficients: {total_slots}')
    print(f'Hamming code parameters: n={n}, r={args.r}')
    print(f'Maximum embeddable bits: {msg_len}')
    print(f'Embedding efficiency: {args.r}/{n} = {args.r/n:.4f} bits per modification')
    
    embed_bits = generate_random_bits(msg_len, seed=args.seed)
    stego, modified_positions = f5_embed(cover, embed_bits, args.r, seed=args.seed)
    
    print(f'Modified positions: {len(modified_positions)}')
    print(f'Modification rate: {len(modified_positions)/total_slots:.4f}')
    
    os.makedirs(os.path.dirname(args.stego_path), exist_ok=True)
    cv2.imwrite(args.stego_path, stego)
    print(f'Stego saved to {args.stego_path}')
    
    extract_bits = f5_extract(stego, msg_len, args.r, seed=args.seed)
    acc = calculate_accuracy(embed_bits, extract_bits)
    print(f'Extracted {len(extract_bits)}/{msg_len} bits. Accuracy: {acc:.4f}')
