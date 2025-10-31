from utils import *
import cv2
from typing import List
def zigzag_scan(block):
    """Perform ZigZag scan of an 8x8 block"""
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
def inverse_zigzag(arr):
    """Inverse ZigZag: 1D array -> 8x8 block"""
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
# JPEG-like quantization table
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]], dtype=np.float32)

def get_ac_coeffs(image: np.ndarray) -> List[np.float32]:
    H, W = image.shape
    ac_list = []
    dc_list = []
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = image[i: i+8, j: j+8].astype(np.float32)
            dct_block = cv2.dct(block)
            quant_block = np.round(dct_block / Q)
            zz = zigzag_scan(quant_block)
            dc_coeff = zz[0]
            ac_coeffs = zz[1:] # 跳过DC
            ac_list.append(ac_coeffs)
            dc_list.append(dc_coeff)
    return ac_list, dc_list

def reconstruct_blocks(image: np.ndarray, coeff_list: List[np.ndarray]):
    H, W = image.shape
    idx = 0
    recon_image = np.zeros((H, W), dtype=np.float32)
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            if idx > len(ac_list):
                break
            coeff = coeff_list[idx]
            quant_block = inverse_zigzag(coeff)
            block = quant_block * Q
            recon_block = cv2.idct(block)
            recon_image[i: i+8, j: j+8] = recon_block
            idx += 1
            
    recon_image = np.clip(recon_image, 0, 255).astype(np.uint8)
    return recon_image

def count_nonzero_ac(ac_list: List[np.float32]):
    cnt = 0
    for ac in ac_list:
        cnt += np.sum(ac != 0)
    return cnt

def combine_dc_ac(dc_list, ac_list):
    full_list = []
    for dc, ac in zip(dc_list, ac_list):
        full_block = np.zeros(64, dtype=np.float32)
        full_block[0] = dc        # DC放在索引0
        full_block[1:] = ac       # AC放在1-63
        full_list.append(full_block)
    return full_list

ratio = 0.5
cover = load_gray_image('pics/lena_gray.png')
H, W = cover.shape
stego = cover.copy()
ac_list, dc_list = get_ac_coeffs(cover)
nonzeor_ac_cnt = count_nonzero_ac(ac_list)
bit_length = int(H * W * ratio) if nonzeor_ac_cnt >= int(H * W * ratio) else nonzeor_ac_cnt# 控制嵌入比特数量
embed_bits = generate_random_bits(bit_length)

bit_idx = 0
new_ac_list = []

for blk_ac in ac_list:
    new_blk_ac = blk_ac.copy()
    for i, ac in enumerate(new_blk_ac):
        if ac == 0:
            continue

        if bit_idx >= len(embed_bits):
            break

        bit = int(embed_bits[bit_idx])

        # F4嵌入逻辑：
        # 正奇/负偶 -> 1，正偶/负奇 -> 0
        if (ac > 0 and ac % 2 != bit) or (ac < 0 and (-ac) % 2 == bit):
            # 修改一个单位
            new_blk_ac[i] += -1 if ac > 0 else 1

        bit_idx += 1

    new_ac_list.append(new_blk_ac)

    if bit_idx >= len(embed_bits):
        break

coeff_list = combine_dc_ac(dc_list, new_ac_list)
stego = reconstruct_blocks(cover, coeff_list)
cv2.imwrite('res/lena_gray_f4_stego.png', stego)

stego = load_gray_image('res/lena_gray_f4_stego.png')

def f4_extract(bit_length: int, stego: np.ndarray) -> str:
    H, W = stego.shape
    bits = []

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            if len(bits) >= bit_length:
                break

            block = stego[i:i+8, j:j+8].astype(np.float32)
            dct_block = cv2.dct(block)
            quant_block = np.round(dct_block / Q)
            zz = zigzag_scan(quant_block)

            # 跳过DC
            for ac in zz[1:]:
                if ac == 0:
                    continue

                # F4逻辑：正奇/负偶 → 1，正偶/负奇 → 0
                if (ac > 0 and ac % 2 == 1) or (ac < 0 and (-ac) % 2 == 0):
                    bits.append('1')
                else:
                    bits.append('0')

                if len(bits) >= bit_length:
                    break

        if len(bits) >= bit_length:
            break

    print(f"Extracted {len(bits)}/{bit_length} bits.")
    return ''.join(bits[:bit_length])

extract_bits = f4_extract(bit_length, stego)

accuracy = np.mean([b1==b2 for b1, b2 in zip(embed_bits, extract_bits)])
print(f"Extraction accuracy: {accuracy:.4f}")
        