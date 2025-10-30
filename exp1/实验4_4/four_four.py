import cv2 as cv
import numpy as np

# ====================================================================
# A. 辅助函数
# ====================================================================

def load_image_gray(image_path):
    """加载灰度图像为 float32 数组，用于 DCT 运算"""
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法加载图片: {image_path}")
        return None
    return img.astype(np.float32)

def save_image(image_array, save_path, quality=90):
    """保存图像到文件（自动裁剪并转换为 0~255 的 uint8），支持JPEG质量设置"""
    norm_img = np.clip(image_array, 0, 255).astype(np.uint8)
    if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
        cv.imwrite(save_path, norm_img, [cv.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv.imwrite(save_path, norm_img)
    print(f"已保存图像到: {save_path}")

def message_to_bits(message):
    """将字符串消息转换为一个比特列表，末尾加上终止符 '$$$'"""
    bits = []
    message_with_terminator = message + '$$$' 
    for char in message_with_terminator:
        binary_char = bin(ord(char))[2:].zfill(8)
        bits.extend([int(b) for b in binary_char])
    return bits

def bits_to_message(bits):
    """将比特流转换为字符串消息，遇到终止符 '$$$' 则停止并去除终止符"""
    message_bytes = []
    # ASCII('$') = 36 = 00100100
    expected_terminator_bin = '00100100' * 3 
    
    # 查找终止符的循环
    i = 0
    while i + 7 < len(bits):
        byte_bits = bits[i:i+8]
        binary_string = "".join(map(str, byte_bits))
        
        if binary_string == '00100100':
            # 检查整个终止符 '$$$' (24 比特)
            if i + 23 < len(bits):
                terminator_bits = bits[i:i+24]
                terminator_string = "".join(map(str, terminator_bits))
                if terminator_string == expected_terminator_bin:
                    # 找到了终止符，退出解码循环
                    break
        
        char_code = int(binary_string, 2)
        try:
            message_bytes.append(chr(char_code))
        except ValueError:
            pass 
        i += 8
            
    secret_message = "".join(message_bytes)
    return secret_message

def psnr_uint8(orig_uint8, recon_uint8):
    """计算两张 uint8 灰度图的 PSNR（dB）"""
    orig = np.asarray(orig_uint8, dtype=np.float32)
    recon = np.asarray(recon_uint8, dtype=np.float32)
    if orig.shape != recon.shape:
        h = min(orig.shape[0], recon.shape[0])
        w = min(orig.shape[1], recon.shape[1])
        orig = orig[:h, :w]
        recon = recon[:h, :w]
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10.0 * np.log10((PIXEL_MAX ** 2) / mse)

# ====================================================================
# B. 嵌入与提取功能
# ====================================================================

def embed_dct_two_point(image_array, message, C1_pos=(3,4), C2_pos=(4,3), block_size=8, delta=25.0):
    """
    DCT 域两点法隐写嵌入功能。delta为可调参数
    """
    H, W = image_array.shape
    stego_image = image_array.copy()
    bits = message_to_bits(message)
    bit_idx = 0
    rows, cols = H // block_size, W // block_size

    for i in range(rows):
        for j in range(cols):
            if bit_idx >= len(bits):
                return stego_image
            r0, r1 = i*block_size, (i+1)*block_size
            c0, c1 = j*block_size, (j+1)*block_size
            dct_block = cv.dct(stego_image[r0:r1, c0:c1])

            avg = (dct_block[C1_pos] + dct_block[C2_pos]) / 2.0
            if bits[bit_idx] == 1:
                dct_block[C1_pos] = avg + delta / 2.0
                dct_block[C2_pos] = avg - delta / 2.0
            else:
                dct_block[C1_pos] = avg - delta / 2.0
                dct_block[C2_pos] = avg + delta / 2.0

            stego_image[r0:r1, c0:c1] = cv.idct(dct_block)
            bit_idx += 1
    return stego_image

def extract_dct_two_point(stego_image, C1_pos=(3,4), C2_pos=(4,3), block_size=8):
    """
    DCT 域两点法隐写提取功能。
    包含提前终止循环的优化。
    """
    H, W = stego_image.shape
    bits = []
    rows, cols = H // block_size, W // block_size

    for i in range(rows):
        for j in range(cols):
            r0, r1 = i*block_size, (i+1)*block_size
            c0, c1 = j*block_size, (j+1)*block_size
            dct_block = cv.dct(stego_image[r0:r1, c0:c1])
            bits.append(1 if dct_block[C1_pos] > dct_block[C2_pos] else 0)

    return bits_to_message(bits)

# ====================================================================
# C. 主程序测试（已加入 PSNR 输出）
# ====================================================================

carrier_image_path = 'dy_picture.jpg'
secret_message_dct = "This is my secret message for DCT two-point stego!" 

carrier_image_gray = load_image_gray(carrier_image_path)

if carrier_image_gray is not None:
    try:
        orig_uint8 = np.clip(carrier_image_gray, 0, 255).astype(np.uint8)
        for delta in [5, 10, 20, 30, 40, 50]:
            # 1) 嵌入（得到 float32 stego，未压缩）
            stego_image_dct = embed_dct_two_point(carrier_image_gray, secret_message_dct, delta=delta)

            # 2) 计算保存前 PSNR（反映嵌入引入的失真）
            stego_pre_uint8 = np.clip(stego_image_dct, 0, 255).astype(np.uint8)
            psnr_pre = psnr_uint8(orig_uint8, stego_pre_uint8)

            # 3) 保存为 JPEG（评估经过有损压缩后的效果），并重新加载
            stego_save_path = f"stego_dct_two_point_{delta}.jpg"
            save_image(stego_image_dct, stego_save_path, quality=75)  # 若想无损改为 PNG

            loaded_stego = load_image_gray(stego_save_path)
            if loaded_stego is None:
                print(f"\ndelta={delta} 保存或加载失败，跳过")
                continue
            loaded_uint8 = np.clip(loaded_stego, 0, 255).astype(np.uint8)

            # 4) 计算保存后 PSNR（反映嵌入+压缩后的总失真）
            psnr_post = psnr_uint8(orig_uint8, loaded_uint8)

            # 5) 提取并判断是否一致
            extracted_message_dct = extract_dct_two_point(loaded_stego)
            success = (extracted_message_dct == secret_message_dct)
            status_text = "✅ 提取一致" if success else "❌ 提取不一致"

            # 6) 输出
            pre_text = "inf" if psnr_pre == float('inf') else f"{psnr_pre:.2f} dB"
            post_text = "inf" if psnr_post == float('inf') else f"{psnr_post:.2f} dB"
            print(f"\ndelta={delta}  PSNR_before_save={pre_text}   PSNR_after_save={post_text}   {status_text}")
            # 可选：显示或保存差异图以便可视化（见建议）

    except ValueError as e:
        print(e)