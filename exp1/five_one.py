import math

import cv2 as cv
import numpy as np
from PIL import Image

# 假设你已经定义了 load_image_gray, save_image, message_to_bits, bits_to_message 函数
# 定义一个函数，用于加载图像
def load_image(image_path):
    """加载图像并转换为NumPy数组"""
    try:
        img = Image.open(image_path).convert('RGB')  # 确保是RGB格式
        img_array = np.array(img)
        print(f"图像 {image_path} 加载成功，形状: {img_array.shape}")
        return img_array
    except FileNotFoundError:
        print(f"错误：未找到文件 {image_path}。请检查路径。")
        return None
      
def message_to_bits(message):
    """将字符串消息转换为一个比特列表 (List[int])，前 32 比特存储消息字节长度"""
    message_bytes = message.encode("utf-8")
    length = len(message_bytes)
    header_bits = [int(b) for b in format(length, "032b")]
    payload_bits = []
    for byte in message_bytes:
        payload_bits.extend(int(b) for b in format(byte, "08b"))
    bits = header_bits + payload_bits
    print(f"消息 '{message}' 转换为比特流，共 {len(bits)} 比特 (含 32 位长度头)。")
    return bits
  
# 确保加载的图像是灰度图，因为 DCT 通常在灰度图或 YCbCr 的 Y 通道上进行
def load_image_gray(image_path):
    """加载图像并转换为灰度图 (NumPy 数组)"""
    try:
        # 使用 PIL 加载并转换为灰度，保证兼容性
        img = Image.open(image_path).convert('L') 
        img_array = np.array(img, dtype=np.float32) # 转换为float32进行DCT运算
        print(f"图像 {image_path} 加载成功，形状: {img_array.shape}")
        return img_array
    except FileNotFoundError:
        print(f"错误：未找到文件 {image_path}。请检查路径。")
        return None

def save_image(img_array, path):
    """将 NumPy 数组保存为图像，并确保数据类型正确"""
    # 将浮点数转换为 uint8，并确保值在 0-255 之间
    img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_pil.save(path)
    print(f"图像已保存为 {path}")
    

def binarize_image(image_array, threshold=128):
    """
    将灰度图二值化为纯黑（0）和纯白（255）图像。
    注意：在二值图像隐写中，我们通常将目标像素（例如文字或图案）视为黑色（0），背景视为白色（255）。
    """
    # 转换为 uint8
    img_uint8 = np.clip(image_array, 0, 255).astype(np.uint8)
    # OpenCV 的 cv.THRESH_BINARY_INV 是反向二值化：
    # > Threshold 的像素设为 0（黑色，前景）
    # <= Threshold 的像素设为 255（白色，背景）
    _, binary_img = cv.threshold(img_uint8, threshold, 255, cv.THRESH_BINARY_INV)
    return binary_img.astype(np.uint8)

DEFAULT_BLOCK_SIZE = 8
TARGET_HIGH = 0.52
TARGET_LOW = 0.48


def embed_binary_density(binary_img, message, block_size=DEFAULT_BLOCK_SIZE, max_change_ratio=0.05):
    """
    二值图像隐写嵌入功能：基于区域密度（黑像素比例）。
    
    参数:
    - binary_img: 纯黑白图像 NumPy 数组 (0 或 255, uint8)
    - max_change_ratio: 最大允许修改的像素比例 (例如 5% = 0.05)
    
    返回:
    - stego_binary_img: 隐写后的二值图像 NumPy 数组
    """
    H, W = binary_img.shape
    stego_binary_img = binary_img.copy()
    
    secret_bits = message_to_bits(message)
    total_bits = len(secret_bits)
    
    block_rows = H // block_size
    block_cols = W // block_size
    max_capacity = block_rows * block_cols
    
    if total_bits > max_capacity:
        raise ValueError(f"错误：消息过长。需要 {total_bits} 比特，但最大容量为 {max_capacity} 比特。")

    bit_index = 0
    # 块内总像素数
    block_total_pixels = block_size * block_size
    # 遍历图像的 8x8 块
    for i in range(block_rows):
        for j in range(block_cols):
            if bit_index >= total_bits:
                break
                
            r_start, r_end = i * block_size, (i + 1) * block_size
            c_start, c_end = j * block_size, (j + 1) * block_size
            
            block = stego_binary_img[r_start:r_end, c_start:c_end]
            
            # 1. 计算当前黑像素（0）比例 P_black
            # 记住：0 是黑色，255 是白色
            P_black = np.sum(block == 0) / block_total_pixels
            max_changes = max(1, int(np.ceil(block_total_pixels * max_change_ratio)))

            bit_to_embed = secret_bits[bit_index]

            # 在嵌入阶段，去掉 success/max_changes 判断，直接执行像素翻转
            if bit_to_embed == 1:
                required_black = math.ceil(TARGET_HIGH * block_total_pixels)
                current_black = int(np.sum(block == 0))
                missing_black = max(0, required_black - current_black)
                white_indices = np.column_stack(np.where(block == 255))
                if missing_black > len(white_indices):
                    missing_black = len(white_indices)
                if missing_black > 0:
                    flip_indices = np.random.choice(len(white_indices), missing_black, replace=False)
                    for idx in flip_indices:
                        r, c = white_indices[idx]
                        block[r, c] = 0
            elif bit_to_embed == 0:
                allowed_black = math.floor(TARGET_LOW * block_total_pixels)
                current_black = int(np.sum(block == 0))
                excess_black = max(0, current_black - allowed_black)
                black_indices = np.column_stack(np.where(block == 0))
                if excess_black > len(black_indices):
                    excess_black = len(black_indices)
                if excess_black > 0:
                    flip_indices = np.random.choice(len(black_indices), excess_black, replace=False)
                    for idx in flip_indices:
                        r, c = black_indices[idx]
                        block[r, c] = 255
            stego_binary_img[r_start:r_end, c_start:c_end] = block
            bit_index += 1
        
        if bit_index >= total_bits:
            break

    if bit_index < total_bits:
        raise RuntimeError(
            f"嵌入失败：仅嵌入 {bit_index} / {total_bits} 比特。请提高 max_change_ratio 或进一步缩短消息。"
        )

    print(f"二值图像嵌入完成。共嵌入 {bit_index} 比特。")
    return stego_binary_img
  
def bits_to_message(bits):
    """根据前 32 位长度头还原消息，如果长度不足返回 ("", False)"""
    if len(bits) < 32:
        return "", False

    length_bits = bits[:32]
    byte_length = int("".join(map(str, length_bits)), 2)
    payload_bits = bits[32:32 + byte_length * 8]

    if len(payload_bits) < byte_length * 8:
        return "", False

    message_bytes = bytearray()
    for i in range(0, len(payload_bits), 8):
        byte_bits = payload_bits[i:i + 8]
        message_bytes.append(int("".join(map(str, byte_bits)), 2))

    try:
        message = message_bytes.decode("utf-8")
    except UnicodeDecodeError:
        message = message_bytes.decode("utf-8", errors="replace")

    return message, True


# --- 主程序测试：二值化和嵌入 ---
# 假设载体图像是 'dy_picture.jpg'
carrier_image_path = 'dy_picture.jpg'
carrier_image_gray = load_image_gray(carrier_image_path)

if carrier_image_gray is not None:
    # 1. 二值化
    binary_img = binarize_image(carrier_image_gray, threshold=120)
    save_image(binary_img, "binary_carrier.png")
    
    secret_message_binary = "Binary test"
    
    try:
        # 2. 嵌入
        stego_binary_img = embed_binary_density(binary_img, secret_message_binary)
        save_image(stego_binary_img, "stego_binary_density.png")
        
    except (ValueError, RuntimeError) as e:
        print(e)
        
def extract_binary_density(
    stego_binary_img,
    block_size=DEFAULT_BLOCK_SIZE,
    threshold=0.5,
):
    H, W = stego_binary_img.shape
    block_rows = H // block_size
    block_cols = W // block_size
    extracted_bits = []
    block_total_pixels = block_size * block_size

    required_bits = None

    for i in range(block_rows):
        for j in range(block_cols):
            r_start, r_end = i * block_size, (i + 1) * block_size
            c_start, c_end = j * block_size, (j + 1) * block_size
            block = stego_binary_img[r_start:r_end, c_start:c_end]
            P_black = np.sum(block == 0) / block_total_pixels

            bit = 1 if P_black >= threshold else 0
            extracted_bits.append(bit)

            # 读取长度头后确定需要的总比特数
            if required_bits is None and len(extracted_bits) >= 32:
                length_bits = extracted_bits[:32]
                byte_length = int("".join(map(str, length_bits)), 2)
                required_bits = 32 + byte_length * 8

            if required_bits is not None and len(extracted_bits) >= required_bits:
                break

        if required_bits is not None and len(extracted_bits) >= required_bits:
            break

    secret_message, completed = bits_to_message(extracted_bits)

    if not completed:
        print("⚠️ 提取的比特流长度不足以还原完整消息。")

    print(f"提取完成，消息长度: {len(secret_message)} 字符。")
    return secret_message

# --- 主程序测试：提取 ---
if 'stego_binary_img' in locals():
    print("-" * 30)
    # 模拟从磁盘加载隐写图
    loaded_stego_binary = load_image_gray("stego_binary_density.png")
    
    if loaded_stego_binary is not None:
        # 确保加载的图像是二值化的，将其再次转换为 0 或 255 的 uint8 格式
        # 注意：load_image_gray 返回 float32，需要转回 uint8 才能进行 == 0 的操作
        loaded_stego_binary = np.clip(loaded_stego_binary, 0, 255).astype(np.uint8)
        loaded_stego_binary = np.where(loaded_stego_binary > 127, 255, 0).astype(np.uint8)

        # 执行提取
        extracted_message_binary = extract_binary_density(loaded_stego_binary)
        print(f"原始消息: '{secret_message_binary}'")
        print(f"提取消息: '{extracted_message_binary}'")

        if extracted_message_binary == secret_message_binary:
            print("✅ 二值图像隐写成功，提取消息与原始消息一致！")
        else:
            print("❌ 警告：二值图像提取消息与原始消息不一致。")