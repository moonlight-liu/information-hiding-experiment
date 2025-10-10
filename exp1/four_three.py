import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

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
    img_pil = Image.fromarray(img_uint8, 'L')
    img_pil.save(path)
    print(f"图像已保存为 {path}")
    
def process_dct(image_array, mask_size, block_size=8):
    """
    对图像进行分块DCT，并删除高频系数。
    
    参数:
    - image_array: 灰度图像 NumPy 数组 (float32)
    - mask_size: 保留的左上角低频区域大小 (e.g., 3 表示保留 3x3 的低频系数)
    - block_size: DCT 分块大小 (默认为 8)
    
    返回:
    - processed_image: 处理后的图像 NumPy 数组
    """
    H, W = image_array.shape
    processed_image = np.zeros_like(image_array)

    # 1. 创建高频删除掩码
    # 这是一个 8x8 的矩阵，只有左上角 mask_size x mask_size 区域是 1，其余是 0
    mask = np.zeros((block_size, block_size), dtype=np.float32)
    mask[:mask_size, :mask_size] = 1.0

    # 2. 遍历图像的 8x8 块
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            # 确保不越界（处理非 8 的倍数的边缘）
            block = image_array[i:i+block_size, j:j+block_size]
            
            # 如果边缘块不是 8x8，则跳过或填充。这里为简化，假设图像大小是 8 的倍数，
            # 否则需要进行边缘填充。
            if block.shape != (block_size, block_size):
                processed_image[i:i+block.shape[0], j:j+block.shape[1]] = block
                continue
                
            # 3. DCT 变换 (使用 cv2.dct)
            dct_block = cv.dct(block)
            
            # 4. 删除高频系数：应用掩码
            masked_dct_block = dct_block * mask
            
            # 5. IDCT 逆变换 (使用 cv2.idct)
            idct_block = cv.idct(masked_dct_block)
            
            # 6. 重新组合
            processed_image[i:i+block_size, j:j+block_size] = idct_block
            
    return processed_image

# --- 主程序测试 ---
carrier_image_path = 'dy_picture.jpg'
carrier_image_gray = load_image_gray(carrier_image_path)

if carrier_image_gray is not None:
    # 原始图像保存
    save_image(carrier_image_gray, "original_gray.png")
    
    # 保留 3x3 低频系数，其余高频删除
    processed_image_3x3 = process_dct(carrier_image_gray, mask_size=3)
    save_image(processed_image_3x3, "dct_high_freq_removed_3x3.png")

    # 保留 6x6 低频系数，观察失真程度
    processed_image_6x6 = process_dct(carrier_image_gray, mask_size=6)
    save_image(processed_image_6x6, "dct_high_freq_removed_6x6.png")