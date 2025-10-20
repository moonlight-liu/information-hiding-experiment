import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 解决 Matplotlib 中文/符号(如 α、负号)显示乱码问题：自动选择常见中文字体
def _setup_chinese_font():
    candidates = [
        'Microsoft YaHei',  # 微软雅黑 (Windows 常见)
        'SimHei',           # 黑体
        'SimSun',           # 宋体
        'Noto Sans CJK SC', # 思源黑体简体
        'Arial Unicode MS'  # 全字库（若安装）
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块
            print(f"Matplotlib 使用中文字体: {name}")
            return
    # 若没有找到中文字体，至少保证负号正常，提示可能乱码
    plt.rcParams['axes.unicode_minus'] = False
    print("警告: 未检测到常见中文字体，图中文字可能出现乱码。可安装 'Microsoft YaHei' 或 'SimHei' 字体。")

_setup_chinese_font()

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

# --- DCT域两点法嵌入与提取 ---
def message_to_bits(message: str):
    """字符串 -> 比特流，并在末尾添加终止符 $$$"""
    bits = []
    message_with_terminator = message + '$$$'
    for char in message_with_terminator:
        binary_char = bin(ord(char))[2:].zfill(8)
        bits.extend([int(b) for b in binary_char])
    return bits

def bits_to_message(bits):
    """比特流 -> 字符串，遇到终止符 $$$ 即停止"""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        char = chr(int(''.join(map(str, byte)), 2))
        chars.append(char)
        if ''.join(chars[-3:]) == '$$$':
            return ''.join(chars[:-3])
    return ''.join(chars)

def embed_dct_two_point(image_array: np.ndarray, message: str, C1_pos=(3,4), C2_pos=(4,3), block_size=8, alpha=25.0):
    """
    DCT 域两点法嵌入：在每个 8x8 块通过调节两个系数的相对大小来表达 1/0。
    alpha 为强度，越大鲁棒性越好但失真越大。
    """
    H, W = image_array.shape
    stego = np.array(image_array, dtype=np.float32)
    bits = message_to_bits(message)
    # 容量检查：一个块可嵌入一位
    blocks_y, blocks_x = H // block_size, W // block_size
    capacity = blocks_y * blocks_x
    if len(bits) > capacity:
        raise ValueError(f"消息过长，所需比特 {len(bits)} > 容量 {capacity} (每块1比特)")

    bit_idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            if bit_idx >= len(bits):
                break
            i, j = by * block_size, bx * block_size
            block = stego[i:i+block_size, j:j+block_size]
            dct_block = cv.dct(block)
            avg = (dct_block[C1_pos] + dct_block[C2_pos]) / 2.0
            if bits[bit_idx] == 1:
                dct_block[C1_pos] = avg + alpha/2
                dct_block[C2_pos] = avg - alpha/2
            else:
                dct_block[C1_pos] = avg - alpha/2
                dct_block[C2_pos] = avg + alpha/2
            idct_block = cv.idct(dct_block)
            stego[i:i+block_size, j:j+block_size] = idct_block
            bit_idx += 1
    return stego

def extract_dct_two_point(image_array: np.ndarray, message_length: int, C1_pos=(3,4), C2_pos=(4,3), block_size=8):
    """
    提取：逐块比较两个系数的大小，C1>C2 记为 1，否则 0。
    message_length 为需要提取的比特数（与嵌入时相同）。
    """
    H, W = image_array.shape
    blocks_y, blocks_x = H // block_size, W // block_size
    bits = []
    bit_idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            if bit_idx >= message_length:
                break
            i, j = by * block_size, bx * block_size
            block = image_array[i:i+block_size, j:j+block_size]
            dct_block = cv.dct(block.astype(np.float32))
            bit = 1 if dct_block[C1_pos] > dct_block[C2_pos] else 0
            bits.append(bit)
            bit_idx += 1
    return bits

def save_jpeg(img_array, path, quality=90):
    """以指定 JPEG 质量保存图像"""
    img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
    cv.imwrite(path, img_uint8, [int(cv.IMWRITE_JPEG_QUALITY), int(quality)])

def bit_error_rate(bits1, bits2):
    """计算两比特流的误码率（按最短长度对齐）"""
    bits1 = np.array(bits1)
    bits2 = np.array(bits2)
    min_len = min(len(bits1), len(bits2))
    if min_len == 0:
        return 1.0
    return float(np.mean(bits1[:min_len] != bits2[:min_len]))

# --- 主实验流程 ---
if __name__ == "__main__":
    carrier_image_path = 'dy_picture.jpg'
    carrier_image_gray = load_image_gray(carrier_image_path)
    if carrier_image_gray is None:
        raise SystemExit(1)

    # 保存原图灰度
    save_image(carrier_image_gray, "original_gray.png")

    # 1）随 α 增大，图像失真变化（保存对比）
    message = "信息隐藏实验测试ABC123"
    alphas = [5, 10, 20, 30, 40, 50]
    fig = plt.figure(figsize=(12, 6))
    # 先放原图
    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(np.clip(carrier_image_gray, 0, 255).astype(np.uint8), cmap='gray')
    ax.set_title('原图')
    ax.axis('off')
    # 生成不同 alpha 的 stego 图
    for idx, alpha in enumerate(alphas, start=2):
        stego = embed_dct_two_point(carrier_image_gray, message, alpha=alpha)
        save_image(stego, f"stego_alpha_{alpha}.png")
        ax = fig.add_subplot(2, 4, idx if idx <= 8 else 8)
        ax.imshow(np.clip(stego, 0, 255).astype(np.uint8), cmap='gray')
        ax.set_title(f'α={alpha}')
        ax.axis('off')
        if idx >= 8:
            break
    plt.tight_layout()
    plt.savefig("alpha_distortion_compare.png")
    plt.show()

    # 2）不同质量因子与 α 下的误码率曲线
    qualities = [100, 90, 80, 70, 60, 50, 40, 30]
    results = {a: [] for a in alphas}
    original_bits = message_to_bits(message)
    message_length = len(original_bits)

    for alpha in alphas:
        stego = embed_dct_two_point(carrier_image_gray, message, alpha=alpha)
        ber_list = []
        for q in qualities:
            save_jpeg(stego, 'temp_stego.jpg', quality=q)
            jpeg_img = load_image_gray('temp_stego.jpg')
            extracted_bits = extract_dct_two_point(jpeg_img, message_length)
            ber = bit_error_rate(original_bits, extracted_bits)
            ber_list.append(ber)
            print(f"alpha={alpha}, quality={q}, BER={ber:.4f}")
        results[alpha] = ber_list

    # 绘制 BER 曲线
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        plt.plot(qualities, results[alpha], marker='o', label=f'α={alpha}')
    plt.xlabel('JPEG质量因子')
    plt.ylabel('误码率 (BER)')
    plt.title('不同α下误码率随JPEG质量因子的变化')
    plt.gca().invert_xaxis()  # 质量从高到低
    plt.grid(True)
    plt.legend()
    plt.savefig("ber_vs_quality.png")
    plt.show()