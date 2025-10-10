import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

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

# 示例使用 (假设你的图像路径是 'lenna.png')
carrier_image_path = 'dy_picture.jpg'
carrier_image = load_image(carrier_image_path)
if carrier_image is not None:
    print(f"图像数据类型: {carrier_image.dtype}")
    
# 设计随机取点的算法，随机选取像素点嵌入秘密信息 。

# 提取秘密信息 。

# 画出随机位置（可选，但有助于理解） 。

# 对比隐写前后图像直方图，分析LSB隐写导致的值对效应 。

def message_to_bits(message):
    """将字符串消息转换为一个比特列表 (List[int])"""
    bits = []
    # 首先，编码字符串：在消息末尾添加一个结束标记（例如：'$$$'）
    # 这样提取时就知道何时停止。
    message_with_terminator = message + '$$$'
    
    # 将每个字符转换为8位的ASCII码，然后分解为比特
    for char in message_with_terminator:
        # 使用bin()将字符转换为二进制字符串（如 '0b1000001'）
        # [2:].zfill(8) 截断 '0b'，并补齐到8位
        binary_char = bin(ord(char))[2:].zfill(8)
        # 将8位二进制字符串转换为8个整数 (0或1)
        bits.extend([int(b) for b in binary_char])
    print(f"消息 '{message}' 转换为比特流，共 {len(bits)} 比特。")
    return bits

def embed_random_lsb(carrier_image_array, message, seed):
    """
    随机 LSB 隐写嵌入功能。
    
    参数:
    - carrier_image_array: 载体图像的 NumPy 数组
    - message: 要隐藏的字符串消息
    - seed: 用于随机数生成器的整数种子，作为提取时的密钥
    
    返回:
    - stego_image_array: 隐写后的图像 NumPy 数组
    """
    print("LSB隐写嵌入开始...")
    # 1. 将图像展平，方便随机选取像素
    flat_image = carrier_image_array.flatten()
    
    # 2. 将秘密消息转换为比特流
    secret_bits = message_to_bits(message)
    total_bits = len(secret_bits)
    
    # 3. 检查容量
    max_capacity = len(flat_image)
    if total_bits > max_capacity:
        raise ValueError(f"错误：消息过长。需要 {total_bits} 比特，但最大容量为 {max_capacity} 比特。")

    # 4. 初始化随机数生成器并生成随机选取索引
    # 使用用户提供的种子确保可重现性
    random.seed(seed)
    
    # 生成从 0 到 max_capacity-1 的所有索引
    all_indices = list(range(max_capacity))
    
    # 随机选择 total_bits 个不重复的索引作为嵌入位置
    # random.sample 是一个高效的随机不重复采样方法
    random_indices = random.sample(all_indices, total_bits)
    
    print(f"载体图像总容量 (比特): {max_capacity}")
    print(f"秘密消息长度 (比特): {total_bits}")
    print(f"使用的随机种子 (密钥): {seed}")

    # 5. 执行嵌入
    # 遍历随机索引和秘密比特
    for i in range(total_bits):
        index = random_indices[i]
        bit = secret_bits[i]
        
        # 获取当前像素值
        original_pixel = flat_image[index]
        
        # 将 LSB 清零（与 11111110 (254) 进行按位与操作）
        # 0xFE 是 254 的十六进制表示
        cleared_pixel = original_pixel & 0xFE
        
        # 将秘密比特位设置到 LSB（与 0 或 1 进行按位或操作）
        new_pixel = cleared_pixel | bit
        
        # 替换像素
        flat_image[index] = new_pixel

    # 6. 将一维数组恢复为原始图像形状
    stego_image_array = flat_image.reshape(carrier_image_array.shape)
    
    # 7. 打印随机选取的位置（用于调试和理解）
    # 创建一个与原图相同大小的画布，将修改过的像素标为红色
    # 实际操作中，为了隐私，不应该在最终图片上直接标出。
    # 这里的可视化主要用于理解“画出随机位置”的要求
    random_indices_3d = np.unravel_index(random_indices, carrier_image_array.shape)
    visualization_map = np.zeros_like(carrier_image_array, dtype=np.uint8)
    visualization_map[random_indices_3d[0], random_indices_3d[1], 0] = 255  # R通道
    visualization_map[random_indices_3d[0], random_indices_3d[1], 1] = 0    # G通道
    visualization_map[random_indices_3d[0], random_indices_3d[1], 2] = 0    # B通道
    
    print("嵌入完成。")
    return stego_image_array, visualization_map

def extract_random_lsb(stego_image_array, seed, max_bits=10000):
    """
    随机 LSB 隐写提取功能。
    
    参数:
    - stego_image_array: 隐写后的图像 NumPy 数组
    - seed: 用于随机数生成器的整数种子（密钥）
    - max_bits: 最大的提取比特数，防止无限循环
    
    返回:
    - secret_message: 提取到的字符串秘密消息
    """
    print("LSB隐写提取开始...")
    # 1. 将图像展平
    flat_image = stego_image_array.flatten()
    max_capacity = len(flat_image)
    
    # 2. 初始化随机数生成器并生成随机选取索引
    random.seed(seed)
    all_indices = list(range(max_capacity))
    
    # 我们不知道嵌入了多少比特，所以假设最大提取 max_bits 比特
    # 实际上，我们可以预先在消息头嵌入长度信息，但为了简单，我们使用 '$$$' 终止符
    
    # 3. 提取比特
    extracted_bits = []
    
    # 由于不知道长度，我们必须不断随机采样，直到达到终止符或容量上限
    # 随机采样一个比 max_bits 略大的数量，确保能取到所有被嵌入的比特
    total_extract_count = min(max_capacity, max_bits)
    random_indices = random.sample(all_indices, total_extract_count)
    
    # 4. 遍历随机索引并提取 LSB
    for index in random_indices:
        # 提取 LSB：与 00000001 (1) 进行按位与操作
        bit = flat_image[index] & 0x01 
        extracted_bits.append(bit)
        
        # 检查是否可以形成一个完整的字符 (8比特)
        if len(extracted_bits) % 8 == 0 and len(extracted_bits) >= 8:
            # 尝试解码最近的 8 比特
            byte_bits = extracted_bits[-8:]
            binary_string = "".join(map(str, byte_bits))
            char_code = int(binary_string, 2)
            
            # 检查终止符
            if chr(char_code) == '$':
                # 检查是否是 '$$$'
                if len(extracted_bits) >= 24: # 至少3个字符
                    # 解码最近的 24 比特 (3个字符)
                    terminator_bits = extracted_bits[-24:]
                    terminator_string = "".join(map(str, terminator_bits))
                    
                    # 检查 '$$$' 标记
                    # ASCII('$') = 36 = 00100100
                    expected_terminator_bin = '00100100' * 3 
                    
                    if terminator_string == expected_terminator_bin:
                        # 找到了终止符，提取结束
                        # 移除终止符的 24 比特
                        extracted_bits = extracted_bits[:-24]
                        break
    
    # 5. 将比特流转换为字符串
    message_bytes = []
    for i in range(0, len(extracted_bits) // 8 * 8, 8):
        byte_bits = extracted_bits[i:i+8]
        binary_string = "".join(map(str, byte_bits))
        char_code = int(binary_string, 2)
        try:
            message_bytes.append(chr(char_code))
        except ValueError:
            # 忽略无法解码的字符
            pass 
            
    secret_message = "".join(message_bytes)
    print(f"提取完成，消息长度: {len(secret_message)} 字符。")
    print("LSB隐写提取完成。")
    return secret_message
  
def plot_histograms(original_image, stego_image, channel=0):
    """
    绘制原始图像和隐写图像指定通道的直方图，用于观察值对效应。
    
    参数:
    - original_image: 原始图像 NumPy 数组
    - stego_image: 隐写图像 NumPy 数组
    - channel: 要分析的颜色通道 (0=R, 1=G, 2=B)
    """
    channel_names = ['Red', 'Green', 'Blue']
    
    # 提取指定通道数据
    original_channel = original_image[:, :, channel].flatten()
    stego_channel = stego_image[:, :, channel].flatten()
    
    plt.figure(figsize=(12, 6))
    
    # 原始图像直方图
    plt.subplot(1, 2, 1)
    # bins=256 确保每个灰度级都有一个条形
    plt.hist(original_channel, bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title(f'Original Image {channel_names[channel]} Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    # 隐写图像直方图
    plt.subplot(1, 2, 2)
    plt.hist(stego_channel, bins=256, range=(0, 256), color='red', alpha=0.7)
    plt.title(f'Stego Image {channel_names[channel]} Channel Histogram (LSB)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    print("已生成直方图。")
    
    
# --- 主程序部分 ---
carrier_image_path = 'dy_picture.jpg'
carrier_image = load_image(carrier_image_path)
secret_message = "my message"
secret_key_seed = 12345  # 随机种子作为密钥

if carrier_image is not None:
    try:
        # 执行嵌入
        stego_image, random_map = embed_random_lsb(carrier_image, secret_message, secret_key_seed)
        
        # 保存隐写图像
        stego_img_pil = Image.fromarray(stego_image)
        stego_img_pil.save("stego_random_lsb.png")
        print("隐写图像已保存为 stego_random_lsb.png")

        # 保存可视化图像（可选）
        random_map_pil = Image.fromarray(random_map)
        random_map_pil.save("random_selection_map.png")
        print("随机选取位置图已保存为 random_selection_map.png")
        
        # --- 提取测试部分 ---
        print("-" * 30)
        extracted_message = extract_random_lsb(stego_image, secret_key_seed)
        print(f"原始消息: '{secret_message}'")
        print(f"提取消息: '{extracted_message}'")

        if extracted_message == secret_message:
            print("LSB隐写成功，提取消息与原始消息一致！")
        else:
            print("警告：提取消息与原始消息不一致。请检查代码逻辑。")

        # --- 直方图对比部分 ---
        print("-" * 30)
        print("正在绘制原始图像与隐写图像的直方图对比...")
        # 分别对R、G、B通道绘制直方图
        for channel in range(3):
            plot_histograms(carrier_image, stego_image, channel=channel)

    except ValueError as e:
        print(e)