import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from wsvd_detect import detect_once

# ================= 配置区域 =================
# 1. 原始未加水印的封面图路径 (用于提取低频系数C)
COVER_PATH = r"G:\information_hiding\exp3\input\girl.jpg" 

# 2. StirMark 输出的攻击图片所在文件夹 (请确保路径无误)
STIRMARK_OUTPUT_DIR = r"G:\information_hiding\exp3\StirMarkBenchmark_4_0_129\Media\Output\Images\Set1"

# 3. 你的含水印文件名的一大半 (用于过滤Set2和Set3的干扰文件)
# 如果你的原图叫 girl_watermarked.bmp，这里就填 "girl_watermarked"
TARGET_FILENAME_PREFIX = "girl_watermarked"

# 4. 原始生成水印时的参数 (必须与嵌入时完全一致)
PARAMS = {
    "alpha": 0.5,
    "seed": 1234,
    "wavelet": "db1",
    "level": 1,
    "ratio": 0.8
}
# ===========================================

def parse_stirmark_filename(filename):
    """
    解析文件名以获取攻击类型和强度
    例如: girl_watermarked_JPEG_70.bmp -> attack="JPEG", strength=70
    """
    name_no_ext = os.path.splitext(filename)[0]
    
    # 简单的解析逻辑：StirMark通常把参数放在最后
    parts = name_no_ext.split('_')
    
    # 尝试获取数值参数
    try:
        val = float(parts[-1])
        attack_type = parts[-2] # 例如 JPEG
    except ValueError:
        return "Unknown", 0
        
    return attack_type, val

def run_batch_test():
    # 获取所有bmp图片
    search_path = os.path.join(STIRMARK_OUTPUT_DIR, "*.bmp")
    all_files = glob.glob(search_path)
    
    # 过滤掉 Set2 和 Set3 的图片，只保留我们的目标图片
    attack_files = [f for f in all_files if TARGET_FILENAME_PREFIX in os.path.basename(f)]
    
    if not attack_files:
        print(f"错误：在 {STIRMARK_OUTPUT_DIR} 没有找到包含 '{TARGET_FILENAME_PREFIX}' 的 .bmp 文件。")
        return None

    # 存储结果: {"JPEG": [(30, 0.98), (50, 0.99)], ...}
    results = {}

    print(f"{'File Name':<45} | {'Attack':<15} | {'Val':<5} | {'Corr (Spatial)':<10}")
    print("-" * 90)

    for file_path in attack_files:
        file_name = os.path.basename(file_path)
        
        # 解析文件名
        attack_type, val = parse_stirmark_filename(file_name)
        
        # 统一攻击名称 (StirMark的文件命名有时和Log不一样)
        # 比如 Log里是 Test_MedianCut，文件名可能是 MedianCut
        attack_type_upper = attack_type.upper()  # 转换为大写进行匹配
        if "JPEG" in attack_type_upper: attack_group = "JPEG"
        elif "NOISE" in attack_type_upper: attack_group = "Noise"
        elif "MEDIAN" in attack_type_upper: attack_group = "Median"
        else: continue # 跳过其他不相关的攻击

        try:
            # 调用检测函数
            # 注意：这里我们只关心空域相关性 corr_spatial (第一个返回值)
            corr_spatial, _ = detect_once(
                cover_path=COVER_PATH,
                test_path=file_path,
                alpha=PARAMS["alpha"],
                seed=PARAMS["seed"],
                wavelet=PARAMS["wavelet"],
                level=PARAMS["level"],
                ratio=PARAMS["ratio"]
            )
            
            # 记录数据
            if attack_group not in results:
                results[attack_group] = []
            results[attack_group].append((val, abs(corr_spatial))) # 取绝对值，防止反相
            
            print(f"{file_name:<45} | {attack_group:<15} | {val:<5.1f} | {corr_spatial:.4f}")
            
        except Exception as e:
            print(f"处理出错 {file_name}: {e}")

    return results

def plot_results(results):
    """绘制鲁棒性曲线"""
    plt.figure(figsize=(18, 5))
    
    # 设置中文字体（防止乱码，根据你的系统环境调整）
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    # 1. JPEG 攻击分析
    plt.subplot(1, 3, 1)
    if "JPEG" in results:
        data = sorted(results["JPEG"], key=lambda x: x[0]) 
        xs, ys = zip(*data)
        plt.plot(xs, ys, 'b-o', linewidth=2)
        plt.title("JPEG 压缩鲁棒性")
        plt.xlabel("压缩质量因子 (Quality Factor)")
        plt.ylabel("检测相关性 (Correlation)")
        plt.grid(True)
        plt.ylim(0, 1.1)

    # 2. 噪声攻击分析
    plt.subplot(1, 3, 2)
    if "Noise" in results:
        data = sorted(results["Noise"], key=lambda x: x[0])
        xs, ys = zip(*data)
        plt.plot(xs, ys, 'r-o', linewidth=2)
        plt.title("加噪攻击鲁棒性")
        plt.xlabel("噪声强度 (Noise Level)")
        plt.ylabel("检测相关性 (Correlation)")
        plt.grid(True)
        plt.ylim(0, 1.1)

    # 3. 中值滤波分析
    plt.subplot(1, 3, 3)
    if "Median" in results:
        data = sorted(results["Median"], key=lambda x: x[0])
        xs, ys = zip(*data)
        plt.plot(xs, ys, 'g-o', linewidth=2)
        plt.title("中值滤波鲁棒性")
        plt.xlabel("滤波窗口大小 (3x3, 5x5...)")
        plt.ylabel("检测相关性 (Correlation)")
        plt.grid(True)
        plt.ylim(0, 1.1)

    plt.tight_layout()
    output_img = r"G:\information_hiding\exp3\output\robustness_report.png"
    plt.savefig(output_img)
    print(f"\n图表已保存至: {output_img}")
    plt.show()

if __name__ == "__main__":
    print("开始 W-SVD 鲁棒性批量测试...")
    data = run_batch_test()
    if data:
        plot_results(data)
    else:
        print("未获取到有效数据，请检查路径设置。")