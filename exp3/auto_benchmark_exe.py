import os
import subprocess
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
from wsvd import wavemarksvd_like, save_rgb_image_float01, load_rgb_image_float01
from wsvd_detect import detect_once

# ==================== 核心配置区域 (请修改这里) ====================

# 1. StirMark 根目录
ROOT_DIR = r"G:\information_hiding\exp3\StirMarkBenchmark_4_0_129"

# 2. 原始封面图片路径 (脚本读取这张图，加水印后覆盖到 StirMark 的输入目录)
COVER_PATH = r"G:\information_hiding\exp3\input\girl.jpg"

# =================================================================

# 自动推导路径
EXE_DIR = os.path.join(ROOT_DIR, "Bin", "Benchmark")
EXE_PATH = os.path.join(EXE_DIR, "StirMark Benchmark.exe")

# 输入输出目录 (StirMark 读取和写入的地方)
# 脚本会将生成的水印图放到这里，StirMark 就会对它进行攻击
INPUT_SET1 = os.path.join(ROOT_DIR, "Media", "Input", "Images", "Set1")
OUTPUT_SET1 = os.path.join(ROOT_DIR, "Media", "Output", "Images", "Set1")

# 结果保存目录
RESULT_DIR = "output/benchmark_results"

# 默认参数 (控制变量时的基准值)
DEFAULT_PARAMS = {
    "alpha": 0.5,
    "seed": 1234,
    "wavelet": "db1",
    "level": 1,
    "ratio": 0.8
}

# 实验设计：定义要对比的5组参数
EXPERIMENTS = {
    "1_Alpha": {
        "param": "alpha",
        "values": [0.1, 0.3, 0.5, 0.8], 
        "label": "强度因子 Alpha"
    },
    "2_Ratio": {
        "param": "ratio",
        "values": [0.6, 0.8, 0.95],
        "label": "嵌入比例 d/n"
    },
    "3_Wavelet": {
        "param": "wavelet",
        "values": ["db1", "haar", "sym2"],
        "label": "小波基函数"
    },
    "4_Level": {
        "param": "level",
        "values": [1, 2, 3],
        "label": "分解层数"
    },
    "5_Seed": {
        "param": "seed",
        "values": [1234, 5678, 9999],
        "label": "随机种子"
    }
}

def clean_output_dir():
    """清理 StirMark 输出目录，防止读取到上一轮实验的旧图片"""
    if os.path.exists(OUTPUT_SET1):
        files = glob.glob(os.path.join(OUTPUT_SET1, "*.*"))
        for f in files:
            try: os.remove(f)
            except: pass

def parse_stirmark_filename(filename):
    """解析文件名，提取攻击类型和参数"""
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    
    try:
        val = float(parts[-1])
        attack_type = parts[-2].upper()
    except:
        return "UNKNOWN", 0
    
    if "JPEG" in attack_type: return "JPEG", val
    if "NOISE" in attack_type: return "Noise", val
    if "MEDIAN" in attack_type: return "Median", val
    
    return "UNKNOWN", 0

def run_single_experiment(exp_key, config, cover_arr):
    param_name = config["param"]
    values = config["values"]
    print(f"\n{'='*20} 开始实验: {config['label']} {'='*20}")
    
    group_data = {}

    for val in values:
        print(f"  >>> 正在测试 {param_name} = {val} ...")
        
        # 1. 准备当前参数
        current_params = DEFAULT_PARAMS.copy()
        current_params[param_name] = val
        
        # 2. 清理 Output 文件夹 (确保拿到的是最新生成的攻击图)
        clean_output_dir()
        
        # 3. 生成水印图 -> 覆盖到 StirMark 的 Input 文件夹
        # 注意：这里我们生成 bmp 格式，确保文件名固定
        target_input_file = os.path.join(INPUT_SET1, "girl_watermarked.bmp")
        
        try:
            # 调用 wsvd 生成水印
            res = wavemarksvd_like(cover_arr, **current_params)
            # 保存到 StirMark 的输入目录
            save_rgb_image_float01(res["watermarkimagergb"], target_input_file)
        except Exception as e:
            print(f"    [错误] 生成水印失败: {e}")
            continue

        # 4. 运行 StirMark EXE
        # 这里不再操作 ini 文件，完全信任你已经配置好的 Profiles/SMBsettings.ini
        try:
            # cwd=EXE_DIR 很重要，模拟你在 Bin/Benchmark 下运行命令
            subprocess.run([EXE_PATH], cwd=EXE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        except Exception as e:
            print(f"    [错误] StirMark 运行异常: {e}")
            continue

        # 5. 遍历 Output 文件夹进行检测
        attacked_files = glob.glob(os.path.join(OUTPUT_SET1, "*.bmp"))
        
        # 简单检查一下是否生成了文件
        if len(attacked_files) < 5:
            print(f"    [警告] Output 文件夹图片数量过少 ({len(attacked_files)})，StirMark 可能没跑通！")
            # 不 continue，尝试检测已有的

        results = {"JPEG": [], "Noise": [], "Median": []}
        
        for fpath in attacked_files:
            fname = os.path.basename(fpath)
            # 过滤干扰文件
            if "girl_watermarked" not in fname:
                continue
                
            atype, strength = parse_stirmark_filename(fname)
            
            if atype in results:
                try:
                    # 调用检测函数 (注意使用当前轮次的 params)
                    corr, _ = detect_once(COVER_PATH, fpath, **current_params)
                    results[atype].append((strength, abs(corr)))
                except:
                    pass
        
        # 排序
        for k in results:
            results[k].sort(key=lambda x: x[0])
            
        group_data[val] = results
        print(f"    检测完成: JPEG({len(results['JPEG'])}) Noise({len(results['Noise'])}) Median({len(results['Median'])})")

    return group_data

def plot_experiment_result(exp_key, config, group_data):
    """画图并保存"""
    if not group_data:
        return

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"W-SVD 参数分析: {config['label']}", fontsize=16)
    
    attacks_map = [("JPEG", "抗 JPEG 压缩", "质量因子 Q"), 
                   ("Noise", "抗加噪攻击", "噪声强度"), 
                   ("Median", "抗中值滤波", "窗口大小")]
    
    for i, (atype, title, xlabel) in enumerate(attacks_map):
        ax = axes[i]
        
        for val, results in group_data.items():
            if atype in results and results[atype]:
                data_points = results[atype]
                xs, ys = zip(*data_points)
                ax.plot(xs, ys, marker='o', label=f"{config['param']}={val}")
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("相关性 (Correlation)")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()

    out_file = os.path.join(RESULT_DIR, f"{exp_key}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[保存] 结果图已保存至: {out_file}")

def main():
    print("=== W-SVD 全自动 StirMark 评测系统 (V2: 依赖 Profiles 配置) ===")
    
    # 0. 准备工作
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    if not os.path.exists(COVER_PATH):
        print(f"[错误] 找不到原图: {COVER_PATH}")
        return
        
    cover_arr = load_rgb_image_float01(COVER_PATH)
    
    # 1. 循环执行所有实验
    for exp_key, config in EXPERIMENTS.items():
        group_data = run_single_experiment(exp_key, config, cover_arr)
        plot_experiment_result(exp_key, config, group_data)
        
    print("\n所有测试结束！请打开 output/benchmark_results 查看图表。")

if __name__ == "__main__":
    main()