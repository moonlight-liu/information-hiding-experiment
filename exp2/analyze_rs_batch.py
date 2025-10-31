import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils import load_gray_image
from rs_analyze import rs_statistics


"""
批量运行 RS 分析并绘制结果的辅助脚本。

主要功能：
- 对文件夹内一系列以不同嵌入比率命名的图像运行 RS 估计（调用 `rs_statistics`）。
- 生成若干图表：估计值 vs 真值、绝对误差图以及两种方法（LSBR/LSBM）对比图。

本文件仅做批量分析与绘图，不修改图像数据。
"""

def analyze_folder(folder: Path, prefix: str, ratios, group_size, mask):
    """
    对指定文件夹中按命名规则（prefix_ratio.png）排列的图像逐一运行 RS 估计。

    参数：
    - folder: Path 对象，目标文件夹路径。
    - prefix: 文件名前缀（如 'lsbr_stego'），文件名按 f'{prefix}_{ratio:.1f}.png' 查找。
    - ratios: 可迭代的比率列表（用于构造文件名并作为真值）。
    - group_size, mask: 传递给 rs_statistics 的参数。

    返回：按 ratios 顺序的估计值列表（若文件缺失则填 nan）。
    """
    est = []
    for r in ratios:
        f = folder / f'{prefix}_{r:.1f}.png'
        if not f.exists():
            print(f'Warning skip: {f}')
            est.append(float('nan')); continue
        img = load_gray_image(str(f))
        p_hat = rs_statistics(img, group_size, mask)
        est.append(float(p_hat))
        print(f'Processed {f.name}: p_hat={p_hat:.4f}')
    return est

def plot_ratio_vs_est(ratios, est, save_path, title):
    """
    绘制估计嵌入率 p_hat 与真实嵌入比率 ratio 的对比图并保存。

    参数：ratios（x 轴真值）、est（y 轴估计值）、save_path（输出文件路径）、title（图标题）。
    输出：把图保存到 save_path，并打印保存信息。
    """
    plt.figure(figsize=(8,5))
    plt.plot(ratios, est, 'o-', label='Estimated p')
    plt.plot(ratios, ratios, '--', label='Ground truth')
    plt.xlabel('Embed ratio'); plt.ylabel('RS estimated p')
    plt.title(title); plt.ylim(0, 1.01); plt.grid(True, ls='--', alpha=0.4)
    plt.legend(); os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f'Saved: {save_path}')

def plot_abs_error(ratios, est, save_path, title):
    """
    绘制估计误差的绝对值图（|p_hat - ratio|）并保存。

    输入参数同上，返回值无，函数会保存并打印输出路径。
    """
    import numpy as np
    abs_err = np.abs(np.array(est, dtype=float) - np.array(ratios, dtype=float))
    plt.figure(figsize=(8,5))
    plt.plot(ratios, abs_err, 'o-', color='crimson', label='|p_hat - ratio|')
    plt.xlabel('Embed ratio'); plt.ylabel('Absolute error')
    plt.title(title); plt.ylim(0, 1.01); plt.grid(True, ls='--', alpha=0.4)
    plt.legend(); os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f'Saved: {save_path}')

def plot_both_methods(ratios, lsbr_est, lsbm_est, save_path):
    """
    绘制并对比两种方法（LSBR 与 LSBM）在不同嵌入比率下的 RS 估计结果。

    输入：ratios、两组估计值、保存路径。输出图包含真值参考线以便直观比较。
    """
    plt.figure(figsize=(8,5))
    plt.plot(ratios, ratios, '--', color='gray', label='Ground truth')
    plt.plot(ratios, lsbr_est, 'o-', label='LSBR (RS est)')
    plt.plot(ratios, lsbm_est, 's-', label='LSBM (RS est)')
    plt.xlabel('Embed ratio'); plt.ylabel('RS estimated p')
    plt.title('RS Estimated p vs Ratio (LSBR vs LSBM)')
    plt.ylim(0, 1.01); plt.grid(True, ls='--', alpha=0.4)
    plt.legend(); os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f'Saved: {save_path}')

def sanitize_estimates(values):
    """
    清洗估计值列表：将无穷或非法值转换为 NaN，并裁剪到 [0,1] 区间。

    返回清洗后的列表（与输入长度相同，类型为 float 列表，可包含 NaN）。
    """
    arr = np.array(values, dtype=float)
    # 将非有限值设为 NaN，再裁剪到 [0,1]
    arr[~np.isfinite(arr)] = np.nan
    arr = np.clip(arr, 0.0, 1.0)
    return arr.tolist()

def main():
    """
    批量分析主流程：
    - 在仓库的 `LSBR` 与 `LSBM` 子文件夹中寻找命名为 `{prefix}_{ratio:.1f}.png` 的文件；
    - 对每个比率调用 RS 统计并收集估计值；
    - 绘制并保存估计值 vs 真值图、绝对误差图，以及两种方法的对比图。
    """
    base = Path(__file__).resolve().parent
    ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    group_size, mask = 4, [1,0,0,1]

    lsbr_est = analyze_folder(base/'LSBR', 'lsbr_stego', ratios, group_size, mask)
    lsbr_est = sanitize_estimates(lsbr_est)
    plot_ratio_vs_est(ratios, lsbr_est, str(base/'res'/'rs_est_vs_ratio_lsbr.png'),
                      'RS Estimated p vs Ratio (LSBR)')
    plot_abs_error(ratios, lsbr_est, str(base/'res'/'rs_abs_err_lsbr.png'),
                   'RS Absolute Error vs Ratio (LSBR)')

    lsbm_est = analyze_folder(base/'LSBM', 'lsbm_stego', ratios, group_size, mask)
    lsbm_est = sanitize_estimates(lsbm_est)
    plot_ratio_vs_est(ratios, lsbm_est, str(base/'res'/'rs_est_vs_ratio_lsbm.png'),
                      'RS Estimated p vs Ratio (LSBM)')
    plot_abs_error(ratios, lsbm_est, str(base/'res'/'rs_abs_err_lsbm.png'),
                   'RS Absolute Error vs Ratio (LSBM)')

    # 合并对比图
    plot_both_methods(ratios, lsbr_est, lsbm_est, str(base/'res'/'rs_est_vs_ratio_both.png'))

if __name__ == '__main__':
    main()