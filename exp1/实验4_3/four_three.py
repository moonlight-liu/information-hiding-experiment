import os
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# 尝试设置中文字体，优先使用 Windows 常见字体；若未找到则打印警告
_chinese_fonts = ['Microsoft YaHei', 'SimHei', 'STHeiti', 'Arial Unicode MS']
for _f in _chinese_fonts:
    if any(_f in fname.name for fname in font_manager.fontManager.ttflist):
        rcParams['font.sans-serif'] = [_f]
        break
else:
    print("警告：未找到常用中文字体，图片中的中文可能仍出现乱码。可安装 'SimHei' 或 'Microsoft YaHei'。")

# 允许负号正常显示
rcParams['axes.unicode_minus'] = False

def load_image_gray(image_path):
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return None
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img).astype(np.uint8)
    print(f"图像 {image_path} 加载成功，形状: {arr.shape}")
    return arr

def save_image(img_array, path):
    img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)
    # 不要过多输出，保持简短
    print(f"已保存: {path}")

def dct2(image_f):
    # image_f: float32 2D
    return cv.dct(image_f)

def idct2(coeff_f):
    return cv.idct(coeff_f)

def vis_dct_magnitude(coeffs):
    mag = np.log(np.abs(coeffs) + 1.0)
    # 归一化到 0-255
    mag = mag - mag.min()
    if mag.max() > 0:
        mag = mag / mag.max() * 255.0
    return mag.astype(np.uint8)

def apply_freq_cutoff(coeffs, cutoff):
    """
    全图频域系数矩阵 coeffs（低频在左上）。
    保留 u+v < cutoff 的系数，其余置零。
    """
    H, W = coeffs.shape
    u = np.arange(H)[:, None]
    v = np.arange(W)[None, :]
    mask = (u + v) < cutoff
    return coeffs * mask.astype(coeffs.dtype), mask

def psnr_uint8(orig, recon):
    orig_f = orig.astype(np.float32)
    recon_f = recon.astype(np.float32)
    mse = np.mean((orig_f - recon_f) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10.0 * np.log10((PIXEL_MAX ** 2) / mse)

if __name__ == "__main__":
    carrier_image_path = "dy_picture.jpg"
    out_dir = "hf_results"
    os.makedirs(out_dir, exist_ok=True)

    img = load_image_gray(carrier_image_path)
    if img is None:
        raise SystemExit(1)

    # 全图 2D DCT
    img_f = img.astype(np.float32)
    coeffs = dct2(img_f)       # NxM 频域矩阵
    coeffs_vis = vis_dct_magnitude(coeffs)
    save_image(coeffs_vis, os.path.join(out_dir, "dct_full_magnitude.png"))

    # 示例：截断高频后（用于图3展示，选择一个中等 cutoff）
    cutoff_example = 20
    coeffs_trunc_ex, _ = apply_freq_cutoff(coeffs, cutoff_example)
    coeffs_trunc_vis = vis_dct_magnitude(coeffs_trunc_ex)
    recon_ex = idct2(coeffs_trunc_ex)
    recon_ex = np.clip(recon_ex, 0, 255)
    save_image(coeffs_trunc_vis, os.path.join(out_dir, f"dct_trunc_uplusv_lt_{cutoff_example}.png"))
    save_image(recon_ex, os.path.join(out_dir, f"recon_uplusv_lt_{cutoff_example}.png"))

    # 生成 cutoff 从 10 到 90（步长 10）的 9 张重构图（图4），并计算 PSNR 与能量保留
    cutoffs = list(range(10, 100, 10))  # 10,20,...,90 共9个
    recon_list = []
    psnr_list = []
    energy_ratio = []
    total_energy = np.sum(coeffs ** 2)

    for c in cutoffs:
        coeffs_c, mask = apply_freq_cutoff(coeffs, c)
        recon_c = idct2(coeffs_c)
        recon_c = np.clip(recon_c, 0, 255)
        recon_list.append(recon_c.astype(np.uint8))
        psnr_val = psnr_uint8(img, recon_c.astype(np.uint8))
        psnr_list.append(psnr_val)
        retained = np.sum(coeffs_c ** 2)
        energy_ratio.append(retained / total_energy if total_energy > 0 else 0.0)
        save_image(recon_c, os.path.join(out_dir, f"recon_cutoff_{c}.png"))

    # 保存并绘制图4：3x3 网格重构图
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3))
    for idx, recon in enumerate(recon_list):
        r = idx // cols
        cc = idx % cols
        axes[r, cc].imshow(recon, cmap='gray', vmin=0, vmax=255)
        axes[r, cc].set_title(f"cutoff={cutoffs[idx]}")
        axes[r, cc].axis('off')
    plt.suptitle("cutoff 10..90 的 DCT 高频截断重构（3x3）", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_grid = os.path.join(out_dir, "recon_grid_3x3.png")
    plt.savefig(out_grid, dpi=200)
    print(f"汇总图已保存: {out_grid}")
    plt.close(fig)

    # 绘制 PSNR 与能量保留趋势图
    fig2, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(cutoffs, psnr_list, marker='o', color='C0', label='PSNR (dB)')
    ax1.set_xlabel('cutoff (u+v threshold)')
    ax1.set_ylabel('PSNR (dB)', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax2 = ax1.twinx()
    ax2.plot(cutoffs, energy_ratio, marker='s', color='C1', label='Energy ratio')
    ax2.set_ylabel(' retained energy ratio', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax1.set_title('PSNR 与 能量保留 随 cutoff 变化')
    fig2.tight_layout()
    out_trend = os.path.join(out_dir, "psnr_energy_trend.png")
    fig2.savefig(out_trend, dpi=200)
    print(f"趋势图已保存: {out_trend}")
    plt.close(fig2)

    # 输出简要结果（PSNR 列表）
    for c, p, e in zip(cutoffs, psnr_list, energy_ratio):
        print(f"cutoff={c:2d}  PSNR={p:.2f} dB   energy_ratio={e:.4f}")

    print("完成。结果保存在目录：", out_dir)