from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn
import pywt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

from wsvd import load_rgb_image_float01, wavemarksvd_like


def normalized_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算两个矩阵的归一化相关系数
    
    数学公式: d = (W · W'^T) / (||W|| · ||W'||)
    
    Args:
        x: 第一个矩阵（如原始水印模板W）
        y: 第二个矩阵（如待测水印模板W'）
    
    Returns:
        归一化相关系数，范围[-1, 1]，值越接近1表示相似度越高
    """
    x_flat = x.ravel().astype(np.float64)  # 展平为一维数组
    y_flat = y.ravel().astype(np.float64)

    num = float(np.dot(x_flat, y_flat))    # 计算内积 W·W'^T
    denom = float(np.linalg.norm(x_flat) * np.linalg.norm(y_flat))  # 计算模长乘积
    if denom == 0.0:
        return 0.0
    return num / denom  # 归一化相关系数


def compute_watermark_templates(
    cover_rgb: np.ndarray,
    test_rgb: np.ndarray,
    alpha: float,
    seed: int,
    wavelet: str,
    level: int,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算原始水印模板W和待测水印模板W'
    
    检测原理：
    1. 从原始封面图像提取低频系数C (realCA)
    2. 重走嵌入步骤得到理论含水印系数A (waterCA)
    3. 从待测图像提取低频系数B (CA_test)
    4. 原始水印模板: W = A - C
    5. 待测水印模板: W' = B - C
    
    Args:
        cover_rgb: 原始封面RGB图像
        test_rgb: 待检测RGB图像
        alpha, seed, wavelet, level, ratio: 水印嵌入参数
    
    Returns:
        (realwatermark, testwatermark): 原始水印模板W和待测水印模板W'
    """
    # === 步骤1: 提取红色通道 ===
    cover_r = cover_rgb[:, :, 0]  # 原始封面图像红色通道
    test_r = test_rgb[:, :, 0]    # 待测图像红色通道

    # === 步骤2: 重走嵌入过程获取理论含水印系数A ===
    # 使用相同参数对原始图像进行水印嵌入，获取理论上的含水印低频系数
    embed_result = wavemarksvd_like(
        cover_rgb,
        alpha=alpha,
        seed=seed,
        wavelet=wavelet,
        level=level,
        ratio=ratio,
    )
    waterCA = embed_result["waterCA"]  # 理论含水印低频系数A

    # === 步骤3: 提取待测图像的低频系数B ===
    coeffs_test = pywt.wavedec2(test_r, wavelet, level=level)
    CA_test = coeffs_test[0]  # 待测图像低频系数B

    # === 步骤4: 提取原始封面图像的低频系数C ===
    coeffs_real = pywt.wavedec2(cover_r, wavelet, level=level)
    realCA = coeffs_real[0]  # 原始封面低频系数C

    # === 步骤5: 计算水印模板 ===
    realwatermark = waterCA - realCA    # 原始水印模板: W = A - C
    testwatermark = CA_test - realCA    # 待测水印模板: W' = B - C
    return realwatermark, testwatermark


def detect_once(
    cover_path: str | Path,
    test_path: str | Path,
    alpha: float,
    seed: int,
    wavelet: str = "db1",
    level: int = 1,
    ratio: float = 0.8,  # 与嵌入代码保持一致
)-> Tuple[float, float]:
    """
    对单个种子进行水印检测
    
    检测流程：
    1. 计算原始水印模板W和待测水印模板W'
    2. 计算空域相关性d = corr(W, W')
    3. 计算DCT域相关性d^ = corr(DCT(W), DCT(W'))
    
    Args:
        cover_path: 原始封面图像路径
        test_path: 待检测图像路径
        alpha: 嵌入强度（需与嵌入时一致）
        seed: 随机种子（需与嵌入时一致）
        wavelet: 小波类型
        level: 小波分解层数
        ratio: 替换比例
    
    Returns:
        (corr_coef, corr_DCTcoef): 空域相关性和DCT域相关性
    """
    # === 步骤1: 加载图像 ===
    cover_rgb = load_rgb_image_float01(cover_path)  # 原始封面图像
    test_rgb = load_rgb_image_float01(test_path)    # 待检测图像

    # === 步骤2: 计算水印模板 ===
    W, Wp = compute_watermark_templates(
        cover_rgb,
        test_rgb,
        alpha=alpha,
        seed=seed,
        wavelet=wavelet,
        level=level,
        ratio=ratio,
    )
    
    # === 步骤3: 计算空域相关性d ===
    # 直接计算原始水印模板W和待测水印模板W'的相关性
    corr_coef = normalized_correlation(W, Wp)

    # === 步骤4: 计算DCT域相关性d^ ===
    # 对水印模板进行二维DCT变换
    W_dct = dctn(W, type=2, norm=None)    # 原始水印模板的DCT变换
    Wp_dct = dctn(Wp, type=2, norm=None)  # 待测水印模板的DCT变换

    # 选取DCT变换后的左上角子块（通常32x32或更小）
    h, w = W_dct.shape
    d_block = min(32, max(h, w))  # 子块大小，最大32
    Wb = W_dct[:d_block, :d_block].copy()   # 原始水印DCT子块
    Wpb = Wp_dct[:d_block, :d_block].copy() # 待测水印DCT子块
    
    # 将DC分量（0,0位置）置零，只考虑AC分量
    Wb[0, 0] = 0.0
    Wpb[0, 0] = 0.0

    # 计算DCT域相关性
    corr_DCTcoef = normalized_correlation(Wb, Wpb)

    print(f"检测 seed={seed} 的结果：")
    print(f"  小波系数相关性 corr_coef     = {corr_coef:.6f}")
    print(f"  DCT 后小波系数相关性 corr_DCT = {corr_DCTcoef:.6f}")
    return corr_coef, corr_DCTcoef


def scan_seeds(
    cover_path: str | Path,
    test_path: str | Path,
    alpha: float,
    wavelet: str,
    level: int,
    ratio: float,
    seed_start: int,
    seed_end: int,
    out_plot: str | Path | None = None,
) -> Tuple[List[int], np.ndarray]:
    """
    扫描多个种子进行水印检测，生成"种子-相关性值"SC图
    
    目的：通过扫描不同的seed值来：
    1. 确定最佳检测阈值τ
    2. 观察相关性值的分布特征
    3. 区分有水印和无水印的图像
    
    检测策略：
    - 如果待测图像确实含有对应seed的水印，该seed的相关性会明显较高
    - 其他seed的相关性应该较低（接近随机水平）
    - 通过统计分析确定合适的阈值τ
    
    Args:
        cover_path: 原始封面图像路径
        test_path: 待检测图像路径
        alpha: 嵌入强度
        wavelet: 小波类型
        level: 小波分解层数
        ratio: 替换比例
        seed_start: 起始种子值
        seed_end: 结束种子值
        out_plot: 输出SC图的路径
    
    Returns:
        (seeds, ds_spatial_arr): 种子列表和对应的空域相关性数组
    """
    cover_rgb = load_rgb_image_float01(cover_path)
    test_rgb = load_rgb_image_float01(test_path)

    seeds: List[int] = []
    ds_spatial: List[float] = []
    ds_dct: List[float] = []

    print(
        f"开始种子扫描：seed ∈ [{seed_start}, {seed_end}]，"
        f"alpha={alpha}, wavelet={wavelet}, level={level}, ratio={ratio}"
    )

    # === 种子扫描主循环 ===
    for s in range(seed_start, seed_end + 1):
        # 计算当前种子s对应的水印模板
        W, Wp = compute_watermark_templates(
            cover_rgb,
            test_rgb,
            alpha=alpha,
            seed=s,  # 当前扫描的种子
            wavelet=wavelet,
            level=level,
            ratio=ratio,
        )
        
        # 计算空域相关性
        d_spatial = normalized_correlation(W, Wp)

        # 计算DCT域相关性
        W_dct = dctn(W, type=2, norm=None)    # 原始水印DCT变换
        Wp_dct = dctn(Wp, type=2, norm=None)  # 待测水印DCT变换

        h, w = W_dct.shape
        d_block = min(32, max(h, w))          # 选择子块大小
        Wb = W_dct[:d_block, :d_block].copy()   # 原始水印DCT子块
        Wpb = Wp_dct[:d_block, :d_block].copy() # 待测水印DCT子块
        
        # 置零DC分量，只保留AC分量
        Wb[0, 0] = 0.0
        Wpb[0, 0] = 0.0

        d_dct = normalized_correlation(Wb, Wpb)  # DCT域相关性

        # 记录结果
        seeds.append(s)
        ds_spatial.append(d_spatial)
        ds_dct.append(d_dct)
        print(f"  seed={s:3d} -> d={d_spatial:.4f}, d^={d_dct:.4f}")

    ds_spatial_arr = np.array(ds_spatial, dtype=np.float64)
    ds_dct_arr = np.array(ds_dct, dtype=np.float64)

    abs_spatial = np.abs(ds_spatial_arr)
    abs_dct = np.abs(ds_dct_arr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(seeds, abs_spatial, marker="o")
    ax1.set_ylabel("相关性 d")
    ax1.set_title("小波系数相关性分析（空域）")
    ax1.grid(True, alpha=0.3)

    ax2.plot(seeds, abs_dct, marker="o")
    ax2.set_xlabel("种子")
    ax2.set_ylabel("相关性 d^")
    ax2.set_title("DCT 变换后小波系数相关性分析")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("“种子-相关性值”SC 图（空域与 DCT 域）")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out_plot is not None:
        out_plot = Path(out_plot)
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, dpi=300, bbox_inches="tight")
        print(f"SC 曲线已保存到: {out_plot}")
    else:
        plt.show()

    plt.close(fig)
    return seeds, ds_spatial_arr

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # === 单次检测模式 ===
    p_detect = subparsers.add_parser("detect", help="检测指定种子的水印")
    p_detect.add_argument("--cover", default="input/girl.jpg", help="原始封面图像路径")
    p_detect.add_argument("--test", default="output/girl_watermarked.jpg", help="待检测图像路径")
    p_detect.add_argument("--alpha", type=float, default=0.5, help="嵌入强度（需与嵌入时一致）")
    p_detect.add_argument("--seed", type=int, default=1234, help="随机种子（需与嵌入时一致）")
    p_detect.add_argument("--wavelet", type=str, default="db1", help="小波类型")
    p_detect.add_argument("--level", type=int, default=1, help="小波分解层数")
    p_detect.add_argument("--ratio", type=float, default=0.8, help="替换比例")

    # === 种子扫描模式 ===
    p_scan = subparsers.add_parser("scan", help="扫描多个种子生成SC图")
    p_scan.add_argument("--cover", default="input/girl.jpg", help="原始封面图像路径")
    p_scan.add_argument("--test", default="output/girl_watermarked.jpg", help="待检测图像路径")
    p_scan.add_argument("--alpha", type=float, default=0.5, help="嵌入强度")
    p_scan.add_argument("--wavelet", type=str, default="db1", help="小波类型")
    p_scan.add_argument("--level", type=int, default=1, help="小波分解层数")
    p_scan.add_argument("--ratio", type=float, default=0.8, help="替换比例")
    p_scan.add_argument("--seed-start", type=int, default=1230, help="起始种子值")
    p_scan.add_argument("--seed-end", type=int, default=1240, help="结束种子值")
    p_scan.add_argument("--out-plot", type=str, default="output/sc_plot.png", help="SC图保存路径")

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "detect":
        detect_once(
            cover_path=args.cover,
            test_path=args.test,
            alpha=args.alpha,
            seed=args.seed,
            wavelet=args.wavelet,
            level=args.level,
            ratio=args.ratio,
        )
    elif args.mode == "scan":
        scan_seeds(
            cover_path=args.cover,
            test_path=args.test,
            alpha=args.alpha,
            wavelet=args.wavelet,
            level=args.level,
            ratio=args.ratio,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
            out_plot=args.out_plot,
        )
    else:
        parser.error(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
