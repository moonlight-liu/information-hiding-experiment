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
    x_flat = x.ravel().astype(np.float64)
    y_flat = y.ravel().astype(np.float64)

    num = float(np.dot(x_flat, y_flat))
    denom = float(np.linalg.norm(x_flat) * np.linalg.norm(y_flat))
    if denom == 0.0:
        return 0.0
    return num / denom


def compute_watermark_templates(
    cover_rgb: np.ndarray,
    test_rgb: np.ndarray,
    alpha: float,
    seed: int,
    wavelet: str,
    level: int,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    cover_r = cover_rgb[:, :, 0]
    test_r = test_rgb[:, :, 0]

    embed_result = wavemarksvd_like(
        cover_rgb,
        alpha=alpha,
        seed=seed,
        wavelet=wavelet,
        level=level,
        ratio=ratio,
    )
    waterCA = embed_result["waterCA"]

    coeffs_test = pywt.wavedec2(test_r, wavelet, level=level)
    CA_test = coeffs_test[0]

    coeffs_real = pywt.wavedec2(cover_r, wavelet, level=level)
    realCA = coeffs_real[0]

    realwatermark = waterCA - realCA
    testwatermark = CA_test - realCA
    return realwatermark, testwatermark


def detect_once(
    cover_path: str | Path,
    test_path: str | Path,
    alpha: float,
    seed: int,
    wavelet: str = "db1",
    level: int = 1,
    ratio: float = 0.8,
)-> Tuple[float, float]:
    cover_rgb = load_rgb_image_float01(cover_path)
    test_rgb = load_rgb_image_float01(test_path)

    W, Wp = compute_watermark_templates(
        cover_rgb,
        test_rgb,
        alpha=alpha,
        seed=seed,
        wavelet=wavelet,
        level=level,
        ratio=ratio,
    )
    corr_coef = normalized_correlation(W, Wp)

    W_dct = dctn(W, type=2, norm=None)
    Wp_dct = dctn(Wp, type=2, norm=None)

    h, w = W_dct.shape
    d_block = min(32, max(h, w))
    Wb = W_dct[:d_block, :d_block].copy()
    Wpb = Wp_dct[:d_block, :d_block].copy()
    Wb[0, 0] = 0.0
    Wpb[0, 0] = 0.0

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
    cover_rgb = load_rgb_image_float01(cover_path)
    test_rgb = load_rgb_image_float01(test_path)

    seeds: List[int] = []
    ds_spatial: List[float] = []
    ds_dct: List[float] = []

    print(
        f"开始种子扫描：seed ∈ [{seed_start}, {seed_end}]，"
        f"alpha={alpha}, wavelet={wavelet}, level={level}, ratio={ratio}"
    )

    for s in range(seed_start, seed_end + 1):
        W, Wp = compute_watermark_templates(
            cover_rgb,
            test_rgb,
            alpha=alpha,
            seed=s,
            wavelet=wavelet,
            level=level,
            ratio=ratio,
        )
        d_spatial = normalized_correlation(W, Wp)

        W_dct = dctn(W, type=2, norm=None)
        Wp_dct = dctn(Wp, type=2, norm=None)

        h, w = W_dct.shape
        d_block = min(32, max(h, w))
        Wb = W_dct[:d_block, :d_block].copy()
        Wpb = Wp_dct[:d_block, :d_block].copy()
        Wb[0, 0] = 0.0
        Wpb[0, 0] = 0.0

        d_dct = normalized_correlation(Wb, Wpb)

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

    p_detect = subparsers.add_parser("detect")
    p_detect.add_argument("--cover", required=True)
    p_detect.add_argument("--test", required=True)
    p_detect.add_argument("--alpha", type=float, default=0.1)
    p_detect.add_argument("--seed", type=int, default=1)
    p_detect.add_argument("--wavelet", type=str, default="db1")
    p_detect.add_argument("--level", type=int, default=1)
    p_detect.add_argument("--ratio", type=float, default=0.8)

    p_scan = subparsers.add_parser("scan")
    p_scan.add_argument("--cover", required=True)
    p_scan.add_argument("--test", required=True)
    p_scan.add_argument("--alpha", type=float, default=0.1)
    p_scan.add_argument("--wavelet", type=str, default="db1")
    p_scan.add_argument("--level", type=int, default=1)
    p_scan.add_argument("--ratio", type=float, default=0.8)
    p_scan.add_argument("--seed-start", type=int, default=1)
    p_scan.add_argument("--seed-end", type=int, default=20)
    p_scan.add_argument("--out-plot", type=str, default=None)

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
