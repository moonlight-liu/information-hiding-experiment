from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import pywt


def load_rgb_image_float01(path: str | Path) -> np.ndarray:
    """
    加载RGB图像并归一化到[0,1]范围
    
    Args:
        path: 图像文件路径
    
    Returns:
        归一化的RGB图像数组，形状为(H, W, 3)，数值范围[0,1]
    """
    path = Path(path)
    img = Image.open(path).convert("RGB")  # 确保是RGB格式
    arr = np.asarray(img, dtype=np.float64) / 255.0  # 归一化到[0,1]
    return arr


def save_rgb_image_float01(arr: np.ndarray, path: str | Path) -> None:
    """
    保存归一化的RGB图像数组到文件
    
    Args:
        arr: 归一化的RGB图像数组，数值范围[0,1]
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建目录
    arr_clipped = np.clip(arr, 0.0, 1.0)  # 裁剪到有效范围[0,1]
    img = Image.fromarray((arr_clipped * 255.0).round().astype(np.uint8), mode="RGB")  # 转换为uint8
    img.save(path)


def pad_to_square(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    将图像填充为方阵，便于后续SVD分解
    
    数学原理：SVD分解通常在方阵上进行效果更好，因此需要将非方形的低频系数矩阵
    通过零填充扩展为方阵。
    
    Args:
        image: 输入的2D图像数组
    
    Returns:
        padded: 填充后的方形数组
        (row, col): 原始图像尺寸，用于后续裁剪
    """
    row, col = image.shape
    standard = max(row, col)  # 取较大的维度作为方阵边长
    padded = np.zeros((standard, standard), dtype=image.dtype)
    
    # 将原图像放置在左上角
    if row <= col:
        padded[:row, :] = image
    else:
        padded[:, :col] = image
    return padded, (row, col)


def wavemarksvd_like(
    data_rgb: np.ndarray,
    alpha: float,
    seed: int,
    wavelet: str,
    level: int,
    ratio: float,
) -> Dict[str, np.ndarray]:
    """
    W-SVD水印嵌入主函数 - 基于小波变换和奇异值分解的数字水印技术
    
    数学原理：
    1. 小波分解：img -> LL(低频), LH,HL,HH(高频)
    2. SVD分解：CA = U·Σ·V^T (其中CA为低频系数LL)
    3. 正交矩阵替换：用随机正交矩阵替换U,V的部分列
    4. 水印生成：watermark = U_new·diag(σ_tilda)·V_new^T
    5. 嵌入：CA_watermarked = CA + α·watermark
    
    Args:
        data_rgb: 输入RGB图像，形状(H,W,3)，数值范围[0,1]
        alpha: 水印嵌入强度因子，控制水印可见性与鲁棒性平衡
        seed: 随机种子，确保水印的可重复性
        wavelet: 小波类型，如'db1','haar'等
        level: 小波分解层数，通常为1-3
        ratio: 替换比例，决定替换正交矩阵的列数比例
    
    Returns:
        包含水印图像和相关参数的字典
    """
    # === 步骤1: 载体图像预处理 ===
    data = np.asarray(data_rgb, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("wavemarksvd_like 期望输入为 RGB 图像（H×W×3）")

    # 选择红色通道作为载体（也可选择其他通道或灰度化）
    datared = data[:, :, 0]  # 提取红色通道
    row, col = datared.shape

    # === 步骤2: 小波分解和低频提取 ===
    # 对原始图像进行小波分解，获取真实的低频系数尺寸（用于最终裁剪）
    coeffs_real = pywt.wavedec2(datared, wavelet, level=level)
    CA_real = coeffs_real[0]  # 低频系数LL
    real_CA_shape = CA_real.shape  # 记录真实低频系数形状

    # 为确保SVD计算效果，将图像填充为方阵
    new, orig_shape = pad_to_square(datared)

    # 对填充后的方形图像进行小波分解
    coeffs = pywt.wavedec2(new, wavelet, level=level)
    CA = coeffs[0]  # 提取低频系数（LL子带）
    d1, d2 = CA.shape
    if d1 != d2:
        raise RuntimeError(f"最深层 CA 非方阵：shape={CA.shape}")
    d = d1  # 方阵维度

    # === 步骤3: 低频系数归一化 ===
    # 将低频系数归一化到[0,1]范围，便于后续水印嵌入和防止溢出
    CAmin = float(CA.min())
    CAmax = float(CA.max())
    eps = 1e-12  # 防止除零
    CA_norm = (CA - CAmin) / (CAmax - CAmin + eps)  # 归一化: [CAmin,CAmax] -> [0,1]

    # === 步骤4: SVD分解 ===
    # 对归一化的低频系数进行奇异值分解: CA_norm = U·Σ·V^T
    U, sigma, Vt = np.linalg.svd(CA_norm, full_matrices=True)
    V = Vt.T  # 转置得到V矩阵

    # === 步骤5: 计算替换列数 ===
    # 根据ratio参数确定要替换的正交矩阵列数
    np_cap = int(round(d * ratio))  # 替换列数 = 矩阵维度 × 替换比例
    np_cap = max(1, min(np_cap, d))  # 确保在合理范围[1, d]内

    # === 步骤6: 生成随机正交矩阵 ===
    # 使用固定种子生成伪随机数，确保水印的可重复性
    rng = np.random.RandomState(seed)
    
    # 生成随机矩阵并通过QR分解得到正交矩阵
    # 数学原理: 任意矩阵A可通过QR分解得到A=Q·R，其中Q为正交矩阵
    M_V = rng.rand(d, np_cap) - 0.5  # 生成[-0.5, 0.5]范围的随机矩阵
    Q_V, _ = np.linalg.qr(M_V)       # QR分解得到正交矩阵Q_V
    
    M_U = rng.rand(d, np_cap) - 0.5  # 同样生成U对应的随机矩阵
    Q_U, _ = np.linalg.qr(M_U)       # QR分解得到正交矩阵Q_U

    # === 步骤7: 正交矩阵列替换 ===
    # 保存原始U,V矩阵副本，用于后续相关性计算
    U2 = U.copy()
    V2 = V.copy()

    # 用随机生成的正交矩阵列替换原始SVD矩阵的后np_cap列
    # 数学原理: 替换后的矩阵仍保持正交性，但改变了原始的奇异向量结构
    U[:, d - np_cap : d] = Q_U[:, :np_cap]  # 替换U矩阵的后np_cap列
    V[:, d - np_cap : d] = Q_V[:, :np_cap]  # 替换V矩阵的后np_cap列

    # === 步骤8: 生成水印奇异值 ===
    # 生成新的随机奇异值序列
    sigma_rand = rng.rand(d)                    # 生成随机数序列
    sigma_sorted = np.sort(sigma_rand)[::-1]    # 降序排列（符合奇异值特性）
    sigma_tilda = alpha * sigma_sorted          # 乘以嵌入强度因子α

    # === 步骤9: 生成水印模板 ===
    # 使用修改后的正交矩阵和新奇异值重构水印模板
    # 数学公式: watermark = U_new · diag(σ_tilda) · V_new^T
    watermark = U @ np.diag(sigma_tilda) @ V.T

    # === 步骤10: 计算正交矩阵相关性 ===
    # 相关性用于评估水印嵌入对原始矩阵结构的影响程度
    def corr2(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个矩阵的皮尔逊相关系数"""
        a_flat = a.ravel().astype(np.float64)
        b_flat = b.ravel().astype(np.float64)
        if a_flat.size == 0 or b_flat.size == 0:
            return 0.0
        a_mean = a_flat.mean()
        b_mean = b_flat.mean()
        a_z = a_flat - a_mean  # 中心化
        b_z = b_flat - b_mean
        denom = float(np.linalg.norm(a_z) * np.linalg.norm(b_z))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_z, b_z) / denom)  # 皮尔逊相关系数

    # 计算修改前后U,V矩阵的相关性
    correlationU = corr2(U, U2)  # U矩阵相关性
    correlationV = corr2(V, V2)  # V矩阵相关性

    # === 步骤11: 水印嵌入 ===
    # 将水印模板叠加到归一化的低频系数上
    CA_tilda_norm = CA_norm + watermark  # 加性水印嵌入
    CA_tilda_norm = np.clip(CA_tilda_norm, 0.0, 1.0)  # 裁剪到[0,1]范围，防止溢出

    # === 步骤12: 反归一化 ===
    # 将嵌入水印后的归一化系数恢复到原始灰度范围
    CA_tilda_real = (CAmax - CAmin) * CA_tilda_norm + CAmin  # [0,1] -> [CAmin,CAmax]

    # 裁剪到真实的低频系数尺寸（去除之前的填充）
    h_ll, w_ll = real_CA_shape
    waterCA = CA_tilda_real[:h_ll, :w_ll]  # 裁剪为原始低频系数尺寸

    # === 步骤13: 生成纯水印图像（可选，用于可视化） ===
    # 使用水印模板和原始高频系数重构纯水印图像
    coeffs_wm = [watermark] + list(coeffs[1:])  # 用水印替换低频，保持高频不变
    watermark2_full = pywt.waverec2(coeffs_wm, wavelet)  # 小波逆变换
    
    # 裁剪到原始图像尺寸
    r0, c0 = orig_shape
    if r0 <= c0:
        watermark2 = watermark2_full[:r0, :]
    else:
        watermark2 = watermark2_full[:, :c0]

    # === 步骤14: 重构含水印图像 ===
    # 使用嵌入水印后的低频系数和原始高频系数重构图像
    coeffs_new = [CA_tilda_real] + list(coeffs[1:])  # 替换低频系数
    watermarked_padded = pywt.waverec2(coeffs_new, wavelet)  # 小波逆变换重构图像
    
    # 裁剪到原始图像尺寸
    if r0 <= c0:
        watermarkimage = watermarked_padded[:r0, :]
    else:
        watermarkimage = watermarked_padded[:, :c0]

    # === 步骤15: 生成RGB水印图像 ===
    # 将含水印的单通道图像合成为RGB图像
    watermarkimagergb = data.copy()  # 复制原始RGB数据
    watermarkimagergb[:, :, 0] = watermarkimage  # 替换红色通道

    # === 步骤16: 整理输出信息 ===
    # 收集所有中间结果和参数，便于分析和调试
    info: Dict[str, np.ndarray] = {
        "CA": CA,                                    # 原始低频系数
        "CA_norm": CA_norm,                          # 归一化低频系数
        "CA_tilda_norm": CA_tilda_norm,             # 嵌入水印后的归一化系数
        "CA_tilda_real": CA_tilda_real,             # 反归一化后的含水印系数
        "watermark": watermark,                      # 水印模板
        "waterCA": waterCA,                          # 裁剪后的含水印低频系数
        "watermark2": watermark2,                    # 纯水印图像
        "CAmin": np.array([CAmin]),                 # 低频系数最小值
        "CAmax": np.array([CAmax]),                 # 低频系数最大值
        "U": U,                                      # 修改后的左奇异向量矩阵
        "V": V,                                      # 修改后的右奇异向量矩阵
        "sigma_tilda": sigma_tilda,                 # 水印奇异值
        "correlationU": np.array([correlationU]),   # U矩阵相关性
        "correlationV": np.array([correlationV]),   # V矩阵相关性
        "real_CA_shape": np.array(real_CA_shape),   # 真实低频系数形状
    }

    return {
        "watermarkimagergb": watermarkimagergb,  # 含水印的RGB图像
        "watermarkimage": watermarkimage,        # 含水印的单通道图像
        "waterCA": waterCA,                      # 含水印的低频系数
        "watermark2": watermark2,                # 纯水印图像
        "correlationU": correlationU,            # U矩阵相关性
        "correlationV": correlationV,            # V矩阵相关性
        "info": info,                            # 详细参数信息
    }

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_embed = subparsers.add_parser("embed")
    p_embed.add_argument("--cover", default="input/girl.jpg", help="输入图像路径")
    p_embed.add_argument("--out", default="output/girl_watermarked.jpg", help="输出水印图像路径")
    p_embed.add_argument("--alpha", type=float, default=0.5, help="嵌入强度 (0.1-1.0)")
    p_embed.add_argument("--seed", type=int, default=1234, help="随机种子")
    p_embed.add_argument("--wavelet", type=str, default="db1", help="小波类型")
    p_embed.add_argument("--level", type=int, default=1, help="小波分解层数")
    p_embed.add_argument("--ratio", type=float, default=0.8, help="替换比例 (0.1-1.0)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "embed":
        cover_rgb = load_rgb_image_float01(args.cover)
        result = wavemarksvd_like(
            cover_rgb,
            alpha=args.alpha,
            seed=args.seed,
            wavelet=args.wavelet,
            level=args.level,
            ratio=args.ratio,
        )
        watermarkimagergb = result["watermarkimagergb"]
        info = result["info"]
        save_rgb_image_float01(watermarkimagergb, args.out)
        print(f"嵌入完成，已保存含水印 RGB 图像到: {args.out}")
        print(
            f"  最深层 CA 形状: {info['CA'].shape}, "
            f"CA 范围: [{info['CAmin'][0]:.6f}, {info['CAmax'][0]:.6f}], "
            f"U/V 相关系数: "
            f"corrU={info['correlationU'][0]:.4f}, "
            f"corrV={info['correlationV'][0]:.4f}"
        )

    else:
        parser.error(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()


