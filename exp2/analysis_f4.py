#!/usr/bin/env python3
"""
F4 隐写算法分布统计与 RS 检测对比脚本（单一嵌入率）。

功能概述：
1. 生成 stego 图像（调用 f4Pro.f4_embed）。
2. 对比 cover 与 stego 的：
    - 像素灰度直方图分布与 JS / KL 统计（去除卡方）。
   - JPEG 量化后 AC 系数（合并所有块）分布、零系数比例、奇偶性变化。
3. 计算 AC 系数的符号与奇偶性映射统计：
   - 正/负、奇/偶、F4 规则下表示 0/1 的比例变化。
4. 调用 RS 分析（使用 rs_analyze.rs_statistics）分别对 cover 与 stego 估计嵌入率指标，用于观察抗检测能力。
5. 将结果保存为 CSV + JSON + 图像（PNG）。
6. 验证嵌入后提取的比特串与原始嵌入串一致性（准确率）。


使用示例：
python analysis_f4.py --cover_path pics/lena_gray.png --ratio 0.5 --seed 2025 --out_dir res/f4_analysis

输出文件：
- metrics.csv / metrics.json
- pixel_hist.png
- ac_hist.png
- ac_parity_bar.png
"""
import os
import csv
import json
import argparse
from typing import Dict, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt

from f4Pro import f4_embed, f4_extract_from_image, get_ac_dc_coeffs, Q
from utils import load_gray_image, calculate_accuracy
from rs_analyze import rs_statistics

EPS = 1e-12


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def prob_from_hist(hist: np.ndarray) -> np.ndarray:
    total = np.sum(hist)
    if total == 0:
        return np.zeros_like(hist, dtype=np.float64)
    p = hist.astype(np.float64) / total
    return p


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # KL(p||q)
    p_ = p + EPS
    q_ = q + EPS
    return float(np.sum(p_ * np.log(p_ / q_)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def compute_pixel_histograms(cover: np.ndarray, stego: np.ndarray) -> Dict[str, float]:
    hist_cover, _ = np.histogram(cover.flatten(), bins=256, range=(0,256))
    hist_stego, _ = np.histogram(stego.flatten(), bins=256, range=(0,256))
    p_cover = prob_from_hist(hist_cover)
    p_stego = prob_from_hist(hist_stego)
    return {
        'pixel_js': js_divergence(p_cover, p_stego),
        'pixel_kl_cover_stego': kl_divergence(p_cover, p_stego),
        'pixel_kl_stego_cover': kl_divergence(p_stego, p_cover)
    }, hist_cover, hist_stego


def flatten_all_ac(ac_list) -> np.ndarray:
    return np.concatenate([ac.astype(np.int32) for ac in ac_list]) if ac_list else np.array([], dtype=np.int32)


def compute_ac_histograms(ac_cover: np.ndarray, ac_stego: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    if ac_cover.size == 0 or ac_stego.size == 0:
       return {'ac_js': 0.0, 'ac_kl_cover_stego': 0.0, 'ac_kl_stego_cover': 0.0}, np.array([]), np.array([]), np.array([])
    vmin = int(min(ac_cover.min(), ac_stego.min()))
    vmax = int(max(ac_cover.max(), ac_stego.max()))
    # clamp range to avoid huge tails
    span = vmax - vmin
    if span > 200:  # limit extreme range for visualization
        vmin = max(vmin, -100)
        vmax = min(vmax, 100)
    bins = vmax - vmin + 1
    hist_cover, edges = np.histogram(ac_cover, bins=bins, range=(vmin, vmax+1))
    hist_stego, _ = np.histogram(ac_stego, bins=bins, range=(vmin, vmax+1))
    p_cover = prob_from_hist(hist_cover)
    p_stego = prob_from_hist(hist_stego)
    metrics = {
        'ac_js': js_divergence(p_cover, p_stego),
        'ac_kl_cover_stego': kl_divergence(p_cover, p_stego),
        'ac_kl_stego_cover': kl_divergence(p_stego, p_cover),
        'ac_range_min': vmin,
        'ac_range_max': vmax
    }
    return metrics, hist_cover, hist_stego, edges


def parity_symbol_stats(ac_vals: np.ndarray) -> Dict[str, int]:
    if ac_vals.size == 0:
        return {'pos':0,'neg':0,'zero':0,'odd_pos':0,'even_pos':0,'odd_neg':0,'even_neg':0,'encode_1':0,'encode_0':0}
    pos = np.sum(ac_vals > 0)
    neg = np.sum(ac_vals < 0)
    zero = np.sum(ac_vals == 0)
    odd_pos = np.sum((ac_vals > 0) & (ac_vals % 2 != 0))
    even_pos = np.sum((ac_vals > 0) & (ac_vals % 2 == 0))
    odd_neg = np.sum((ac_vals < 0) & (np.abs(ac_vals) % 2 != 0))
    even_neg = np.sum((ac_vals < 0) & (np.abs(ac_vals) % 2 == 0))
    # F4 抽取规则：正奇 或 负偶 -> 1，其余 -> 0
    encode_1 = odd_pos + even_neg
    encode_0 = even_pos + odd_neg
    return {
        'pos': int(pos), 'neg': int(neg), 'zero': int(zero),
        'odd_pos': int(odd_pos), 'even_pos': int(even_pos),
        'odd_neg': int(odd_neg), 'even_neg': int(even_neg),
        'encode_1': int(encode_1), 'encode_0': int(encode_0)
    }


def dict_diff(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: b.get(k,0) - a.get(k,0) for k in set(a) | set(b)}


def save_bar_parity(stats_cover: Dict[str,int], stats_stego: Dict[str,int], out_path: str):
    labels = ['encode_1','encode_0','odd_pos','even_pos','odd_neg','even_neg']
    cover_vals = [stats_cover[l] for l in labels]
    stego_vals = [stats_stego[l] for l in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, cover_vals, width, label='cover')
    plt.bar(x + width/2, stego_vals, width, label='stego')
    plt.xticks(x, labels, rotation=20)
    plt.ylabel('count')
    plt.title('AC parity & F4 symbol counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_hist_plot(hist_cover: np.ndarray, hist_stego: np.ndarray, edges: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(10,5))
    centers = edges[:-1]
    plt.plot(centers, hist_cover, label='cover', alpha=0.7)
    plt.plot(centers, hist_stego, label='stego', alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.xlabel('value')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_pixel_hist(hist_cover: np.ndarray, hist_stego: np.ndarray, out_path: str):
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(256), hist_cover, label='cover', alpha=0.7)
    plt.plot(np.arange(256), hist_stego, label='stego', alpha=0.7)
    plt.title('Pixel gray histogram')
    plt.legend()
    plt.xlabel('gray level')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_analysis(cover_path: str, ratio: float, seed: int, out_dir: str) -> Dict[str, float]:
    ensure_dir(out_dir)
    cover = load_gray_image(cover_path)
    stego, embed_bits = f4_embed(cover, ratio, seed=seed)
    # 提取校验
    extracted_bits = f4_extract_from_image(stego, len(embed_bits))
    extraction_acc = calculate_accuracy(embed_bits, extracted_bits) if len(embed_bits) == len(extracted_bits) else 0.0
    cv2.imwrite(os.path.join(out_dir, 'stego.png'), stego)

    # pixel hist metrics
    pixel_metrics, pixel_hist_cover, pixel_hist_stego = compute_pixel_histograms(cover, stego)
    save_pixel_hist(pixel_hist_cover, pixel_hist_stego, os.path.join(out_dir, 'pixel_hist.png'))

    # AC coeff lists
    ac_cover_list, dc_cover_list, _, _ = get_ac_dc_coeffs(cover)
    ac_stego_list, dc_stego_list, _, _ = get_ac_dc_coeffs(stego)
    ac_cover = flatten_all_ac(ac_cover_list)
    ac_stego = flatten_all_ac(ac_stego_list)

    ac_metrics, ac_hist_cover, ac_hist_stego, ac_edges = compute_ac_histograms(ac_cover, ac_stego)
    if ac_hist_cover.size > 0:
        save_hist_plot(ac_hist_cover, ac_hist_stego, ac_edges, 'Quantized AC coefficient histogram', os.path.join(out_dir, 'ac_hist.png'))

    # parity & symbol stats
    parity_cover = parity_symbol_stats(ac_cover)
    parity_stego = parity_symbol_stats(ac_stego)
    save_bar_parity(parity_cover, parity_stego, os.path.join(out_dir, 'ac_parity_bar.png'))
    parity_diff = dict_diff(parity_cover, parity_stego)

    # zero / non-zero changes
    nonzero_cover = int(np.sum(ac_cover != 0))
    nonzero_stego = int(np.sum(ac_stego != 0))

    # RS analysis
    mask = [1,0,0,1]
    rs_cover = rs_statistics(cover, group_size=4, mask=mask)
    rs_stego = rs_statistics(stego, group_size=4, mask=mask)

    # assemble metrics
    metrics = {
        'cover_path': cover_path,
        'ratio': ratio,
        'seed': seed,
        'embedded_bits': len(embed_bits),
        'extracted_bits': len(extracted_bits),
        'extraction_accuracy': extraction_acc,
        **pixel_metrics,
        **ac_metrics,
        'nonzero_ac_cover': nonzero_cover,
        'nonzero_ac_stego': nonzero_stego,
        'nonzero_ac_delta': nonzero_stego - nonzero_cover,
        'rs_estimate_cover': float(rs_cover),
        'rs_estimate_stego': float(rs_stego),
        'rs_estimate_delta': float(rs_stego - rs_cover)
    }

    # add parity stats flattened with prefix
    for k,v in parity_cover.items():
        metrics[f'parity_cover_{k}'] = v
    for k,v in parity_stego.items():
        metrics[f'parity_stego_{k}'] = v
    for k,v in parity_diff.items():
        metrics[f'parity_diff_{k}'] = v

    # save CSV
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['key','value'])
        for k,v in metrics.items():
            writer.writerow([k,v])
    # save JSON
    json_path = os.path.join(out_dir, 'metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='F4 single-rate distribution & RS analysis')
    parser.add_argument('--cover_path', type=str, default='pics/lena_gray.png')
    parser.add_argument('--ratio', type=float, default=0.5, help='嵌入率（相对于非零 AC 槽）')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--out_dir', type=str, default='res/f4_analysis')
    args = parser.parse_args()

    metrics = run_analysis(args.cover_path, args.ratio, args.seed, args.out_dir)
    print('Analysis metrics summary:')
    for k in sorted(metrics.keys()):
        if k.startswith('pixel_') or k.startswith('ac_') or k in ('rs_estimate_cover','rs_estimate_stego','rs_estimate_delta','extraction_accuracy'):
            print(f'  {k}: {metrics[k]}')
    print(f'Full metrics saved to {args.out_dir}')


if __name__ == '__main__':
    main()
