#!/usr/bin/env python3
"""
F5与F4算法性能对比分析脚本
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils import load_gray_image, generate_random_bits, calculate_accuracy
from f4Pro import f4_embed, f4_extract_from_image
from f5 import f5_embed, f5_extract
from rs_analyze import rs_statistics

def compare_algorithms():
    """
    比较F4和F5算法的性能
    """
    cover_path = 'pics/lena_gray.png'
    cover = load_gray_image(cover_path)
    
    print("="*60)
    print("F4 vs F5 Steganography Comparison")
    print("="*60)
    
    # F4测试
    print("\n1. F4 Algorithm:")
    f4_stego, f4_embed_bits = f4_embed(cover, ratio=0.3, seed=2025)
    cv2.imwrite('res/comparison_f4_stego.png', f4_stego)
    f4_extract_bits = f4_extract_from_image(f4_stego, len(f4_embed_bits))
    f4_acc = calculate_accuracy(f4_embed_bits, f4_extract_bits)
    
    print(f"   Embedded bits: {len(f4_embed_bits)}")
    print(f"   Extraction accuracy: {f4_acc:.4f}")
    
    # F5测试（不同r值）
    for r in [2, 3, 4]:
        print(f"\n2. F5 Algorithm (r={r}):")
        n = 2**r - 1
        f5_stego, f5_modified = f5_embed(cover, f4_embed_bits[:len(f4_embed_bits)//n*n*(r//3 if r>=3 else 1)], r, seed=2025)
        cv2.imwrite(f'res/comparison_f5_r{r}_stego.png', f5_stego)
        f5_extract_bits = f5_extract(f5_stego, len(f4_embed_bits)//n*n*(r//3 if r>=3 else 1), r, seed=2025)
        f5_acc = calculate_accuracy(f4_embed_bits[:len(f5_extract_bits)], f5_extract_bits)
        
        print(f"   Hamming({n}, {r})")
        print(f"   Embedded bits: {len(f5_extract_bits)}")
        print(f"   Modified positions: {len(f5_modified)}")
        print(f"   Extraction accuracy: {f5_acc:.4f}")
        print(f"   Efficiency: {r}/{n} = {r/n:.4f} bits per modification")
    
    # RS分析对比
    print(f"\n3. RS Analysis:")
    print("   Cover image:")
    rs_cover = rs_statistics(cover, 4, [1,0,0,1])
    print(f"   Estimated rate: {rs_cover:.4f}")
    
    print("   F4 stego:")
    rs_f4 = rs_statistics(f4_stego, 4, [1,0,0,1])
    print(f"   Estimated rate: {rs_f4:.4f}")
    
    print("   F5 stego (r=3):")
    rs_f5 = rs_statistics(f5_stego, 4, [1,0,0,1])
    print(f"   Estimated rate: {rs_f5:.4f}")

if __name__ == '__main__':
    compare_algorithms()