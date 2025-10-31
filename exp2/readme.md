# 代码对应

一、卡方分析 → chi_square_analyze.py
说明：该脚本实现了对灰度图像的卡方统计量计算（按偶/奇灰度对对比），并输出 p 值及可视化曲线（chi2_stat, p）。

二、RS分析 → rs_analyze.py
说明：实现了 RS (Regular/Singular) 分析，包括像素组相关性计算、翻转操作与基于方程求解嵌入率 p 的逻辑（函数 rs_statistics）。

三、LSBM → lsbm.py
说明：实现了 LSB Matching（LSBM）嵌入/提取（lsbm_embed, lsbm_extract），并包含嵌入位置可视化与直方图比较等辅助函数。

参考/基础 LSB 替换 → lsb.py
说明：实现了最简单的 LSB 替换嵌入/提取（顺序位置），以及可视化/直方图函数。可作为教学或基线实现。

LSB Replacement（基于随机位置） → lsbr.py
说明：实现了“LSB 随机位置嵌入/提取”（使用随机种子复现位置），与 LSB/LSBM 配套用于对比实验。

六、F4隐写 → f4.py
说明：该脚本包含对图像按 8x8 分块做 DCT、量化并进行 ZigZag 扫描的函数；实现了 F4 嵌入逻辑（注释“F4嵌入逻辑”）和对应的提取函数 f4_extract，并可重构 JPEG-like 块以保存为伪 stego 图像（适用于灰度图的实验实现）。

Jsteg隐写 — 未找到专门实现文件
F3隐写 — 未找到专门实现文件
F5隐写 — 未找到专门实现文件

## 课堂验收

### 第一部分 卡方分析

实验步骤 I：LSBR 和 LSBM 隐写图像的生成 (准备分析数据)
隐写分析需要对不同嵌入率的隐写图像进行检测。首先需要利用 lsbr.py 和 lsbm.py 生成这些图像。

### 第二部分 RS分析
