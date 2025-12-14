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