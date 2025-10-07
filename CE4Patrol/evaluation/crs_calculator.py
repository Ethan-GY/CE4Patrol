def calculate_crs(A, L, R, C, alpha=0.3, beta=0.5, gamma=0.2):
    """
    计算综合可靠性得分 (Composite Reliability Score - CRS).
    A: 准确率分 (0或1)
    L: 逻辑一致性分 (0到1)
    R: 推荐行动分 (0到1)
    C: 模型置信度 (0到1)
    """
    if not isinstance(C, (int, float)) or C < 0 or C > 1:
        C = 0.5 # 如果置信度无效，给一个中性值
        
    core_score = (alpha * A + beta * L + gamma * R)
    crs_score = core_score * C
    return crs_score