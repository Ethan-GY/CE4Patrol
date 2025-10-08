"""
CRS (Context Reasoning Score) calculator for CE4Patrol.
"""

import logging
from typing import Dict, Any, List


class CRSCalculator:
    """上下文推理分数计算器"""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.5, gamma: float = 0.2):
        """
        初始化CRS计算器
        
        Args:
            alpha: 准确率权重 (A)
            beta: 逻辑一致性权重 (L)
            gamma: 行动可靠性权重 (R)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.logger = logging.getLogger(__name__)
        
        # 验证权重和
        total_weight = alpha + beta + gamma
        if abs(total_weight - 1.0) > 1e-6:
            self.logger.warning(f"Weight sum is {total_weight}, expected 1.0")
    
    def compute_crs(self, accuracy: float, logic_consistency: float, 
                    action_reliability: float, confidence: float) -> float:
        """
        计算CRS分数
        
        Args:
            accuracy: 准确率/F1分数 (A)
            logic_consistency: 逻辑一致性分数 (L)
            action_reliability: 行动可靠性分数 (R)
            confidence: 置信度 (C)
            
        Returns:
            CRS分数 = (α*A + β*L + γ*R) * C
        """
        try:
            # 计算基础分数
            base_score = (
                self.alpha * accuracy +
                self.beta * logic_consistency +
                self.gamma * action_reliability
            )
            
            # 应用置信度权重
            crs_score = base_score * confidence
            
            # 确保分数在0-1范围内
            crs_score = max(0.0, min(1.0, crs_score))
            
            self.logger.debug(
                f"CRS calculation: A={accuracy:.3f}, L={logic_consistency:.3f}, "
                f"R={action_reliability:.3f}, C={confidence:.3f} -> CRS={crs_score:.3f}"
            )
            
            return crs_score
            
        except Exception as e:
            self.logger.error(f"Error computing CRS: {e}")
            return 0.0
    
    def compute_batch_crs(self, metrics_list: List[Dict[str, float]]) -> List[float]:
        """
        批量计算CRS分数
        
        Args:
            metrics_list: 指标列表，每个元素包含accuracy, logic_consistency, 
                         action_reliability, confidence
            
        Returns:
            CRS分数列表
        """
        crs_scores = []
        
        for metrics in metrics_list:
            crs = self.compute_crs(
                accuracy=metrics.get("accuracy", 0.0),
                logic_consistency=metrics.get("logic_consistency", 0.0),
                action_reliability=metrics.get("action_reliability", 0.0),
                confidence=metrics.get("confidence", 0.0)
            )
            crs_scores.append(crs)
        
        return crs_scores
    
    def get_crs_breakdown(self, accuracy: float, logic_consistency: float,
                         action_reliability: float, confidence: float) -> Dict[str, float]:
        """
        获取CRS分数分解
        
        Args:
            accuracy: 准确率
            logic_consistency: 逻辑一致性
            action_reliability: 行动可靠性
            confidence: 置信度
            
        Returns:
            包含各组件分数的字典
        """
        return {
            "accuracy_component": self.alpha * accuracy,
            "logic_component": self.beta * logic_consistency,
            "action_component": self.gamma * action_reliability,
            "base_score": (
                self.alpha * accuracy +
                self.beta * logic_consistency +
                self.gamma * action_reliability
            ),
            "confidence": confidence,
            "final_crs": self.compute_crs(accuracy, logic_consistency, action_reliability, confidence)
        }
    
    def analyze_crs_distribution(self, crs_scores: List[float]) -> Dict[str, Any]:
        """
        分析CRS分数分布
        
        Args:
            crs_scores: CRS分数列表
            
        Returns:
            分布统计信息
        """
        if not crs_scores:
            return {"error": "Empty CRS scores list"}
        
        import numpy as np
        
        return {
            "count": len(crs_scores),
            "mean": np.mean(crs_scores),
            "std": np.std(crs_scores),
            "min": np.min(crs_scores),
            "max": np.max(crs_scores),
            "median": np.median(crs_scores),
            "q25": np.percentile(crs_scores, 25),
            "q75": np.percentile(crs_scores, 75)
        }
