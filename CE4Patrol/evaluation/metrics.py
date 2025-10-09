"""
Metrics evaluation module for CE4Patrol.
Fuses the best of both approaches:
- Rigorous binary F1 score for accuracy (is_anomaly).
- User's brilliant `used_clauses` for logic consistency.
- Semantic similarity (Sentence-BERT) for action reliability.
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util

class MetricsEvaluator:
    """指标评估器 (融合版)"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 加载 Sentence-BERT 模型，这步可能需要一些时间
        try:
            self.st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model: {e}")
            self.st_model = None

    def _get_is_anomaly(self, response: Dict[str, Any]) -> bool:
        """从响应或真值中判断是否为异常"""
        return response.get("anomaly_type", "无异常") != "无异常"

    def _calculate_accuracy(self, parsed_response: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """计算二分类 F1 准确率 (是否检出异常)"""
        try:
            pred_is_anomaly = self._get_is_anomaly(parsed_response)
            true_is_anomaly = ground_truth.get("is_anomaly", False)
            # 使用 f1_score 计算二分类指标，pos_label=True 表示我们关心的是正确识别出"异常"
            return float(f1_score([true_is_anomaly], [pred_is_anomaly], pos_label=True, zero_division=0))
        except Exception as e:
            self.logger.error(f"Error calculating accuracy: {e}")
            return 0.0

    def _calculate_logic_consistency(self, parsed_response: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """计算逻辑一致性 (完全采纳您的优秀方案)"""
        try:
            predicted_clauses = set(parsed_response.get("used_clauses", []))
            true_clauses = set(ground_truth.get("used_clauses", []))
            
            if not true_clauses:
                return 1.0 if not predicted_clauses else 0.0
            
            intersection = len(predicted_clauses.intersection(true_clauses))
            union = len(predicted_clauses.union(true_clauses))
            
            # 使用 Jaccard 相似度来评估逻辑一致性
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating logic consistency: {e}")
            return 0.0

    def _calculate_action_reliability(self, parsed_response: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """计算行动可靠性 (使用语义相似度)"""
        if not self.st_model:
            self.logger.warning("SentenceTransformer model not available. Skipping action reliability.")
            return 0.0
        
        try:
            pred_actions = parsed_response.get("recommended_action", [])
            true_actions = ground_truth.get("recommended_action", [])

            if not true_actions and not pred_actions:
                return 1.0
            if not true_actions or not pred_actions:
                return 0.0

            # 编码句子
            pred_embeddings = self.st_model.encode(pred_actions, convert_to_tensor=True)
            true_embeddings = self.st_model.encode(true_actions, convert_to_tensor=True)

            # 计算余弦相似度矩阵
            cos_scores = util.cos_sim(pred_embeddings, true_embeddings)

            # 计算类似 BERTScore 的 F1
            # Precision: for each predicted action, find its best match in true actions
            p_scores = cos_scores.max(axis=1).values
            p_mean = p_scores.mean().item()

            # Recall: for each true action, find its best match in predicted actions
            r_scores = cos_scores.max(axis=0).values
            r_mean = r_scores.mean().item()

            # F1 Score
            f1 = 2 * (p_mean * r_mean) / (p_mean + r_mean) if (p_mean + r_mean) > 0 else 0.0
            return f1
        except Exception as e:
            self.logger.error(f"Error calculating action reliability: {e}")
            return 0.0
    
    def _extract_confidence(self, parsed_response: Dict[str, Any]) -> float:
        """提取置信度 (采纳您的方案)"""
        try:
            confidence = parsed_response.get("confidence", 0.0)
            return max(0.0, min(1.0, float(confidence)))
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not parse confidence: {confidence}. Defaulting to 0.0. Error: {e}")
            return 0.0

    def evaluate_case(self, vlm_response: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        metrics = {}
        parsed_response = self._parse_model_json(vlm_response)
        
        if not parsed_response:
            return self._get_default_metrics()

        metrics["accuracy"] = self._calculate_accuracy(parsed_response, ground_truth)
        metrics["logic_consistency"] = self._calculate_logic_consistency(parsed_response, ground_truth)
        metrics["action_reliability"] = self._calculate_action_reliability(parsed_response, ground_truth)
        metrics["confidence"] = self._extract_confidence(parsed_response)
        
        return metrics

    def _parse_model_json(self, vlm_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # This is a simplified version of your robust parser from vlm_caller.
        # It assumes the response has been pre-parsed once.
        if "error" in vlm_response: return None
        if "raw_response" in vlm_response:
            try: return json.loads(vlm_response["raw_response"])
            except: return None
        return vlm_response
    
    def _get_default_metrics(self) -> Dict[str, float]:
        return {"accuracy": 0.0, "logic_consistency": 0.0, "action_reliability": 0.0, "confidence": 0.0}