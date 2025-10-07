from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 建议在类初始化时加载模型，避免重复加载
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def calculate_accuracy_f1(y_true, y_pred):
    """计算异常检测的F1分数 (A)"""
    return f1_score(y_true, y_pred)

def calculate_logic_similarity(model_reason, gt_reason_keywords, context_rules):
    """
    基于规则匹配计算逻辑相似度 (L).
    检查模型推理是否引用了正确的上下文信息。
    """
    score = 0.0
    # 检查是否提到了关键的实体/状态词
    if any(keyword in model_reason for keyword in gt_reason_keywords):
        score += 0.5
    
    # 检查是否明确引用了相关的规则ID
    relevant_rule_ids = [rule['rule_id'] for rule in context_rules if any(kw in rule['description'] for kw in gt_reason_keywords)]
    if any(rule_id in model_reason for rule_id in relevant_rule_ids):
        score += 0.5
        
    return score # 分数在 [0, 0.5, 1.0]

def calculate_action_reliability(model_action, gt_action_keywords):
    """
    使用Sentence-BERT计算推荐行动的可靠性 (R).
    """
    if not model_action or not gt_action_keywords:
        return 0.0
    
    # 将关键词列表转换为句子
    gt_action_sentence = " ".join(gt_action_keywords)
    
    # 计算嵌入
    embedding1 = sbert_model.encode(model_action, convert_to_tensor=True)
    embedding2 = sbert_model.encode(gt_action_sentence, convert_to_tensor=True)
    
    # 计算余弦相似度
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    
    # 将 [-1, 1] 区间的分数归一化到 [0, 1]
    return (cosine_score + 1) / 2
