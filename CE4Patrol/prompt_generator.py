"""
Prompt generator module with enhanced CoT instructions.
"""
import logging
from typing import Dict, Any, List

class PromptGenerator:
    """提示生成器 (增强指令版)"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def build_prompt(self, case: Dict[str, Any], enable_cot: bool, 
                    inject_layers: List[str], anomaly_types_enum: List[str]) -> str:
        context = case['context']
        scene_info = context['scene_info']
        case_id = case['case_id']

        context_parts = []
        if 'S' in inject_layers: context_parts.append(self._build_spatiotemporal_context(context))
        if 'R' in inject_layers: context_parts.append(self._build_rules_context(context))
        if 'D' in inject_layers: context_parts.append(self._build_decision_context(context))
        
        ambiguity_info = self._build_semantic_ambiguity(context)
        
        template = self._build_cot_prompt if enable_cot else self._build_direct_prompt
        return template(case_id, scene_info, context_parts, ambiguity_info, anomaly_types_enum)

    def _build_spatiotemporal_context(self, context: Dict[str, Any]) -> str:
        """构建时空上下文"""
        st = context["spatiotemporal"]
        return f"""
时空上下文 (Spatiotemporal Context):
- 时区: {st['timezone']}
- 时间提示: {st['time_hint']}
- GPS坐标: {st['gps']}
"""
    
    def _build_rules_context(self, context: Dict[str, Any]) -> str:
        """构建规则上下文"""
        rules = context["rules"]
        clauses_text = "\n".join([
            f"- {clause['clause_id']}: {clause['logic']} (权重: {clause['weight']})"
            for clause in rules["logic_clauses"]
        ])
        
        return f"""
规则上下文 (Rules Context):
- 手册参考: {', '.join(rules['manual_refs'])}
- 逻辑子句:
{clauses_text}
"""
    
    def _build_decision_context(self, context: Dict[str, Any]) -> str:
        """构建决策上下文"""
        decision = context["decision"]
        playbook = decision["playbook"]
        
        risk_levels_text = f"""
- 高风险动作: {', '.join(playbook['risk_levels']['high'])}
- 中风险动作: {', '.join(playbook['risk_levels']['medium'])}
- 低风险动作: {', '.join(playbook['risk_levels']['low'])}
"""
        
        return f"""
决策上下文 (Decision Context):
- 正常参考: {', '.join(decision['normal_refs'])}
- 操作手册:
{risk_levels_text}
- 触发阈值: {playbook['trigger']}
"""
    
    def _build_semantic_ambiguity(self, context: Dict[str, Any]) -> str:
        """构建语义歧义信息"""
        ambiguity = context["semantic_ambiguity"]
        objects_text = "\n".join([
            f"- {obj['name']}: 风险等级 {obj['risk']}"
            for obj in ambiguity["small_objects"]
        ])
        
        return f"""
语义歧义对象 (Semantic Ambiguity Objects):
{objects_text}
"""
    
    def _build_cot_prompt(self, case_id: str, scene_info: Dict, context_parts: List[str], 
                         ambiguity_info: str, anomaly_types_enum: List[str]) -> str:
        context_text = "\n".join(context_parts) if context_parts else "无额外上下文"
        return f"""你是一个专业的工业安全巡检专家。请分析以下场景图像，识别潜在的"薛定谔异常"。

场景信息:
- 案例ID: {case_id}
- 描述: {scene_info['description']}

{context_text}
{ambiguity_info}

请严格按照以下步骤进行思维链分析：

1.  **图像观察 (Image Observation):** 详细、客观地描述图像中的关键元素，如设备状态、环境布局、人员活动等。
2.  **上下文对齐 (Context Alignment):** 将观察到的内容与提供的上下文信息（时空、规则、决策）进行逐一比对，找出潜在的冲突点或不一致之处。
3.  **异常推理 (Anomaly Inference):**
    *   基于步骤2的发现，明确判断是否存在异常。
    *   如果存在异常，**必须明确引用触发判断的 `logic_clauses` 中的 `clause_id`**。
    *   如果无异常，请说明理由。
4.  **行动推荐 (Action Recommendation):**
    *   根据推理出的风险等级，**严格参考 `decision.playbook`** 生成具体的、可执行的行动建议列表。
    *   如果无异常，推荐 "无需处理" 或 "继续观察"。
5.  **最终输出构建 (Final JSON Output):**
    *   综合以上分析，构建一个严格的JSON对象。
    *   `anomaly_type` 字段的值**必须**是以下之一: {anomaly_types_enum}


请仅输出一个有效的JSON对象，不要包含任何额外的解释性文本。JSON格式如下:
{{
  "anomaly_type": "string",
  "reason": "string",
  "recommended_action": ["string"],
  "confidence": float (0.0-1.0),
  "used_clauses": ["string"]
}}"""
    
    def _build_direct_prompt(self, case_id: str, scene_info: Dict[str, Any], context_parts: List[str],
                            ambiguity_info: str, anomaly_types_enum: List[str]) -> str:
        """构建直接提示"""
        context_text = "\n".join(context_parts) if context_parts else "无额外上下文"
        
        return f"""你是一个专业的工业安全巡检专家。请分析以下场景图像，识别潜在的"薛定谔异常"。

场景信息:
- 场景ID: {scene_info['scene_id']}
- 描述: {scene_info['description']}

{context_text}

{ambiguity_info}

请直接分析图像并输出JSON格式结果，包含以下字段：
- `anomaly_type` 字段的值**必须**是以下之一: {anomaly_types_enum}
- `reason` 字段: 判断理由
- `recommended_action` 字段: 推荐行动列表
- `confidence` 字段: 置信度（0-1）
- `used_clauses` 字段: 使用的逻辑子句ID列表

请确保输出为有效的JSON格式。"""
