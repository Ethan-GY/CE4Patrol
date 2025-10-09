"""
Prompt generator module with enhanced CoT instructions.
"""
import logging
from typing import Dict, Any, List
from .data_models import PatrolCase

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

    def _build_context_string(self, case: PatrolCase, config: dict) -> str:
        """根据实验配置构建上下文注入字符串"""
        context_parts = []
        if config.get("S", False):
            context_parts.append(self._build_spatiotemporal_context(case))
        if config.get("R", False):
            context_parts.append(self._build_rules_context(case))
        if config.get("D", False):
            context_parts.append(self._build_decision_context(case))
        
        return "\n\n".join(filter(None, context_parts))

    def _build_spatiotemporal_context(self, case: PatrolCase) -> str:
        """构建时空上下文的文本描述"""
        st_context = case.context.spatiotemporal
        return (
            f"### 1. 时空背景 (S)\n"
            f"- **巡检位置:** {st_context.location}\n"
            f"- **巡检时间:** {st_context.timestamp}"
        )
    
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
    
    def _build_decision_context(self, case: PatrolCase) -> str:
        """构建决策辅助上下文的文本描述"""
        decision_context = case.context.decision
        
        # 构建正常状态文本参考
        normal_refs_str = "\n".join([f"- {ref}" for ref in decision_context.normal_text_refs])
        
        # 构建正常状态图像参考
        normal_images_str = "无"
        if decision_context.normal_image_refs:
            normal_images_str = "\n".join([f"- `{path}`" for path in decision_context.normal_image_refs])

        # 构建行动预案
        playbook_str = "\n".join([f"- **{level.upper()}风险:** {action}" for level, action in decision_context.playbook.items()])

        # 构建语义模糊部分
        ambiguity_str = ""
        if decision_context.semantic_ambiguity:
            ambiguity_info = decision_context.semantic_ambiguity
            possible_options = ", ".join([f"{item['name']} (风险: {item['risk']})" for item in ambiguity_info['possible_interpretations']])
            ambiguity_str = (
                f"\n\n**特别注意 - 语义模糊对象:**\n"
                f"- **对象描述:** {ambiguity_info['object_description']}\n"
                f"- **可能解释:** {possible_options}\n"
                f"- **分析指引:** {ambiguity_info['instruction']}"
            )

        return (
            f"### 3. 决策辅助信息 (D)\n"
            f"**正常状态参考:**\n{normal_refs_str}\n"
            f"**正常状态参考图片路径:**\n{normal_images_str}\n\n"
            f"**标准行动预案 (Playbook):**\n{playbook_str}"
            f"{ambiguity_str}"
        )
        
    
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
