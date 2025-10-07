PROMPT_TEMPLATE = """
“System_prompt”：你是一个工业专家安防系统，负责安防异常检测。请严格按照一下步骤分析图像，并输出结构化JSON。

第一步：图像描述
- 详细描述图像中的关键元素、状态和环境。聚焦于可能与安全规则相关的对象

第二步：注入上下文信息
- 阅读并理解以下提供的上下文信息：
--- 上下文信息 ---
{context_str}
---
- 将图像观察到的内容与上下文信息（特别是时空信息和安全规则）进行关联。

第三步：推理与判断
- 基于上下文和观察，判断是否存在异常（是/否）。
- 若异常，从以下列表选择最匹配的异常类型：
   {anomaly_type_list_str}
- 解释判断原因（必须引用上下文中的具体规则或时空信息）。
- 从以下预案中选择或组合最推荐的行动：
   {action_protocols_str}

【输出格式】（严格使用JSON，不要任何额外文本）
{{
  "is_anomaly": true/false,
  "anomaly_type": "string 或 null",
  "reason": "string",
  "recommended_actions": ["string1", ...],
  "confidence": "float (0-1)"  # 异常判断的置信度
}}
"""

import json

class PromptGenerator:
    def __init__(self, cot_template):
        self.cot_template = cot_template

    def _format_context(self, context, use_spatiotemporal, use_rules, use_decision):
        """Formats the context parts for injection into the prompt."""
        parts = []
        if use_spatiotemporal and 'spatiotemporal' in context:
            parts.append(f"- **时空信息**: {json.dumps(context['spatiotemporal'], ensure_ascii=False)}")
        if use_rules and 'security_rules' in context:
            parts.append(f"- **安全规则**: {json.dumps(context['security_rules'], ensure_ascii=False)}")
        if use_decision and 'decision' in context:
            parts.append(f"- **决策预案参考**: {json.dumps(context['decision'], ensure_ascii=False)}")
        
        if not parts:
            return "无"
        return "\n".join(parts)

    def generate(self, case_context, use_cot=True, **ablation_flags):
        """
        Generates a VLM prompt for a given case.
        ablation_flags: {'use_spatiotemporal': bool, 'use_rules': bool, 'use_decision': bool}
        """
        context_str = self._format_context(
            case_context['context'],
            ablation_flags.get('use_spatiotemporal', True),
            ablation_flags.get('use_rules', True),
            ablation_flags.get('use_decision', True)
        )

        if use_cot:
            # CoT 模板驱动
            prompt = self.cot_template.format(context_str=context_str)
        else:
            # 简单上下文拼接
            prompt = (
                "你是一个工业安防AI助手。请基于以下上下文信息和图像，判断是否存在异常。\n\n"
                f"--- 上下文信息 ---\n{context_str}\n\n"
                "--- 任务指令 ---\n"
                "请直接分析图像并判断。请以JSON格式输出你的结论，包含`anomaly_type`, `reason`, `recommended_action`, `confidence`字段。"
            )
        return prompt