PROMPT_TEMPLATE = """
“System_prompt”：你是一个工业专家安防系统，负责安防异常检测。请严格按照一下步骤分析图像，并输出结构化JSON。

【图像描述】
{visual_description}

【可用上下文信息】
<在此动态插入从JSON知识库中检索到的相关上下文>
- 位置: {location} ({zone_type})
- 时间: {timestamp} ({shift})
- 安全规则: {safety_rules_str}
- 正常参考: {normal_reference_note}
- 允许行为: {allowed_activities_str}

【任务要求】
1. 判断是否存在异常（是/否）。
2. 若异常，从以下列表选择最匹配的异常类型：
   {anomaly_type_list_str}
3. 解释判断原因（必须引用上下文中的具体规则或时空信息）。
4. 从以下预案中选择或组合最推荐的行动：
   {action_protocols_str}

【输出格式】（严格使用JSON，不要任何额外文本）
{{
  "is_anomaly": true/false,
  "anomaly_type": "string 或 null",
  "reason": "string",
  "recommended_actions": ["string1", ...]
}}
"""

def build_prompt(sample, anomaly_type_list):
    # 格式化列表为字符串
    anomaly_type_list_str = "\n   ".join([f"- {t}" for t in anomaly_type_list])
    safety_rules_str = "; ".join(sample['context']['space']['safety_rules'])
    allowed_activities_str = "; ".join(sample['context']['time']['allowed_activities'])
    action_protocols_str = "\n   ".join([
        f"- {step}" for step in sample['context']['action_protocols']['if_abnormal']
    ])

    return PROMPT_TEMPLATE.format(
        visual_description=sample['visual_description'],
        location=sample['context']['space']['location'],
        zone_type=sample['context']['space']['zone_type'],
        timestamp=sample['context']['time']['timestamp'],
        shift=sample['context']['time']['shift'],
        safety_rules_str=safety_rules_str,
        normal_reference_note=f"参考图: {sample['context']['space']['normal_reference_image']}",
        allowed_activities_str=allowed_activities_str,
        anomaly_type_list_str=anomaly_type_list_str,
        action_protocols_str=action_protocols_str
    )