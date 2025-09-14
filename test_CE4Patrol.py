import json
import os
import time
import openai
from typing import List, Dict, Any
from pathlib import Path

# ==================== 配置区 ====================
OPENAI_API_KEY = "your-openai-api-key-here"  # ← 替换为您的API密钥
MODEL = "gpt-4o"  # 支持 gpt-4o, gpt-4-turbo
DATA_FILE = "cogguard_bench.json"
IMAGE_DIR = "images"  # 图像存放目录
OUTPUT_DIR = "results"
SLEEP_DELAY = 1.5  # 避免速率限制

# 创建输出目录
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# 初始化 OpenAI 客户端
openai.api_key = OPENAI_API_KEY

# ==================== 提示词构建函数 ====================
def build_prompt(current_img_path: str, ref_img_path: str, rule: str, action_plan: List[str], env: Dict) -> str:
    """
    构建 CogGuard 结构化提示词（支持四组上下文）
    :param current_img_path: 当前图像路径
    :param ref_img_path: 参考图像路径
    :param rule: 文本规则
    :param action_plan: 行动预案列表
    :param env: 环境信息字典
    :return: 完整提示字符串
    """
    prompt_base = f"""你是一名安全巡逻机器狗，当前位于【{env['gps']}】，时间为【{env['timestamp']}】，光照为【{env['lighting']}】，天气为【{env['weather']}】。

{rule}

请对比当前图像与下方正常参考图（ID: {Path(ref_img_path).stem}）：
[插入正常参考图]

现在，请逐步推理：
1. 描述图像中可见的物体和状态；
2. 对比当前状态与参考图的差异；
3. 是否违反任何已知规则？
4. 若存在异常，给出具体原因；
5. 推荐下一步行动（从以下预案中选择最匹配的一项）：
   - {', '.join(action_plan)}

输出格式必须为严格JSON，不允许额外文本：
{{
  "anomaly_detected": true/false,
  "reason": "详细解释，必须引用规则和参考图",
  "action": "从预案中选择的唯一行动",
  "confidence": 0.0–1.0
}}
"""

    return prompt_base


# ==================== 评估函数：计算 CRS ====================
def compute_crs(response: Dict, ground_truth: Dict) -> float:
    """
    计算综合推理得分 CRS = (Accuracy×0.3 + Logic×0.5 + Action×0.2) × Confidence
    """
    # Accuracy: 是否正确识别异常
    accuracy = 1.0 if response.get("anomaly_detected") == ground_truth["expected_anomaly"] else 0.0

    # Logic: 专家评分（来自标注）
    logic = ground_truth["expert_logic_score"]

    # Action: 是否选择了正确的行动（精确匹配）
    expected_action = ground_truth["expected_action"]
    predicted_action = response.get("action", "")
    action_score = 1.0 if any(expected_action in act for act in ground_truth["action_plan"]) and predicted_action.strip() in ground_truth["action_plan"] else 0.0

    # Confidence: 模型自评
    confidence = response.get("confidence", 0.5)  # 默认0.5若缺失

    crs = (accuracy * 0.3 + logic * 0.5 + action_score * 0.2) * confidence
    return round(crs, 4)


# ==================== 主程序：执行四组消融实验 ====================
def run_ablation_experiment():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 四组上下文配置
    context_levels = [
        ("A (Baseline)", "", "", ""),  # 仅图像 + 通用问题
        ("B (+Text)", "{rule}", "", ""),  # + 规则
        ("C (+Visual)", "{rule}", "{ref_img}", ""),  # + 规则 + 参考图
        ("D (+Action)", "{rule}", "{ref_img}", "{action_plan}")  # + 全部
    ]

    results = []

    for test_case in data:
        current_img = test_case["current_img"]
        ref_img = test_case["ref_img"]
        rule = test_case["rule_text"]
        actions = test_case["action_plan"]
        env = test_case["environment"]

        for level_name, rule_placeholder, ref_placeholder, action_placeholder in context_levels:
            # 构建提示词
            if level_name == "A (Baseline)":
                prompt = "你是一名安全巡逻机器狗。请判断这张图片中是否存在任何安全隐患？输出JSON格式：{anomaly_detected: true/false, reason: ..., action: ..., confidence: ...}"
            elif level_name == "B (+Text)":
                prompt = build_prompt("", "", rule, actions, env)
            elif level_name == "C (+Visual)":
                prompt = build_prompt("", ref_img, rule, actions, env)
            else:  # D (+Action)
                prompt = build_prompt("", ref_img, rule, actions, env)

            # 调用模型
            try:
                print(f"[{level_name}] Processing {test_case['scene_id']}...")
                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"file://{os.path.abspath(os.path.join(IMAGE_DIR, current_img))}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500,
                    temperature=0.1
                )

                content = response.choices[0].message.content.strip()

                # 解析 JSON
                import re
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if not json_match:
                    raise ValueError("无法解析模型输出为JSON")

                llm_output = json.loads(json_match.group(1))

                # 计算CRS
                crs = compute_crs(llm_output, test_case)

                result = {
                    "scene_id": test_case["scene_id"],
                    "context_level": level_name,
                    "model_response": llm_output,
                    "crs": crs,
                    "raw_response": content,
                    "ground_truth": {
                        "expected_anomaly": test_case["expected_anomaly"],
                        "expected_action": test_case["expected_action"],
                        "expert_logic_score": test_case["expert_logic_score"]
                    }
                }
                results.append(result)

                print(f"✅ Success: {test_case['scene_id']} | CRS={crs}")
                time.sleep(SLEEP_DELAY)

            except Exception as e:
                print(f"❌ Error on {test_case['scene_id']} ({level_name}): {e}")
                results.append({
                    "scene_id": test_case["scene_id"],
                    "context_level": level_name,
                    "model_response": {"error": str(e)},
                    "crs": 0.0,
                    "raw_response": "",
                    "ground_truth": {
                        "expected_anomaly": test_case["expected_anomaly"],
                        "expected_action": test_case["expected_action"],
                        "expert_logic_score": test_case["expert_logic_score"]
                    }
                })

    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 所有实验完成！结果已保存至：{output_path}")

    # 汇总统计
    from collections import defaultdict
    crs_by_level = defaultdict(list)
    for r in results:
        crs_by_level[r["context_level"]].append(r["crs"])

    print("\n📊 综合CRS统计：")
    for level, scores in crs_by_level.items():
        avg = sum(scores) / len(scores)
        print(f"{level}: {avg:.4f} (n={len(scores)})")


if __name__ == "__main__":
    run_ablation_experiment()