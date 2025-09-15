# test_cogguard_international.py
# 使用 OpenAI GPT-4o 多模态 API（支持图像+文本输入）
# 需要：pip install openai python-dotenv

import json
import os
import time
import base64
from pathlib import Path
from dotenv import load_dotenv
import openai

# ==================== 配置区 ====================
load_dotenv()  # 从 .env 文件读取 API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ← 在 .env 中填写：OPENAI_API_KEY=your-key-here
MODEL = "gpt-4o"  # 支持 gpt-4o, gpt-4-turbo
DATA_FILE = "cogguard_bench_v2.json"
IMAGE_DIR = "images"
OUTPUT_DIR = "results/international"
SLEEP_DELAY = 1.5  # 避免速率限制（GPT-4o 限制约 3 RPM）

# 创建输出目录
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 初始化 OpenAI 客户端
openai.api_key = OPENAI_API_KEY

# ==================== 辅助函数：将图片转为 Base64 ====================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==================== 构建提示词函数 ====================
def build_prompt(current_img_path: str, ref_img_path: str, rule: str, action_plan: List[str], env: Dict, context_level: str) -> list:
    """
    根据上下文层级构建多模态提示（用于GPT-4o）
    :param context_level: 'A', 'B', 'C', 'D', 'E'
    :return: messages 列表，供 openai.ChatCompletion.create 使用
    """
    base_message = [
        {
            "role": "system",
            "content": "你是一名安全巡逻机器狗。请根据图像、规则、环境和预案，进行严谨推理，输出严格JSON格式。"
        }
    ]

    user_content = []

    # 添加当前图像（所有组都包含）
    current_base64 = encode_image(current_img_path)
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{current_base64}"}
    })

    # 构建文本提示
    text_parts = []

    if context_level in ['B', 'C', 'D', 'E']:
        text_parts.append(f"安全规则：{rule}")

    if context_level in ['C', 'D', 'E']:
        text_parts.append(f"正常参考状态（ID: {Path(ref_img_path).stem}）：[图像已提供]")

    if context_level in ['D', 'E']:
        actions_str = "\n".join([f"- {act}" for act in action_plan])
        text_parts.append(f"可选行动预案：\n{actions_str}")

    if context_level == 'E':
        env_info = (
            f"环境信息：\n"
            f"- 位置：{env['gps']}\n"
            f"- 时间：{env['timestamp']}\n"
            f"- 光照：{env['lighting']}\n"
            f"- 天气：{env['weather']}\n"
            f"- 是否工作时间：{'是' if env['is_work_hours'] else '否'}\n"
            f"- 上次巡逻：{env['last_patrol_time']}"
        )
        text_parts.append(env_info)

    if not text_parts:
        text_parts.append("请判断这张图片中是否存在任何安全隐患？")

    user_content.append({
        "type": "text",
        "text": "\n\n".join(text_parts) + "\n\n请逐步推理并输出以下JSON格式：\n{\n  \"anomaly_detected\": true/false,\n  \"reason\": \"详细解释，必须引用规则和参考图\",\n  \"action\": \"从预案中选择的唯一行动\",\n  \"confidence\": 0.0–1.0\n}"
    })

    return base_message + [{"role": "user", "content": user_content}]

# ==================== 计算 CRS 函数（与国内版完全一致）====================
def compute_crs(response: Dict, ground_truth: Dict) -> float:
    accuracy = 1.0 if response.get("anomaly_detected") == ground_truth["expected_anomaly"] else 0.0

    logic_base = ground_truth["expert_logic_score"]

    # 环境修正因子：检查模型是否合理使用了环境信息
    reason_lower = response.get("reason", "").lower()
    env_keywords = ["time", "hour", "night", "rain", "weather", "gps", "location", "patrol", "work hours"]
    has_env_mention = any(kw in reason_lower for kw in env_keywords)

    # 如果是需要环境判断的类型，且模型提到了环境 → 加分
    if ground_truth["type"] in ["ambiguous_object_with_context", "false_positive_confounder", "rule_violation_with_confounding_env"]:
        if has_env_mention and response["anomaly_detected"] == ground_truth["expected_anomaly"]:
            logic_final = min(1.0, logic_base + 0.1)
        elif has_env_mention and response["anomaly_detected"] != ground_truth["expected_anomaly"]:
            logic_final = max(0.0, logic_base - 0.2)
        else:
            logic_final = logic_base
    else:
        logic_final = logic_base

    action_plan = ground_truth["action_plan"]
    predicted_action = response.get("action", "")
    action_score = 1.0 if any(predicted_action.strip() in act for act in action_plan) else 0.0

    confidence = response.get("confidence", 0.5)
    crs = (accuracy * 0.3 + logic_final * 0.5 + action_score * 0.2) * confidence
    return round(crs, 4)

# ==================== 主程序：五组消融实验 ====================
def run_ablation_experiment():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    context_levels = [
        ("A (Baseline)", False, False, False, False),
        ("B (+Text)", True, False, False, False),
        ("C (+Visual)", True, True, False, False),
        ("D (+Action)", True, True, True, False),
        ("E (+Env)", True, True, True, True)
    ]

    results = []

    for test_case in data:
        current_img = test_case["current_img"]
        ref_img = test_case["ref_img"]
        rule = test_case["rule_text"]
        actions = test_case["action_plan"]
        env = test_case["environment"]

        for level_name, use_text, use_ref, use_action, use_env in context_levels:
            try:
                print(f"[{level_name}] Processing {test_case['scene_id']}...")
                messages = build_prompt(current_img, ref_img, rule, actions, env, level_name[0])

                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.1,
                    top_p=0.9
                )

                content = response.choices[0].message.content.strip()

                # 解析 JSON（容错处理）
                import re
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if not json_match:
                    raise ValueError(f"无法解析响应为JSON: {content[:200]}...")

                llm_output = json.loads(json_match.group(1))

                # 强制转换字段（避免类型错误）
                llm_output["anomaly_detected"] = bool(llm_output.get("anomaly_detected", False))
                llm_output["confidence"] = float(llm_output.get("confidence", 0.5))
                llm_output["action"] = str(llm_output.get("action", ""))

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
                        "expert_logic_score": test_case["expert_logic_score"],
                        "type": test_case["type"]
                    }
                }
                results.append(result)

                print(f"✅ Success: {test_case['scene_id']} | {level_name} | CRS={crs}")
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
                        "expert_logic_score": test_case["expert_logic_score"],
                        "type": test_case["type"]
                    }
                })

    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 国际版实验完成！结果已保存至：{output_path}")

    # 汇总统计
    from collections import defaultdict
    crs_by_level = defaultdict(list)
    for r in results:
        crs_by_level[r["context_level"]].append(r["crs"])

    print("\n📊 国际版 CRSSummary：")
    for level, scores in crs_by_level.items():
        avg = sum(scores) / len(scores)
        print(f"{level}: {avg:.4f} (n={len(scores)})")

if __name__ == "__main__":
    run_ablation_experiment()