# test_cogguard_china.py
# 使用 Qwen2-VL-7B 本地模型（国内可用，无需API）
# 需要：pip install transformers torch accelerate pillow

import json
import os
import time
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ==================== 配置区 ====================
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # 或 Qwen/Qwen-VL-Chat
DATA_FILE = "cogguard_bench_v2.json"
IMAGE_DIR = "images"
OUTPUT_DIR = "results/china"
SLEEP_DELAY = 0.5  # 本地模型较快，可降低延迟

# 创建输出目录
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 加载模型和tokenizer（首次运行会自动下载）
print("🔄 正在加载 Qwen2-VL 模型，请耐心等待...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("✅ 模型加载完成！")

# ==================== 构建提示词函数 ====================
def build_prompt(current_img_path: str, ref_img_path: str, rule: str, action_plan: List[str], env: Dict, context_level: str) -> str:
    """
    构建 Qwen-VL 的纯文本提示（图像通过 tokenizer 自动处理）
    """
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

    prompt = "\n\n".join(text_parts) + "\n\n请逐步推理并输出以下JSON格式：\n{\n  \"anomaly_detected\": true/false,\n  \"reason\": \"详细解释，必须引用规则和参考图\",\n  \"action\": \"从预案中选择的唯一行动\",\n  \"confidence\": 0.0–1.0\n}"

    return prompt

# ==================== 计算 CRS 函数（与国际版完全一致）====================
def compute_crs(response: Dict, ground_truth: Dict) -> float:
    accuracy = 1.0 if response.get("anomaly_detected") == ground_truth["expected_anomaly"] else 0.0

    logic_base = ground_truth["expert_logic_score"]

    reason_lower = response.get("reason", "").lower()
    env_keywords = ["time", "hour", "night", "rain", "weather", "gps", "location", "patrol", "work hours"]
    has_env_mention = any(kw in reason_lower for kw in env_keywords)

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
        current_img_path = test_case["current_img"]
        ref_img_path = test_case["ref_img"]
        rule = test_case["rule_text"]
        actions = test_case["action_plan"]
        env = test_case["environment"]

        for level_name, use_text, use_ref, use_action, use_env in context_levels:
            try:
                print(f"[{level_name}] Processing {test_case['scene_id']}...")

                prompt = build_prompt(current_img_path, ref_img_path, rule, actions, env, level_name[0])

                # 加载当前图像
                current_image = Image.open(current_img_path).convert("RGB")

                # Qwen-VL 输入格式：[image, text]
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "image": current_image, "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)

                # 生成响应
                outputs = model.generate(inputs, max_new_tokens=512, do_sample=False)
                response_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

                # 解析 JSON
                import re
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if not json_match:
                    raise ValueError(f"无法解析响应为JSON: {response_text[:200]}...")

                llm_output = json.loads(json_match.group(1))

                # 强制类型转换
                llm_output["anomaly_detected"] = bool(llm_output.get("anomaly_detected", False))
                llm_output["confidence"] = float(llm_output.get("confidence", 0.5))
                llm_output["action"] = str(llm_output.get("action", ""))

                crs = compute_crs(llm_output, test_case)

                result = {
                    "scene_id": test_case["scene_id"],
                    "context_level": level_name,
                    "model_response": llm_output,
                    "crs": crs,
                    "raw_response": response_text,
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

    print(f"\n🎉 国内版实验完成！结果已保存至：{output_path}")

    # 汇总统计
    from collections import defaultdict
    crs_by_level = defaultdict(list)
    for r in results:
        crs_by_level[r["context_level"]].append(r["crs"])

    print("\n📊 国内版 CRSSummary：")
    for level, scores in crs_by_level.items():
        avg = sum(scores) / len(scores)
        print(f"{level}: {avg:.4f} (n={len(scores)})")

if __name__ == "__main__":
    run_ablation_experiment()