import wandb

# 假设你有一组提示词要测试
prompt_candidates = [
    "Is there any anomaly in this image?",
    "Detect defects or irregularities.",
    "Point out anything unusual.",
    "Are there visual abnormalities?",
    "Look for anomalies and describe them.",
    # ... 你可以加更多
]

# 遍历每个提示词
for prompt in prompt_candidates:
    
    # ========== 1. 初始化 W&B Run ==========
    wandb.init(
        project="vlm-anomaly-detection",   # 项目名：建议按论文/课题命名
        name=f"prompt-{hash(prompt) % 10000}",  # 实验名：可自定义，比如 "prompt-1", "prompt-with-detail"
        config={
            "model": "LLaVA-1.5",          # 你的 VLM 模型
            "dataset": "MVTec-AD",         # 数据集
            "prompt": prompt,              # ✅ 关键！记录当前提示词
            "temperature": 0.0,            # 如果有采样参数
            "max_new_tokens": 50,
        }
    )

    # ========== 2. 用当前 prompt 跑实验 ==========
    success_count = 0
    total_count = 0

    for image_path in test_images:  # 假设你有测试图像列表
        # 伪代码：调用你的 VLM 模型
        response = vlm_model(image_path, prompt)
        
        # 伪代码：判断是否成功检测异常（根据你的评估逻辑）
        is_success = evaluate_response(response, image_path)
        
        if is_success:
            success_count += 1
        total_count += 1

        if not is_success:
            # 上传失败的图像 + 模型回答，方便后期分析为什么失败
            wandb.log({
                "failed_example": wandb.Image(image_path),
                "model_response": response
            })

        # （可选）记录每张图的结果，用于后期分析
        wandb.log({"image_success": int(is_success)})

        
    # ========== 3. 计算并记录成功率 ==========
    success_rate = success_count / total_count if total_count > 0 else 0.0
    wandb.log({"success_rate": success_rate})

    # ========== 4. 结束当前 Run ==========
    wandb.finish()

    print(f"Prompt: {prompt} → Success Rate: {success_rate:.2%}")