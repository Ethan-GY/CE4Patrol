import json
import pandas as pd
from ce4patrol.context_loader import load_data
from ce4patrol.prompt_generator import PromptGenerator, COT_TEMPLATE
from ce4patrol.vlm_caller import call_qwen_vl_api
from ce4patrol.evaluation.metrics import calculate_accuracy_f1, calculate_logic_similarity, calculate_action_reliability
from ce4patrol.evaluation.crs_calculator import calculate_crs
from ce4patrol.analysis.visualizer import plot_results
from tqdm import tqdm

def run_experiment():
    # 1. 加载数据
    all_cases, all_ground_truths = load_data('data/context.json', 'data/ground_truth.json')
    prompt_gen = PromptGenerator(COT_TEMPLATE)
    api_key = "YOUR_DASHSCOPE_API_KEY"

    # 2. 定义实验配置 (主实验 + 消融实验)
    experiment_configs = {
        "CE4Patrol_Full": {"use_cot": True, "use_spatiotemporal": True, "use_rules": True, "use_decision": True},
        "No_CoT":         {"use_cot": False, "use_spatiotemporal": True, "use_rules": True, "use_decision": True},
        "No_Rules":       {"use_cot": True, "use_spatiotemporal": True, "use_rules": False, "use_decision": True},
        "No_Spatio":      {"use_cot": True, "use_spatiotemporal": False, "use_rules": True, "use_decision": True},
        "Base_VLM":       {"use_cot": False, "use_spatiotemporal": False, "use_rules": False, "use_decision": False}, # 基础模型能力
    }
    
    results = []

    # 3. 运行实验
    for case_id, case_data in tqdm(all_cases.items(), desc="Processing Cases"):
        gt_data = all_ground_truths[case_id]
        
        for config_name, config_params in experiment_configs.items():
            # 生成提示词
            prompt = prompt_gen.generate(case_data, **config_params)
            
            # 调用VLM
            model_output = call_qwen_vl_api(api_key, case_data['image_path'], prompt)
            
            if model_output:
                # 评估
                is_pred_anomaly = model_output.get('anomaly_type') != '无异常'
                accuracy = 1 if (is_pred_anomaly == gt_data['is_anomaly']) else 0
                
                logic_sim = calculate_logic_similarity(model_output.get('reason', ''), gt_data['ground_truth']['reason_keywords'], case_data['context']['security_rules'])
                
                action_rel = calculate_action_reliability(model_output.get('recommended_action', ''), gt_data['ground_truth']['action_keywords'])
                
                confidence = model_output.get('confidence', 0.0)
                
                # 计算CRS
                crs = calculate_crs(accuracy, logic_sim, action_rel, confidence)
                
                results.append({
                    "case_id": case_id,
                    "category": case_data['category'],
                    "config": config_name,
                    "accuracy": accuracy,
                    "logic_similarity": logic_sim,
                    "action_reliability": action_rel,
                    "confidence": confidence,
                    "crs": crs,
                    "model_output": model_output
                })

    # 4. 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/experiment_results.csv', index=False)
    
    # 5. 可视化分析
    plot_results(results_df)

if __name__ == "__main__":
    run_experiment()