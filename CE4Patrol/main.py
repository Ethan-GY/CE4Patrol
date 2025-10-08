#!/usr/bin/env python3
"""
CE4Patrol Fused: Main experiment runner.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# 导入融合后的模块
from ce4patrol.context_loader import ContextLoader
from ce4patrol.prompt_generator import PromptGenerator
from ce4patrol.vlm_caller import VLMCaller
from ce4patrol.evaluation.metrics import MetricsEvaluator
from ce4patrol.evaluation.crs_calculator import CRSCalculator
from ce4patrol.analysis.visualizer import ExperimentVisualizer
from ce4patrol.analysis.case_analyzer import CaseAnalyzer

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    layers: List[str]  # ['S', 'R', 'D'] for spatiotemporal, rules, decision
    enable_cot: bool


# 定义所有消融实验
ABLATIONS = [
    ExperimentConfig("CE4_FULL", ["S", "R", "D"], True),
    ExperimentConfig("NoCoT", ["S", "R", "D"], False),
    ExperimentConfig("SR_only", ["S", "R"], True),
    ExperimentConfig("S_only", ["S"], True),
    ExperimentConfig("R_only", ["R"], True),
    ExperimentConfig("D_only", ["D"], True),
    ExperimentConfig("None", [], True),
]


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_experiment(config: ExperimentConfig, dataset: Dict[str, Any], 
                  vlm_caller: VLMCaller, metrics_evaluator: MetricsEvaluator) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Running experiment: {config.name}")
    
    prompt_generator = PromptGenerator()
    crs_calculator = CRSCalculator()
    
    results = {"config": dataclasses.asdict(config), "scene_results": []}
    anomaly_types_enum = dataset["metadata"].get("anomaly_types_enum", [])
    
    for case in dataset["cases"]:
        case_id = case["case_id"]
        image_path = case["image_path"]
        
        if not Path(image_path).exists():
            logger.warning(f"Image not found for case {case_id}: {image_path}, skipping.")
            continue
            
        prompt = prompt_generator.build_prompt(
            case=case,
            enable_cot=config.enable_cot,
            inject_layers=config.layers,
            anomaly_types_enum=anomaly_types_enum
        )
        
        vlm_response = vlm_caller.call_vlm(prompt, image_path)
        
        if "error" in vlm_response:
            logger.error(f"VLM call failed for case {case_id}: {vlm_response['error']}")
            continue
            
        gt_case = case["ground_truth"]
        metrics = metrics_evaluator.evaluate_case(vlm_response, gt_case)
        crs = crs_calculator.compute_crs(**metrics)
        
        scene_result = {
            "scene_id": case_id,
            "vlm_response": vlm_response,
            "ground_truth": gt_case,
            "metrics": metrics,
            "crs": crs
        }
        results["scene_results"].append(scene_result)
        logger.info(f"Case {case_id} completed - CRS: {crs:.3f}")
    
    return results

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if any(not os.getenv(var) for var in ["CE4_API_KEY", "CE4_API_BASE", "CE4_MODEL"]):
        logger.error("Missing required environment variables. Please check your .env file.")
        return
    
    Path("results").mkdir(exist_ok=True)
    
    logger.info("Loading dataset...")
    context_loader = ContextLoader()
    dataset = context_loader.load_dataset("data/dataset.json")
    
    vlm_caller = VLMCaller()
    # 初始化一次，避免重复加载模型
    metrics_evaluator = MetricsEvaluator()
    
    all_results = [run_experiment(config, dataset, vlm_caller, metrics_evaluator) for config in ABLATIONS]
    
    output_path = "results/experiment_outputs.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Experiment results saved to {output_path}")
    
    logger.info("Generating visualizations...")
    visualizer = ExperimentVisualizer()
    visualizer.create_radar_chart(all_results)
    visualizer.create_heatmap(all_results)
    visualizer.create_crs_distribution(all_results)
    
    logger.info("Generating case analysis report...")
    case_analyzer = CaseAnalyzer()
    case_analyzer.analyze_cases(all_results)
    
    logger.info("All experiments completed successfully!")

if __name__ == "__main__":
    main()