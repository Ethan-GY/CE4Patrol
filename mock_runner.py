#!/usr/bin/env python3
"""
CE4Patrol 完整实验模拟
使用模拟的VLM响应来展示完整的实验流程
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_vlm_responses():
    """创建模拟的VLM响应"""
    
    # 模拟不同实验配置的响应
    mock_responses = {
        "CE4_FULL": {
            "device_room_evening": {
                "anomaly_type": "设备异常",
                "reason": "LED指示灯显示红色异常状态，防火门处于开启状态，监控摄像头被部分遮挡，这些都是设备异常的表现",
                "recommended_action": [
                    "检查LED指示灯电路连接",
                    "关闭防火门并检查门禁系统",
                    "调整监控摄像头角度",
                    "记录所有异常设备状态",
                    "通知设备维护部门"
                ],
                "confidence": 0.92,
                "used_clauses": ["LED_COLOR_CONFLICT", "FIRE_DOOR_OPEN", "CAMERA_OCCLUSION"]
            },
            "office_night": {
                "anomaly_type": "环境异常",
                "reason": "夜间办公区域灯光异常亮起，检测到人员活动，需要核实是否为授权加班",
                "recommended_action": [
                    "核实人员身份和授权",
                    "检查灯光控制系统",
                    "确认是否有加班申请",
                    "记录异常情况",
                    "通知安全部门"
                ],
                "confidence": 0.88,
                "used_clauses": ["LIGHTS_AT_NIGHT", "PERSON_AFTER_HOURS"]
            }
        },
        "NoCoT": {
            "device_room_evening": {
                "anomaly_type": "设备异常",
                "reason": "LED指示灯颜色异常，防火门开启",
                "recommended_action": [
                    "检查LED电路",
                    "关闭防火门",
                    "记录状态"
                ],
                "confidence": 0.85,
                "used_clauses": ["LED_COLOR_CONFLICT", "FIRE_DOOR_OPEN"]
            },
            "office_night": {
                "anomaly_type": "环境异常",
                "reason": "夜间灯光异常，人员活动",
                "recommended_action": [
                    "核实身份",
                    "检查灯光",
                    "记录情况"
                ],
                "confidence": 0.82,
                "used_clauses": ["LIGHTS_AT_NIGHT", "PERSON_AFTER_HOURS"]
            }
        },
        "SR_only": {
            "device_room_evening": {
                "anomaly_type": "设备异常",
                "reason": "基于时空和规则上下文，LED指示灯颜色与预期不符，防火门在非紧急情况下开启",
                "recommended_action": [
                    "检查LED指示灯电路",
                    "关闭防火门",
                    "记录设备状态"
                ],
                "confidence": 0.87,
                "used_clauses": ["LED_COLOR_CONFLICT", "FIRE_DOOR_OPEN"]
            },
            "office_night": {
                "anomaly_type": "环境异常",
                "reason": "夜间办公区域灯光异常，检测到人员活动",
                "recommended_action": [
                    "核实人员身份",
                    "检查灯光控制系统",
                    "记录异常情况"
                ],
                "confidence": 0.84,
                "used_clauses": ["LIGHTS_AT_NIGHT", "PERSON_AFTER_HOURS"]
            }
        },
        "S_only": {
            "device_room_evening": {
                "anomaly_type": "设备异常",
                "reason": "基于时空上下文分析，晚间时段设备状态异常",
                "recommended_action": [
                    "检查设备状态",
                    "记录异常"
                ],
                "confidence": 0.75,
                "used_clauses": ["LED_COLOR_CONFLICT"]
            },
            "office_night": {
                "anomaly_type": "环境异常",
                "reason": "夜间时段检测到异常活动",
                "recommended_action": [
                    "核实情况",
                    "记录活动"
                ],
                "confidence": 0.78,
                "used_clauses": ["LIGHTS_AT_NIGHT"]
            }
        },
        "R_only": {
            "device_room_evening": {
                "anomaly_type": "设备异常",
                "reason": "根据规则逻辑，LED颜色冲突和防火门开启都违反了安全规则",
                "recommended_action": [
                    "检查LED指示灯电路",
                    "关闭防火门",
                    "记录违规情况"
                ],
                "confidence": 0.89,
                "used_clauses": ["LED_COLOR_CONFLICT", "FIRE_DOOR_OPEN"]
            },
            "office_night": {
                "anomaly_type": "环境异常",
                "reason": "违反夜间安全规则，灯光异常和人员活动需要核实",
                "recommended_action": [
                    "核实人员身份和授权",
                    "检查灯光控制系统",
                    "记录违规情况"
                ],
                "confidence": 0.86,
                "used_clauses": ["LIGHTS_AT_NIGHT", "PERSON_AFTER_HOURS"]
            }
        },
        "D_only": {
            "device_room_evening": {
                "anomaly_type": "设备异常",
                "reason": "根据决策手册，需要采取高风险处理行动",
                "recommended_action": [
                    "立即通知安全部门",
                    "启动应急响应程序",
                    "记录详细事件信息"
                ],
                "confidence": 0.91,
                "used_clauses": ["LED_COLOR_CONFLICT", "FIRE_DOOR_OPEN"]
            },
            "office_night": {
                "anomaly_type": "环境异常",
                "reason": "根据决策手册，需要采取中风险处理行动",
                "recommended_action": [
                    "记录异常情况",
                    "通知相关责任人",
                    "安排后续检查"
                ],
                "confidence": 0.83,
                "used_clauses": ["LIGHTS_AT_NIGHT", "PERSON_AFTER_HOURS"]
            }
        },
        "None": {
            "device_room_evening": {
                "anomaly_type": "设备异常",
                "reason": "检测到设备状态异常",
                "recommended_action": [
                    "检查设备",
                    "记录状态"
                ],
                "confidence": 0.70,
                "used_clauses": []
            },
            "office_night": {
                "anomaly_type": "环境异常",
                "reason": "检测到环境异常",
                "recommended_action": [
                    "核实情况",
                    "记录异常"
                ],
                "confidence": 0.72,
                "used_clauses": []
            }
        }
    }
    
    return mock_responses

def run_mock_experiment():
    """运行模拟实验"""
    
    print("🚀 开始运行CE4Patrol完整实验模拟")
    print("=" * 60)
    
    # 导入必要的模块
    from ce4patrol.context_loader import ContextLoader
    from ce4patrol.prompt_generator import PromptGenerator
    from ce4patrol.evaluation.metrics import MetricsEvaluator
    from ce4patrol.evaluation.crs_calculator import CRSCalculator
    from ce4patrol.analysis.visualizer import ExperimentVisualizer
    from ce4patrol.analysis.case_analyzer import CaseAnalyzer
    
    # 加载数据
    logger.info("加载上下文数据和真实标签...")
    context_loader = ContextLoader()
    context_data = context_loader.load_context("data/context.json")
    
    with open("data/ground_truth.json", 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    # 创建模拟响应
    mock_responses = create_mock_vlm_responses()
    
    # 初始化组件
    prompt_generator = PromptGenerator()
    metrics_evaluator = MetricsEvaluator()
    crs_calculator = CRSCalculator()
    
    # 定义实验配置
    experiments = [
        {"name": "CE4_FULL", "layers": ["S", "R", "D"], "enable_cot": True},
        {"name": "NoCoT", "layers": ["S", "R", "D"], "enable_cot": False},
        {"name": "SR_only", "layers": ["S", "R"], "enable_cot": True},
        {"name": "S_only", "layers": ["S"], "enable_cot": True},
        {"name": "R_only", "layers": ["R"], "enable_cot": True},
        {"name": "D_only", "layers": ["D"], "enable_cot": True},
        {"name": "None", "layers": [], "enable_cot": True},
    ]
    
    all_results = []
    
    # 运行所有实验
    for exp_config in experiments:
        logger.info(f"开始实验: {exp_config['name']}")
        
        experiment_result = {
            "config": exp_config,
            "scene_results": []
        }
        
        # 处理每个场景
        for scene in context_data["scenes"]:
            scene_id = scene["scene_id"]
            image_path = f"data/images/{scene_id}.jpg"
            
            # 获取模拟的VLM响应
            vlm_response = mock_responses[exp_config["name"]][scene_id]
            
            # 获取真实标签
            gt_scene = ground_truth["scenes"].get(scene_id)
            if not gt_scene:
                logger.warning(f"没有找到场景 {scene_id} 的真实标签")
                continue
            
            # 计算指标
            metrics = metrics_evaluator.evaluate_scene(vlm_response, gt_scene)
            
            # 计算CRS
            crs = crs_calculator.compute_crs(
                accuracy=metrics["accuracy"],
                logic_consistency=metrics["logic_consistency"],
                action_reliability=metrics["action_reliability"],
                confidence=metrics["confidence"]
            )
            
            scene_result = {
                "scene_id": scene_id,
                "vlm_response": vlm_response,
                "ground_truth": gt_scene,
                "metrics": metrics,
                "crs": crs
            }
            
            experiment_result["scene_results"].append(scene_result)
            
            logger.info(f"场景 {scene_id} 完成 - CRS: {crs:.3f}")
        
        all_results.append(experiment_result)
        logger.info(f"实验 {exp_config['name']} 完成")
    
    # 保存实验结果
    output_path = "results/experiment_outputs.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"实验结果已保存到 {output_path}")
    
    # 生成可视化
    logger.info("生成可视化图表...")
    visualizer = ExperimentVisualizer()
    visualizer.create_confusion_matrix(all_results)
    visualizer.create_radar_chart(all_results)
    visualizer.create_heatmap(all_results)
    visualizer.create_crs_distribution(all_results)
    
    # 生成案例分析报告
    logger.info("生成案例分析报告...")
    case_analyzer = CaseAnalyzer()
    case_analyzer.analyze_cases(all_results)
    
    # 显示实验结果摘要
    print("\n" + "=" * 60)
    print("📊 实验结果摘要")
    print("=" * 60)
    
    for result in all_results:
        config_name = result["config"]["name"]
        scene_results = result["scene_results"]
        
        if scene_results:
            avg_crs = sum(scene["crs"] for scene in scene_results) / len(scene_results)
            avg_accuracy = sum(scene["metrics"]["accuracy"] for scene in scene_results) / len(scene_results)
            avg_logic = sum(scene["metrics"]["logic_consistency"] for scene in scene_results) / len(scene_results)
            avg_action = sum(scene["metrics"]["action_reliability"] for scene in scene_results) / len(scene_results)
            avg_confidence = sum(scene["metrics"]["confidence"] for scene in scene_results) / len(scene_results)
            
            print(f"\n🔬 {config_name}:")
            print(f"   CRS: {avg_crs:.3f}")
            print(f"   准确率: {avg_accuracy:.3f}")
            print(f"   逻辑一致性: {avg_logic:.3f}")
            print(f"   行动可靠性: {avg_action:.3f}")
            print(f"   置信度: {avg_confidence:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ 完整实验模拟完成！")
    print("📁 结果文件:")
    print("   - results/experiment_outputs.json (原始结果)")
    print("   - results/analysis_report.md (分析报告)")
    print("   - results/*.png (可视化图表)")
    print("=" * 60)

if __name__ == "__main__":
    run_mock_experiment()
