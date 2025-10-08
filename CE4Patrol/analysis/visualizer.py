"""
Visualization module for CE4Patrol experiment results.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd


class ExperimentVisualizer:
    """实验可视化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')
    
    def create_confusion_matrix(self, all_results: List[Dict[str, Any]]) -> None:
        """
        创建混淆矩阵
        
        Args:
            all_results: 所有实验结果
        """
        try:
            # 收集所有预测和真实标签
            predictions = []
            true_labels = []
            
            for experiment in all_results:
                for scene_result in experiment.get("scene_results", []):
                    vlm_response = scene_result.get("vlm_response", {})
                    ground_truth = scene_result.get("ground_truth", {})
                    
                    pred_type = vlm_response.get("anomaly_type", "未知")
                    true_type = ground_truth.get("anomaly_type", "未知")
                    
                    predictions.append(pred_type)
                    true_labels.append(true_type)
            
            if not predictions:
                self.logger.warning("No data for confusion matrix")
                return
            
            # 创建混淆矩阵
            from sklearn.metrics import confusion_matrix
            
            # 获取所有唯一标签
            all_labels = sorted(list(set(predictions + true_labels)))
            
            cm = confusion_matrix(true_labels, predictions, labels=all_labels)
            
            # 绘制混淆矩阵
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=all_labels, yticklabels=all_labels)
            plt.title('混淆矩阵 - 异常类型分类', fontsize=16, fontweight='bold')
            plt.xlabel('预测标签', fontsize=12)
            plt.ylabel('真实标签', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # 保存图像
            output_path = "results/confusion_matrix.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confusion matrix saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating confusion matrix: {e}")
    
    def create_radar_chart(self, all_results: List[Dict[str, Any]]) -> None:
        """
        创建雷达图显示各项指标
        
        Args:
            all_results: 所有实验结果
        """
        try:
            # 计算每个实验的平均指标
            experiment_metrics = {}
            
            for experiment in all_results:
                config_name = experiment["config"]["name"]
                scene_results = experiment.get("scene_results", [])
                
                if not scene_results:
                    continue
                
                # 计算平均指标
                metrics = [scene["metrics"] for scene in scene_results]
                avg_metrics = {
                    "accuracy": np.mean([m.get("accuracy", 0) for m in metrics]),
                    "logic_consistency": np.mean([m.get("logic_consistency", 0) for m in metrics]),
                    "action_reliability": np.mean([m.get("action_reliability", 0) for m in metrics]),
                    "confidence": np.mean([m.get("confidence", 0) for m in metrics]),
                    "crs": np.mean([scene.get("crs", 0) for scene in scene_results])
                }
                
                experiment_metrics[config_name] = avg_metrics
            
            if not experiment_metrics:
                self.logger.warning("No data for radar chart")
                return
            
            # 创建雷达图
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
            
            # 定义指标标签
            metrics_labels = ["准确率", "逻辑一致性", "行动可靠性", "置信度", "CRS"]
            metrics_keys = ["accuracy", "logic_consistency", "action_reliability", "confidence", "crs"]
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            # 为每个实验绘制雷达图
            colors = plt.cm.Set3(np.linspace(0, 1, len(experiment_metrics)))
            
            for i, (exp_name, metrics) in enumerate(experiment_metrics.items()):
                values = [metrics[key] for key in metrics_keys]
                values += values[:1]  # 闭合图形
                
                ax.plot(angles, values, 'o-', linewidth=2, label=exp_name, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            # 设置标签和网格
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_labels)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.grid(True)
            
            # 添加图例和标题
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            plt.title('实验指标雷达图对比', fontsize=16, fontweight='bold', pad=20)
            
            # 保存图像
            output_path = "results/radar_chart.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Radar chart saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating radar chart: {e}")
    
    def create_heatmap(self, all_results: List[Dict[str, Any]]) -> None:
        """
        创建实验×指标热力图
        
        Args:
            all_results: 所有实验结果
        """
        try:
            # 准备数据
            experiment_names = []
            metrics_data = []
            
            for experiment in all_results:
                config_name = experiment["config"]["name"]
                scene_results = experiment.get("scene_results", [])
                
                if not scene_results:
                    continue
                
                experiment_names.append(config_name)
                
                # 计算平均指标
                metrics = [scene["metrics"] for scene in scene_results]
                avg_metrics = [
                    np.mean([m.get("accuracy", 0) for m in metrics]),
                    np.mean([m.get("logic_consistency", 0) for m in metrics]),
                    np.mean([m.get("action_reliability", 0) for m in metrics]),
                    np.mean([m.get("confidence", 0) for m in metrics]),
                    np.mean([scene.get("crs", 0) for scene in scene_results])
                ]
                
                metrics_data.append(avg_metrics)
            
            if not metrics_data:
                self.logger.warning("No data for heatmap")
                return
            
            # 创建DataFrame
            metrics_labels = ["准确率", "逻辑一致性", "行动可靠性", "置信度", "CRS"]
            df = pd.DataFrame(metrics_data, index=experiment_names, columns=metrics_labels)
            
            # 创建热力图
            plt.figure(figsize=(12, 8))
            sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd',
                       cbar_kws={'label': '分数'})
            plt.title('实验配置 × 指标热力图', fontsize=16, fontweight='bold')
            plt.xlabel('指标', fontsize=12)
            plt.ylabel('实验配置', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # 保存图像
            output_path = "results/metrics_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Metrics heatmap saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating heatmap: {e}")
    
    def create_crs_distribution(self, all_results: List[Dict[str, Any]]) -> None:
        """
        创建CRS分数分布图
        
        Args:
            all_results: 所有实验结果
        """
        try:
            # 收集CRS分数
            crs_data = {}
            
            for experiment in all_results:
                config_name = experiment["config"]["name"]
                scene_results = experiment.get("scene_results", [])
                
                crs_scores = [scene.get("crs", 0) for scene in scene_results]
                if crs_scores:
                    crs_data[config_name] = crs_scores
            
            if not crs_data:
                self.logger.warning("No CRS data for distribution plot")
                return
            
            # 创建箱线图
            plt.figure(figsize=(12, 8))
            
            # 准备数据
            labels = list(crs_data.keys())
            data = list(crs_data.values())
            
            # 绘制箱线图
            box_plot = plt.boxplot(data, labels=labels, patch_artist=True)
            
            # 设置颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.title('CRS分数分布对比', fontsize=16, fontweight='bold')
            plt.xlabel('实验配置', fontsize=12)
            plt.ylabel('CRS分数', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图像
            output_path = "results/crs_distribution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"CRS distribution plot saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating CRS distribution plot: {e}")
