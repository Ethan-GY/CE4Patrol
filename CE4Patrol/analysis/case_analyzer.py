"""
Case analysis module for CE4Patrol to automatically categorize errors.
"""
import logging
from typing import Dict, Any, List
from pathlib import Path

class CaseAnalyzer:
    """案例分析器，自动筛选低分样本并按错误类型分类"""
    def __init__(self, low_crs_threshold: float = 0.6):
        self.logger = logging.getLogger(__name__)
        self.threshold = low_crs_threshold

    def _categorize_error(self, scene_result: Dict[str, Any]) -> List[str]:
        """为单个案例分析错误类型"""
        errors = []
        metrics = scene_result['metrics']
        pred_is_anomaly = scene_result['vlm_response'].get("anomaly_type", "无异常") != "无异常"
        true_is_anomaly = scene_result['ground_truth'].get("is_anomaly", False)

        # 1. 分类错误 (最严重的错误)
        if pred_is_anomaly != true_is_anomaly:
            error_type = "漏报 (False Negative)" if true_is_anomaly else "误报 (False Positive)"
            errors.append(f"[分类错误] {error_type}")

        # 2. 逻辑断裂
        if metrics.get('logic_consistency', 1.0) < 0.5:
            errors.append("[逻辑断裂] VLM未能引用正确的决策依据子句")

        # 3. 行动失当
        if metrics.get('action_reliability', 1.0) < 0.6:
            errors.append("[行动失当] 推荐的行动与标准预案语义偏差较大")

        # 4. 信心误判
        confidence = metrics.get('confidence', 0.5)
        if pred_is_anomaly == true_is_anomaly and confidence < 0.5:
            errors.append("[信心不足] 判断正确，但置信度过低")
        if pred_is_anomaly != true_is_anomaly and confidence > 0.7:
            errors.append("[过度自信] 判断错误，但置信度过高")

        return errors if errors else ["综合表现不佳"]

    def analyze_cases(self, all_results: List[Dict[str, Any]]):
        report_path = Path("results/analysis_report.md")
        report_content = ["# CE4Patrol 案例分析报告\n\n"]
        
        self.logger.info(f"Analyzing cases with CRS threshold < {self.threshold}")

        for experiment in all_results:
            config_name = experiment["config"]["name"]
            report_content.append(f"## 实验配置: {config_name}\n\n")
            
            low_crs_cases = [
                res for res in experiment.get("scene_results", [])
                if res.get("crs", 1.0) < self.threshold
            ]

            if not low_crs_cases:
                report_content.append("此配置下所有案例的CRS分数均高于阈值，表现良好。\n\n")
                continue

            report_content.append(f"发现 {len(low_crs_cases)} 个低分案例:\n\n")
            
            for case in sorted(low_crs_cases, key=lambda x: x['crs']):
                case_id = case['scene_id']
                crs_score = case['crs']
                errors = self._categorize_error(case)
                
                report_content.append(f"### 案例: `{case_id}` (CRS: {crs_score:.3f})\n")
                report_content.append(f"**错误类型**: {', '.join(errors)}\n\n")
                
                report_content.append("**VLM 响应:**\n```json\n" + json.dumps(case['vlm_response'], ensure_ascii=False, indent=2) + "\n```\n")
                report_content.append("**真实标签 (Ground Truth):**\n```json\n" + json.dumps(case['ground_truth'], ensure_ascii=False, indent=2) + "\n```\n")
                report_content.append("---\n")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_content))
        
        self.logger.info(f"Case analysis report saved to {report_path}")