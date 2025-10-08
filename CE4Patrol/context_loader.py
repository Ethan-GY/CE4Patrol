"""
Context loader module for loading and validating the case-centric dataset.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# --- Pydantic Models for Validation ---

class SceneInfo(BaseModel):
    description: str

class SpatiotemporalContext(BaseModel):
    timezone: str
    time_hint: str
    gps: Dict[str, float]

class LogicClause(BaseModel):
    clause_id: str
    logic: str
    weight: float = Field(..., ge=0.0, le=1.0)

class RulesContext(BaseModel):
    manual_refs: List[str]
    logic_clauses: List[LogicClause]

class RiskLevel(BaseModel):
    high: Optional[List[str]] = None
    medium: Optional[List[str]] = None
    low: Optional[List[str]] = None

class Playbook(BaseModel):
    risk_levels: RiskLevel

class DecisionContext(BaseModel):
    normal_refs: List[str]
    playbook: Playbook

class SmallObject(BaseModel):
    name: str
    risk: str

class SemanticAmbiguity(BaseModel):
    small_objects: Optional[List[SmallObject]] = None

class ContextData(BaseModel):
    scene_info: SceneInfo
    spatiotemporal: SpatiotemporalContext
    rules: RulesContext
    decision: DecisionContext
    semantic_ambiguity: SemanticAmbiguity

class GroundTruth(BaseModel):
    is_anomaly: bool
    anomaly_type: str
    reason: str
    recommended_action: List[str]
    used_clauses: List[str]

class Case(BaseModel):
    case_id: str
    image_path: str
    context: ContextData
    ground_truth: GroundTruth

class DataSet(BaseModel):
    cases: List[Case]
    metadata: Dict[str, Any]

class ContextLoader:
    """上下文与数据集加载器"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        加载并验证完整的 case-centric 数据集
        Args:
            dataset_path: 数据集文件路径 (e.g., 'data/dataset.json')
        Returns:
            验证后的数据集字典
        """
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            self.logger.error(f"Dataset file not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            validated_data = DataSet(**raw_data)
            self.logger.info(f"Successfully loaded and validated dataset with {len(validated_data.cases)} cases.")
            return validated_data.dict()
        except Exception as e:
            self.logger.error(f"Error loading or validating dataset: {e}", exc_info=True)
            raise