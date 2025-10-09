"""
Pydantic models for ensuring data integrity throughout the CE4Patrol project.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- 1. SpatiotemporalContext: 升级为语义化结构 ---
class SpatiotemporalContext(BaseModel):
    """时空上下文，已升级为语义化信息"""
    location: str = Field(..., description="具有语义的地理位置，例如 'Server Room B' 或 '5号仓库门口'")
    timestamp: str = Field(..., description="事件发生的精确时间戳, e.g., '2024-05-21 23:15:00'")

class SceneInfo(BaseModel):
    """场景元信息"""
    description: str = Field(..., description="对场景的宏观描述")

class RuleClause(BaseModel):
    """单条规则条款"""
    clause_id: str
    clause: str
    weight: float = Field(0.5, ge=0, le=1)

class RulesContext(BaseModel):
    """规则上下文"""
    manual_refs: List[str]
    logic_clauses: List[RuleClause]

# --- 2. DecisionContext: 增加视觉参考 ---
class DecisionContext(BaseModel):
    """决策辅助上下文，增加正常状态的视觉参考"""
    normal_text_refs: List[str] = Field(..., description="对 '正常' 状态的文本描述")
    normal_image_refs: Optional[List[str]] = Field(default=[], description="指向 '正常' 状态参考图片的路径 (新)")
    playbook: Dict[str, str] = Field(..., description="按风险等级划分的行动预案")
    semantic_ambiguity: Optional[Dict[str, Any]] = None

class Context(BaseModel):
    """完整的上下文信息"""
    scene_info: SceneInfo
    spatiotemporal: SpatiotemporalContext
    rules: RulesContext
    decision: DecisionContext

class GroundTruth(BaseModel):
    """标准答案"""
    is_anomaly: bool
    anomaly_type: List[str]
    reason: List[str]
    used_clauses: List[str]
    recommended_action: List[str]

class PatrolCase(BaseModel):
    """单个巡检案例"""
    case_id: str
    image_path: str
    description: str
    context: Context
    ground_truth: GroundTruth

class DataSet(BaseModel):
    """完整的数据集"""
    dataset_name: str
    version: str
    cases: List[PatrolCase]