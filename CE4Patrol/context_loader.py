"""
Context loader module for loading and validating the case-centric dataset.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError
 
# --- Pydantic Models for Validation ---
from .data_models import DataSet 

class ContextLoader:
    """上下文与数据集加载器"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, dataset_path: str) -> DataSet:
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