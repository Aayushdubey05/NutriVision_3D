from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FoodPrediction:
    label: str
    confidence: float
    class_index: int


@dataclass
class TopKPrediction:
    """Top-K predictions for a single crop."""
    predictions: list[FoodPrediction]  # Sorted by confidence descending
    top1: FoodPrediction
    is_uncertain: bool  # True if top1 confidence < threshold
    threshold_used: float

    @property
    def label(self) -> str:
        """Return top-1 label or 'uncertain' if below threshold."""
        return self.top1.label if not self.is_uncertain else "uncertain"

    @property
    def confidence(self) -> float:
        return self.top1.confidence


@dataclass
class ClassificationResult:
    predictions: list[FoodPrediction]  # Backward compatible: top-1 only
    topk_predictions: list[TopKPrediction]  # New: top-K for each crop
    inference_time: float
