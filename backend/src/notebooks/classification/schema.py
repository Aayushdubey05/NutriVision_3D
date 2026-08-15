from dataclasses import dataclass
import numpy as np


@dataclass 
class FoodPrediction:
    label: str
    confidence: float
    class_index: float

@dataclass
class ClassificationResult:
    predictions: list[FoodPrediction]
    inference_time: float