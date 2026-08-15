from dataclasses import dataclass
from typing import Tuple
import numpy as np



@dataclass
class FoodMask:
    mask: np.ndarray
    bbox: tuple[int, int,int,int]
    crop: np.ndarray
    area:int
    score: float
    confidence: float


@dataclass
class SegmentationResult:
    masks: list[FoodMask]
    image_height: int
    image_width: int
    inference_time: float
