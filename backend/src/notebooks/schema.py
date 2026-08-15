from dataclasses import dataclass
import numpy as np 

@dataclass
class FoodInstance:
    mask: np.ndarray
    bbox: list[int]
    crop: np.ndarray
    area: int
    label: str|None = None
    confidence: float| None= None
    depth_map: np.ndarray | None= None
    volume: float| None= None
    weight: float| None = None
    nutrition: dict| None = None

