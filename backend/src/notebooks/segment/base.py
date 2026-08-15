import numpy as np 
from abc import ABC, abstractmethod
from .schema import SegmentationResult


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray) -> SegmentationResult:
        pass



















