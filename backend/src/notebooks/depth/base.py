from abc import ABC, abstractmethod
import numpy as np 

from schema import DepthResult

class BaseDepthEstimator(ABC):

    @abstractmethod
    def estimate(self, image: np.ndarray) -> DepthResult:
        pass
