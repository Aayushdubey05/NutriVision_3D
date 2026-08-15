from abc import ABC, abstractmethod
import numpy as np
from .schema import ClassificationResult

class BaseClassifier(ABC):
    @abstractmethod
    def classify(
        self,
        crops: list[np.ndarray],
    ) -> ClassificationResult:
        pass

    @abstractmethod
    def unload(self):
        pass
