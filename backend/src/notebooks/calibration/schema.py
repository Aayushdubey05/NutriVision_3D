from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..segment.schema import FoodMask


@dataclass(frozen=True)
class CalibrationResult:
    mm_per_pixel: float
    mm_per_depth_unit: float
    quality: Literal["high", "medium", "low"]
    plate_diameter_mm: float
    plate_mask: "FoodMask | None"
    confidence: float
    warnings: list[str]
    plane_fit_r2: float | None = None