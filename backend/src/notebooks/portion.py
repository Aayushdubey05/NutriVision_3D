"""Convert a segmented food region and a calibrated depth map into volume."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PortionEstimate:
    volume_ml: float
    weight_g: float
    footprint_area_cm2: float
    mean_height_cm: float

class PortionEstimator:
    """Estimate volume from a top-down, calibrated depth image.

    Depth Anything supplies relative depth. `mm_per_depth_unit` must be
    calibrated for the capture setup; `mm_per_pixel` comes from a known plate
    diameter or reference object. Without both, grams are not meaningful.
    """
    def __init__(self, mm_per_pixel: float, mm_per_depth_unit: float) -> None:
        if mm_per_pixel <= 0 or mm_per_depth_unit <= 0:
            raise ValueError("Calibration values must be positive.")
        self.mm_per_pixel = float(mm_per_pixel)
        self.mm_per_depth_unit = float(mm_per_depth_unit)

    def estimate(self, mask: np.ndarray, depth_map: np.ndarray, density_g_per_ml: float,
                 reference_depth: float | None = None, food_is_closer_when: str = "larger") -> PortionEstimate:
        if mask.shape != depth_map.shape:
            raise ValueError("Mask and depth map must have identical dimensions.")
        if density_g_per_ml <= 0:
            raise ValueError("Food density must be positive.")
        pixels = mask.astype(bool)
        if not np.any(pixels):
            raise ValueError("Mask has no foreground pixels.")
        if reference_depth is None:
            background = depth_map[~pixels]
            reference_depth = float(np.median(background)) if background.size else float(np.min(depth_map))
        signed_height = depth_map - reference_depth
        if food_is_closer_when == "smaller":
            signed_height = -signed_height
        elif food_is_closer_when != "larger":
            raise ValueError("food_is_closer_when must be 'larger' or 'smaller'.")
        height_mm = np.maximum(signed_height[pixels], 0) * self.mm_per_depth_unit
        pixel_area_mm2 = self.mm_per_pixel ** 2
        volume_ml = float(np.sum(height_mm) * pixel_area_mm2 / 1000.0)
        return PortionEstimate(volume_ml, volume_ml * density_g_per_ml,
            float(np.count_nonzero(pixels) * pixel_area_mm2 / 100.0), float(np.mean(height_mm) / 10.0))
