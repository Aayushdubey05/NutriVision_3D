from __future__ import annotations
import logging
import numpy as np
import cv2
from typing import Literal, TYPE_CHECKING
from .schema import CalibrationResult
from ..config import PLATE_DETECTION, PLATE_DEFAULTS

if TYPE_CHECKING:
    from ..segment.schema import FoodMask

LOGGER = logging.getLogger(__name__)


class PlateDetector:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or PLATE_DETECTION
        self.min_circularity = self.config.get("min_circularity", 0.6)
        self.min_area_fraction = self.config.get("min_area_fraction", 0.15)
        self.max_area_fraction = self.config.get("max_area_fraction", 0.95)
        self.plate_thickness_mm = self.config.get("plate_thickness_mm", 3.0)
        self.min_confidence = self.config.get("min_confidence_threshold", 0.5)
        self.scoring_weights = self.config.get("scoring_weights", {
            "area": 0.4,
            "circularity": 0.3,
            "centered": 0.2,
            "border_touch": 0.1,
        })

    def _calculate_circularity(self, mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        return (4 * np.pi * area) / (perimeter ** 2)

    def _calculate_centeredness(self, mask: np.ndarray, image_shape: tuple) -> float:
        h, w = image_shape[:2]
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] == 0:
            return 0.0
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        dist_from_center = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
        max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
        return 1.0 - (dist_from_center / max_dist)

    def _calculate_border_touch(self, mask: np.ndarray, image_shape: tuple) -> float:
        h, w = image_shape[:2]
        border_pixels = 5
        top = mask[:border_pixels, :].any()
        bottom = mask[-border_pixels:, :].any()
        left = mask[:, :border_pixels].any()
        right = mask[:, -border_pixels:].any()
        touches = sum([top, bottom, left, right])
        return touches / 4.0

    def _calculate_area_score(self, mask: np.ndarray, image_shape: tuple) -> float:
        h, w = image_shape[:2]
        image_area = h * w
        mask_area = mask.sum()
        area_fraction = mask_area / image_area
        if area_fraction < self.min_area_fraction or area_fraction > self.max_area_fraction:
            return 0.0
        return min(area_fraction / self.max_area_fraction, 1.0)

    def select_plate_mask(self, masks: list["FoodMask"], image_shape: tuple) -> tuple["FoodMask | None", float]:
        if not masks:
            return None, 0.0

        best_mask = None
        best_score = 0.0

        for food_mask in masks:
            mask = food_mask.mask.astype(bool)
            if mask.sum() == 0:
                continue

            circularity = self._calculate_circularity(mask)
            if circularity < self.min_circularity:
                continue

            area_score = self._calculate_area_score(mask, image_shape)
            if area_score == 0.0:
                continue

            centeredness = self._calculate_centeredness(mask, image_shape)
            border_touch = self._calculate_border_touch(mask, image_shape)

            score = (
                self.scoring_weights["area"] * area_score +
                self.scoring_weights["circularity"] * circularity +
                self.scoring_weights["centered"] * centeredness +
                self.scoring_weights["border_touch"] * border_touch
            )

            if score > best_score:
                best_score = score
                best_mask = food_mask

        if best_mask is None or best_score < self.min_confidence:
            return None, 0.0

        return best_mask, best_score

    def estimate_diameter_px(self, plate_mask: "FoodMask") -> float:
        mask = plate_mask.mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            x, y, w, h = plate_mask.bbox
            return max(w, h)
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            x, y, w, h = plate_mask.bbox
            return max(w, h)
        (_, _), (ma, MA), _ = cv2.fitEllipse(contour)
        return max(ma, MA)

    def estimate_mm_per_pixel(self, plate_mask: "FoodMask", plate_diameter_mm: float | None = None) -> float:
        if plate_diameter_mm is None:
            plate_diameter_mm = PLATE_DEFAULTS.get("standard_thali", 280.0)
        diameter_px = self.estimate_diameter_px(plate_mask)
        if diameter_px <= 0:
            return 0.0
        return plate_diameter_mm / diameter_px

    def estimate_mm_per_depth_unit(self, plate_mask: "FoodMask", depth_map: np.ndarray) -> tuple[float, float | None]:
        """
        Estimate mm per depth unit using plate-to-background depth difference.
        
        For top-down view, the plate region and background (table) should be at 
        different depths. The difference in median depth represents the physical
        height difference (plate thickness + any food).
        
        Falls back to configurable default if plate/background separation unclear.
        """
        mask_bool = plate_mask.mask.astype(bool)
        plate_depths = depth_map[mask_bool]
        if plate_depths.size == 0:
            return 100.0, None

        # Background = everything not in plate mask
        bg_mask = ~mask_bool
        if not bg_mask.any():
            return 100.0, None
        bg_depths = depth_map[bg_mask]
        
        plate_median = float(np.median(plate_depths))
        bg_median = float(np.median(bg_depths))
        
        # Depth difference (in relative depth units)
        depth_diff = abs(plate_median - bg_median)
        
        if depth_diff <= 1e-6:
            # Plate and background at same depth - cannot determine scale
            return 100.0, 0.0
        
        # Assume typical plate+food height: 20-30mm for Indian thali with food
        # This is a rough estimate; user can override via config
        assumed_height_mm = self.config.get("assumed_plate_food_height_mm", 25.0)
        mm_per_depth_unit = assumed_height_mm / depth_diff
        
        # Quality metric: how well separated are plate and background depths?
        plate_std = float(np.std(plate_depths))
        bg_std = float(np.std(bg_depths))
        separation = depth_diff / max(plate_std, bg_std, 1e-6)
        r2 = min(separation / 10.0, 1.0)  # Normalize: separation > 10σ = perfect
        
        return mm_per_depth_unit, r2

    def calibrate(
        self,
        masks: list["FoodMask"],
        depth_map: np.ndarray,
        image_shape: tuple,
        plate_diameter_mm: float | None = None
    ) -> CalibrationResult:
        warnings = []
        plate_mask, confidence = self.select_plate_mask(masks, image_shape)

        if plate_mask is None:
            warnings.append("No plate detected; using default 280mm thali calibration")
            mm_per_pixel = (plate_diameter_mm or PLATE_DEFAULTS["standard_thali"]) / (image_shape[1] * 0.8)
            mm_per_depth_unit = 100.0
            quality: Literal["high", "medium", "low"] = "low"
            return CalibrationResult(
                mm_per_pixel=mm_per_pixel,
                mm_per_depth_unit=mm_per_depth_unit,
                quality=quality,
                plate_diameter_mm=plate_diameter_mm or PLATE_DEFAULTS["standard_thali"],
                plate_mask=None,
                confidence=0.0,
                warnings=warnings,
                plane_fit_r2=None
            )

        mm_per_pixel = self.estimate_mm_per_pixel(plate_mask, plate_diameter_mm)
        mm_per_depth_unit, r2 = self.estimate_mm_per_depth_unit(plate_mask, depth_map)

        if mm_per_pixel <= 0 or mm_per_depth_unit <= 0:
            warnings.append("Invalid calibration values; using defaults")
            mm_per_pixel = (plate_diameter_mm or PLATE_DEFAULTS["standard_thali"]) / (image_shape[1] * 0.8)
            mm_per_depth_unit = 100.0
            quality = "low"
        elif confidence >= 0.7 and r2 is not None and r2 >= 0.8:
            quality = "high"
        else:
            quality = "medium"
            if r2 is not None and r2 < 0.8:
                warnings.append(f"Weak depth plane fit (R²={r2:.2f})")
            if confidence < 0.7:
                warnings.append(f"Low plate detection confidence ({confidence:.2f})")

        return CalibrationResult(
            mm_per_pixel=mm_per_pixel,
            mm_per_depth_unit=mm_per_depth_unit,
            quality=quality,
            plate_diameter_mm=plate_diameter_mm or PLATE_DEFAULTS["standard_thali"],
            plate_mask=plate_mask,
            confidence=confidence,
            warnings=warnings,
            plane_fit_r2=r2
        )