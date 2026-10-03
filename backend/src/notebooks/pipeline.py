"""End-to-end: SAM2 regions -> EfficientNet classes -> depth -> nutrition."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from .classification.classifier import FoodClassifier
from .depth.depth_anything import DepthAnythingEstimator
from .nutrition import NutritionFacts, NutritionLookup
from .portion import PortionEstimate, PortionEstimator
from .segment.sam2_segmenter import SAM2Segmenter

@dataclass(frozen=True)
class FoodAnalysis:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    portion: PortionEstimate
    nutrition: NutritionFacts

@dataclass(frozen=True)
class FoodPortion:
    label: str
    weight_g: float
    volume_ml: float
    calories_kcal: float
    protein_g: float
    carbohydrates_g: float
    fat_g: float
    confidence: float
    bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "weight_g": round(self.weight_g, 1),
            "volume_ml": round(self.volume_ml, 1),
            "calories_kcal": round(self.calories_kcal, 1),
            "protein_g": round(self.protein_g, 1),
            "carbohydrates_g": round(self.carbohydrates_g, 1),
            "fat_g": round(self.fat_g, 1),
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
        }

@dataclass(frozen=True)
class CalibrationInfo:
    quality: Literal["high", "medium", "low"]
    plate_diameter_mm: float
    mm_per_pixel: float
    mm_per_depth_unit: float
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "quality": self.quality,
            "plate_diameter_mm": round(self.plate_diameter_mm, 1),
            "mm_per_pixel": round(self.mm_per_pixel, 4),
            "mm_per_depth_unit": round(self.mm_per_depth_unit, 2),
            "warnings": self.warnings,
        }

@dataclass(frozen=True)
class AnalysisResult:
    items: list[FoodPortion]
    totals: NutritionFacts
    calibration: CalibrationInfo

    def to_dict(self) -> dict:
        return {
            "items": [item.to_dict() for item in self.items],
            "totals": {
                "calories_kcal": round(self.totals.calories_kcal, 1),
                "protein_g": round(self.totals.protein_g, 1),
                "carbohydrates_g": round(self.totals.carbohydrates_g, 1),
                "fat_g": round(self.totals.fat_g, 1),
            },
            "calibration": self.calibration.to_dict(),
        }

class NutritionPipeline:
    def __init__(self, segmenter: SAM2Segmenter, classifier: FoodClassifier,
                 depth_estimator: DepthAnythingEstimator, portion_estimator: PortionEstimator,
                 nutrition_lookup: NutritionLookup, densities_g_per_ml: dict[str, float],
                 default_density_g_per_ml: float = 1.0) -> None:
        self.segmenter, self.classifier, self.depth_estimator = segmenter, classifier, depth_estimator
        self.portion_estimator, self.nutrition_lookup = portion_estimator, nutrition_lookup
        self.densities, self.default_density = densities_g_per_ml, default_density_g_per_ml

    def analyze(self, image: np.ndarray, reference_depth: float | None = None) -> list[FoodAnalysis]:
        segmentation = self.segmenter.segment(image)
        if not segmentation.masks:
            return []
        predictions = self.classifier.classify([item.crop for item in segmentation.masks]).predictions
        depth = self.depth_estimator.estimate(image).depth_map
        analyses = []
        for region, prediction in zip(segmentation.masks, predictions):
            portion = self.portion_estimator.estimate(region.mask, depth,
                self.densities.get(prediction.label, self.default_density), reference_depth)
            analyses.append(FoodAnalysis(prediction.label, prediction.confidence, region.bbox, portion,
                self.nutrition_lookup.lookup(prediction.label, portion.weight_g)))
        return analyses

    def unload(self) -> None:
        for model in (self.segmenter, self.classifier, self.depth_estimator):
            model.unload()
