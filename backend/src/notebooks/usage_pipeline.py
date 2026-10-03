"""Example entry point for a calibrated food-image analysis request."""
from __future__ import annotations
import cv2
import numpy as np
import threading
from enum import Enum
from .calibration import PlateDetector
from .classification import FoodClassifier
from .config import CLASSIFICATION_CONFIG, FOOD_DENSITY_DB, PLATE_DETECTION
from .depth import DepthAnythingEstimator
from .nutrition import NutritionFacts, NutritionLookup
from .pipeline import AnalysisResult, CalibrationInfo, FoodPortion, NutritionPipeline
from .portion import PortionEstimator
from .segment import SAM2Segmenter

# Replace these sample records with USDA/IFCT values for every model class.
NUTRITION_PER_100G = {
    "default": {"calories_kcal": 150.0, "protein_g": 5.0, "carbohydrates_g": 20.0, "fat_g": 5.0},
    "aloo_group": {"calories_kcal": 130.0, "protein_g": 3.5, "carbohydrates_g": 18.0, "fat_g": 6.0},
    "chicken_group": {"calories_kcal": 220.0, "protein_g": 18.0, "carbohydrates_g": 8.0, "fat_g": 14.0},
    "sweet_white_group": {"calories_kcal": 180.0, "protein_g": 4.0, "carbohydrates_g": 30.0, "fat_g": 5.0},
    "sweet_brown_group": {"calories_kcal": 300.0, "protein_g": 3.0, "carbohydrates_g": 40.0, "fat_g": 15.0},
    "bread_group": {"calories_kcal": 250.0, "protein_g": 8.0, "carbohydrates_g": 50.0, "fat_g": 3.0},
    "dal_group": {"calories_kcal": 160.0, "protein_g": 9.0, "carbohydrates_g": 20.0, "fat_g": 5.0},
    "rice_group": {"calories_kcal": 180.0, "protein_g": 4.0, "carbohydrates_g": 38.0, "fat_g": 4.0},
}

_ZERO_NUTRITION = NutritionFacts(0.0, 0.0, 0.0, 0.0)

# Classification config
CONFIDENCE_THRESHOLD = CLASSIFICATION_CONFIG["confidence_threshold"]
TOP_K = CLASSIFICATION_CONFIG["top_k"]
MERGE_TARGETS = CLASSIFICATION_CONFIG["merge_targets"]

# Mode enum
class AnalysisMode(Enum):
    AUTO = "auto"
    SINGLE = "single"
    THALI = "thali"

# Global singleton for model caching
class _ModelCache:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._lock = threading.Lock()
        self._models = None
        self._pipelines = {}
        self._initialized = True
    
    def get_models(self):
        with self._lock:
            if self._models is None:
                self._models = {
                    "segmenter": SAM2Segmenter(),
                    "classifier": FoodClassifier(
                        confidence_threshold=CONFIDENCE_THRESHOLD,
                        top_k=TOP_K,
                    ),
                    "depth_estimator": DepthAnythingEstimator(),
                }
        return self._models
    
    def get_pipeline(self, calibration):
        cache_key = (round(calibration.mm_per_pixel, 6), round(calibration.mm_per_depth_unit, 2))
        with self._lock:
            if cache_key not in self._pipelines:
                models = self.get_models()
                portion_estimator = PortionEstimator(
                    calibration.mm_per_pixel, calibration.mm_per_depth_unit
                )
                nutrition_lookup = NutritionLookup(NUTRITION_PER_100G, default_key="default")
                self._pipelines[cache_key] = NutritionPipeline(
                    models["segmenter"], models["classifier"], models["depth_estimator"],
                    PortionEstimator(calibration.mm_per_pixel, calibration.mm_per_depth_unit),
                    nutrition_lookup, FOOD_DENSITY_DB,
                )
            return self._pipelines[cache_key]
    
    def clear(self):
        with self._lock:
            for p in self._pipelines.values():
                p.unload()
            if self._models:
                for m in self._models.values():
                    if hasattr(m, 'unload'):
                        m.unload()
            self._models = None
            self._pipelines = {}

_model_cache = _ModelCache()


def _apply_class_merge(label: str, topk_preds: list, is_uncertain: bool) -> tuple[str, float, list]:
    if is_uncertain:
        alternatives = [(p.label, p.confidence) for p in topk_preds[:3]]
        return "uncertain", topk_preds[0].confidence, alternatives
    merged_label = MERGE_TARGETS.get(label, label)
    return merged_label, topk_preds[0].confidence, []


def _convert_to_food_portion(analysis, topk_pred) -> FoodPortion:
    p = analysis.portion
    n = analysis.nutrition
    final_label, final_conf, _ = _apply_class_merge(
        analysis.label, topk_pred.predictions, topk_pred.is_uncertain
    )
    return FoodPortion(
        label=final_label,
        weight_g=p.weight_g,
        volume_ml=p.volume_ml,
        calories_kcal=n.calories_kcal,
        protein_g=n.protein_g,
        carbohydrates_g=n.carbohydrates_g,
        fat_g=n.fat_g,
        confidence=final_conf,
        bbox=analysis.bbox,
    )


def _compute_totals(items: list[FoodPortion]) -> NutritionFacts:
    return NutritionFacts(
        sum(i.calories_kcal for i in items),
        sum(i.protein_g for i in items),
        sum(i.carbohydrates_g for i in items),
        sum(i.fat_g for i in items),
    )


def analyze_image(
    image_path: str,
    plate_diameter_mm: float | None = None,
    mode: AnalysisMode | str = AnalysisMode.AUTO,
) -> AnalysisResult:
    """
    Analyze a food image with automatic plate detection and calibration.
    
    Args:
        image_path: Path to the image file.
        plate_diameter_mm: Optional plate diameter in mm. Defaults to standard thali (280mm).
        mode: AnalysisMode.AUTO (detect plate), AnalysisMode.SINGLE (skip plate), 
              AnalysisMode.THALI (require plate)
    
    Returns:
        AnalysisResult with food items, totals, and calibration info.
    """
    if isinstance(mode, str):
        mode = AnalysisMode(mode)
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 1. Segment ONCE
    models = _ModelCache().get_models()
    segmenter = models["segmenter"]
    segmentation = segmenter.segment(image)

    if not segmentation.masks:
        return AnalysisResult(
            items=[],
            totals=NutritionFacts(0,0,0,0),
            calibration=CalibrationInfo(
                quality="low",
                plate_diameter_mm=plate_diameter_mm or 280.0,
                mm_per_pixel=0.0,
                mm_per_depth_unit=0.0,
                warnings=["No food detected in image"],
            ),
        )

    # 2. Depth estimation
    depth_estimator = models["depth_estimator"]
    depth_result = depth_estimator.estimate(image)

    # 3. Calibration based on mode
    if mode == AnalysisMode.SINGLE:
        # Single food: no plate needed, use defaults
        h, w = image.shape[:2]
        calibration = type('Calibration', (), {
            'quality': 'low',
            'plate_diameter_mm': plate_diameter_mm or 280.0,
            'mm_per_pixel': (plate_diameter_mm or 280.0) / (w * 0.8),
            'mm_per_depth_unit': 100.0,
            'warnings': ['Single-food mode: using default calibration'],
        })()
    else:
        # Thali or auto mode: try plate detection
        plate_detector = PlateDetector(PLATE_DETECTION)
        calibration = plate_detector.calibrate(
            segmentation.masks,
            depth_result.depth_map,
            image.shape,
            plate_diameter_mm,
        )
        
        if mode == AnalysisMode.THALI and calibration.quality == "low":
            return AnalysisResult(
                items=[],
                totals=NutritionFacts(0,0,0,0),
                calibration=CalibrationInfo(
                    quality="low",
                    plate_diameter_mm=plate_diameter_mm or 280.0,
                    mm_per_pixel=0.0,
                    mm_per_depth_unit=0.0,
                    warnings=["THALI mode: No plate detected"],
                ),
            )
    
    # 4. Classification on crops from INITIAL segmentation
    crops = [item.crop for item in segmentation.masks]
    classifier = _ModelCache().get_models()["classifier"]
    classification_result = classifier.classify(crops)
    topk_results = classification_result.topk_predictions

    # 5. Get cached pipeline for this calibration
    pipeline = _ModelCache().get_pipeline(calibration)
    # Override pipeline's segmenter to use our masks
    pipeline.segmenter = models["segmenter"]

    try:
        analyses = pipeline.analyze(image)
    except Exception as e:
        _ModelCache().clear()
        raise

    # 6. Convert with merge + uncertainty
    items = []
    for i, analysis in enumerate(analyses):
        topk_pred = topk_results[i] if i < len(topk_results) else None
        if topk_pred:
            items.append(_convert_to_food_portion(analysis, topk_pred))
        else:
            p, n = analysis.portion, analysis.nutrition
            items.append(FoodPortion(
                label=analysis.label, weight_g=p.weight_g, volume_ml=p.volume_ml,
                calories_kcal=n.calories_kcal, protein_g=n.protein_g,
                carbohydrates_g=n.carbohydrates_g, fat_g=n.fat_g,
                confidence=analysis.confidence, bbox=analysis.bbox,
            ))

    totals = NutritionFacts(
        sum(i.calories_kcal for i in items),
        sum(i.protein_g for i in items),
        sum(i.carbohydrates_g for i in items),
        sum(i.fat_g for i in items),
    )

    return AnalysisResult(
        items=items,
        totals=totals,
        calibration=CalibrationInfo(
            quality=calibration.quality,
            plate_diameter_mm=calibration.plate_diameter_mm,
            mm_per_pixel=calibration.mm_per_pixel,
            mm_per_depth_unit=calibration.mm_per_depth_unit,
            warnings=calibration.warnings,
        ),
    )