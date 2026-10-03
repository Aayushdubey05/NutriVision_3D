"""
FastAPI service for NutriVision food analysis.
Loads models on startup and keeps them cached in memory.
"""
from __future__ import annotations
import os
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path for imports - MUST BE FIRST
import sys
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from notebooks.calibration import PlateDetector
from notebooks.classification import FoodClassifier
from notebooks.config import CLASSIFICATION_CONFIG, FOOD_DENSITY_DB, PLATE_DETECTION
from notebooks.depth import DepthAnythingEstimator
from notebooks.nutrition import NutritionFacts, NutritionLookup
from notebooks.pipeline import AnalysisResult, CalibrationInfo, FoodPortion, NutritionPipeline
from notebooks.portion import PortionEstimator
from notebooks.segment import SAM2Segmenter

# Nutrition database
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

# Config
CONFIDENCE_THRESHOLD = CLASSIFICATION_CONFIG["confidence_threshold"]
TOP_K = CLASSIFICATION_CONFIG["top_k"]
MERGE_TARGETS = CLASSIFICATION_CONFIG["merge_targets"]


class AnalysisMode(str, Enum):
    AUTO = "auto"
    SINGLE = "single"
    THALI = "thali"


class FoodItemResponse(BaseModel):
    label: str
    weight_g: float
    volume_ml: float
    calories_kcal: float
    protein_g: float
    carbohydrates_g: float
    fat_g: float
    confidence: float
    bbox: list[int]


class CalibrationResponse(BaseModel):
    quality: str
    plate_diameter_mm: float
    mm_per_pixel: float
    mm_per_depth_unit: float
    warnings: list[str]


class AnalysisResponse(BaseModel):
    items: list[FoodItemResponse]
    totals: dict
    calibration: CalibrationResponse


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    cache_stats: dict


# Global model cache (thread-safe singleton)
class ModelCache:
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
        self._load_time = None
        self._initialized = True

    def load_models(self):
        """Load all heavy models. Called on startup."""
        with self._lock:
            if self._models is not None:
                return self._models

            print("[STARTUP] Loading models...")
            start = time.time()

            self._models = {
                "segmenter": SAM2Segmenter(),
                "classifier": FoodClassifier(
                    confidence_threshold=CLASSIFICATION_CONFIG["confidence_threshold"],
                    top_k=CLASSIFICATION_CONFIG["top_k"],
                ),
                "depth_estimator": DepthAnythingEstimator(),
            }

            self._load_time = time.time() - start
            print(f"[STARTUP] Models loaded in {self._load_time:.1f}s")
            return self._models

    def get_models(self):
        with self._lock:
            if self._models is None:
                return self.load_models()
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
                    portion_estimator, nutrition_lookup, FOOD_DENSITY_DB,
                )
            return self._pipelines[cache_key]

    def get_stats(self):
        with self._lock:
            return {
                "models_loaded": self._models is not None,
                "load_time_seconds": self._load_time,
                "pipelines_cached": len(self._pipelines),
            }

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
            self._load_time = None


_model_cache = ModelCache()


# Pydantic models for request/response
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


# FastAPI app
app = FastAPI(
    title="NutriVision API",
    description="Food image analysis with portion estimation and nutrition calculation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("[STARTUP] Initializing NutriVision API...")
    _model_cache.load_models()
    print("[STARTUP] Ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    _model_cache.clear()
    print("[SHUTDOWN] Models unloaded")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    stats = _model_cache.get_stats()
    return HealthResponse(
        status="healthy" if stats["models_loaded"] else "loading",
        models_loaded=stats["models_loaded"],
        cache_stats=stats,
    )


@app.get("/status")
async def status():
    """Detailed status endpoint."""
    return _model_cache.get_stats()


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_food(
    file: UploadFile = File(...),
    mode: str = Form(default="auto"),
    plate_diameter_mm: Optional[float] = Form(default=None),
):
    """
    Analyze a food image.
    
    Args:
        file: Image file (JPG, PNG)
        mode: "auto" | "single" | "thali"
        plate_diameter_mm: Optional plate diameter for calibration
    """
    # Validate mode
    try:
        analysis_mode = AnalysisMode(mode)
    except ValueError:
        raise HTTPException(400, f"Invalid mode: {mode}. Must be 'auto', 'single', or 'thali'")

    # Read image
    try:
        contents = await file.read()
        nparr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if nparr is None:
            raise HTTPException(400, "Invalid image file")
    except Exception as e:
        raise HTTPException(400, f"Failed to read image: {e}")

    try:
        result = _analyze_image_array(nparr, analysis_mode, plate_diameter_mm)
        return AnalysisResponse(**result.to_dict())
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")


def _analyze_image_array(image: np.ndarray, mode: AnalysisMode, plate_diameter_mm: Optional[float]) -> AnalysisResult:
    """Core analysis logic - reused from usage_pipeline."""
    from notebooks.usage_pipeline import _apply_class_merge, _convert_to_food_portion
    
    models = _model_cache.get_models()
    segmenter = models["segmenter"]
    segmentation = segmenter.segment(image)

    if not segmentation.masks:
        return AnalysisResult(
            items=[],
            totals=NutritionFacts(0, 0, 0, 0),
            calibration=CalibrationInfo(
                quality="low",
                plate_diameter_mm=plate_diameter_mm or 280.0,
                mm_per_pixel=0.0,
                mm_per_depth_unit=0.0,
                warnings=["No food detected in image"],
            ),
        )

    # Depth estimation
    depth_estimator = models["depth_estimator"]
    depth_result = depth_estimator.estimate(image)

    # Calibration based on mode
    if mode == AnalysisMode.SINGLE:
        h, w = image.shape[:2]
        calibration = type('Calibration', (), {
            'quality': 'low',
            'plate_diameter_mm': plate_diameter_mm or 280.0,
            'mm_per_pixel': (plate_diameter_mm or 280.0) / (w * 0.8),
            'mm_per_depth_unit': 100.0,
            'warnings': ['Single-food mode: using default calibration'],
        })()
    else:
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
                totals=NutritionFacts(0, 0, 0, 0),
                calibration=CalibrationInfo(
                    quality="low",
                    plate_diameter_mm=plate_diameter_mm or 280.0,
                    mm_per_pixel=0.0,
                    mm_per_depth_unit=0.0,
                    warnings=["THALI mode: No plate detected"],
                ),
            )

    # Classification on crops
    crops = [item.crop for item in segmentation.masks]
    classifier = _model_cache.get_models()["classifier"]
    classification_result = classifier.classify(crops)
    topk_results = classification_result.topk_predictions

    # Get cached pipeline
    pipeline = _model_cache.get_pipeline(calibration)
    pipeline.segmenter = models["segmenter"]

    try:
        analyses = pipeline.analyze(image)
    except Exception as e:
        _model_cache.clear()
        raise

    # Convert results
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


if __name__ == "__main__":
    import numpy as np
    uvicorn.run(app, host="0.0.0.0", port=8000)