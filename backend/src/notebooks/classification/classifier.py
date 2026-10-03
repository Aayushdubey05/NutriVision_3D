from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import timm
import torch
import numpy as np

from .base import BaseClassifier
from .schema import FoodPrediction, ClassificationResult, TopKPrediction
from .utils import (
    resize_image,
    image_to_tensor,
    softmax,
    get_top_prediction,
    validate_image,
    bgr_to_rgb,
    get_device,
)


LOGGER = logging.getLogger(__name__)


DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_TOP_K = 3


class FoodClassifier(BaseClassifier):
    """
    Food classification using a fine-tuned EfficientNetV2-S model.
    """

    DEFAULT_MODEL = (
        Path(__file__).resolve().parent.parent
        / "weights"
        / "efficientnetv2_best.pth"
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        image_size: int = 384,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:

        self.device = torch.device(device) if device else get_device()
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

        model_path = Path(model_path) if model_path else self.DEFAULT_MODEL

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found:\n{model_path}"
            )

        LOGGER.info(
            "Loading Food Classifier on %s...",
            self.device,
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.classes = checkpoint["classes"]

        self.model = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=False,
            num_classes=len(self.classes),
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.model.to(self.device)
        self.model.eval()

        LOGGER.info("Food Classifier Loaded Successfully.")

    def _prepare_image(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        """
        Preprocess a single crop.
        """

        validate_image(image)

        image = bgr_to_rgb(image)

        image = resize_image(
            image,
            (
                self.image_size,
                self.image_size,
            ),
        )

        return image_to_tensor(
            image,
            self.device,
        )

    def _get_top_k_predictions(self, probs: torch.Tensor) -> TopKPrediction:
        """Get top-K predictions for a single crop's probability distribution."""
        topk_conf, topk_idx = torch.topk(probs, self.top_k, dim=0)
        
        topk_preds = [
            FoodPrediction(
                label=self.classes[int(topk_idx[i].item())],
                confidence=float(topk_conf[i].item()),
                class_index=int(topk_idx[i].item()),
            )
            for i in range(self.top_k)
        ]
        
        top1 = topk_preds[0]
        is_uncertain = top1.confidence < self.confidence_threshold
        
        return TopKPrediction(
            predictions=topk_preds,
            top1=top1,
            is_uncertain=is_uncertain,
            threshold_used=self.confidence_threshold,
        )

    @torch.inference_mode()
    def classify(
        self,
        crops: list[np.ndarray],
    ) -> ClassificationResult:
        """
        Classify multiple food crops using batch inference.
        Returns both top-1 (backward compatible) and top-K predictions.
        """

        if len(crops) == 0:
            raise ValueError(
                "No crops provided for classification."
            )

        start_time = time.perf_counter()

        batch = torch.cat(
            [
                self._prepare_image(crop)
                for crop in crops
            ],
            dim=0,
        )

        logits = self.model(batch)

        probabilities = softmax(logits)

        predictions: list[FoodPrediction] = []
        topk_predictions: list[TopKPrediction] = []

        for probs in probabilities:
            topk = self._get_top_k_predictions(probs)
            topk_predictions.append(topk)
            # Backward compatible: use top-1 label (or "uncertain")
            predictions.append(
                FoodPrediction(
                    label=topk.label,
                    confidence=topk.confidence,
                    class_index=topk.top1.class_index,
                )
            )

        inference_time = (
            time.perf_counter() - start_time
        )

        LOGGER.info(
            "Classified %d food item(s) in %.3f sec.",
            len(predictions),
            inference_time,
        )

        return ClassificationResult(
            predictions=predictions,
            topk_predictions=topk_predictions,
            inference_time=inference_time,
        )

    def unload(self) -> None:
        """
        Free model resources.
        """

        LOGGER.info(
            "Unloading Food Classifier..."
        )

        self.model = None
        self.classes = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()