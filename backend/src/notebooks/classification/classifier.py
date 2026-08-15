from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import timm
import torch
import numpy as np

from .base import BaseClassifier
from .schema import FoodPrediction, ClassificationResult
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
    ) -> None:

        self.device = torch.device(device) if device else get_device()
        self.image_size = image_size

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

    @torch.inference_mode()
    def classify(
        self,
        crops: list[np.ndarray],
    ) -> ClassificationResult:
        """
        Classify multiple food crops using batch inference.
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

        for probs in probabilities:

            class_index, confidence = get_top_prediction(
                probs.unsqueeze(0)
            )

            predictions.append(
                FoodPrediction(
                    label=self.classes[class_index],
                    confidence=confidence,
                    class_index=class_index,
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