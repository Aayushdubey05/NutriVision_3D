from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
from sam2_rrepo.sam2.build_sam import build_sam2
from sam2_rrepo.sam2.automatic_mask_generator import (SAM2AutomaticMaskGenerator,)
from .base import BaseSegmenter
from .schema import FoodMask,SegmentationResult
from .utils import(
    crop_from_bbox,
    clip_bbox,
    calculate_mask_area,
    is_validd_area,
)

from ..depth.utils import (
    get_device,
    validate_image,
    bgr_to_rgb,
)

LOGGER = logging.getLogger(__name__)

class SAM2Segmenter(BaseSegmenter):
    DEFAULT_CONFIG = "sam2_rrepo/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
    DEFAULT_CHECKPOINT = "sam2_rrepo/checkpoints/sam2.1_hiera_small.pt"

    def __init__(self,
                 checkpoint: Optional[str] = None,
                 config: Optional[str] = None,
                 device: Optional[str] = None,
                 min_area: int = 5000,
                 points_per_side: int = 16,
                 pred_iou_thresh: float = 0.88 
                 ) -> None:
        self.device = torch.device(device) if device else get_device()
        self.min_area = min_area
        checkpoint = checkpoint or self.DEFAULT_CHECKPOINT
        config = config or self.DEFAULT_CONFIG 

        LOGGER.info(
            "Loading SAM2 on %s",
            self.device,
        )

        self.model = build_sam2(
            config_file = str(config),
            ckpt_path = str(checkpoint),
            device = self.device,
        )

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model = self.model,
            points_per_side = points_per_side,
            pred_iou_thresh = pred_iou_thresh,
        )

        LOGGER.info("SAM2 loaded successfully")

    def _prepare_image(
            self,
            image: np.ndarray,
    ) -> np.ndarray:

        validate_image(image)
        return bgr_to_rgb(image)

    
    @torch.inference_mode()
    def segment(self, image: np.ndarray) -> SegmentationResult:
        start_time = time.perf_counter()
        rgb_image = self._prepare_image(image)
        masks = self.mask_generator.generate(rgb_image)
        food_masks = list[FoodMask] = []

        for mask_data in masks:
            segmentation = mask_data["segmentation"]
            area = calculate_mask_area(segmentation)
            if not is_validd_area(area, self.min_area):
                continue

            bbox = tuple(map(int, mask_data["bbox"]))
            bbox = clip_bbox(
                bbox,
                image.shape[:2]
            )
            crop = crop_from_bbox(
                image,
                bbox,
            )
            food_masks.append(
                FoodMask(
                    mask=segmentation.astype(np.uint8),
                    bbox=bbox,
                    crop=crop,
                    area=area,
                    confidence=float(mask_data["predicted_iou"]),
                )
            )

        inference_time = time.perf_counter() - start_time
        LOGGER.info(
            "Detected %d food objects in %.2f seconds.",
            len(food_masks),
            inference_time,
        )

        return SegmentationResult(
            masks=food_masks,
            image_height=image.shape[0],
            image_width=image.shape[1],
            inference_time=inference_time,
        ) 

    def visualize(
        self,
        image: np.ndarray,
        result: SegmentationResult,
    ) -> np.ndarray:

        canvas = image.copy()
        for food in result.masks:
            x,y,w,h = food.bbox
            cv2.rectangle(
                canvas,
                (x,y),
                (x+w, y+h),
                (0, 255, 0),
                2,
            )


            cv2.putText(
                canvas,
                f"{food.area}",
                (x, y- 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return canvas

    def unload(self) -> None:
        LOGGER.info("Unloading SAM2 Model.")
        self.model = None
        self.mask_generator = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()        
