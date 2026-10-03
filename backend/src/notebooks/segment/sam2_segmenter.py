from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
from src.sam2_rrepo.sam2.build_sam import build_sam2
from src.sam2_rrepo.sam2.automatic_mask_generator import (SAM2AutomaticMaskGenerator,)
from .base import BaseSegmenter
from .schema import FoodMask, SegmentationResult
from .utils import (
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
    DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
    DEFAULT_CHECKPOINT = "src/sam2_rrepo/checkpoints/sam2.1_hiera_small.pt"

    def __init__(self,
                 checkpoint: Optional[str] = None,
                 config: Optional[str] = None,
                 device: Optional[str] = None,
                 min_area: int = 5000,
                 points_per_side: int = 32,
                 pred_iou_thresh: float = 0.86,
                 stability_score_thresh: float = 0.92,
                 box_nms_thresh: float = 0.7,
                 crop_n_layers: int = 1,
                 crop_nms_thresh: float = 0.7,
                 crop_overlap_ratio: float = 512 / 1500,
                 crop_n_points_downscale_factor: int = 1,
                 output_dir: str | Path = "src/notebooks/data/outputs/segment"
                 ) -> None:
        self.device = torch.device(device) if device else get_device()
        self.min_area = min_area
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint or self.DEFAULT_CHECKPOINT
        config = config or self.DEFAULT_CONFIG

        LOGGER.info("Loading SAM2 on %s", self.device)

        self.model = build_sam2(
            config_file=str(config),
            ckpt_path=str(checkpoint),
            device=self.device,
        )

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
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
        food_masks: list[FoodMask] = []

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
            x, y, w, h = bbox
            crop = crop_from_bbox(image, bbox).copy()
            crop_mask = segmentation[y:y + h, x:x + w]
            crop[~crop_mask.astype(bool)] = 0
            food_masks.append(
                FoodMask(
                    mask=segmentation.astype(np.uint8),
                    bbox=bbox,
                    crop=crop,
                    area=area,
                    score=float(mask_data.get("stability_score", 0.0)),
                    confidence=float(mask_data["predicted_iou"]),
                )
            )

        # Post-process: NMS + merge nearby similar masks
        food_masks = self._post_process_masks(food_masks, image.shape[:2])

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

    def _post_process_masks(self, masks: list[FoodMask], image_shape: tuple[int, int]) -> list[FoodMask]:
        """Apply NMS and merge nearby masks of similar appearance."""
        if len(masks) <= 1:
            return masks

        # Sort by confidence descending
        masks = sorted(masks, key=lambda m: m.confidence, reverse=True)

        # Box NMS
        keep = []
        for m in masks:
            overlap = False
            for k in keep:
                if self._box_iou(m.bbox, k.bbox) > 0.5:
                    overlap = True
                    break
            if not overlap:
                keep.append(m)

        # Merge nearby masks with similar color histogram (likely same food type)
        merged = []
        used = set()
        for i, m1 in enumerate(keep):
            if i in used:
                continue
            group = [m1]
            for j, m2 in enumerate(keep[i+1:], i+1):
                if j in used:
                    continue
                # Check proximity (centroids close) and appearance similarity
                if self._should_merge(m1, m2, image_shape):
                    group.append(m2)
                    used.add(j)
            if len(group) > 1:
                merged.append(self._merge_masks(group, image_shape))
            else:
                merged.append(m1)
            used.add(i)

        return merged

    def _box_iou(self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def _mask_centroid(self, mask: FoodMask) -> tuple[float, float]:
        x, y, w, h = mask.bbox
        return x + w / 2, y + h / 2

    def _color_histogram(self, crop: np.ndarray, bins: int = 16) -> np.ndarray:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _should_merge(self, m1: FoodMask, m2: FoodMask, image_shape: tuple[int, int]) -> bool:
        # Centroid distance normalized by image diagonal
        c1 = self._mask_centroid(m1)
        c2 = self._mask_centroid(m2)
        dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
        diag = np.hypot(image_shape[1], image_shape[0])
        if dist / diag > 0.15:  # Too far apart
            return False

        # Color similarity (Bhattacharyya distance)
        h1 = self._color_histogram(m1.crop)
        h2 = self._color_histogram(m2.crop)
        bc = np.sum(np.sqrt(h1 * h2 + 1e-10))
        if bc < 0.6:  # Dissimilar appearance
            return False

        # Aspect ratio similarity
        ar1 = m1.bbox[2] / max(m1.bbox[3], 1)
        ar2 = m2.bbox[2] / max(m2.bbox[3], 1)
        if abs(ar1 - ar2) / max(ar1, ar2) > 0.5:
            return False

        return True

    def _merge_masks(self, masks: list[FoodMask], image_shape: tuple[int, int]) -> FoodMask:
        # Union of all masks
        combined_mask = np.zeros(image_shape, dtype=np.uint8)
        min_x, min_y = image_shape[1], image_shape[0]
        max_x, max_y = 0, 0
        total_area = 0
        for m in masks:
            combined_mask = cv2.bitwise_or(combined_mask, m.mask)
            x, y, w, h = m.bbox
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
            total_area += m.area
        bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        x, y, w, h = bbox
        crop = crop_from_bbox(combined_mask, bbox).copy() * 255
        # Use highest confidence mask's crop for classification
        best = max(masks, key=lambda m: m.confidence)
        return FoodMask(
            mask=combined_mask,
            bbox=bbox,
            crop=best.crop,
            area=total_area,
            score=max(m.score for m in masks),
            confidence=max(m.confidence for m in masks),
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

    def save_segmentation(
        self,
        image: np.ndarray,
        result: SegmentationResult,
        prefix: str = "seg"
    ) -> list[Path]:
        """Save segmentation outputs: crops, masks, and visualization."""
        saved = []
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base = f"{prefix}_{timestamp}"

        for i, food in enumerate(result.masks):
            crop_path = self.output_dir / f"{base}_{i}_crop.jpg"
            cv2.imwrite(str(crop_path), food.crop)
            saved.append(crop_path)

            mask_path = self.output_dir / f"{base}_{i}_mask.png"
            cv2.imwrite(str(mask_path), food.mask * 255)
            saved.append(mask_path)

        vis = self.visualize(image, result)
        vis_path = self.output_dir / f"{base}_viz.jpg"
        cv2.imwrite(str(vis_path), vis)
        saved.append(vis_path)

        LOGGER.info("Saved %d segmentation outputs to %s", len(saved), self.output_dir)
        return saved

    def unload(self) -> None:
        LOGGER.info("Unloading SAM2 Model.")
        self.model = None
        self.mask_generator = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()        
