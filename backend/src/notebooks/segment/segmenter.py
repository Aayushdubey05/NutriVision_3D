from pathlib import Path

import cv2
import numpy as np 

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class FoodSegmenter:
    def __init__(self, checkpoint, config, min_area=5000):
        self.model = build_sam2(
            config, checkpoint, device="cpu"
        )

        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.model,
            points_per_side=32,
            pred_iou_thresh=0.88
        )

    def segment(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(rgb)
        foods = []

        for m in masks:
            segmentation = m["segmentation"]
            area = m["area"]

            x,y,w,h = m["bbox"]

            crop = image[
                int(y): int(y+h),
                int(x): int(x+w)
            ]

            foods.append({
                "mask": segmentation.astype(np.uint8),
                "bbox": [int(x), int(y), int(w), int(h)],
                "crop": crop,
                "area": int(area),            
            })

            
    