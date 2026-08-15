from __future__ import annotations
from typing import Tuple, Optional
import logging 
import cv2
import numpy as np 
import torch 
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation, 
)
from base import BaseDepthEstimator
from schema import DepthResult
from utils import (get_device, 
                    validate_image, 
                    bgr_to_rgb, 
                    normalize_depth, 
                    resize_depth_map,)

LOGGER = logging.getLogger(__name__)

class DepthAnythingEstimator(BaseDepthEstimator):
    DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self,
                 model_name: str= DEFAULT_MODEL,
                 device: Optional[str] = None) -> None:
        self.device = torch.device(device) if device else get_device()
        LOGGER.info(
            "Loading Depth Anything model on %s",
            self.device
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        LOGGER.info("Depth model loaded successfully.")

    def _prepare_inputs(
            self,
            image: np.ndarray
    ) -> dict:
        rgb = bgr_to_rgb(image)
        inputs = self.processor(
            images = rgb,
            return_tensors="pt",
        )

        return {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

    @torch.inference_mode()
    def estimate(self, image: np.ndarray) -> DepthResult:
        validate_image(image)
        original_height, original_width = image.shape[:2]

        inputs = self._prepare_inputs(image)

        if self.device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16            
                ):
                    outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth
        depth = (
        predicted_depth
        .squeeze()
        .cpu()
        .numpy()
        )

        depth = resize_depth_map(depth, (original_height, original_width),)
        normalized = normalize_depth(depth)

        return DepthResult(
            depth_map=depth,
            normalized_depth=normalized,
            image_height=original_height,
            image_width=original_width,
        )

    def visualize(self, image: np.ndarray,) -> np.ndarray:
        from utils import apply_colormap
        result = self.estimate(image)
        return apply_colormap(
            result.normalized_depth
        )

    def unload(self):
        del self.model
        torch.cuda.empty_cache()



        
        