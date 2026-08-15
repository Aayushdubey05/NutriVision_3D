from __future__ import annotations
# from ..depth.utils import get_device, validate_image, bgr_to_rgb,rgb_to_bgr
import cv2 
from typing import Tuple
import numpy as np
import torch

def get_image_size(image: np.ndarray) -> Tuple[int, int]:
    return image.shape[:2]


def clip_bbox(
        bbox: Tuple[int, int, int ,int],
        image_shape: Tuple[int , int]
) -> Tuple[int, int, int, int]:

    x,y,w,h = bbox
    height, width = image_shape
    x = max(0,x)
    y = max(0,y)
    w = min(w, width-x)
    h = min(h,height-y)

    return x,y,w,h

def crop_from_bbox(
        image:np.ndarray,
        bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    x,y,w,h = bbox
    return image[y:y+h, x:x+w]

def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8)) * 255

def calculate_mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))

# def filter_small_masks(
#         masks: list,
#         min_area: int
# ) -> list:
#     return [
#         mask for mask in masks if mask["area"] >= min_area
#     ]

def is_validd_area(area: int,min_area: int ) -> bool:
    return area >= min_area

