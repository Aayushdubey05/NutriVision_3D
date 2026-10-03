from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np
import torch

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else  "cpu")

def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray but received {type(image)}"
        )

    if image.ndim != 3:
        raise ValueError(
            "Input image must have shape (H, W, 3) " 
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Input image must contain exactly 3 channels."
        )


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return h,w

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    d_min = depth.min()
    d_max = depth.max()

    if d_max - d_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)

    depth = (depth-d_min) / (d_max-d_min)
    return depth

def resize_depth_map(
    depth: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    height, width = target_size
    if depth.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth.squeeze(2)
    elif depth.ndim != 2:
        return np.zeros((height, width), dtype=np.float32)
    # OpenCV doesn't support float16, convert to float32
    if depth.dtype == np.float16:
        depth = depth.astype(np.float32)
    return cv2.resize(
        depth,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

#  Visualizer
def depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    depth = normalize_depth(depth)
    return (depth*255).astype(np.uint8)

def apply_colormap(depth: np.ndarray) -> np.ndarray:
    depth_uint8 = depth_to_uint8(depth)

    return cv2.applyColorMap(
        depth_uint8,
        cv2.COLORMAP_INFERNO
    )
