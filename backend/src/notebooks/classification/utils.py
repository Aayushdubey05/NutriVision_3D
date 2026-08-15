from __future__ import annotations
import cv2
import numpy as np 
import torch

def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Input image must be a numpy array."
        )
    if image.size == 0:
        raise ValueError(
            "Input image is empty"
        )

    if image.ndim != 3:
        raise ValueError(
            "Input image must have 3 dimesions (H, W, C)."
        )
    

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

def resize_image(
    image: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_LINEAR
    )

def normalize_image(
    image: np.ndarray,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    image = image.astype(np.float32)/255.0
    image = (
        image-np.asarray(mean, dtype=np.float32)
    )/np.asarray(std, dtype=np.float32)

    return image

def image_to_tensor(
        image: np.ndarray,
        device: torch.device,
) -> torch.Tensor:
    tensor = (
        torch.from_numpy(image).permute(2,0,1).float()/255.0
    )
    tensor = tensor.unsqueeze(0)
    return tensor.to(device)

def softmax(logits: torch.Tensor)-> torch.Tensor:
    return torch.softmax(
        logits,
        dim=1,
    )

def get_top_prediction(
    probabilities: torch.Tensor,
) -> tuple[int, float]:
    confidence, index = torch.max(
        probabilities,
        dim=1,
    )

    return(
        int(index.item()),
        float(confidence.item()),
    )

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