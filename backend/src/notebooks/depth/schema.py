from dataclasses import dataclass
import numpy as np 

@dataclass
class DepthResult:
    depth_map: np.ndarray
    normalized_depth: np.ndarray
    image_height: int
    image_width: int
    # inference_time: float
    

