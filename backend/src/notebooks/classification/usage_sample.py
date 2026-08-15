from __future__ import annotations

import logging
from pathlib import Path

import cv2

from .classifier import FoodClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

LOGGER = logging.getLogger(__name__)


IMAGE_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "Indian_food_yolo"
    / "train"
    / "aloo_gobi"
)


def main() -> None:

    print("=" * 60)
    print("Food Classification Demo")
    print("=" * 60)

    image_paths = sorted(
        IMAGE_DIR.glob("*.*")
    )

    if not image_paths:
        print("No images found.")
        return

    print(f"Found {len(image_paths)} image(s).")

    print("\nLoading Food Classifier...")
    classifier = FoodClassifier()
    print("Food Classifier Loaded Successfully!\n")

    try:

        for index, image_path in enumerate(image_paths, start=1):

            print("-" * 60)
            print(f"[{index}/{len(image_paths)}] {image_path.name}")

            image = cv2.imread(str(image_path))

            if image is None:
                print("Failed to load image.")
                continue

            result = classifier.classify([image])

            prediction = result.predictions[0]

            print(f"Prediction      : {prediction.label}")
            print(f"Confidence      : {prediction.confidence:.4f}")
            print(f"Class Index     : {prediction.class_index}")
            print(
                f"Inference Time  : "
                f"{result.inference_time:.4f} sec"
            )

    finally:
        classifier.unload()

    print("\n" + "=" * 60)
    print("Classification Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()