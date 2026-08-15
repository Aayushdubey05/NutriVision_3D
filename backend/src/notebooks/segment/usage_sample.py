import time
from pathlib import Path

import cv2

from .sam2_segmenter import SAM2Segmenter


# Base directory of this file
BASE_DIR = Path(__file__).resolve().parent

IMAGE_FOLDER = (
    BASE_DIR.parent
    / "data"
    / "Indian_food_yolo"
    / "train"
    / "aloo_gobi"
)

OUTPUT_FOLDER = (
    BASE_DIR.parent
    / "data"
    / "outputs"
    / "segment"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def collect_images(folder: Path):
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))

    return sorted(images)


def main():

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAM2 Segmentation Demo")
    print("=" * 60)

    print(f"Image Folder : {IMAGE_FOLDER}")
    print(f"Output Folder: {OUTPUT_FOLDER}")

    if not IMAGE_FOLDER.exists():
        print("\nImage folder does not exist.")
        return

    image_paths = collect_images(IMAGE_FOLDER)

    if not image_paths:
        print("\nNo images found.")
        return

    print(f"\nFound {len(image_paths)} image(s).\n")
    print("Creating SAM2...")
    segmenter = SAM2Segmenter(
        device="cpu",      # change to "cuda" if available
        min_area=5000,
    )
    print("SAM2 Created!")

    print("Creating the image loop")

    total_masks = 0

    try:
        for idx, image_path in enumerate(image_paths, start=1):

            print("-" * 60)
            print(f"[{idx}/{len(image_paths)}] {image_path.name}")

            image = cv2.imread(str(image_path))
            print("Image loaded")

            if image is None:
                print("Failed to load image.")
                continue

            start = time.perf_counter()
            print("Calling segment")
            result = segmenter.segment(image)
            print("segment finished")
            elapsed = time.perf_counter() - start

            total_masks += len(result.masks)

            print(f"Detected Objects : {len(result.masks)}")
            print(f"Inference Time  : {elapsed:.3f} sec")

            visualization = segmenter.visualize(
                image,
                result,
            )

            output_path = OUTPUT_FOLDER / f"{image_path.stem}_segment.png"

            cv2.imwrite(
                str(output_path),
                visualization,
            )

            print(f"Saved -> {output_path.name}")

        print("\n" + "=" * 60)
        print("Processing Complete")
        print("=" * 60)
        print(f"Images Processed : {len(image_paths)}")
        print(f"Total Masks      : {total_masks}")
        print(f"Results Saved To : {OUTPUT_FOLDER}")

    finally:
        segmenter.unload()


if __name__ == "__main__":
    main()