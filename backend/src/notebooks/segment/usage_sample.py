import argparse
import logging
import time
from pathlib import Path

import cv2
import torch

from .sam2_segmenter import SAM2Segmenter


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_IMAGE_FOLDER = BASE_DIR.parent / "data" / "Indian_food_yolo" / "train"
DEFAULT_OUTPUT_FOLDER = BASE_DIR.parent / "src"/ "notebooks" / "data" / "outputs" / "segment"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: Path) -> list[Path]:
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
    return sorted(images)


def collect_all_class_images(root: Path) -> list[Path]:
    """Recursively collect images from all class subdirectories."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(root.rglob(f"*{ext}"))
    return sorted(images)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 Segmentation Demo")
    parser.add_argument(
        "--input", "-i", type=Path, default=None,
        help="Single image file or directory (default: all classes under Indian_food_yolo/train)"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT_FOLDER,
        help="Output directory for segmentation results"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--min-area", type=int, default=5000,
        help="Minimum mask area in pixels"
    )
    parser.add_argument(
        "--points-per-side", type=int, default=32,
        help="SAM2 grid points per side (higher = more masks, slower)"
    )
    parser.add_argument(
        "--pred-iou-thresh", type=float, default=0.86,
        help="Predicted IoU threshold (lower = more masks)"
    )
    parser.add_argument(
        "--stability-thresh", type=float, default=0.92,
        help="Stability score threshold (higher = more stable)"
    )
    parser.add_argument(
        "--box-nms-thresh", type=float, default=0.7,
        help="Box NMS threshold (lower = stricter deduplication)"
    )
    parser.add_argument(
        "--crop-layers", type=int, default=1,
        help="Number of crop layers for multi-scale detection"
    )
    parser.add_argument(
        "--crop-nms-thresh", type=float, default=0.7,
        help="Crop NMS threshold"
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Recursively scan subdirectories for images"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
        help="Logging level"
    )
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    if args.input is None:
        input_path = DEFAULT_IMAGE_FOLDER
        logger.info("No input specified, using default: %s", input_path)
    else:
        input_path = args.input

    output_path = args.output
    output_path.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    logger.info("Using device: %s", device)

    if input_path.is_file():
        image_paths = [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    elif input_path.is_dir():
        if args.recursive:
            image_paths = collect_all_class_images(input_path)
        else:
            image_paths = collect_images(input_path)
    else:
        logger.error("Input path does not exist: %s", input_path)
        return

    if not image_paths:
        logger.warning("No images found.")
        return

    logger.info("Found %d image(s)", len(image_paths))
    logger.info("Output directory: %s", output_path)

    logger.info("Initializing SAM2...")
    segmenter = SAM2Segmenter(
        device=device,
        min_area=args.min_area,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        output_dir=output_path,
    )
    logger.info("SAM2 initialized successfully")

    total_masks = 0
    processed = 0

    try:
        for idx, image_path in enumerate(image_paths, start=1):
            logger.info("[%d/%d] %s", idx, len(image_paths), image_path.name)

            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Failed to load image: %s", image_path)
                continue

            start = time.perf_counter()
            result = segmenter.segment(image)
            elapsed = time.perf_counter() - start

            total_masks += len(result.masks)
            processed += 1

            logger.info("Detected objects: %d | Inference: %.3fs", len(result.masks), elapsed)

            saved = segmenter.save_segmentation(image, result, prefix=image_path.stem)
            logger.info("Saved %d output files", len(saved))

    finally:
        segmenter.unload()
        logger.info("SAM2 unloaded")

    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("Images processed : %d", processed)
    logger.info("Total masks      : %d", total_masks)
    logger.info("Results saved to : %s", output_path)


if __name__ == "__main__":
    main()