import time
from pathlib import Path

import cv2
import numpy as np

from src.notebooks.usage_pipeline import analyze_image, NUTRITION_PER_100G
from src.notebooks.config import CLASSIFICATION_CONFIG


BASE_DIR = Path(__file__).resolve().parent

# Test multiple classes to evaluate accuracy
TEST_CLASSES = [
    "adhirasam",
    "aloo_gobi",
    "aloo_matar",
    "aloo_shimla_mirch",
    "butter_chicken",
    "chicken_tikka_masala",
    "dal_makhani",
    "biryani",
    "roti",
    "naan",
]

IMAGE_FOLDER = (
    BASE_DIR.parent
    / "notebooks"
    / "data"
    / "Indian_food_yolo"
    / "test"
)

OUTPUT_FOLDER = (
    BASE_DIR.parent
    / "notebooks"
    / "data"
    / "outputs"
    / "pipeline"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def collect_images(folder: Path, class_names: list[str]):
    images = []
    for class_name in class_names:
        class_dir = folder / class_name
        if class_dir.exists():
            for ext in IMAGE_EXTENSIONS:
                images.extend(class_dir.glob(f"*{ext}"))
    return sorted(images)


def save_results_to_txt(image_path: Path, result, output_folder: Path, true_class: str):
    output_path = output_folder / f"{image_path.stem}_analysis.txt"

    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Image: {image_path.name}\n")
        f.write(f"True Class: {true_class}\n")
        cal = result.calibration
        f.write(f"Calibration Quality: {cal.quality.upper()}\n")
        f.write(f"Plate Diameter: {cal.plate_diameter_mm:.1f} mm\n")
        f.write(f"mm_per_pixel: {cal.mm_per_pixel:.6f}\n")
        f.write(f"mm_per_depth_unit: {cal.mm_per_depth_unit:.2f}\n")
        if cal.warnings:
            f.write("Warnings:\n")
            for w in cal.warnings:
                f.write(f"  - {w}\n")
        f.write("=" * 60 + "\n\n")

        if not result.items:
            f.write("No food item detected.\n")
            return

        f.write(f"Detected {len(result.items)} food item(s):\n\n")

        for i, item in enumerate(result.items, 1):
            f.write(f"Item {i}: {item.label}\n")
            f.write(f"  Confidence:     {item.confidence:.2%}\n")
            f.write(f"  BBox:           {item.bbox}\n")
            f.write(f"  Volume:         {item.volume_ml:.1f} ml\n")
            f.write(f"  Weight:         {item.weight_g:.1f} g\n")
            f.write(f"  Calories:       {item.calories_kcal:.1f} kcal\n")
            f.write(f"  Protein:        {item.protein_g:.1f} g\n")
            f.write(f"  Carbohydrates:  {item.carbohydrates_g:.1f} g\n")
            f.write(f"  Fat:            {item.fat_g:.1f} g\n")
            if item.label == "uncertain":
                f.write(f"  *** UNCERTAIN - Top alternatives in topk_predictions ***\n")
            f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("TOTAL NUTRITION:\n")
        f.write(f"  Calories:     {result.totals.calories_kcal:.1f} kcal\n")
        f.write(f"  Protein:      {result.totals.protein_g:.1f} g\n")
        f.write(f"  Carbohydrates: {result.totals.carbohydrates_g:.1f} g\n")
        f.write(f"  Fat:          {result.totals.fat_g:.1f} g\n")


def get_true_class(image_path: Path) -> str:
    """Extract true class from folder structure."""
    return image_path.parent.name


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Full Pipeline Test (Auto-Calibration + Phase 1 Classification)")
    print("=" * 60)

    print(f"Image Folder : {IMAGE_FOLDER}")
    print(f"Output Folder: {OUTPUT_FOLDER}")
    print(f"Test Classes : {', '.join(TEST_CLASSES)}")

    if not IMAGE_FOLDER.exists():
        print("\nImage folder does not exist.")
        return

    image_paths = collect_images(IMAGE_FOLDER, TEST_CLASSES)

    if not image_paths:
        print("\nNo images found.")
        return

    print(f"\nFound {len(image_paths)} image(s).\n")
    print("Running pipeline...")

    total_items = 0
    successful = 0
    failed = 0
    uncertain_count = 0
    correct_count = 0
    total_with_prediction = 0

    try:
        for idx, image_path in enumerate(image_paths, start=1):
            print("-" * 60)
            print(f"[{idx}/{len(image_paths)}] {image_path.name}")

            image = cv2.imread(str(image_path))

            if image is None:
                print("Failed to load image.")
                failed += 1
                continue

            true_class = get_true_class(image_path)
            print(f"True Class     : {true_class}")

            start = time.perf_counter()
            try:
                result = analyze_image(str(image_path))
                elapsed = time.perf_counter() - start

                if result.items:
                    total_items += len(result.items)
                    successful += 1
                    
                    # Track accuracy
                    for item in result.items:
                        total_with_prediction += 1
                        predicted = item.label
                        if predicted == "uncertain":
                            uncertain_count += 1
                            print(f"Predicted      : UNCERTAIN (conf: {item.confidence:.2%})")
                            # Get top alternatives from topk_predictions if available
                        else:
                            # Apply same merge logic for comparison
                            from src.notebooks.config import CLASSIFICATION_CONFIG
                            merge_targets = CLASSIFICATION_CONFIG["merge_targets"]
                            true_merged = merge_targets.get(true_class, true_class)
                            pred_merged = merge_targets.get(predicted, predicted)
                            if true_merged == pred_merged:
                                correct_count += 1
                                print(f"Predicted      : {predicted} (conf: {item.confidence:.2%}) [CORRECT]")
                            else:
                                print(f"Predicted      : {predicted} (conf: {item.confidence:.2%}) [WRONG]")
                    
                    print(f"Calibration    : {result.calibration.quality.upper()}")
                    print(f"Inference Time : {elapsed:.3f} sec")

                    save_results_to_txt(image_path, result, OUTPUT_FOLDER, true_class)
                    print(f"Saved -> {OUTPUT_FOLDER / f'{image_path.stem}_analysis.txt'}")
                else:
                    print("No food item detected.")
                    failed += 1

            except Exception as e:
                elapsed = time.perf_counter() - start
                print(f"ERROR: {type(e).__name__}: {e}")
                failed += 1

    finally:
        print("\n" + "=" * 60)
        print("Processing Complete")
        print("=" * 60)
        print(f"Images Processed : {len(image_paths)}")
        print(f"Successful       : {successful}")
        print(f"Failed           : {failed}")
        print(f"Total Items      : {total_items}")
        print(f"Results Saved To : {OUTPUT_FOLDER}")
        print()
        print("--- Classification Accuracy ---")
        print(f"Total Predictions: {total_with_prediction}")
        print(f"Correct          : {correct_count}")
        print(f"Uncertain        : {uncertain_count}")
        print(f"Wrong            : {total_with_prediction - correct_count - uncertain_count}")
        if total_with_prediction > 0:
            certain = total_with_prediction - uncertain_count
            if certain > 0:
                print(f"Acc (certain)    : {correct_count/certain*100:.1f}%")
            print(f"Acc (overall)    : {correct_count/total_with_prediction*100:.1f}%")
            print(f"Uncertain rate   : {uncertain_count/total_with_prediction*100:.1f}%")


if __name__ == "__main__":
    main()