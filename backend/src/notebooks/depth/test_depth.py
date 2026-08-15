from pathlib import Path
import cv2
from depth_anything import DepthAnythingEstimator

def main():
    estimator = DepthAnythingEstimator(device="cpu")
    IMAGE_FOLDER = Path("./data/indian_food_yolo/train/dal_tadka")
    OUTPUT_FOLDER = Path("./data/outputs/depth/dal_tadka")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    IMAGE_EXTENSION = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    )

    image_paths = []

    for extension in IMAGE_EXTENSION:
        image_paths.extend(
            IMAGE_FOLDER.glob(f"*{extension}")
        )

    if len(image_paths) == 0:
        print("NO images FOUND.")
        return

    for image_path in image_paths:
        print(f"\n Processing: {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            print("Unable to load the image")
            continue

        result = estimator.estimate(image)
        depth_vis = estimator.visualize(image)
        output_path = OUTPUT_FOLDER / f"{image_path.stem}_depth.png"
        cv2.imwrite(
            str(output_path),
            depth_vis,
        )
        print("Saved: ", output_path)
        print("Depth Shape: ", result.depth_map.shape)
        print("Depth MIN: ", result.depth_map.min())
        print("Depth Max: ", result.depth_map.max())


    print("\n finished")




if __name__ == "__main__":
    main()
