import sys
sys.path.insert(0, 'src')
from notebooks.usage_pipeline import analyze_image, AnalysisMode
import json
from pathlib import Path

OUTPUT_DIR = Path('src/notebooks/data/outputs/pipeline')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

test_cases = [
    ('aloo_gobi', 'src/notebooks/data/Indian_food_yolo/test/aloo_gobi/00fce13f41.jpg', AnalysisMode.SINGLE),
    ('adhirasam', 'src/notebooks/data/Indian_food_yolo/test/adhirasam/06c639bab2.jpg', AnalysisMode.SINGLE),
    ('dal_makhani', 'src/notebooks/data/Indian_food_yolo/test/dal_makhani/dal_makhani_0.jpg', AnalysisMode.SINGLE),
    ('chicken_thali', 'src/notebooks/data/Indian_food_yolo/test/sutar_feni/Chicken-Thali-Indian-Thali-Recipe7.jpg', AnalysisMode.THALI),
]

print("Starting Phase 1 Testing...")
print("=" * 60)

for name, path, mode in test_cases:
    print(f"\nTesting {name} with mode={mode.value}...")
    result = analyze_image(path, mode=mode)
    
    stem = Path(path).stem
    output_file = OUTPUT_DIR / (stem + f'_{mode.value}.json')
    with open(output_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"  Calibration: {result.calibration.quality}")
    print(f"  mm_per_pixel: {result.calibration.mm_per_pixel:.4f}")
    print(f"  Items: {len(result.items)}")
    for item in result.items:
        if item.confidence > 0.1:
            print(f"    {item.label}: {item.confidence:.1%}, {item.weight_g:.1f}g")
    print(f"  Saved: {output_file.name}")

print("\n" + "=" * 60)
print("Phase 1 Testing Complete!")
print("Output files saved to:", OUTPUT_DIR)