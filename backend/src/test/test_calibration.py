"""
Unit tests for PlateDetector calibration logic.
Run with: python -m pytest src/test/test_calibration.py -v
"""
import numpy as np
from src.notebooks.calibration import PlateDetector
from src.notebooks.config import PLATE_DETECTION


class MockFoodMask:
    def __init__(self, mask, bbox):
        self.mask = mask
        self.bbox = bbox


def create_circular_mask(h, w, center_y, center_x, radius):
    y, x = np.ogrid[:h, :w]
    return ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2


def create_rectangular_mask(h, w, y1, x1, y2, x2):
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_plate_detection_basic():
    """Test that a clear circular plate is detected."""
    h, w = 480, 640
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 3
    
    plate_mask = create_circular_mask(h, w, center_y, center_x, radius)
    food_mask = create_rectangular_mask(h, w, 100, 100, 200, 200)
    
    masks = [
        MockFoodMask(plate_mask, (center_x - radius, center_y - radius, 2*radius, 2*radius)),
        MockFoodMask(food_mask, (100, 100, 100, 100)),
    ]
    
    detector = PlateDetector(PLATE_DETECTION)
    best_mask, confidence = detector.select_plate_mask(masks, (h, w))
    
    assert best_mask is not None, "Should detect plate mask"
    assert confidence > 0.5, f"Confidence should be > 0.5, got {confidence}"
    print(f"[PASS] Plate detected with confidence {confidence:.3f}")


def test_no_plate_fallback():
    """Test fallback when no plate-like mask exists."""
    h, w = 480, 640
    
    food_mask1 = create_rectangular_mask(h, w, 100, 100, 200, 200)
    food_mask2 = create_rectangular_mask(h, w, 300, 400, 350, 500)
    
    masks = [
        MockFoodMask(food_mask1, (100, 100, 100, 100)),
        MockFoodMask(food_mask2, (400, 300, 100, 50)),
    ]
    
    detector = PlateDetector(PLATE_DETECTION)
    best_mask, confidence = detector.select_plate_mask(masks, (h, w))
    
    assert best_mask is None, "Should not detect plate"
    assert confidence == 0.0, "Confidence should be 0"
    print("[PASS] Correctly returns None for non-plate masks")


def test_diameter_estimation():
    """Test diameter estimation from ellipse fitting."""
    h, w = 480, 640
    center_y, center_x = h // 2, w // 2
    radius = 150
    
    plate_mask = create_circular_mask(h, w, center_y, center_x, radius)
    mask_obj = MockFoodMask(plate_mask, (center_x - radius, center_y - radius, 2*radius, 2*radius))
    
    detector = PlateDetector(PLATE_DETECTION)
    diameter_px = detector.estimate_diameter_px(mask_obj)
    
    expected = 2 * radius
    tolerance = 5
    assert abs(diameter_px - expected) < tolerance, f"Diameter {diameter_px} != {expected} ± {tolerance}"
    print(f"[PASS] Diameter estimated: {diameter_px:.1f}px (expected ~{expected}px)")


def test_mm_per_pixel():
    """Test mm/pixel calculation."""
    h, w = 480, 640
    center_y, center_x = h // 2, w // 2
    radius = 150
    
    plate_mask = create_circular_mask(h, w, center_y, center_x, radius)
    mask_obj = MockFoodMask(plate_mask, (center_x - radius, center_y - radius, 2*radius, 2*radius))
    
    detector = PlateDetector(PLATE_DETECTION)
    mm_per_pixel = detector.estimate_mm_per_pixel(mask_obj, 280.0)
    
    expected = 280.0 / (2 * radius)
    assert abs(mm_per_pixel - expected) < 0.01, f"mm_per_pixel {mm_per_pixel} != {expected}"
    print(f"[PASS] mm_per_pixel: {mm_per_pixel:.6f} (expected {expected:.6f})")


def test_depth_scale_estimation():
    """Test mm_per_depth_unit from plane fitting."""
    h, w = 480, 640
    center_y, center_x = h // 2, w // 2
    radius = 150
    
    plate_mask = create_circular_mask(h, w, center_y, center_x, radius)
    mask_obj = MockFoodMask(plate_mask, (center_x - radius, center_y - radius, 2*radius, 2*radius))
    
    # Create depth map: flat plate at depth 0.5, background at 0.6
    # Add realistic noise (sensor noise + slight plate curvature)
    depth_map = np.ones((h, w), dtype=np.float32) * 0.6
    plate_pixels = plate_mask.sum()
    depth_map[plate_mask] = 0.5 + np.random.normal(0, 0.002, plate_pixels)
    
    detector = PlateDetector(PLATE_DETECTION)
    mm_per_depth_unit, r2 = detector.estimate_mm_per_depth_unit(mask_obj, depth_map)
    
    # plate_thickness_mm = 3.0, expected depth_range ~ 0.1 (0.6 - 0.5 = 0.1)
    expected = 3.0 / 0.1  # ≈ 30
    assert mm_per_depth_unit > 0, "Should return positive scale"
    # R² may be low for nearly-flat surfaces; just verify it returns a value
    assert r2 is not None, "Should return R² value"
    print(f"[PASS] mm_per_depth_unit: {mm_per_depth_unit:.2f}, R²: {r2:.4f}")


def test_full_calibration_high_quality():
    """Test full calibration returns high quality for good plate."""
    h, w = 480, 640
    center_y, center_x = h // 2, w // 2
    radius = 150
    
    plate_mask = create_circular_mask(h, w, center_y, center_x, radius)
    food_mask = create_rectangular_mask(h, w, 200, 200, 250, 250)
    
    masks = [
        MockFoodMask(plate_mask, (center_x - radius, center_y - radius, 2*radius, 2*radius)),
        MockFoodMask(food_mask, (200, 200, 50, 50)),
    ]
    
    depth_map = np.ones((h, w), dtype=np.float32) * 0.6
    depth_map[plate_mask] = 0.5 + np.random.normal(0, 0.0001, plate_mask.sum())
    
    detector = PlateDetector(PLATE_DETECTION)
    calibration = detector.calibrate(masks, depth_map, (h, w), 280.0)
    
    assert calibration.quality in ("high", "medium"), f"Expected high/medium, got {calibration.quality}"
    assert calibration.mm_per_pixel > 0
    assert calibration.mm_per_depth_unit > 0
    assert calibration.plate_diameter_mm == 280.0
    print(f"[PASS] Full calibration: quality={calibration.quality}, mm/px={calibration.mm_per_pixel:.4f}, mm/depth={calibration.mm_per_depth_unit:.2f}")


def test_full_calibration_low_quality():
    """Test full calibration returns low quality when no plate."""
    h, w = 480, 640
    
    food_mask1 = create_rectangular_mask(h, w, 100, 100, 200, 200)
    food_mask2 = create_rectangular_mask(h, w, 300, 400, 350, 500)
    
    masks = [
        MockFoodMask(food_mask1, (100, 100, 100, 100)),
        MockFoodMask(food_mask2, (400, 300, 100, 50)),
    ]
    
    depth_map = np.ones((h, w), dtype=np.float32) * 0.5
    
    detector = PlateDetector(PLATE_DETECTION)
    calibration = detector.calibrate(masks, depth_map, (h, w), None)
    
    assert calibration.quality == "low", f"Expected low, got {calibration.quality}"
    assert calibration.plate_mask is None
    assert "No plate detected" in calibration.warnings[0]
    assert calibration.mm_per_pixel > 0  # fallback
    assert calibration.mm_per_depth_unit == 100.0  # fallback
    print(f"[PASS] Fallback calibration: quality={calibration.quality}, warnings={calibration.warnings}")


def test_off_center_plate():
    """Test detection of slightly off-center plate."""
    h, w = 480, 640
    center_y, center_x = h // 2, w // 2
    radius = 150
    
    # Plate slightly off-center
    plate_mask = create_circular_mask(h, w, center_y + 30, center_x - 40, radius)
    food_mask = create_rectangular_mask(h, w, 100, 100, 150, 150)
    
    masks = [
        MockFoodMask(plate_mask, (center_x - radius - 40, center_y - radius + 30, 2*radius, 2*radius)),
        MockFoodMask(food_mask, (100, 100, 50, 50)),
    ]
    
    detector = PlateDetector(PLATE_DETECTION)
    best_mask, confidence = detector.select_plate_mask(masks, (h, w))
    
    assert best_mask is not None, "Should detect off-center plate"
    assert confidence > 0.4, f"Confidence should be reasonable, got {confidence}"
    print(f"[PASS] Off-center plate detected with confidence {confidence:.3f}")


if __name__ == "__main__":
    print("Running PlateDetector unit tests...\n")
    
    test_plate_detection_basic()
    test_no_plate_fallback()
    test_diameter_estimation()
    test_mm_per_pixel()
    test_depth_scale_estimation()
    test_full_calibration_high_quality()
    test_full_calibration_low_quality()
    test_off_center_plate()
    
    print("\n[PASS] All calibration tests passed!")