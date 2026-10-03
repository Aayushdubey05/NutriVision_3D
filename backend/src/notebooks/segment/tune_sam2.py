"""Auto-tune SAM2 segmentation parameters on a validation set with ground truth."""
from __future__ import annotations
import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .sam2_segmenter import SAM2Segmenter
from .schema import FoodMask, SegmentationResult

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VAL_DIR = BASE_DIR.parent / "data" / "Indian_food_yolo" / "val"
DEFAULT_OUTPUT_DIR = BASE_DIR.parent / "data" / "outputs" / "tune"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


PARAM_GRID = {
    "points_per_side": [16, 32, 48],
    "pred_iou_thresh": [0.82, 0.86, 0.88],
    "stability_score_thresh": [0.90, 0.92, 0.95],
    "box_nms_thresh": [0.6, 0.7, 0.8],
    "crop_n_layers": [0, 1],
    "min_area": [3000, 5000, 8000],
}


OPTUNA_PARAM_SPACE = {
    "points_per_side": {"type": "int", "low": 8, "high": 64, "step": 8},
    "pred_iou_thresh": {"type": "float", "low": 0.75, "high": 0.95, "step": 0.01},
    "stability_score_thresh": {"type": "float", "low": 0.85, "high": 0.98, "step": 0.01},
    "box_nms_thresh": {"type": "float", "low": 0.5, "high": 0.9, "step": 0.05},
    "crop_n_layers": {"type": "categorical", "choices": [0, 1, 2]},
    "min_area": {"type": "int", "low": 1000, "high": 15000, "step": 1000},
}


def collect_val_images(root: Path) -> list[Path]:
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(root.rglob(f"*{ext}"))
    return sorted(images)


def load_gt_masks(gt_path: Path) -> list[np.ndarray]:
    """Load ground truth masks from JSON (COCO format) or PNG directory."""
    if gt_path.suffix == ".json":
        with open(gt_path) as f:
            data = json.load(f)
        masks = []
        for ann in data.get("annotations", []):
            if "segmentation" in ann:
                h, w = ann.get("height", 512), ann.get("width", 512)
                mask = np.zeros((h, w), dtype=np.uint8)
                for seg in ann["segmentation"]:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                masks.append(mask)
        return masks
    elif gt_path.is_dir():
        masks = []
        for ext in IMAGE_EXTENSIONS:
            for p in gt_path.glob(f"*{ext}"):
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    masks.append((m > 127).astype(np.uint8))
        return masks
    return []


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0


def compute_ap(pred_masks: list[FoodMask], gt_masks: list[np.ndarray], iou_thresh: float = 0.5) -> float:
    """Compute Average Precision for instance segmentation."""
    if not gt_masks or not pred_masks:
        return 0.0 if gt_masks else 1.0

    pred_sorted = sorted(pred_masks, key=lambda m: m.confidence, reverse=True)
    gt_matched = [False] * len(gt_masks)
    tp = 0
    fp = 0

    for pred in pred_sorted:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gt in enumerate(gt_masks):
            if gt_matched[gi]:
                continue
            if pred.mask.shape != gt.shape:
                gt_resized = cv2.resize(gt.astype(np.uint8), (pred.mask.shape[1], pred.mask.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
            else:
                gt_resized = gt
            iou = mask_iou(pred.mask.astype(bool), gt_resized.astype(bool))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def evaluate_config(
    segmenter: SAM2Segmenter,
    image_paths: list[Path],
    gt_dir: Path | None = None,
) -> dict[str, float]:
    total_f1 = 0.0
    total_time = 0.0
    total_masks = 0
    count = 0

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        start = time.perf_counter()
        result = segmenter.segment(image)
        elapsed = time.perf_counter() - start

        total_time += elapsed
        total_masks += len(result.masks)

        if gt_dir:
            gt_masks = load_gt_masks(gt_dir / img_path.stem)
            if gt_masks:
                f1 = compute_ap(result.masks, gt_masks)
                total_f1 += f1
                count += 1

    return {
        "avg_f1": total_f1 / max(count, 1),
        "avg_time_sec": total_time / max(len(image_paths), 1),
        "avg_masks_per_image": total_masks / max(len(image_paths), 1),
        "images_evaluated": len(image_paths),
        "images_with_gt": count,
    }


def run_grid_search(
    image_paths: list[Path],
    gt_dir: Path | None,
    device: str,
    output_csv: Path,
    max_combos: int | None = None,
) -> list[dict[str, Any]]:
    import itertools

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    if max_combos:
        combos = combos[:max_combos]

    logging.info("Testing %d parameter combinations on %d images", len(combos), len(image_paths))

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        logging.info("[%d/%d] %s", i + 1, len(combos), params)

        segmenter = SAM2Segmenter(device=device, output_dir=DEFAULT_OUTPUT_DIR / "tmp", **params)
        metrics = evaluate_config(segmenter, image_paths, gt_dir)
        segmenter.unload()

        row = {**params, **metrics}
        results.append(row)
        logging.info("  F1: %.3f | Time: %.3fs | Masks/img: %.1f",
                     metrics["avg_f1"], metrics["avg_time_sec"], metrics["avg_masks_per_image"])

        # Append to CSV incrementally
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    return results


def find_best_config(results: list[dict], metric: str = "avg_f1") -> dict | None:
    if not results:
        return None
    return max(results, key=lambda r: r.get(metric, 0))


def _suggest_params(trial: "optuna.Trial") -> dict:
    params = {}
    for name, spec in OPTUNA_PARAM_SPACE.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step", 0.01))
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def run_bayesian_opt(
    image_paths: list[Path],
    gt_dir: Path | None,
    device: str,
    output_csv: Path,
    n_trials: int = 50,
    metric: str = "avg_f1",
    timeout: int | None = None,
) -> list[dict[str, Any]]:
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not installed. Run: pip install optuna")

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_params(trial)
        logging.info("Trial %d: %s", trial.number, params)

        segmenter = SAM2Segmenter(device=device, output_dir=DEFAULT_OUTPUT_DIR / f"trial_{trial.number}", **params)
        metrics = evaluate_config(segmenter, image_paths, gt_dir)
        segmenter.unload()

        row = {**params, **metrics, "trial": trial.number}
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

        logging.info("  %s: %.4f | Time: %.3fs | Masks/img: %.1f",
                     metric, metrics.get(metric, 0), metrics["avg_time_sec"], metrics["avg_masks_per_image"])

        return metrics.get(metric, 0.0)

    study_name = f"sam2_tune_{metric}"
    storage = f"sqlite:///{output_csv.with_suffix('.db')}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    logging.info("Starting Bayesian optimization: %d trials", n_trials)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logging.info("Best trial: %s", study.best_trial.params)
    logging.info("Best value: %.4f", study.best_value)

    # Export all trials to CSV
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_csv, index=False)

    # Return best config
    best_params = study.best_trial.params
    segmenter = SAM2Segmenter(device=device, output_dir=DEFAULT_OUTPUT_DIR / "best", **best_params)
    best_metrics = evaluate_config(segmenter, image_paths, gt_dir)
    segmenter.unload()

    best_result = {**best_params, **best_metrics}
    with open(output_csv.with_name("best_config.json"), "w") as f:
        json.dump(best_result, f, indent=2)

    return [best_result]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 Hyperparameter Tuning")
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR,
                        help="Validation images directory")
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="Ground truth masks directory (COCO JSON or PNG dir)")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR / "tune_results.csv",
                        help="Output CSV for results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for inference")
    parser.add_argument("--mode", choices=["grid", "bayes"], default="grid",
                        help="Search mode: grid (exhaustive) or bayes (Optuna Bayesian optimization)")
    parser.add_argument("--max-combos", type=int, default=None,
                        help="Grid mode: limit number of combinations to test")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Bayes mode: number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Bayes mode: timeout in seconds")
    parser.add_argument("--metric", choices=["avg_f1", "avg_time_sec", "avg_masks_per_image"],
                        default="avg_f1", help="Metric to optimize")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser.parse_args()


def get_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    device = get_device(args.device)
    logger.info("Device: %s | Mode: %s", device, args.mode)

    image_paths = collect_val_images(args.val_dir)
    if not image_paths:
        logger.error("No validation images found in %s", args.val_dir)
        return

    logger.info("Found %d validation images", len(image_paths))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "bayes":
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not installed. Run: pip install optuna")
            return
        results = run_bayesian_opt(
            image_paths=image_paths,
            gt_dir=args.gt_dir,
            device=device,
            output_csv=args.output,
            n_trials=args.n_trials,
            metric=args.metric,
            timeout=args.timeout,
        )
    else:
        results = run_grid_search(
            image_paths=image_paths,
            gt_dir=args.gt_dir,
            device=device,
            output_csv=args.output,
            max_combos=args.max_combos,
        )

    best = find_best_config(results, args.metric)
    if best:
        logger.info("=" * 60)
        logger.info("BEST CONFIG (by %s):", args.metric)
        for k, v in best.items():
            logger.info("  %s: %s", k, v)
        logger.info("=" * 60)

        best_path = args.output.with_name("best_config.json")
        with open(best_path, "w") as f:
            json.dump(best, f, indent=2)
        logger.info("Best config saved to %s", best_path)


if __name__ == "__main__":
    main()