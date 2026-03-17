"""
Comparative evaluation: measure the impact of polygon-based location priors.

Runs predictions on a test set:
1. WITHOUT polygon priors (baseline)
2. WITH polygon priors (using lat/lon)

Compares metrics side-by-side to quantify improvement.
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--filelist",
        type=str,
        default=str(PROJECT_ROOT / "data" / "splits" / "raw_audio_holdout_filelist.txt"),
        help="Filelist of test files (one path per line).",
    )
    p.add_argument(
        "--location-map-json",
        type=str,
        default="",
        help="Optional JSON mapping filepath -> {lat, lon}. If not provided, infers from filename (Pugu/Rondo -> Tanzania).",
    )
    p.add_argument(
        "--tanzania-lat",
        type=float,
        default=-6.8,
        help="Representative latitude for Tanzania (default: -6.8, near Pugu/Rondo).",
    )
    p.add_argument(
        "--tanzania-lon",
        type=float,
        default=39.28,
        help="Representative longitude for Tanzania (default: 39.28, near Pugu/Rondo).",
    )
    p.add_argument(
        "--classifier-threshold",
        type=float,
        default=0.2,
        help="Confidence threshold for classifier.",
    )
    p.add_argument(
        "--detector-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for detector.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "evaluations"),
        help="Output directory for CSVs and summary.",
    )
    return p.parse_args()


def load_location_map(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    """Load filepath -> {lat, lon} mapping from JSON."""
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, dict) and "lat" in v and "lon" in v}
    return {}


def infer_location_from_filename(filepath: str, tz_lat: float, tz_lon: float) -> Optional[Tuple[float, float]]:
    """Infer approximate lat/lon from filename tokens (Pugu/Rondo -> Tanzania)."""
    import re
    name = Path(filepath).name.lower()
    if re.search(r"\bpugu\b", name) or re.search(r"\brondo\b", name):
        return (tz_lat, tz_lon)
    return None


def run_predictions(
    filelist_path: Path,
    out_csv: Path,
    location_map: Dict[str, Dict[str, float]],
    tz_lat: float,
    tz_lon: float,
    classifier_threshold: float,
    detector_threshold: float,
    use_polygons: bool,
    location_map_json_path: Optional[str] = None,
) -> bool:
    """Run predict_3stage_with_context.py and return success status."""
    script = PROJECT_ROOT / "scripts" / "predict_3stage_with_context.py"
    cmd = [
        sys.executable,
        str(script),
        "--filelist",
        str(filelist_path),
        "--out-csv",
        str(out_csv),
        "--classifier-threshold",
        str(classifier_threshold),
        "--detector-threshold",
        str(detector_threshold),
    ]

    # If using polygons: per-file locations from JSON, or single lat/lon fallback.
    if use_polygons:
        if location_map_json_path and Path(location_map_json_path).exists():
            cmd.extend(["--location-map-json", str(Path(location_map_json_path).resolve())])
        else:
            has_explicit = any(
                infer_location_from_filename(k, tz_lat, tz_lon) or (k in location_map)
                for k in filelist_path.read_text(encoding="utf-8").splitlines()
                if k.strip()
            )
            if has_explicit:
                cmd.extend(["--lat", str(tz_lat), "--lon", str(tz_lon)])
            else:
                return False

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"ERROR: Prediction failed:\n{result.stderr}")
            return False
        return out_csv.exists()
    except subprocess.TimeoutExpired:
        print("ERROR: Prediction timed out")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def analyze_csv(csv_path: Path) -> Dict:
    """Analyze prediction CSV and return metrics."""
    if not csv_path.exists():
        return {}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    total = len(rows)
    passed_detector = sum(1 for r in rows if r.get("detector_result") == "galago")
    confident = sum(1 for r in rows if r.get("species_result") not in ("uncertain", "error", "N/A", "not_classified"))
    uncertain = sum(1 for r in rows if r.get("species_result") == "uncertain")

    # Top-1 accuracy (requires ground truth from filepath)
    # For now, just report coverage and confidence stats
    top1_probs = []
    for r in rows:
        p = r.get("top1_prob", "N/A")
        if p != "N/A":
            try:
                top1_probs.append(float(p))
            except ValueError:
                pass

    avg_top1_prob = sum(top1_probs) / len(top1_probs) if top1_probs else 0.0

    return {
        "total_files": total,
        "passed_detector": passed_detector,
        "confident_predictions": confident,
        "uncertain_predictions": uncertain,
        "coverage_pct": (confident / total * 100) if total > 0 else 0.0,
        "avg_top1_prob": avg_top1_prob,
        "detector_pass_rate": (passed_detector / total * 100) if total > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()

    filelist_path = Path(args.filelist)
    if not filelist_path.exists():
        print(f"ERROR: Filelist not found: {filelist_path}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    location_map = {}
    if args.location_map_json:
        location_map = load_location_map(Path(args.location_map_json))

    print("=" * 70)
    print("Polygon Priors Impact Evaluation")
    print("=" * 70)
    print(f"\nTest set: {filelist_path.name}")
    print(f"Files: {len(filelist_path.read_text(encoding='utf-8').strip().splitlines())}")
    print(f"\nThresholds:")
    print(f"  Detector: {args.detector_threshold}")
    print(f"  Classifier: {args.classifier_threshold}")

    # Run 1: Baseline (no polygon priors)
    print("\n" + "=" * 70)
    print("Run 1: Baseline (no polygon priors)")
    print("=" * 70)
    baseline_csv = out_dir / "baseline_no_polygons.csv"
    success = run_predictions(
        filelist_path=filelist_path,
        out_csv=baseline_csv,
        location_map=location_map,
        tz_lat=args.tanzania_lat,
        tz_lon=args.tanzania_lon,
        classifier_threshold=args.classifier_threshold,
        detector_threshold=args.detector_threshold,
        use_polygons=False,
    )
    if not success:
        print("ERROR: Baseline run failed")
        return 1

    baseline_metrics = analyze_csv(baseline_csv)
    print(f"\nBaseline Results:")
    for k, v in baseline_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Run 2: With polygon priors
    print("\n" + "=" * 70)
    print("Run 2: With polygon priors")
    print("=" * 70)
    polygon_csv = out_dir / "with_polygon_priors.csv"
    success = run_predictions(
        filelist_path=filelist_path,
        out_csv=polygon_csv,
        location_map=location_map,
        tz_lat=args.tanzania_lat,
        tz_lon=args.tanzania_lon,
        classifier_threshold=args.classifier_threshold,
        detector_threshold=args.detector_threshold,
        use_polygons=True,
        location_map_json_path=args.location_map_json or None,
    )
    if not success:
        print("WARNING: Polygon run failed or skipped (no location data)")
        polygon_metrics = {}
    else:
        polygon_metrics = analyze_csv(polygon_csv)
        print(f"\nWith Polygon Priors Results:")
        for k, v in polygon_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

    # Comparison
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)

    if polygon_metrics:
        print(f"\n{'Metric':<30} {'Baseline':>15} {'With Polygons':>15} {'Change':>15}")
        print("-" * 75)

        for key in baseline_metrics.keys():
            if key not in polygon_metrics:
                continue
            base_val = baseline_metrics[key]
            poly_val = polygon_metrics[key]
            if isinstance(base_val, float) and isinstance(poly_val, float):
                change = poly_val - base_val
                change_pct = (change / base_val * 100) if base_val != 0 else 0.0
                print(f"{key:<30} {base_val:>15.2f} {poly_val:>15.2f} {change:>+15.2f} ({change_pct:+.1f}%)")
            else:
                print(f"{key:<30} {base_val:>15} {poly_val:>15}")

        # Key insights
        print("\n" + "=" * 70)
        print("Key Insights:")
        print("=" * 70)

        coverage_change = polygon_metrics.get("coverage_pct", 0) - baseline_metrics.get("coverage_pct", 0)
        avg_prob_change = polygon_metrics.get("avg_top1_prob", 0) - baseline_metrics.get("avg_top1_prob", 0)

        print(f"\nCoverage: {coverage_change:+.1f}% change")
        if coverage_change > 0:
            print("  [OK] Polygon priors increased coverage (more confident predictions)")
        elif coverage_change < 0:
            print("  [WARN] Polygon priors decreased coverage (more uncertain predictions)")

        print(f"\nAverage Top-1 Probability: {avg_prob_change:+.3f} change")
        if avg_prob_change > 0:
            print("  [OK] Polygon priors increased confidence scores")
        elif avg_prob_change < 0:
            print("  [WARN] Polygon priors decreased confidence scores")

    # Save summary JSON
    summary = {
        "baseline": baseline_metrics,
        "with_polygons": polygon_metrics,
        "config": {
            "filelist": str(filelist_path),
            "classifier_threshold": args.classifier_threshold,
            "detector_threshold": args.detector_threshold,
            "tanzania_lat": args.tanzania_lat,
            "tanzania_lon": args.tanzania_lon,
        },
    }
    summary_path = out_dir / "polygon_priors_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
