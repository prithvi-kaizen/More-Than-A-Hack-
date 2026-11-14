"""Import the pinned official metric without loading any target labels."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / 'vendor/mtah/evaluation/possession_metrics.py'
spec = importlib.util.spec_from_file_location('mtah_official_metrics', path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
