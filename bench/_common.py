"""Shared helpers for the bench scripts: machine info, thresholds, results and the trained model.

Every bench script trains the corpus, tokenizer and n-gram model from scratch into a temporary
directory (a few seconds), so results never depend on state left behind by another run.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from grammar_constrained_decoder.training.pipeline import LoadedModel, load_model, train_pipeline

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "bench" / "results"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

__all__ = [
    "RESULTS_DIR",
    "ROOT",
    "check_thresholds",
    "machine",
    "percentile",
    "report_and_check",
    "train_model",
    "write_results",
]


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def train_model(workdir: Path) -> LoadedModel:
    train_pipeline(workdir)
    return load_model(workdir / "registry")


def write_results(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    document = {"machine": machine(), **payload}
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def check_thresholds(section: str, metrics: dict[str, float]) -> list[str]:
    """Compare ``metrics`` against ``thresholds.json[section]`` ({"max": x} or {"min": y})."""
    thresholds: dict[str, dict[str, dict[str, float]]] = json.loads(
        THRESHOLDS.read_text(encoding="utf-8")
    )
    problems = []
    for name, limits in thresholds.get(section, {}).items():
        if name not in metrics:
            problems.append(f"{section}.{name}: metric missing")
            continue
        value = metrics[name]
        if "max" in limits and value > limits["max"]:
            problems.append(f"{section}.{name} = {value} exceeds the maximum {limits['max']}")
        if "min" in limits and value < limits["min"]:
            problems.append(f"{section}.{name} = {value} is below the minimum {limits['min']}")
    return problems


def report_and_check(section: str, metrics: dict[str, float], *, check: bool) -> int:
    print(f"\n{section} summary:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    if not check:
        return 0
    problems = check_thresholds(section, metrics)
    for problem in problems:
        print(f"REGRESSION: {problem}")
    if problems:
        return 1
    print(f"{section}: all metrics within thresholds")
    return 0
