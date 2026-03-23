"""Process saved simulation results and compute metrics for all versions.

Usage (from repo root):
    uv run python -m comparison.process_results

Reads ``results/<version>_results.npz`` + ``results/<version>_scalars.json``
for each version, computes metrics, and saves ``results/<version>_metrics.json``.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

# Ensure repo root is importable
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparison.metrics import compute_all_metrics, save_metrics

RESULTS_DIR = REPO_ROOT / "results"


def load_version_results(version_tag: str) -> dict | None:
    """Load saved results for a version from .npz + .json files.

    Returns
    -------
    dict or None if files not found.
    """
    npz_path = RESULTS_DIR / f"{version_tag}_results.npz"
    scalars_path = RESULTS_DIR / f"{version_tag}_scalars.json"

    if not npz_path.exists():
        print(f"  [SKIP] {version_tag}: {npz_path} not found")
        return None

    # Load numpy arrays
    with np.load(npz_path) as data:
        results = {k: data[k] for k in data.files}

    # Load scalars
    if scalars_path.exists():
        with open(scalars_path) as f:
            scalars = json.load(f)
        results.update(scalars)

    return results


def discover_versions() -> list[str]:
    """Discover version tags from saved .npz files in results/."""
    tags = []
    for npz in sorted(RESULTS_DIR.glob("*_results.npz")):
        tag = npz.stem.replace("_results", "")
        tags.append(tag)
    return tags


def main() -> None:
    """Process results for all discovered versions."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    versions = discover_versions()
    if not versions:
        print(f"No results found in {RESULTS_DIR}")
        sys.exit(1)

    print(f"Discovered {len(versions)} version(s): {', '.join(versions)}")

    for tag in versions:
        print(f"\nProcessing {tag}...")
        results = load_version_results(tag)
        if results is None:
            continue

        # Determine dt_sim from saved scalars, or infer from array lengths.
        # v1-v3 use dt_sim=1 s; v4+ use dt_sim=5 s.
        dt_sim = float(results.get("dt_sim", 0.0))
        if dt_sim <= 0:
            # Infer: soc_true length = sim_hours * 3600 / dt_sim + 1
            n_soc = len(results.get("soc_true", []))
            sim_hours = float(results.get("sim_hours", 24))
            if n_soc > 1:
                dt_sim = sim_hours * 3600.0 / (n_soc - 1)
            else:
                dt_sim = 1.0

        metrics = compute_all_metrics(results, tag, dt_sim=dt_sim)
        metrics_path = RESULTS_DIR / f"{tag}_metrics.json"
        save_metrics(metrics, metrics_path)


if __name__ == "__main__":
    main()
