from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CONFIG = Path("configs") / "detector_config.json"
DEFAULT_OUTPUT_DIR = Path("detector_threshold_scan_output")


def build_threshold_grid(thresholds_csv: str | None, threshold_min: float, threshold_max: float, n_thresholds: int) -> np.ndarray:
    if thresholds_csv:
        values = [float(item.strip()) for item in thresholds_csv.split(",") if item.strip()]
        if not values:
            raise ValueError("No threshold values were parsed from --thresholds.")
        return np.asarray(values, dtype=float)

    if n_thresholds < 2:
        raise ValueError("n_thresholds must be at least 2 when using a min/max range.")
    if threshold_max < threshold_min:
        raise ValueError("threshold_max must be >= threshold_min.")
    return np.linspace(threshold_min, threshold_max, n_thresholds)


def run_detector_estimation(
    repo_dir: Path,
    config_path: Path,
    threshold_kev: float,
    output_dir: Path,
) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    analysis = dict(config.get("analysis", {}))
    analysis["energy_threshold_kev"] = float(threshold_kev)
    config["analysis"] = analysis

    with tempfile.TemporaryDirectory(prefix="threshold_scan_", dir=output_dir) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        tmp_config = tmp_dir / "detector_config_threshold.json"
        tmp_output = tmp_dir / "detector_output"

        with open(tmp_config, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        command = [
            sys.executable,
            "detector_estimation.py",
            "--config",
            str(tmp_config),
            "--output-dir",
            str(tmp_output),
        ]
        subprocess.run(command, cwd=repo_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        summary_path = tmp_output / "cf4_detector_summary.json"
        with open(summary_path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def summarize_threshold_point(summary: dict, threshold_kev: float) -> dict[str, float]:
    nuclear = summary["nuclear_integrated_rates"]
    electron = summary.get("electron_integrated_rates") or {}

    nuclear_prompt = float(nuclear["prompt_events_per_year_above_threshold"])
    nuclear_delayed = float(nuclear["delayed_events_per_year_above_threshold"])
    nuclear_total = float(nuclear["total_events_per_year_above_threshold"])
    carbon_events = float(nuclear.get("c_piece_events_per_year_above_threshold", 0.0))
    fluorine_events = float(nuclear.get("f_piece_events_per_year_above_threshold", 0.0))
    fluorine_vector = float(nuclear.get("f_vector_events_per_year_above_threshold", 0.0))
    fluorine_axial = float(nuclear.get("f_axial_events_per_year_above_threshold", 0.0))

    electron_prompt = float(electron.get("prompt_events_per_year_above_threshold", 0.0))
    electron_delayed = float(electron.get("delayed_events_per_year_above_threshold", 0.0))
    electron_total = float(electron.get("total_events_per_year_above_threshold", 0.0))

    combined_prompt = nuclear_prompt + electron_prompt
    combined_delayed = nuclear_delayed + electron_delayed
    combined_total = float(summary["combined_integrated_rates"]["total_events_per_year_above_threshold"])

    return {
        "threshold_kev": float(threshold_kev),
        "threshold_kevee": float(threshold_kev),
        "combined_prompt_events_per_year": combined_prompt,
        "combined_delayed_events_per_year": combined_delayed,
        "combined_total_events_per_year": combined_total,
        "combined_prompt_fraction_of_total": combined_prompt / combined_total if combined_total > 0.0 else 0.0,
        "combined_delayed_fraction_of_total": combined_delayed / combined_total if combined_total > 0.0 else 0.0,
        "nuclear_prompt_events_per_year": nuclear_prompt,
        "nuclear_delayed_events_per_year": nuclear_delayed,
        "nuclear_total_events_per_year": nuclear_total,
        "carbon_events_per_year": carbon_events,
        "fluorine_events_per_year": fluorine_events,
        "carbon_fraction_of_nuclear_total": carbon_events / nuclear_total if nuclear_total > 0.0 else 0.0,
        "fluorine_fraction_of_nuclear_total": fluorine_events / nuclear_total if nuclear_total > 0.0 else 0.0,
        "fluorine_vector_events_per_year": fluorine_vector,
        "fluorine_axial_events_per_year": fluorine_axial,
        "fluorine_vector_fraction_of_fluorine_total": fluorine_vector / fluorine_events if fluorine_events > 0.0 else 0.0,
        "fluorine_axial_fraction_of_fluorine_total": fluorine_axial / fluorine_events if fluorine_events > 0.0 else 0.0,
        "electron_prompt_events_per_year": electron_prompt,
        "electron_delayed_events_per_year": electron_delayed,
        "electron_total_events_per_year": electron_total,
    }


def write_scan_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_set_log_scale(ax, *series: np.ndarray, enabled: bool) -> None:
    if not enabled:
        return
    positive = np.concatenate([np.asarray(values, dtype=float) for values in series])
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size:
        ax.set_yscale("log")
        ax.set_ylim(positive.min() * 0.8, positive.max() * 1.2)


def dominant_fraction_series(
    first: np.ndarray,
    second: np.ndarray,
    first_label: str,
    second_label: str,
) -> tuple[np.ndarray, str]:
    first_mean = float(np.nanmean(first))
    second_mean = float(np.nanmean(second))
    if first_mean >= second_mean:
        return first, first_label
    return second, second_label


def plot_scan(path: Path, rows: list[dict[str, float]], *, log_y: bool = False) -> None:
    threshold = np.asarray([row["threshold_kevee"] for row in rows], dtype=float)
    prompt = np.asarray([row["combined_prompt_events_per_year"] for row in rows], dtype=float)
    delayed = np.asarray([row["combined_delayed_events_per_year"] for row in rows], dtype=float)
    total = np.asarray([row["combined_total_events_per_year"] for row in rows], dtype=float)
    prompt_fraction = np.asarray([100.0 * row["combined_prompt_fraction_of_total"] for row in rows], dtype=float)
    delayed_fraction = np.asarray([100.0 * row["combined_delayed_fraction_of_total"] for row in rows], dtype=float)
    dominant_fraction, dominant_label = dominant_fraction_series(
        prompt_fraction,
        delayed_fraction,
        "prompt / total",
        "delayed / total",
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    ax_top.plot(threshold, prompt, label="prompt", linewidth=1.5)
    ax_top.plot(threshold, delayed, label="delayed", linewidth=1.5)
    ax_top.plot(threshold, total, label="total", linewidth=2.2)
    ax_top.set_ylabel("Expected events / year above threshold")
    ax_top.set_title("Threshold scan with fixed detector configuration")
    ax_top.grid(True, alpha=0.3)
    maybe_set_log_scale(ax_top, prompt, delayed, total, enabled=log_y)
    ax_top.legend()

    ax_bottom.plot(threshold, dominant_fraction, label=dominant_label, linewidth=1.5)
    ax_bottom.set_xlabel("Detection threshold [keVee]")
    ax_bottom.set_ylabel("Fraction [%]")
    ax_bottom.grid(True, alpha=0.3)
    maybe_set_log_scale(ax_bottom, dominant_fraction, enabled=log_y)
    ax_bottom.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_selected_csv(path: Path, rows: list[dict[str, float]], fieldnames: list[str]) -> None:
    trimmed = [{name: row[name] for name in fieldnames} for row in rows]
    write_scan_csv(path, trimmed)


def plot_nuclear_composition_scan(path: Path, rows: list[dict[str, float]], *, log_y: bool = False) -> None:
    threshold = np.asarray([row["threshold_kevee"] for row in rows], dtype=float)
    carbon = np.asarray([row["carbon_events_per_year"] for row in rows], dtype=float)
    fluorine = np.asarray([row["fluorine_events_per_year"] for row in rows], dtype=float)
    total = np.asarray([row["nuclear_total_events_per_year"] for row in rows], dtype=float)
    carbon_fraction = np.asarray([100.0 * row["carbon_fraction_of_nuclear_total"] for row in rows], dtype=float)
    fluorine_fraction = np.asarray([100.0 * row["fluorine_fraction_of_nuclear_total"] for row in rows], dtype=float)
    dominant_fraction, dominant_label = dominant_fraction_series(
        carbon_fraction,
        fluorine_fraction,
        "carbon / nuclear total",
        "fluorine / nuclear total",
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    ax_top.plot(threshold, carbon, label="carbon", linewidth=1.5)
    ax_top.plot(threshold, fluorine, label="fluorine", linewidth=1.5)
    ax_top.plot(threshold, total, label="nuclear total", linewidth=2.2)
    ax_top.set_ylabel("Events / year above threshold")
    ax_top.set_title("Carbon and fluorine contributions to the nuclear total")
    ax_top.grid(True, alpha=0.3)
    maybe_set_log_scale(ax_top, carbon, fluorine, total, enabled=log_y)
    ax_top.legend()

    ax_bottom.plot(threshold, dominant_fraction, label=dominant_label, linewidth=1.5)
    ax_bottom.set_xlabel("Detection threshold [keVee]")
    ax_bottom.set_ylabel("Fraction [%]")
    ax_bottom.grid(True, alpha=0.3)
    maybe_set_log_scale(ax_bottom, dominant_fraction, enabled=log_y)
    ax_bottom.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_fluorine_split_scan(path: Path, rows: list[dict[str, float]], *, log_y: bool = False) -> None:
    threshold = np.asarray([row["threshold_kevee"] for row in rows], dtype=float)
    fluorine_total = np.asarray([row["fluorine_events_per_year"] for row in rows], dtype=float)
    fluorine_vector = np.asarray([row["fluorine_vector_events_per_year"] for row in rows], dtype=float)
    fluorine_axial = np.asarray([row["fluorine_axial_events_per_year"] for row in rows], dtype=float)
    vector_fraction = np.asarray([100.0 * row["fluorine_vector_fraction_of_fluorine_total"] for row in rows], dtype=float)
    axial_fraction = np.asarray([100.0 * row["fluorine_axial_fraction_of_fluorine_total"] for row in rows], dtype=float)
    dominant_fraction, dominant_label = dominant_fraction_series(
        vector_fraction,
        axial_fraction,
        "vector / fluorine total",
        "axial / fluorine total",
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    ax_top.plot(threshold, fluorine_total, label="fluorine total", linewidth=2.2)
    ax_top.plot(threshold, fluorine_vector, label="fluorine vector", linewidth=1.5)
    ax_top.plot(threshold, fluorine_axial, label="fluorine axial", linewidth=1.5)
    ax_top.set_ylabel("Events / year above threshold")
    ax_top.set_title("Fluorine total, vector, and axial contributions")
    ax_top.grid(True, alpha=0.3)
    maybe_set_log_scale(ax_top, fluorine_total, fluorine_vector, fluorine_axial, enabled=log_y)
    ax_top.legend()

    ax_bottom.plot(threshold, dominant_fraction, label=dominant_label, linewidth=1.5)
    ax_bottom.set_xlabel("Detection threshold [keVee]")
    ax_bottom.set_ylabel("Fraction [%]")
    ax_bottom.grid(True, alpha=0.3)
    maybe_set_log_scale(ax_bottom, dominant_fraction, enabled=log_y)
    ax_bottom.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan the detector measured-energy threshold (keVee) by repeatedly running detector_estimation.py."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the detector configuration JSON.")
    parser.add_argument("--thresholds", default=None, help="Comma-separated threshold values in keVee.")
    parser.add_argument("--threshold-min", type=float, default=0.0, help="Minimum threshold in keVee for a linear scan.")
    parser.add_argument("--threshold-max", type=float, default=15.0, help="Maximum threshold in keVee for a linear scan.")
    parser.add_argument("--n-thresholds", type=int, default=16, help="Number of thresholds for a linear scan.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Folder for the scan CSV and plot.")
    parser.add_argument("--log-y", action="store_true", help="Use logarithmic y-scale for the event-count and fraction panels in the plots.")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    config_path = (repo_dir / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    output_dir = (repo_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = build_threshold_grid(args.thresholds, args.threshold_min, args.threshold_max, args.n_thresholds)
    rows: list[dict[str, float]] = []

    for index, threshold_kev in enumerate(thresholds, start=1):
        print(f"[{index}/{len(thresholds)}] threshold = {threshold_kev:.3f} keVee")
        summary = run_detector_estimation(
            repo_dir=repo_dir,
            config_path=config_path,
            threshold_kev=float(threshold_kev),
            output_dir=output_dir,
        )
        rows.append(summarize_threshold_point(summary, float(threshold_kev)))

    csv_path = output_dir / "detector_threshold_scan.csv"
    plot_path = output_dir / "detector_threshold_scan.png"
    composition_csv_path = output_dir / "detector_threshold_nuclear_composition.csv"
    composition_plot_path = output_dir / "detector_threshold_nuclear_composition.png"
    fluorine_split_csv_path = output_dir / "detector_threshold_fluorine_split.csv"
    fluorine_split_plot_path = output_dir / "detector_threshold_fluorine_split.png"

    write_scan_csv(csv_path, rows)
    plot_scan(plot_path, rows, log_y=args.log_y)
    write_selected_csv(
        composition_csv_path,
        rows,
        [
            "threshold_kevee",
            "threshold_kev",
            "carbon_events_per_year",
            "fluorine_events_per_year",
            "nuclear_total_events_per_year",
            "carbon_fraction_of_nuclear_total",
            "fluorine_fraction_of_nuclear_total",
        ],
    )
    plot_nuclear_composition_scan(composition_plot_path, rows, log_y=args.log_y)
    write_selected_csv(
        fluorine_split_csv_path,
        rows,
        [
            "threshold_kevee",
            "threshold_kev",
            "fluorine_events_per_year",
            "fluorine_vector_events_per_year",
            "fluorine_axial_events_per_year",
            "fluorine_vector_fraction_of_fluorine_total",
            "fluorine_axial_fraction_of_fluorine_total",
        ],
    )
    plot_fluorine_split_scan(fluorine_split_plot_path, rows, log_y=args.log_y)

    print()
    print("Saved files:")
    print(f"  {csv_path}")
    print(f"  {plot_path}")
    print(f"  {composition_csv_path}")
    print(f"  {composition_plot_path}")
    print(f"  {fluorine_split_csv_path}")
    print(f"  {fluorine_split_plot_path}")


if __name__ == "__main__":
    main()
