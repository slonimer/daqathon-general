#!/usr/bin/env python3
"""Build a typed parquet cache from ONC scalar CSV exports.

This preparation script supports a single scalar device family or mixed device
folders and writes a namespaced cache bundle that notebooks can reuse.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEVICE_PATTERNS = {
    "ctd": "ConductivityTemperatureDepth",
    "fluorometer": "FluorometerTurbidity",
    "oxygen": "OxygenSensor",
}

DEFAULT_CACHE_STEM = "scalar_session1"

DEFAULT_MEASUREMENT_COLUMNS = [
    "Conductivity (S/m)",
    "Density (kg/m3)",
    "Depth (m)",
    "Practical Salinity (psu)",
    "Pressure (decibar)",
    "Sigma-t (kg/m3)",
    "Sigma-theta (0 dbar) (kg/m3)",
    "Sound Speed (m/s)",
    "Temperature (C)",
]

PREFERRED_MEASUREMENT_COLUMNS = [
    *DEFAULT_MEASUREMENT_COLUMNS,
    "Fluorescence (mg/m3)",
    "Turbidity (NTU)",
    "Dissolved Oxygen (mL/L)",
    "Dissolved Oxygen (umol/L)",
    "Oxygen Saturation (%)",
]

DEFAULT_QC_COLUMNS = [
    "Conductivity QC Flag",
    "Density QC Flag",
    "Depth QC Flag",
    "Practical Salinity QC Flag",
    "Pressure QC Flag",
    "Sigma-t QC Flag",
    "Sigma-theta (0 dbar) QC Flag",
    "Sound Speed QC Flag",
    "Temperature QC Flag",
]

DEFAULT_KEEP_COLUMNS = ["Time UTC", *DEFAULT_MEASUREMENT_COLUMNS, *DEFAULT_QC_COLUMNS]


@dataclass(frozen=True)
class CacheBundlePaths:
    root: Path
    stem: str
    row_level_dir: Path
    window_cache_path: Path
    metadata_path: Path


def normalize_cache_stem(value: str) -> str:
    stem = value.strip()
    if not stem:
        raise ValueError("cache stem must not be empty")
    if any(separator in stem for separator in ("/", "\\")):
        raise ValueError("cache stem must be a simple name, not a path")
    return stem


def build_cache_bundle_paths(cache_root: Path, cache_stem: str = DEFAULT_CACHE_STEM) -> CacheBundlePaths:
    stem = normalize_cache_stem(cache_stem)
    root = cache_root.expanduser().resolve()
    return CacheBundlePaths(
        root=root,
        stem=stem,
        row_level_dir=root / f"{stem}_row_level",
        window_cache_path=root / f"{stem}_windowed_features.parquet",
        metadata_path=root / f"{stem}_metadata.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ONC scalar CSV files for the DAQathon Session 1 notebooks."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--cache-stem",
        default=DEFAULT_CACHE_STEM,
        help="Bundle name used to namespace row, window, and metadata cache artifacts.",
    )
    parser.add_argument("--target-flag", default="Conductivity QC Flag")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument(
        "--measurement-column",
        dest="measurement_columns",
        action="append",
        default=None,
        help="Repeatable measurement column override. Missing columns are ignored rather than treated as errors.",
    )
    parser.add_argument(
        "--merge-tolerance-seconds",
        type=int,
        default=120,
        help="Nearest-merge tolerance in seconds when joining secondary devices onto CTD rows.",
    )
    return parser.parse_args()


def clean_header_value(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("#"):
        cleaned = cleaned[1:]
    cleaned = cleaned.strip().strip('"')
    if cleaned.startswith("Time UTC"):
        return "Time UTC"
    return cleaned


def locate_header(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if row and "Time UTC" in row[0]:
                return line_number, [clean_header_value(value) for value in row]
    raise ValueError(f"Could not locate the tabular header in {path}")


def parse_measurement_columns(raw_values: list[str] | None) -> list[str] | None:
    if not raw_values:
        return None

    measurement_columns: list[str] = []
    for raw_value in raw_values:
        value = raw_value.strip()
        if not value:
            continue
        if value.startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("measurement column JSON must decode to a list")
            candidates = [str(item) for item in parsed]
        else:
            candidates = value.split(",")

        for candidate in candidates:
            column = candidate.strip()
            if column and column not in measurement_columns:
                measurement_columns.append(column)

    return measurement_columns or None


def detect_device(path: Path) -> str:
    name = path.name
    for device, marker in DEVICE_PATTERNS.items():
        if marker in name:
            return device
    return "other"


def read_scalar_csv(
    path: Path,
    sample_rows: int | None,
    required_columns: list[str] | None = None,
    *,
    allow_missing_columns: bool = True,
) -> pd.DataFrame:
    header_line_number, columns = locate_header(path)
    if "Time UTC" not in columns:
        raise ValueError(f"Missing Time UTC in {path}")

    use_columns = None
    if required_columns is not None:
        requested_columns = ["Time UTC", *[column for column in required_columns if column != "Time UTC"]]
        available_columns = [column for column in requested_columns if column in columns]
        missing_columns = [column for column in requested_columns if column not in columns]
        if missing_columns and not allow_missing_columns:
            raise ValueError(f"Missing expected columns in {path}: {missing_columns}")
        use_columns = available_columns

    frame = pd.read_csv(
        path,
        header=None,
        names=columns,
        skiprows=header_line_number,
        usecols=use_columns,
        nrows=sample_rows,
        low_memory=False,
    )

    frame["Time UTC"] = pd.to_datetime(
        frame["Time UTC"], utc=True, errors="coerce", format="ISO8601"
    )
    non_time_columns = [column for column in frame.columns if column != "Time UTC"]
    for column in non_time_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["Time UTC"]).sort_values("Time UTC").reset_index(drop=True)
    frame["source_file"] = path.name
    return frame


def choose_measurement_columns(
    columns: list[str],
    requested_measurement_columns: list[str] | None = None,
) -> list[str]:
    qc_columns = {column for column in columns if column.endswith("QC Flag")}
    excluded = {"Time UTC", "source_file", *qc_columns}

    if requested_measurement_columns:
        return [
            column
            for column in requested_measurement_columns
            if column in columns and column not in excluded
        ]

    ordered = [column for column in PREFERRED_MEASUREMENT_COLUMNS if column in columns and column not in excluded]
    extras = [column for column in columns if column not in excluded and column not in ordered]
    return ordered + extras


def build_window_features(
    frame: pd.DataFrame,
    target_flag: str,
    window_size: int,
    measurement_columns: list[str],
) -> pd.DataFrame:
    work = frame.reset_index(drop=True).copy()
    work["window_id"] = work.index // window_size

    named_aggs: dict[str, tuple[str, str | callable]] = {
        "window_start": ("Time UTC", "min"),
        "window_end": ("Time UTC", "max"),
        "source_file": ("source_file", "first"),
        "row_count": ("Time UTC", "size"),
        "issue_count": (target_flag, lambda values: int(values.isin([3, 4, 9]).sum())),
        "issue_rate": (target_flag, lambda values: float(values.isin([3, 4, 9]).mean())),
        "target_mode": (
            target_flag,
            lambda values: int(values.mode(dropna=True).iloc[0])
            if not values.mode(dropna=True).empty
            else -1,
        ),
    }

    for column in measurement_columns:
        named_aggs[f"{column}_mean"] = (column, "mean")
        named_aggs[f"{column}_std"] = (column, "std")
        named_aggs[f"{column}_min"] = (column, "min")
        named_aggs[f"{column}_max"] = (column, "max")

    return work.groupby("window_id", sort=True).agg(**named_aggs).reset_index(drop=True)


def clear_old_outputs(paths: CacheBundlePaths) -> None:
    if paths.row_level_dir.exists():
        for existing in paths.row_level_dir.glob("*.parquet"):
            existing.unlink()

    for existing_file in (paths.window_cache_path, paths.metadata_path):
        if existing_file.exists():
            existing_file.unlink()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    bundle_paths = build_cache_bundle_paths(args.cache_root, args.cache_stem)
    bundle_paths.root.mkdir(parents=True, exist_ok=True)
    bundle_paths.row_level_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_root.glob("*.csv"))
    if args.max_files is not None:
        csv_files = csv_files[: args.max_files]
    if not csv_files:
        raise SystemExit(f"No CSV files found in {data_root}")

    grouped_paths: dict[str, list[Path]] = {"ctd": [], "fluorometer": [], "oxygen": [], "other": []}
    for csv_path in csv_files:
        grouped_paths[detect_device(csv_path)].append(csv_path)

    if not grouped_paths["ctd"]:
        raise SystemExit("Expected at least one CTD file (ConductivityTemperatureDepth) for target labels")

    requested_measurement_columns = parse_measurement_columns(args.measurement_columns)

    clear_old_outputs(bundle_paths)
    bundle_paths.row_level_dir.mkdir(parents=True, exist_ok=True)

    ctd_parts = [read_scalar_csv(path, args.sample_rows) for path in grouped_paths["ctd"]]
    ctd_frame = pd.concat(ctd_parts, ignore_index=True).sort_values("Time UTC").reset_index(drop=True)

    merged = ctd_frame.copy()
    merge_tolerance = pd.Timedelta(seconds=args.merge_tolerance_seconds)

    for device_name in ("fluorometer", "oxygen", "other"):
        if not grouped_paths[device_name]:
            continue

        secondary_parts = [read_scalar_csv(path, args.sample_rows) for path in grouped_paths[device_name]]
        secondary = pd.concat(secondary_parts, ignore_index=True).sort_values("Time UTC").reset_index(drop=True)

        rename_map = {}
        for column in secondary.columns:
            if column in {"Time UTC", "source_file"}:
                continue
            if column in merged.columns:
                rename_map[column] = f"{column} [{device_name}]"
        if rename_map:
            secondary = secondary.rename(columns=rename_map)

        secondary = secondary.drop(columns=["source_file"], errors="ignore")

        merged = pd.merge_asof(
            merged.sort_values("Time UTC"),
            secondary.sort_values("Time UTC"),
            on="Time UTC",
            direction="nearest",
            tolerance=merge_tolerance,
        )

    merged = merged.sort_values("Time UTC").reset_index(drop=True)
    all_columns = merged.columns.tolist()
    qc_columns = [column for column in all_columns if column.endswith("QC Flag")]

    if args.target_flag not in merged.columns:
        raise SystemExit(f"Target flag '{args.target_flag}' not found after merge")

    measurement_columns = choose_measurement_columns(
        all_columns,
        requested_measurement_columns=requested_measurement_columns,
    )
    missing_measurement_columns = [
        column
        for column in (requested_measurement_columns or [])
        if column not in measurement_columns
    ]

    total_rows = len(merged)
    target_counts = Counter(merged[args.target_flag].dropna().astype(int).tolist())

    processed_files: list[dict[str, object]] = []
    window_frames: list[pd.DataFrame] = []
    for index, (source_file, source_frame) in enumerate(merged.groupby("source_file", sort=True), start=1):
        source_frame = source_frame.sort_values("Time UTC").reset_index(drop=True)
        part_path = bundle_paths.row_level_dir / f"part-{index:03d}.parquet"
        source_frame.to_parquet(part_path, index=False)

        window_frames.append(
            build_window_features(
                source_frame,
                target_flag=args.target_flag,
                window_size=args.window_size,
                measurement_columns=measurement_columns,
            )
        )

        processed_files.append(
            {
                "source_file": source_file,
                "row_count": int(len(source_frame)),
                "time_start": source_frame["Time UTC"].min().isoformat() if not source_frame.empty else None,
                "time_end": source_frame["Time UTC"].max().isoformat() if not source_frame.empty else None,
                "row_level_part": str(part_path),
            }
        )
        print(f"[{index}] wrote {part_path} with {len(source_frame):,} rows")

    window_frame = pd.concat(window_frames, ignore_index=True)
    window_frame.to_parquet(bundle_paths.window_cache_path, index=False)

    metadata = {
        "target_flag": args.target_flag,
        "cache_root": str(bundle_paths.root),
        "cache_stem": bundle_paths.stem,
        "measurement_columns": measurement_columns,
        "requested_measurement_columns": requested_measurement_columns or [],
        "missing_measurement_columns": missing_measurement_columns,
        "qc_columns": qc_columns,
        "processed_file_count": len(processed_files),
        "row_count": int(total_rows),
        "window_count": int(len(window_frame)),
        "sample_rows_per_file": args.sample_rows,
        "window_size": args.window_size,
        "target_distribution": {str(key): int(value) for key, value in sorted(target_counts.items())},
        "issue_fraction": (
            float(sum(target_counts.get(flag, 0) for flag in (3, 4, 9)) / total_rows)
            if total_rows
            else 0.0
        ),
        "processed_files": processed_files,
        "row_level_cache": str(bundle_paths.row_level_dir),
        "window_cache": str(bundle_paths.window_cache_path),
        "row_columns": all_columns,
        "window_columns": window_frame.columns.tolist(),
        "device_file_counts": {device: len(paths) for device, paths in grouped_paths.items()},
        "merge_tolerance_seconds": args.merge_tolerance_seconds,
    }
    bundle_paths.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "row_level_cache": str(bundle_paths.row_level_dir),
                "window_cache": str(bundle_paths.window_cache_path),
                "metadata": str(bundle_paths.metadata_path),
                "cache_stem": bundle_paths.stem,
                "rows": int(total_rows),
                "windows": int(len(window_frame)),
                "target_distribution": {str(key): int(value) for key, value in sorted(target_counts.items())},
                "device_file_counts": {device: len(paths) for device, paths in grouped_paths.items()},
                "missing_measurement_columns": missing_measurement_columns,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()