"""Shared modeling helpers used by the DAQathon notebooks and study scripts."""

from __future__ import annotations

import copy
import itertools
import json
import math
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency in some environments
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

MEASUREMENT_COLUMNS = [
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

DEFAULT_CACHE_STEM = "scalar_session1"
LEGACY_CACHE_STEMS = (DEFAULT_CACHE_STEM, "ctd_session1")

QC_FLAG_MEANINGS = {
    0: "no QC",
    1: "good",
    2: "probably good",
    3: "probably bad",
    4: "bad",
    6: "bad down-sampling",
    7: "averaged",
    8: "interpolated",
    9: "missing / NaN",
}

DEFAULT_FLAG_PALETTE = {
    0: "#94a3b8",
    1: "#1f77b4",
    2: "#60a5fa",
    3: "#ff7f0e",
    4: "#d62728",
    6: "#8b5cf6",
    7: "#14b8a6",
    8: "#a855f7",
    9: "#7f7f7f",
    34: "#e76f51",
}


@dataclass(frozen=True)
class CacheBundlePaths:
    root: Path
    stem: str
    row_level_dir: Path
    window_cache_path: Path
    metadata_path: Path


def _normalize_cache_stem(cache_stem: str) -> str:
    normalized = cache_stem.strip()
    if not normalized:
        raise ValueError("cache stem must not be empty")
    if any(separator in normalized for separator in ("/", "\\")):
        raise ValueError("cache stem must be a simple name, not a path")
    return normalized


def build_cache_bundle_paths(cache_dir: str | Path, cache_stem: str = DEFAULT_CACHE_STEM) -> CacheBundlePaths:
    cache_root = Path(cache_dir).expanduser().resolve()
    stem = _normalize_cache_stem(cache_stem)
    return CacheBundlePaths(
        root=cache_root,
        stem=stem,
        row_level_dir=cache_root / f"{stem}_row_level",
        window_cache_path=cache_root / f"{stem}_windowed_features.parquet",
        metadata_path=cache_root / f"{stem}_metadata.json",
    )


def resolve_cache_bundle_paths(
    cache_dir: str | Path,
    cache_stem: str | None = None,
) -> CacheBundlePaths:
    stems_to_try = [cache_stem] if cache_stem is not None else list(LEGACY_CACHE_STEMS)
    for stem in stems_to_try:
        candidate = build_cache_bundle_paths(cache_dir, stem)
        if candidate.metadata_path.exists():
            return candidate
    return build_cache_bundle_paths(cache_dir, stems_to_try[0])


def resolve_runtime_output_root(
    notebook_root: str | Path,
    *,
    slurm_tmpdir: str | None = None,
    scratch_dir: str | None = None,
) -> Path:
    """Choose the writable runtime output directory for the current environment.

    Precedence:

    1. ``$SLURM_TMPDIR``
    2. ``$SCRATCH``
    3. repo-local ``tmp/session1_outputs``
    """
    if slurm_tmpdir:
        return Path(slurm_tmpdir).expanduser().resolve() / "daqathon" / "session1_outputs"
    if scratch_dir:
        return Path(scratch_dir).expanduser().resolve() / "daqathon" / "session1_outputs"
    return Path(notebook_root).expanduser().resolve() / "tmp" / "session1_outputs"


def _render_copy_progress(
    label: str,
    copied_files: int,
    total_files: int,
    copied_bytes: int,
    total_bytes: int,
) -> None:
    """Render a simple terminal-friendly progress bar for notebook staging steps."""
    total_files = max(total_files, 1)
    total_bytes = max(total_bytes, 1)
    fraction = copied_bytes / total_bytes
    bar_width = 24
    filled = min(bar_width, int(round(bar_width * fraction)))
    bar = "#" * filled + "-" * (bar_width - filled)
    copied_gb = copied_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    message = (
        f"\r{label}: [{bar}] {fraction:6.1%} | "
        f"{copied_files:>4}/{total_files:<4} files | "
        f"{copied_gb:6.2f}/{total_gb:6.2f} GB"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def stage_directory_into_runtime(
    source_dir: str | Path,
    runtime_dir: str | Path,
    *,
    force: bool = False,
    show_progress: bool = False,
    progress_label: str = "Staging files",
) -> dict[str, object]:
    """Copy a read-only source directory into a writable runtime directory.

    This is used for FIR notebook runs where shared project storage is the
    long-lived source of truth, but node-local job storage such as
    ``$SLURM_TMPDIR`` is much faster for interactive work.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    runtime_dir = Path(runtime_dir).expanduser().resolve()

    if source_dir == runtime_dir:
        return {"staged": False, "reason": "runtime directory already points at the source directory"}
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    files_to_copy: list[tuple[Path, Path]] = []
    total_bytes = 0
    copied_files = 0
    copied_bytes = 0
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for source_path in source_dir.rglob("*"):
        relative_path = source_path.relative_to(source_dir)
        destination_path = runtime_dir / relative_path
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue

        should_copy = force or not destination_path.exists()
        if not should_copy and source_path.stat().st_size != destination_path.stat().st_size:
            should_copy = True

        if should_copy:
            files_to_copy.append((source_path, destination_path))
            total_bytes += source_path.stat().st_size

    total_files = len(files_to_copy)
    if show_progress:
        _render_copy_progress(progress_label, 0, total_files, 0, total_bytes)

    for source_path, destination_path in files_to_copy:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        file_size = source_path.stat().st_size
        shutil.copy2(source_path, destination_path)
        copied_files += 1
        copied_bytes += file_size
        if show_progress:
            _render_copy_progress(progress_label, copied_files, total_files, copied_bytes, total_bytes)

    if show_progress:
        if total_files == 0:
            sys.stdout.write(f"{progress_label}: already staged; nothing to copy.\n")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()

    return {
        "staged": True,
        "source_dir": str(source_dir),
        "runtime_dir": str(runtime_dir),
        "copied_files": copied_files,
        "total_files": total_files,
        "copied_gb": round(copied_bytes / (1024 ** 3), 3),
        "total_gb": round(total_bytes / (1024 ** 3), 3),
    }


def stage_cache_into_runtime(
    persistent_cache_dir: str | Path,
    runtime_cache_dir: str | Path,
    *,
    force: bool = False,
    show_progress: bool = False,
    progress_label: str = "Staging cache",
) -> dict[str, object]:
    """Copy a read-only prepared cache into a writable runtime directory."""
    stage_result = stage_directory_into_runtime(
        source_dir=persistent_cache_dir,
        runtime_dir=runtime_cache_dir,
        force=force,
        show_progress=show_progress,
        progress_label=progress_label,
    )
    if stage_result.get("staged"):
        stage_result["runtime_cache_dir"] = str(Path(runtime_cache_dir).expanduser().resolve())
    return stage_result


def evenly_spaced_take(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Take evenly spaced rows from a dataframe while preserving sort order."""
    if limit is None or len(frame) <= limit:
        return frame.reset_index(drop=True)
    indices = np.linspace(0, len(frame) - 1, num=limit, dtype=int)
    return frame.iloc[indices].reset_index(drop=True)


def select_part_paths(part_paths: list[Path], limit: int | None, mode: str = "spread") -> list[Path]:
    """Choose parquet parts either from the front or spread across the full time range."""
    if limit is None or limit >= len(part_paths):
        return part_paths
    if mode == "first":
        return part_paths[:limit]
    if mode == "spread":
        indices = np.linspace(0, len(part_paths) - 1, num=limit, dtype=int)
        selected = []
        seen = set()
        for index in indices:
            candidate = part_paths[int(index)]
            if candidate not in seen:
                selected.append(candidate)
                seen.add(candidate)
        return selected
    raise ValueError(f"Unsupported selection mode: {mode}")


def load_cache_bundle(
    cache_dir: str | Path,
    *,
    cache_stem: str | None = None,
    row_file_limit: int | None = None,
    part_selection_mode: str = "spread",
    rows_per_file: int = 45000,
    issue_rows_per_file: int = 12000,
    window_limit: int | None = None,
    target_flag: str = "Conductivity QC Flag",
    row_columns: list[str] | None = None,
    window_columns: list[str] | None = None,
) -> dict[str, object]:
    """Load metadata plus row- and window-level samples from the prepared cache."""
    bundle_paths = resolve_cache_bundle_paths(cache_dir, cache_stem=cache_stem)
    metadata_path = bundle_paths.metadata_path
    row_cache_dir = bundle_paths.row_level_dir
    window_cache_path = bundle_paths.window_cache_path

    metadata = json.loads(metadata_path.read_text())
    part_paths = sorted(row_cache_dir.glob("*.parquet"))
    if not part_paths:
        raise FileNotFoundError(f"No parquet parts found in {row_cache_dir}")

    selected_paths = select_part_paths(part_paths, limit=row_file_limit, mode=part_selection_mode)
    part_to_source = {
        Path(file_info["row_level_part"]).name: file_info["source_file"]
        for file_info in metadata["processed_files"]
    }
    selected_source_files = {part_to_source[path.name] for path in selected_paths}

    # Sample rows per parquet part so the notebooks stay responsive while still
    # covering the full time span of the deployment.
    row_df = load_row_level_sample(
        selected_paths,
        rows_per_file=rows_per_file,
        issue_rows_per_file=issue_rows_per_file,
        target_flag=target_flag,
        columns=row_columns,
    )

    window_df = pd.read_parquet(window_cache_path, columns=window_columns)
    window_df["window_start"] = pd.to_datetime(window_df["window_start"], utc=True)
    window_df["window_end"] = pd.to_datetime(window_df["window_end"], utc=True)
    window_df = window_df[window_df["source_file"].isin(selected_source_files)].sort_values("window_start")
    window_df = evenly_spaced_take(window_df, window_limit)

    return {
        "metadata": metadata,
        "selected_paths": selected_paths,
        "selected_source_files": selected_source_files,
        "row_df": row_df,
        "window_df": window_df.reset_index(drop=True),
    }


def load_row_level_sample(
    part_paths: list[Path],
    *,
    rows_per_file: int | None,
    issue_rows_per_file: int,
    target_flag: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a row-level sample from each parquet part with optional issue enrichment."""
    row_frames = []
    for path in part_paths:
        frame = pd.read_parquet(path, columns=columns).sort_values("Time UTC").reset_index(drop=True)
        frame["Time UTC"] = pd.to_datetime(frame["Time UTC"], utc=True)
        if rows_per_file is None:
            sampled_frame = frame
        else:
            base_limit = max(rows_per_file - issue_rows_per_file, 0)
            sampled_frame = evenly_spaced_take(frame, base_limit)
            if issue_rows_per_file > 0 and target_flag in frame.columns:
                issue_frame = frame[frame[target_flag].fillna(1).astype(int) != 1].reset_index(drop=True)
                issue_sample = evenly_spaced_take(issue_frame, issue_rows_per_file)
                sampled_frame = pd.concat([sampled_frame, issue_sample], ignore_index=True)
                sampled_frame = sampled_frame.drop_duplicates(subset=["Time UTC"]).sort_values("Time UTC")
                sampled_frame = evenly_spaced_take(sampled_frame, rows_per_file)
        row_frames.append(sampled_frame)
    return pd.concat(row_frames, ignore_index=True).sort_values("Time UTC").reset_index(drop=True)


def build_distribution_frame(metadata: dict[str, object], df: pd.DataFrame, target_flag: str) -> pd.DataFrame:
    """Compare the full target distribution with the currently loaded sample."""
    full_target_counts = pd.Series(
        {int(key): int(value) for key, value in metadata["target_distribution"].items()}
    ).sort_index()
    sample_target_counts = df[target_flag].dropna().astype(int).value_counts().sort_index()
    return pd.DataFrame({"full_cache": full_target_counts, "loaded_sample": sample_target_counts}).fillna(0).astype(int)


def build_model_frame(
    df: pd.DataFrame,
    *,
    target_flag: str,
    task_mode: str,
    model_row_limit: int | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create the baseline tabular feature frame used by the supervised models."""
    model_df = df.copy()
    model_df["issue"] = model_df[target_flag].isin([3, 4, 9]).astype(int)
    model_df["hour_utc"] = model_df["Time UTC"].dt.hour
    model_df["minute_utc"] = model_df["Time UTC"].dt.minute
    model_df["day_of_year"] = model_df["Time UTC"].dt.dayofyear

    # Absolute deltas are a simple way to expose sudden sensor changes to the model.
    for column in MEASUREMENT_COLUMNS:
        model_df[f"{column} abs_delta"] = model_df[column].diff().abs().fillna(0.0)

    feature_columns = MEASUREMENT_COLUMNS + [f"{column} abs_delta" for column in MEASUREMENT_COLUMNS] + [
        "hour_utc",
        "minute_utc",
        "day_of_year",
    ]

    model_df = model_df.dropna(subset=[target_flag]).reset_index(drop=True)
    if model_row_limit is not None and len(model_df) > model_row_limit:
        model_df = evenly_spaced_take(model_df, model_row_limit)

    if task_mode == "multiclass":
        model_df["model_target"] = model_df[target_flag].astype(int)
    elif task_mode == "binary":
        model_df["model_target"] = model_df["issue"].astype(int)
    else:
        raise ValueError(f"Unsupported task mode: {task_mode}")

    active_labels = sorted(model_df["model_target"].dropna().astype(int).unique().tolist())
    return model_df, feature_columns, active_labels


def add_temporal_context_features(
    frame: pd.DataFrame,
    *,
    lag_steps: tuple[int, ...] = (1, 3, 5),
    rolling_windows: tuple[int, ...] = (5, 15),
) -> tuple[pd.DataFrame, list[str]]:
    """Add lag and rolling statistics that summarize recent sensor history."""
    work = frame.sort_values("Time UTC").reset_index(drop=True).copy()
    context_columns: list[str] = []

    for column in MEASUREMENT_COLUMNS:
        for lag in lag_steps:
            feature_name = f"{column} lag_{lag}"
            work[feature_name] = work[column].shift(lag)
            context_columns.append(feature_name)

        for window in rolling_windows:
            mean_name = f"{column} roll_mean_{window}"
            std_name = f"{column} roll_std_{window}"
            work[mean_name] = work[column].rolling(window=window, min_periods=1).mean()
            work[std_name] = work[column].rolling(window=window, min_periods=2).std()
            context_columns.extend([mean_name, std_name])

    work[context_columns] = work[context_columns].replace([np.inf, -np.inf], np.nan)
    return work, context_columns


def apply_target_strategy(frame: pd.DataFrame, target_flag: str, strategy: str) -> tuple[pd.DataFrame, list[int], str]:
    """Map raw QC flags into one of the teaching target strategies."""
    work = frame.copy()
    if strategy == "multiclass_1_3_4_9":
        work["strategy_target"] = work[target_flag].astype(int)
        labels = [1, 3, 4, 9]
        average = "macro"
    elif strategy == "collapsed_1_34_9":
        mapping = {1: 1, 3: 34, 4: 34, 9: 9}
        work["strategy_target"] = work[target_flag].astype(int).map(mapping)
        labels = [1, 34, 9]
        average = "macro"
    elif strategy == "binary_issue":
        work["strategy_target"] = work[target_flag].astype(int).isin([3, 4, 9]).astype(int)
        labels = [0, 1]
        average = "binary"
    else:
        raise ValueError(f"Unsupported target strategy: {strategy}")

    work = work.dropna(subset=["strategy_target"]).copy()
    work["strategy_target"] = work["strategy_target"].astype(int)
    return work, labels, average


def contiguous_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted dataframe into contiguous train/valid/test segments."""
    train_end = int(len(frame) * train_fraction)
    valid_end = int(len(frame) * (train_fraction + validation_fraction))
    train_frame = frame.iloc[:train_end].copy()
    valid_frame = frame.iloc[train_end:valid_end].copy()
    test_frame = frame.iloc[valid_end:].copy()
    return train_frame, valid_frame, test_frame


def report_average(task_mode: str) -> str:
    """Return the F1 averaging mode that matches the task definition."""
    return "binary" if task_mode == "binary" else "macro"


def fit_random_forest(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
    config: dict[str, object],
) -> Pipeline:
    """Train the baseline Random Forest pipeline used in the notebooks."""
    rf_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config["n_estimators"],
                    max_depth=config["max_depth"],
                    min_samples_leaf=config["min_samples_leaf"],
                    min_samples_split=config.get("min_samples_split", 2),
                    max_features=config.get("max_features", "sqrt"),
                    class_weight=config.get("class_weight", "balanced_subsample"),
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    rf_pipeline.fit(train_df[feature_columns], train_df["model_target"])
    return rf_pipeline


def fit_extra_trees(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
    config: dict[str, object],
    target_column: str = "model_target",
) -> Pipeline:
    """Train an ExtraTrees pipeline with the same preprocessing pattern as RF."""
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=config["n_estimators"],
                    max_depth=config["max_depth"],
                    min_samples_leaf=config["min_samples_leaf"],
                    min_samples_split=config.get("min_samples_split", 2),
                    max_features=config.get("max_features", "sqrt"),
                    class_weight=config.get("class_weight", "balanced_subsample"),
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(train_df[feature_columns], train_df[target_column])
    return pipeline


def evaluate_classifier(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    labels: list[int],
    task_mode: str,
) -> dict[str, object]:
    """Compute prediction outputs and summary metrics for a fitted classifier."""
    y_true = frame["model_target"]
    y_pred = pipeline.predict(frame[feature_columns])
    return {
        "f1": float(f1_score(y_true, y_pred, average=report_average(task_mode), zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=False),
        "predictions": y_pred,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels, normalize="true"),
    }


def run_rf_search(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    labels: list[int],
    task_mode: str,
    seed: int,
    search_space: dict[str, list[object]],
) -> tuple[pd.DataFrame, dict[str, object], Pipeline]:
    """Grid-search a small Random Forest search space and keep the best model."""
    keys = list(search_space.keys())
    results = []
    best_score = -math.inf
    best_config = None
    best_model = None

    for trial_index, values in enumerate(itertools.product(*(search_space[key] for key in keys)), start=1):
        config = dict(zip(keys, values))
        model = fit_random_forest(train_df, feature_columns, seed=seed, config=config)
        valid_result = evaluate_classifier(model, valid_df, feature_columns, labels=labels, task_mode=task_mode)
        row = {"trial": trial_index, **config, "validation_f1": valid_result["f1"]}
        results.append(row)
        if valid_result["f1"] > best_score:
            best_score = valid_result["f1"]
            best_config = config
            best_model = model

    result_frame = pd.DataFrame(results).sort_values("validation_f1", ascending=False).reset_index(drop=True)
    return result_frame, best_config, best_model


def clean_source_file_label(value: str) -> str:
    """Trim verbose filename suffixes for cleaner notebook display tables."""
    return str(value).replace(".csv", "").split("_2025")[0].split("_2026")[0]


def _span_boundaries(times: pd.Series, start_index: int, stop_index: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Expand a contiguous run to midpoints with neighboring timestamps for clearer shading."""
    times = pd.to_datetime(times, utc=True).reset_index(drop=True)
    start_time = times.iloc[start_index]
    end_time = times.iloc[stop_index]

    if start_index > 0:
        previous_time = times.iloc[start_index - 1]
        start_time = previous_time + (start_time - previous_time) / 2
    if stop_index < len(times) - 1:
        next_time = times.iloc[stop_index + 1]
        end_time = end_time + (next_time - end_time) / 2

    return start_time, end_time


def _flag_span_boundaries(panel: pd.DataFrame, start_index: int, stop_index: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Delegate QC-flag shading boundaries to the generic span helper."""
    return _span_boundaries(panel["Time UTC"], start_index, stop_index)


def _iter_flag_spans(panel: pd.DataFrame, target_flag: str) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    """Return contiguous non-good QC regions as (flag, span_start, span_end)."""
    flag_values = panel[target_flag].copy()
    if pd.api.types.is_numeric_dtype(flag_values):
        flag_values = flag_values.fillna(9).astype(int)
    else:
        flag_values = pd.to_numeric(flag_values, errors="coerce").fillna(9).astype(int)

    spans: list[tuple[int, pd.Timestamp, pd.Timestamp]] = []
    run_start = 0

    for index in range(1, len(panel) + 1):
        reached_end = index == len(panel)
        if reached_end or flag_values.iloc[index] != flag_values.iloc[run_start]:
            run_flag = int(flag_values.iloc[run_start])
            if run_flag != 1:
                span_start, span_end = _flag_span_boundaries(panel, run_start, index - 1)
                spans.append((run_flag, span_start, span_end))
            run_start = index

    return spans


def parse_optional_utc_datetime(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    """Parse an optional datetime-like value into a timezone-aware UTC timestamp."""
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, utc=True)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def build_row_part_index(metadata: dict[str, object], row_cache_dir: str | Path) -> pd.DataFrame:
    """Convert cache metadata into a searchable dataframe of parquet parts."""
    row_cache_path = Path(row_cache_dir)
    records = []
    for file_info in metadata.get("processed_files", []):
        time_start = parse_optional_utc_datetime(file_info.get("time_start"))
        time_end = parse_optional_utc_datetime(file_info.get("time_end"))
        records.append(
            {
                "source_file": file_info["source_file"],
                "time_start": time_start,
                "time_end": time_end,
                "row_part_path": row_cache_path / Path(str(file_info["row_level_part"])).name,
            }
        )
    return pd.DataFrame(records)


def select_overlapping_row_parts(
    metadata: dict[str, object],
    row_cache_dir: str | Path,
    *,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Return only the parquet parts that overlap the requested time interval."""
    part_index = build_row_part_index(metadata, row_cache_dir)
    if part_index.empty:
        return part_index

    start_ts = parse_optional_utc_datetime(start)
    end_ts = parse_optional_utc_datetime(end)

    if start_ts is not None:
        part_index = part_index[part_index["time_end"].isna() | (part_index["time_end"] >= start_ts)]
    if end_ts is not None:
        part_index = part_index[part_index["time_start"].isna() | (part_index["time_start"] <= end_ts)]
    return part_index.sort_values("time_start").reset_index(drop=True)


def load_rows_for_time_range(
    metadata: dict[str, object],
    row_cache_dir: str | Path,
    *,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load only the row-level parquet parts needed for a selected time interval."""
    overlapping_parts = select_overlapping_row_parts(metadata, row_cache_dir, start=start, end=end)
    if overlapping_parts.empty:
        if columns:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame()

    start_ts = parse_optional_utc_datetime(start)
    end_ts = parse_optional_utc_datetime(end)
    frames = []
    for row in overlapping_parts.itertuples(index=False):
        frame = pd.read_parquet(row.row_part_path, columns=columns).sort_values("Time UTC").reset_index(drop=True)
        frame["Time UTC"] = pd.to_datetime(frame["Time UTC"], utc=True)
        if start_ts is not None:
            frame = frame[frame["Time UTC"] >= start_ts]
        if end_ts is not None:
            frame = frame[frame["Time UTC"] <= end_ts]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        if columns:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values("Time UTC").reset_index(drop=True)


def select_time_range(
    reference_frame: pd.DataFrame,
    *,
    time_column: str = "Time UTC",
    label_column: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    auto_select: bool = True,
    max_points: int = 800,
    preferred_labels: tuple[int, ...] = (4, 3, 9),
) -> dict[str, object]:
    """Choose either an explicit or representative interval for a notebook demo."""
    work = reference_frame.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True).copy()
    if work.empty:
        raise ValueError("Cannot select a time range from an empty frame.")
    work[time_column] = pd.to_datetime(work[time_column], utc=True)

    start_ts = parse_optional_utc_datetime(start)
    end_ts = parse_optional_utc_datetime(end)

    if start_ts is not None or end_ts is not None:
        if start_ts is None:
            start_ts = work[time_column].min()
        if end_ts is None:
            end_ts = work[time_column].max()
        if end_ts < start_ts:
            raise ValueError("Range end must be greater than or equal to range start.")
        explicit_slice = work[(work[time_column] >= start_ts) & (work[time_column] <= end_ts)].copy()
        return {
            "start": start_ts,
            "end": end_ts,
            "slice": explicit_slice.reset_index(drop=True),
            "selection_mode": "explicit",
            "selected_label": None,
        }

    if not auto_select:
        selected = evenly_spaced_take(work, min(max_points, len(work)))
        return {
            "start": selected[time_column].min(),
            "end": selected[time_column].max(),
            "slice": selected.reset_index(drop=True),
            "selection_mode": "manual-full-range",
            "selected_label": None,
        }

    chosen_index = len(work) // 2
    chosen_label = None
    # When auto-selecting, bias toward intervals that contain more informative
    # non-good labels so the demo is visually useful.
    if label_column is not None and label_column in work.columns:
        label_series = pd.to_numeric(work[label_column], errors="coerce")
        for label in preferred_labels:
            candidate_indices = work.index[label_series == label].tolist()
            if candidate_indices:
                chosen_index = candidate_indices[len(candidate_indices) // 2]
                chosen_label = label
                break

    point_count = min(max_points, len(work))
    start_index = max(chosen_index - point_count // 2, 0)
    stop_index = min(start_index + point_count, len(work))
    start_index = max(stop_index - point_count, 0)
    selected = work.iloc[start_index:stop_index].copy().reset_index(drop=True)

    return {
        "start": selected[time_column].min(),
        "end": selected[time_column].max(),
        "slice": selected,
        "selection_mode": "auto",
        "selected_label": chosen_label,
    }


def infer_interval_origin(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    split_frames: dict[str, pd.DataFrame],
    *,
    time_column: str = "Time UTC",
) -> str:
    """Label an interval as train, validation, test, mixed, or outside range."""
    start_ts = parse_optional_utc_datetime(start)
    end_ts = parse_optional_utc_datetime(end)
    if start_ts is None or end_ts is None:
        return "unknown"

    overlaps: list[str] = []
    for split_name, frame in split_frames.items():
        if frame.empty or time_column not in frame.columns:
            continue
        frame_times = pd.to_datetime(frame[time_column], utc=True)
        frame_start = frame_times.min()
        frame_end = frame_times.max()
        if pd.isna(frame_start) or pd.isna(frame_end):
            continue
        if start_ts <= frame_end and end_ts >= frame_start:
            overlaps.append(split_name)

    if not overlaps:
        return "outside modeled range"
    if len(overlaps) == 1:
        return overlaps[0]
    return f"mixed ({', '.join(overlaps)})"


def build_labeled_intervals(
    frame: pd.DataFrame,
    *,
    time_column: str,
    label_column: str,
    fill_value: object | None = None,
) -> pd.DataFrame:
    """Collapse row-by-row labels into contiguous labeled time spans."""
    work = frame[[time_column, label_column]].copy().dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["start", "end", "label"])

    work[time_column] = pd.to_datetime(work[time_column], utc=True)
    if fill_value is not None:
        work[label_column] = work[label_column].fillna(fill_value)
    else:
        work = work.dropna(subset=[label_column]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["start", "end", "label"])

    labels = work[label_column].tolist()
    intervals = []
    run_start = 0
    for index in range(1, len(work) + 1):
        reached_end = index == len(work)
        if reached_end or labels[index] != labels[run_start]:
            span_start, span_end = _span_boundaries(work[time_column], run_start, index - 1)
            intervals.append({"start": span_start, "end": span_end, "label": labels[run_start]})
            run_start = index
    return pd.DataFrame(intervals)


def merge_adjacent_intervals(
    interval_frame: pd.DataFrame,
    *,
    label_column: str = "label",
    start_column: str = "start",
    end_column: str = "end",
) -> pd.DataFrame:
    """Merge touching intervals that share the same label."""
    if interval_frame.empty:
        return interval_frame.copy()

    work = interval_frame.copy().sort_values(start_column).reset_index(drop=True)
    merged = [work.iloc[0].to_dict()]
    for row in work.iloc[1:].itertuples(index=False):
        current = row._asdict()
        previous = merged[-1]
        if current[label_column] == previous[label_column] and current[start_column] <= previous[end_column]:
            previous[end_column] = max(previous[end_column], current[end_column])
        else:
            merged.append(current)
    return pd.DataFrame(merged)


def build_label_palette(labels: list[object], *, palette: dict[object, str] | None = None) -> dict[object, object]:
    """Assign display colors to an ordered set of labels."""
    if palette is not None:
        return {label: palette.get(label, palette.get(int(label), "#64748b")) if isinstance(label, (int, np.integer)) else palette.get(label, "#64748b") for label in labels}

    unique_labels = list(dict.fromkeys(labels))
    color_values = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
    return {label: color_values[index] for index, label in enumerate(unique_labels)}


def plot_time_series_with_bands(
    row_frame: pd.DataFrame,
    *,
    band_specs: list[dict[str, object]],
    measurement_column: str = "Conductivity (S/m)",
    secondary_column: str = "Temperature (C)",
    max_points: int | None = None,
    title: str = "Time-range model demo",
) -> plt.Figure:
    """Plot sensor traces with one or more aligned label-band panels underneath."""
    if row_frame.empty:
        raise ValueError("row_frame must not be empty.")

    plot_frame = row_frame.sort_values("Time UTC").reset_index(drop=True).copy()
    plot_frame["Time UTC"] = pd.to_datetime(plot_frame["Time UTC"], utc=True)
    if max_points is not None:
        plot_frame = evenly_spaced_take(plot_frame, max_points)

    row_count = 1 + len(band_specs)
    height_ratios = [3.6] + [0.9] * len(band_specs)
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(15, 3.2 + 1.3 * len(band_specs)),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if row_count == 1:
        axes = [axes]

    main_axis = axes[0]
    band_axes = axes[1:]
    main_axis.plot(plot_frame["Time UTC"], plot_frame[measurement_column], color="#0f172a", linewidth=1.8, label=measurement_column)
    main_axis.set_ylabel(measurement_column)
    main_axis.grid(alpha=0.25)

    twin_axis = main_axis.twinx()
    twin_axis.plot(plot_frame["Time UTC"], plot_frame[secondary_column], color="#059669", linewidth=1.2, alpha=0.7, label=secondary_column)
    twin_axis.set_ylabel(secondary_column, color="#059669")
    twin_axis.tick_params(axis="y", colors="#059669")

    main_axis.legend(
        handles=[
            Line2D([0], [0], color="#0f172a", linewidth=2, label=measurement_column),
            Line2D([0], [0], color="#059669", linewidth=2, label=secondary_column),
        ],
        loc="upper left",
        frameon=True,
    )

    for axis, spec in zip(band_axes, band_specs):
        intervals = spec["intervals"].copy()
        label_order = list(dict.fromkeys(intervals["label"].tolist())) if not intervals.empty else []
        palette = build_label_palette(label_order, palette=spec.get("palette"))
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.grid(alpha=0.15)
        axis.set_title(str(spec["title"]), loc="left", fontsize=11, pad=4)

        if intervals.empty:
            axis.text(0.01, 0.5, "No intervals in this selected range.", transform=axis.transAxes, va="center")
            continue

        intervals = intervals.sort_values("start").reset_index(drop=True)
        for row in intervals.itertuples(index=False):
            color = palette.get(row.label, "#64748b")
            axis.axvspan(row.start, row.end, color=color, alpha=0.7, linewidth=0)

        if len(intervals) <= 14:
            for row in intervals.itertuples(index=False):
                midpoint = row.start + (row.end - row.start) / 2
                axis.text(midpoint, 0.5, str(row.label), ha="center", va="center", fontsize=9, color="#111827")

        legend_handles = [
            Patch(facecolor=palette.get(label, "#64748b"), edgecolor="none", alpha=0.7, label=str(label))
            for label in label_order[:8]
        ]
        if legend_handles:
            axis.legend(handles=legend_handles, loc="upper left", ncol=min(4, len(legend_handles)), frameon=True)

    axes[-1].set_xlabel("Time UTC")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def compute_interval_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    labels: list[int],
    average: str,
    target_names: list[str] | None = None,
) -> dict[str, object]:
    """Compute the text report, normalized confusion matrix, and F1 for a slice."""
    label_names = target_names or [str(label) for label in labels]
    if average == "binary":
        interval_f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    else:
        interval_f1 = float(f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0))
    return {
        "f1": interval_f1,
        "report_text": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            normalize="true",
        ),
        "display_labels": label_names,
    }


def build_window_classification_interval_data(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    task_mode: str,
    window_size: int,
    label_reduction: str,
    time_column: str = "Time UTC",
) -> dict[str, object]:
    """Prepare fixed windows for window-level CNN or transformer inference."""
    work = frame[[time_column, *feature_columns, target_column]].copy().dropna().sort_values(time_column).reset_index(drop=True)
    usable_rows = (len(work) // window_size) * window_size
    work = work.iloc[:usable_rows].copy()
    if work.empty:
        return {"raw_sequences": np.empty((0, window_size, len(feature_columns)), dtype=np.float32), "window_frame": pd.DataFrame(), "class_labels": []}

    raw_sequences = work[feature_columns].to_numpy(dtype=np.float32).reshape(-1, window_size, len(feature_columns))
    raw_targets = work[target_column].to_numpy().reshape(-1, window_size)
    raw_times = pd.to_datetime(work[time_column], utc=True).to_numpy().reshape(-1, window_size)

    if task_mode == "multiclass":
        true_labels = np.array([reduce_window_target(row, mode=label_reduction) for row in raw_targets], dtype=int)
        class_labels = sorted(np.unique(true_labels).tolist())
    else:
        true_labels = raw_targets.max(axis=1).astype(int)
        class_labels = [0, 1]

    window_frame = pd.DataFrame(
        {
            "window_start": pd.to_datetime(raw_times[:, 0], utc=True),
            "window_end": pd.to_datetime(raw_times[:, -1], utc=True),
            "true_label": true_labels,
        }
    )
    return {
        "raw_sequences": raw_sequences,
        "window_frame": window_frame,
        "class_labels": class_labels,
    }


def build_sequence_label_interval_data(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    time_column: str = "Time UTC",
) -> dict[str, object]:
    """Prepare fixed windows for per-timestamp sequence-label inference."""
    work = frame[[time_column, *feature_columns, target_column]].copy().dropna().sort_values(time_column).reset_index(drop=True)
    usable_rows = (len(work) // window_size) * window_size
    work = work.iloc[:usable_rows].copy()
    if work.empty:
        return {
            "raw_sequences": np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            "raw_targets": np.empty((0, window_size)),
            "raw_times": np.empty((0, window_size), dtype="datetime64[ns]"),
        }

    raw_sequences = work[feature_columns].to_numpy(dtype=np.float32).reshape(-1, window_size, len(feature_columns))
    raw_targets = work[target_column].to_numpy().reshape(-1, window_size)
    raw_times = pd.to_datetime(work[time_column], utc=True).to_numpy().reshape(-1, window_size)
    return {"raw_sequences": raw_sequences, "raw_targets": raw_targets, "raw_times": raw_times}


def predict_cnn_window_model(
    model: nn.Module,
    raw_sequences: np.ndarray,
    *,
    task_mode: str,
    class_labels: list[int],
    device: str,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run a window-classification CNN on raw sequences and decode labels."""
    _require_torch()
    if len(raw_sequences) == 0:
        return np.array([], dtype=int)

    model.eval()
    tensor = torch.from_numpy(np.transpose(raw_sequences, (0, 2, 1))).float()
    normalized = (tensor - torch.as_tensor(channel_mean).float()) / torch.as_tensor(channel_std).float()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(normalized), batch_size):
            batch = normalized[start_index : start_index + batch_size].to(device)
            logits = model(batch)
            if task_mode == "multiclass":
                batch_predictions = logits.argmax(dim=1).cpu().numpy()
                batch_predictions = np.array([class_labels[index] for index in batch_predictions], dtype=int)
            else:
                batch_predictions = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long().cpu().numpy().astype(int)
            predictions.append(batch_predictions)
    return np.concatenate(predictions) if predictions else np.array([], dtype=int)


def predict_transformer_window_model(
    model: nn.Module,
    raw_sequences: np.ndarray,
    *,
    task_mode: str,
    class_labels: list[int],
    device: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run the notebook transformer on raw sequences and decode labels."""
    _require_torch()
    if len(raw_sequences) == 0:
        return np.array([], dtype=int)

    model.eval()
    tensor = torch.from_numpy(raw_sequences).float()
    normalized = (tensor - torch.as_tensor(feature_mean).float()) / torch.as_tensor(feature_std).float()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(normalized), batch_size):
            batch = normalized[start_index : start_index + batch_size].to(device)
            logits = model(batch)
            if task_mode == "multiclass":
                batch_predictions = logits.argmax(dim=1).cpu().numpy()
                batch_predictions = np.array([class_labels[index] for index in batch_predictions], dtype=int)
            else:
                batch_predictions = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long().cpu().numpy().astype(int)
            predictions.append(batch_predictions)
    return np.concatenate(predictions) if predictions else np.array([], dtype=int)


def predict_sequence_label_cnn(
    model: nn.Module,
    raw_sequences: np.ndarray,
    *,
    task_mode: str,
    class_labels: list[int],
    device: str,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    """Run a sequence-labeling CNN and return one prediction per timestamp."""
    _require_torch()
    if len(raw_sequences) == 0:
        return np.empty((0, 0), dtype=int)

    model.eval()
    tensor = torch.from_numpy(np.transpose(raw_sequences, (0, 2, 1))).float()
    normalized = (tensor - torch.as_tensor(channel_mean).float()) / torch.as_tensor(channel_std).float()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(normalized), batch_size):
            batch = normalized[start_index : start_index + batch_size].to(device)
            logits = model(batch)
            if task_mode == "multiclass":
                batch_predictions = logits.argmax(dim=1).cpu().numpy()
                batch_predictions = np.vectorize(lambda index: class_labels[int(index)])(batch_predictions).astype(int)
            else:
                batch_predictions = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long().cpu().numpy().astype(int)
            predictions.append(batch_predictions)
    return np.concatenate(predictions, axis=0) if predictions else np.empty((0, 0), dtype=int)


def plot_flag_examples(
    df: pd.DataFrame,
    *,
    target_flag: str,
    measurement_column: str = "Conductivity (S/m)",
    secondary_column: str = "Temperature (C)",
    points_per_panel: int = 300,
    classes: tuple[int, ...] = (1, 3, 4, 9),
    region_alpha: float = 0.18,
    show_flag_points: bool = True,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot representative QC-flag examples with shaded regions and sensor traces."""
    available_classes = [flag for flag in classes if flag in set(df[target_flag].dropna().astype(int).unique())]
    if not available_classes:
        raise ValueError(f"No requested classes found in {target_flag}")

    work = df.sort_values("Time UTC").reset_index(drop=True).copy()
    fig, axes = plt.subplots(len(available_classes), 1, figsize=(15, 3.6 * len(available_classes)), sharex=False)
    if len(available_classes) == 1:
        axes = [axes]

    example_rows = []
    colors = {1: "#1f77b4", 3: "#ff7f0e", 4: "#d62728", 9: "#7f7f7f"}
    line_color = "#0f172a"
    temp_color = "#059669"
    for axis, flag in zip(axes, available_classes):
        flag_rows = work.index[work[target_flag].fillna(-1).astype(int) == flag].tolist()
        center_index = flag_rows[len(flag_rows) // 2]
        start = max(center_index - points_per_panel // 2, 0)
        stop = min(start + points_per_panel, len(work))
        panel = work.iloc[start:stop].copy()
        panel_spans = _iter_flag_spans(panel, target_flag)
        flags_in_panel = sorted({span_flag for span_flag, _, _ in panel_spans})

        # Shade all non-good QC spans so participants can see the local context
        # around the example class rather than only isolated points.
        for span_flag, span_start, span_end in panel_spans:
            axis.axvspan(
                span_start,
                span_end,
                color=colors.get(span_flag, "#9467bd"),
                alpha=region_alpha,
                linewidth=0,
                zorder=0,
            )

        axis.plot(panel["Time UTC"], panel[measurement_column], color=line_color, linewidth=1.8, label=measurement_column)
        if show_flag_points and flag != 1:
            target_points = panel.loc[panel[target_flag].fillna(-1).astype(int) == flag, ["Time UTC", measurement_column]]
            target_points = target_points.dropna(subset=[measurement_column])
            if not target_points.empty:
                axis.scatter(
                    target_points["Time UTC"],
                    target_points[measurement_column],
                    color=colors.get(flag, "#9467bd"),
                    s=20,
                    zorder=3,
                    label=f"Rows with QC flag {flag}",
                )
        axis.set_title(f"Example around QC flag {flag}: {QC_FLAG_MEANINGS.get(flag, 'unknown')}")
        axis.set_ylabel(measurement_column)
        axis.grid(alpha=0.25)

        twin_axis = axis.twinx()
        twin_axis.plot(panel["Time UTC"], panel[secondary_column], color=temp_color, linewidth=1.2, alpha=0.6, label=secondary_column)
        twin_axis.set_ylabel(secondary_column, color=temp_color)
        twin_axis.tick_params(axis="y", colors=temp_color)

        legend_handles = [
            Line2D([0], [0], color=line_color, linewidth=2, label=measurement_column),
            Line2D([0], [0], color=temp_color, linewidth=2, label=secondary_column),
        ]
        if show_flag_points and flag != 1:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors.get(flag, "#9467bd"),
                    markersize=7,
                    label=f"QC flag {flag} points",
                )
            )
        legend_handles.extend(
            [
                Patch(
                    facecolor=colors.get(span_flag, "#9467bd"),
                    edgecolor="none",
                    alpha=region_alpha,
                    label=f"QC region {span_flag}: {QC_FLAG_MEANINGS.get(span_flag, 'unknown')}",
                )
                for span_flag in flags_in_panel
            ]
        )
        axis.legend(handles=legend_handles, loc="upper left", frameon=True)

        example_rows.append(
            {
                "qc_flag": flag,
                "meaning": QC_FLAG_MEANINGS.get(flag, "unknown"),
                "panel_start": panel["Time UTC"].min(),
                "panel_end": panel["Time UTC"].max(),
                "example_time": work.loc[center_index, "Time UTC"],
                "source_file": clean_source_file_label(work.loc[center_index, "source_file"]),
            }
        )

    axes[-1].set_xlabel("Time UTC")
    fig.suptitle("Representative time-series examples for different QC flags", y=1.02)
    fig.tight_layout()
    return fig, pd.DataFrame(example_rows)


def plot_cluster_window_examples(
    clustered_window_df: pd.DataFrame,
    *,
    source_to_row_part: dict[str, str | Path],
    measurement_column: str = "Conductivity (S/m)",
    secondary_column: str = "Temperature (C)",
    examples_per_cluster: int = 1,
    context_points: int = 1500,
    highlight_alpha: float = 0.22,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Show representative k-means windows inside a wider sensor-time context."""
    required_columns = {"cluster", "window_start", "window_end", "source_file", "distance_to_centroid", "issue_rate"}
    missing = required_columns.difference(clustered_window_df.columns)
    if missing:
        raise ValueError(f"clustered_window_df is missing required columns: {sorted(missing)}")

    work = clustered_window_df.sort_values(["cluster", "distance_to_centroid", "issue_rate"], ascending=[True, True, False]).copy()
    cluster_ids = sorted(work["cluster"].dropna().astype(int).unique().tolist())
    if not cluster_ids:
        raise ValueError("No clusters found in clustered_window_df")

    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
    cluster_palette = {cluster_id: cluster_colors[idx] for idx, cluster_id in enumerate(cluster_ids)}

    # Use the windows closest to each centroid as readable cluster prototypes.
    representative_rows = (
        work.groupby("cluster", group_keys=False)
        .head(examples_per_cluster)
        .reset_index(drop=True)
    )

    figure_row_count = len(representative_rows)
    fig, axes = plt.subplots(figure_row_count, 1, figsize=(15, 3.8 * figure_row_count), sharex=False)
    if figure_row_count == 1:
        axes = [axes]

    example_records = []
    for axis, (_, row) in zip(axes, representative_rows.iterrows()):
        source_file = row["source_file"]
        row_part_path = Path(source_to_row_part[source_file])
        panel = pd.read_parquet(
            row_part_path,
            columns=["Time UTC", measurement_column, secondary_column],
        ).sort_values("Time UTC").reset_index(drop=True)
        panel["Time UTC"] = pd.to_datetime(panel["Time UTC"], utc=True)

        window_start = pd.to_datetime(row["window_start"], utc=True)
        window_end = pd.to_datetime(row["window_end"], utc=True)
        center_time = window_start + (window_end - window_start) / 2
        time_delta = (panel["Time UTC"] - center_time).abs()
        center_index = int(time_delta.idxmin())
        start_index = max(center_index - context_points // 2, 0)
        stop_index = min(start_index + context_points, len(panel))
        context_panel = panel.iloc[start_index:stop_index].copy()

        cluster_id = int(row["cluster"])
        cluster_color = cluster_palette[cluster_id]

        axis.plot(
            context_panel["Time UTC"],
            context_panel[measurement_column],
            color="#0f172a",
            linewidth=1.8,
            label=measurement_column,
        )
        target_points = context_panel[
            (context_panel["Time UTC"] >= window_start) & (context_panel["Time UTC"] <= window_end)
        ].dropna(subset=[measurement_column])
        if not target_points.empty:
            axis.scatter(
                target_points["Time UTC"],
                target_points[measurement_column],
                color=cluster_color,
                s=14,
                alpha=0.9,
                zorder=3,
                label="Points in highlighted k-means window",
            )
        axis.axvspan(
            window_start,
            window_end,
            color=cluster_color,
            alpha=highlight_alpha,
            linewidth=0,
            zorder=0,
        )
        axis.set_ylabel(measurement_column)
        axis.grid(alpha=0.25)

        twin_axis = axis.twinx()
        twin_axis.plot(
            context_panel["Time UTC"],
            context_panel[secondary_column],
            color="#059669",
            linewidth=1.2,
            alpha=0.65,
            label=secondary_column,
        )
        twin_axis.set_ylabel(secondary_column, color="#059669")
        twin_axis.tick_params(axis="y", colors="#059669")

        axis.set_title(
            f"Cluster {cluster_id} example | issue rate={float(row['issue_rate']):.2f} | "
            f"distance={float(row['distance_to_centroid']):.3f}"
        )

        legend_handles = [
            Line2D([0], [0], color="#0f172a", linewidth=2, label=measurement_column),
            Line2D([0], [0], color="#059669", linewidth=2, label=secondary_column),
            Patch(
                facecolor=cluster_color,
                edgecolor="none",
                alpha=highlight_alpha,
                label=f"Highlighted k-means window (Cluster {cluster_id})",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cluster_color,
                markersize=6,
                label="Datapoints used inside that window",
            ),
        ]
        axis.legend(handles=legend_handles, loc="upper left", frameon=True)

        example_records.append(
            {
                "cluster": cluster_id,
                "source_file": clean_source_file_label(source_file),
                "window_start": window_start,
                "window_end": window_end,
                "issue_rate": float(row["issue_rate"]),
                "distance_to_centroid": float(row["distance_to_centroid"]),
                "context_start": context_panel["Time UTC"].min(),
                "context_end": context_panel["Time UTC"].max(),
                "rows_in_highlighted_window": int(len(target_points)),
            }
        )

    axes[-1].set_xlabel("Time UTC")
    fig.suptitle("Representative time-series context for k-means windows", y=1.02)
    fig.tight_layout()
    return fig, pd.DataFrame(example_records)


def fit_kmeans(window_df: pd.DataFrame, *, n_clusters: int, seed: int, n_init: str | int = "auto") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit k-means on window summary features and return assignments plus a summary."""
    cluster_feature_columns = [
        column
        for column in window_df.columns
        if column.endswith("_mean") or column.endswith("_std")
    ]
    cluster_input = window_df[cluster_feature_columns]
    cluster_input = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(cluster_input),
        columns=cluster_feature_columns,
    )
    cluster_scaled = StandardScaler().fit_transform(cluster_input)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    result = window_df.copy()
    result["cluster"] = kmeans.fit_predict(cluster_scaled)
    result["distance_to_centroid"] = kmeans.transform(cluster_scaled).min(axis=1)
    summary = (
        result.groupby("cluster")
        .agg(
            window_count=("cluster", "size"),
            mean_issue_rate=("issue_rate", "mean"),
            max_issue_rate=("issue_rate", "max"),
            avg_distance=("distance_to_centroid", "mean"),
            first_window=("window_start", "min"),
            last_window=("window_end", "max"),
        )
        .sort_index()
    )
    return result, summary


def reduce_window_target(values: np.ndarray, mode: str, severity_order: tuple[int, ...] = (1, 3, 4, 9)) -> int:
    """Reduce row labels inside one window to a single label for window models."""
    severity_rank = {label: index for index, label in enumerate(severity_order)}
    labels = [int(value) for value in values if pd.notna(value)]
    if not labels:
        return severity_order[0]
    if mode == "worst":
        return max(labels, key=lambda label: severity_rank.get(label, -1))
    counts = pd.Series(labels).value_counts()
    tied_labels = counts[counts == counts.max()].index.tolist()
    return max(tied_labels, key=lambda label: severity_rank.get(int(label), -1))


@dataclass
class CnnDataBundle:
    """Normalized numpy arrays and metadata used by the notebook CNN helpers."""
    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    class_labels: list[int]
    window_size: int
    feature_columns: list[str]


def build_cnn_data(
    model_df: pd.DataFrame,
    *,
    task_mode: str,
    window_size: int,
    train_fraction: float,
    validation_fraction: float,
    label_reduction: str,
) -> CnnDataBundle:
    """Build fixed-length windows and normalized tensors for the notebook CNN."""
    cnn_df = model_df[MEASUREMENT_COLUMNS + ["model_target"]].copy().dropna().reset_index(drop=True)
    usable_rows = (len(cnn_df) // window_size) * window_size
    cnn_df = cnn_df.iloc[:usable_rows]
    raw_sequences = cnn_df[MEASUREMENT_COLUMNS].to_numpy(dtype=np.float32).reshape(-1, window_size, len(MEASUREMENT_COLUMNS))
    raw_window_targets = cnn_df["model_target"].to_numpy().reshape(-1, window_size)

    if task_mode == "multiclass":
        window_targets = np.array(
            [reduce_window_target(row, mode=label_reduction) for row in raw_window_targets],
            dtype=np.int64,
        )
        class_labels = sorted(np.unique(window_targets).tolist())
        label_to_index = {label: index for index, label in enumerate(class_labels)}
        encoded_targets = np.array([label_to_index[label] for label in window_targets], dtype=np.int64)
    else:
        class_labels = [0, 1]
        encoded_targets = raw_window_targets.max(axis=1).astype(np.float32)

    sequences = np.transpose(raw_sequences, (0, 2, 1))
    train_end = int(len(sequences) * train_fraction)
    valid_end = int(len(sequences) * (train_fraction + validation_fraction))

    X_train = sequences[:train_end]
    X_valid = sequences[train_end:valid_end]
    X_test = sequences[valid_end:]
    y_train = encoded_targets[:train_end]
    y_valid = encoded_targets[train_end:valid_end]
    y_test = encoded_targets[valid_end:]

    # Fit normalization on the training split only to avoid information leakage.
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train = (X_train - channel_mean) / channel_std
    X_valid = (X_valid - channel_mean) / channel_std
    X_test = (X_test - channel_mean) / channel_std

    return CnnDataBundle(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        class_labels=class_labels,
        window_size=window_size,
        feature_columns=MEASUREMENT_COLUMNS,
    )


def _require_torch() -> None:
    """Raise a clear error when PyTorch is unavailable in the current environment."""
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError("PyTorch is required for the CNN sections.")


def train_cnn_model(
    data: CnnDataBundle,
    *,
    task_mode: str,
    config: dict[str, object],
    seed: int,
    checkpoint_path: Path | None = None,
    device_name: str | None = None,
) -> dict[str, object]:
    """Train the baseline notebook CNN with early stopping and checkpointing."""
    _require_torch()
    torch.manual_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    class TinyQCNet(nn.Module):
        """Minimal 1D CNN used in the teaching notebook."""
        def __init__(self, channels: int, output_dim: int) -> None:
            """Build the convolutional encoder and simple pooled prediction head."""
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(channels, config["conv_channels"][0], kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(config["conv_channels"][0], config["conv_channels"][1], kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(config["conv_channels"][1], output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the CNN forward pass for one batch of normalized windows."""
            return self.net(x)

    if task_mode == "multiclass":
        train_targets_tensor = torch.from_numpy(data.y_train).long()
        valid_targets_tensor = torch.from_numpy(data.y_valid).long()
        test_targets_tensor = torch.from_numpy(data.y_test).long()
        class_counts = np.bincount(data.y_train, minlength=len(data.class_labels)).clip(min=1)
        class_weights = len(data.y_train) / (len(class_counts) * class_counts)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        output_dim = len(data.class_labels)
    else:
        train_targets_tensor = torch.from_numpy(data.y_train.astype(np.float32))
        valid_targets_tensor = torch.from_numpy(data.y_valid.astype(np.float32))
        test_targets_tensor = torch.from_numpy(data.y_test.astype(np.float32))
        positive_count = max(float(data.y_train.sum()), 1.0)
        negative_count = max(float(len(data.y_train) - data.y_train.sum()), 1.0)
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        output_dim = 1

    loader_kwargs: dict[str, object] = {}
    num_workers = int(config.get("num_workers", 0))
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        if "prefetch_factor" in config:
            loader_kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    if config.get("pin_memory", torch.cuda.is_available()):
        loader_kwargs["pin_memory"] = True

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(data.X_train).float(), train_targets_tensor),
        batch_size=config["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(data.X_valid).float(), valid_targets_tensor),
        batch_size=config["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(data.X_test).float(), test_targets_tensor),
        batch_size=config["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    model = TinyQCNet(channels=len(data.feature_columns), output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.get("lr_decay_factor", 0.5),
        patience=config.get("lr_patience", 1),
    )

    best_metric = -math.inf
    best_epoch = 0
    patience_counter = 0
    history = []
    best_state = None

    def run_epoch(loader: DataLoader, training: bool) -> tuple[float, np.ndarray, np.ndarray]:
        """Run one epoch and collect predictions for metrics."""
        model.train(training)
        total_loss = 0.0
        predictions = []
        targets = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            with torch.set_grad_enabled(training):
                logits = model(batch_x)
                if task_mode == "multiclass":
                    loss = loss_fn(logits, batch_y)
                    batch_predictions = logits.argmax(dim=1)
                else:
                    logits = logits.squeeze(-1)
                    loss = loss_fn(logits, batch_y)
                    batch_predictions = (torch.sigmoid(logits) >= 0.5).long()
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    if config.get("gradient_clip_norm"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                    optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            predictions.append(batch_predictions.detach().cpu().numpy())
            targets.append(batch_y.detach().cpu().numpy())
        predictions_array = np.concatenate(predictions) if predictions else np.array([])
        targets_array = np.concatenate(targets) if targets else np.array([])
        average_loss = total_loss / max(len(loader.dataset), 1)
        return average_loss, predictions_array, targets_array

    for epoch in range(1, config["epochs"] + 1):
        train_loss, _, _ = run_epoch(train_loader, training=True)
        valid_loss, valid_preds, valid_targets = run_epoch(valid_loader, training=False)
        valid_f1 = float(f1_score(valid_targets, valid_preds, average=report_average(task_mode), zero_division=0))
        scheduler.step(valid_f1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_f1": valid_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Keep the best checkpoint according to validation F1, not the final epoch.
        if valid_f1 > best_metric + config["min_delta"]:
            best_metric = valid_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "task_mode": task_mode,
                        "class_labels": data.class_labels,
                        "feature_columns": data.feature_columns,
                        "window_size": data.window_size,
                        "config": config,
                    },
                    checkpoint_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                break

    if best_state is None:
        raise RuntimeError("CNN training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_preds, test_targets = run_epoch(test_loader, training=False)

    if task_mode == "multiclass":
        report_labels = list(range(len(data.class_labels)))
        report_names = [str(label) for label in data.class_labels]
    else:
        report_labels = [0, 1]
        report_names = ["0", "1"]

    result = {
        "history": pd.DataFrame(history),
        "best_validation_f1": best_metric,
        "best_epoch": best_epoch,
        "test_loss": float(test_loss),
        "test_predictions": test_preds,
        "test_targets": test_targets,
        "test_report_text": classification_report(
            test_targets,
            test_preds,
            labels=report_labels,
            target_names=report_names,
            zero_division=0,
        ),
        "test_confusion_matrix": confusion_matrix(
            test_targets,
            test_preds,
            labels=report_labels,
            normalize="true",
        ),
        "report_labels": report_labels,
        "report_names": report_names,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    return result


def run_cnn_search(
    model_df: pd.DataFrame,
    *,
    task_mode: str,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    search_space: dict[str, list[object]],
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    """Grid-search a compact CNN configuration space for the advanced notebook."""
    keys = list(search_space.keys())
    results = []
    best_score = -math.inf
    best_config = None
    best_result = None

    for trial_index, values in enumerate(itertools.product(*(search_space[key] for key in keys)), start=1):
        config = dict(zip(keys, values))
        data = build_cnn_data(
            model_df,
            task_mode=task_mode,
            window_size=config["window_size"],
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            label_reduction=config["label_reduction"],
        )
        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"cnn_trial_{trial_index:02d}.pt"
        result = train_cnn_model(
            data,
            task_mode=task_mode,
            config=config,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        row = {
            "trial": trial_index,
            **config,
            "validation_f1": result["best_validation_f1"],
            "best_epoch": result["best_epoch"],
            "device": result["device"],
        }
        results.append(row)
        if result["best_validation_f1"] > best_score:
            best_score = result["best_validation_f1"]
            best_config = config
            best_result = result

    result_frame = pd.DataFrame(results).sort_values("validation_f1", ascending=False).reset_index(drop=True)
    return result_frame, best_config, best_result


def save_pickle(path: str | Path, payload: object) -> None:
    """Persist a Python object with pickle."""
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle)
