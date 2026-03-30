#!/usr/bin/env python3
"""Run a full preprocessing suite on the scraped basketball dataset.

Pipeline steps:
1) Load CSV rows and normalize text/missing values.
2) Deduplicate rows by `entry_id` (keep row with most non-missing features).
3) Preprocess by subtype (`record_type`, `source`) for numeric coercion/imputation/winsorization.
4) Impute missing categorical/id values ("Unknown") and add missingness indicator features.
5) Add engineered features (BMI + age buckets).
6) Build a model-ready table with one-hot categorical encoding and min-max scaling.
7) Exclude ultra-sparse numeric features from model-ready scaling.
8) Persist cleaned dataset, model-ready dataset, and a JSON preprocessing report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_INPUT = Path("data") / "player_dataset.csv"
DEFAULT_CLEAN_OUTPUT = Path("data") / "player_dataset_cleaned.csv"
DEFAULT_MODEL_OUTPUT = Path("data") / "player_dataset_model_ready.csv"
DEFAULT_REPORT = Path("data") / "player_preprocessing_report.json"
DEFAULT_SPARSE_MISSING_THRESHOLD = 0.95
DEFAULT_INDICATOR_MISSING_THRESHOLD = 0.50

MISSING_TOKENS = {"", "na", "n/a", "none", "null", "nan", "-", "--"}
SUBTYPE_COLUMNS = ("record_type", "source")
MISSING_MARKER_PREFIX = "__missing__"

NUMERIC_COLUMNS = [
    "season_end_year",
    "draft_year",
    "age",
    "height_in",
    "weight_lb",
    "wingspan_in",
    "standing_reach_in",
    "body_fat_pct",
    "hand_length_in",
    "hand_width_in",
    "games",
    "games_started",
    "minutes_per_game",
    "points_per_game",
    "assists_per_game",
    "rebounds_per_game",
    "steals_per_game",
    "blocks_per_game",
    "turnovers_per_game",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "per",
    "ts_pct",
    "usg_pct",
    "ows",
    "dws",
    "ws",
    "ws_per_48",
    "bpm",
    "vorp",
    "ast_pct",
    "trb_pct",
    "stl_pct",
    "blk_pct",
    "tov_pct",
    "raw_height_wo_shoes",
    "raw_height_w_shoes",
]

INTEGER_COLUMNS = {"season_end_year", "draft_year", "age", "games", "games_started", "weight_lb"}
PERCENTAGE_COLUMNS = {
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "ts_pct",
    "usg_pct",
    "ast_pct",
    "trb_pct",
    "stl_pct",
    "blk_pct",
    "tov_pct",
    "body_fat_pct",
}

CATEGORICAL_COLUMNS = [
    "record_type",
    "source",
    "team_abbr",
    "position",
    "college",
]

ID_COLUMNS = [
    "entry_id",
    "source_url",
    "season_label",
    "player_id",
    "player_name",
    "birth_date",
    "height_ft_in",
    "wingspan_ft_in",
    "standing_reach_ft_in",
]


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split()).strip()


def is_missing(value: object) -> bool:
    return normalize_text(value).lower() in MISSING_TOKENS


def parse_float(value: object) -> Optional[float]:
    text = normalize_text(value).replace(",", "")
    if text.lower() in MISSING_TOKENS:
        return None
    text = text.rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty values.")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"Input file has no header: {path}")
        rows = [{key: normalize_text(value) for key, value in row.items()} for row in reader]
        return rows, list(reader.fieldnames)


def choose_best_row(existing: Dict[str, object], candidate: Dict[str, object], keys: Iterable[str]) -> Dict[str, object]:
    def completeness_score(row: Dict[str, object]) -> int:
        score = 0
        for key in keys:
            value = row.get(key)
            if isinstance(value, str):
                if not is_missing(value):
                    score += 1
            elif value is not None:
                score += 1
        return score

    return candidate if completeness_score(candidate) > completeness_score(existing) else existing


def deduplicate_rows(rows: List[Dict[str, str]], all_columns: Sequence[str]) -> Tuple[List[Dict[str, str]], int]:
    by_entry: Dict[str, Dict[str, str]] = {}
    duplicate_count = 0
    for row in rows:
        key = row.get("entry_id", "")
        if not key:
            key = f"missing_entry_{len(by_entry)}"
            row["entry_id"] = key
        if key in by_entry:
            duplicate_count += 1
            by_entry[key] = choose_best_row(by_entry[key], row, all_columns)
        else:
            by_entry[key] = row
    return list(by_entry.values()), duplicate_count


def subtype_key(row: Dict[str, object]) -> str:
    parts = []
    for col in SUBTYPE_COLUMNS:
        value = normalize_text(row.get(col, "")) or "Unknown"
        parts.append(value)
    return "|".join(parts)


def group_rows_by_subtype(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[subtype_key(row)].append(row)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def coerce_numeric(rows: List[Dict[str, object]]) -> Dict[str, int]:
    invalid_counts = {col: 0 for col in NUMERIC_COLUMNS}
    for row in rows:
        for col in NUMERIC_COLUMNS:
            raw = row.get(col, "")
            value = parse_float(raw)
            if value is None:
                if not is_missing(raw):
                    invalid_counts[col] += 1
                row[col] = None
                continue
            if col in PERCENTAGE_COLUMNS and value > 1.0 and value <= 100.0:
                value = value / 100.0
            if col in INTEGER_COLUMNS:
                row[col] = int(round(value))
            else:
                row[col] = float(value)
    return invalid_counts


def add_numeric_counts(target: Dict[str, int], source: Dict[str, int]) -> None:
    for col, count in source.items():
        target[col] = target.get(col, 0) + count


def impute_numeric(rows: List[Dict[str, object]]) -> Tuple[Dict[str, object], Dict[str, int]]:
    numeric_imputations: Dict[str, object] = {}
    impute_counts: Dict[str, int] = {}

    for col in NUMERIC_COLUMNS:
        values = [row[col] for row in rows if row.get(col) is not None]
        if values:
            fill_value = median(values)
            if col in INTEGER_COLUMNS:
                fill_value = int(round(fill_value))
        else:
            fill_value = 0
        numeric_imputations[col] = fill_value

        marker = f"{MISSING_MARKER_PREFIX}{col}"
        count = 0
        for row in rows:
            was_missing = row.get(col) is None
            row[marker] = 1 if was_missing else 0
            if was_missing:
                row[col] = fill_value
                count += 1
        impute_counts[col] = count

    return numeric_imputations, impute_counts


def impute_text(rows: List[Dict[str, object]]) -> Dict[str, int]:
    impute_counts: Dict[str, int] = {}

    for col in CATEGORICAL_COLUMNS + ID_COLUMNS:
        count = 0
        for row in rows:
            if is_missing(row.get(col, "")):
                row[col] = "Unknown"
                count += 1
        impute_counts[col] = count

    return impute_counts


def winsorize_numeric(rows: List[Dict[str, object]], lower_q: float, upper_q: float) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for col in NUMERIC_COLUMNS:
        values = sorted(float(row[col]) for row in rows)
        low = quantile(values, lower_q)
        high = quantile(values, upper_q)
        clipped = 0
        for row in rows:
            value = float(row[col])
            bounded = min(max(value, low), high)
            if bounded != value:
                clipped += 1
            row[col] = int(round(bounded)) if col in INTEGER_COLUMNS else bounded
        summary[col] = {"low": low, "high": high, "clipped_rows": clipped}
    return summary


def add_engineered_features(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        height_in = float(row.get("height_in", 0) or 0)
        weight_lb = float(row.get("weight_lb", 0) or 0)
        height_m = height_in * 0.0254
        weight_kg = weight_lb * 0.45359237
        bmi = (weight_kg / (height_m ** 2)) if height_m > 0 else 0.0
        row["bmi"] = round(bmi, 4)

        age = int(row.get("age", 0) or 0)
        if age <= 0:
            bucket = "Unknown"
        elif age <= 22:
            bucket = "18-22"
        elif age <= 27:
            bucket = "23-27"
        elif age <= 32:
            bucket = "28-32"
        else:
            bucket = "33+"
        row["age_bucket"] = bucket


def one_hot_values(rows: List[Dict[str, object]], column: str) -> List[str]:
    values = sorted({normalize_text(row.get(column, "Unknown")) or "Unknown" for row in rows})
    return values


def add_missing_indicators(
    rows: List[Dict[str, object]],
    numeric_impute_counts: Dict[str, int],
    threshold: float,
) -> List[str]:
    if not rows:
        return []

    indicator_columns: List[str] = []
    row_count = len(rows)
    for col in NUMERIC_COLUMNS:
        missing_rate = numeric_impute_counts.get(col, 0) / row_count
        if missing_rate < threshold:
            continue
        marker = f"{MISSING_MARKER_PREFIX}{col}"
        indicator_col = f"missing_{col}"
        for row in rows:
            row[indicator_col] = int(row.get(marker, 0))
        indicator_columns.append(indicator_col)
    return sorted(indicator_columns)


def remove_missing_markers(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        for col in NUMERIC_COLUMNS:
            row.pop(f"{MISSING_MARKER_PREFIX}{col}", None)


def sparse_numeric_columns(
    rows: List[Dict[str, object]],
    numeric_impute_counts: Dict[str, int],
    threshold: float,
) -> List[str]:
    if not rows:
        return []
    row_count = len(rows)
    return sorted(
        [col for col in NUMERIC_COLUMNS if (numeric_impute_counts.get(col, 0) / row_count) >= threshold]
    )


def ordered_unique(values: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def to_model_ready(
    rows: List[Dict[str, object]],
    scale_columns: List[str],
    indicator_columns: List[str],
) -> Tuple[List[Dict[str, object]], List[str]]:
    model_rows: List[Dict[str, object]] = []
    one_hot_map = {col: one_hot_values(rows, col) for col in CATEGORICAL_COLUMNS + ["age_bucket"]}

    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for col in scale_columns:
        values = [float(row[col]) for row in rows]
        mins[col] = min(values)
        maxs[col] = max(values)

    for row in rows:
        model_row: Dict[str, object] = {
            "entry_id": row["entry_id"],
            "player_name": row["player_name"],
        }

        for col in scale_columns:
            value = float(row[col])
            low, high = mins[col], maxs[col]
            scaled = 0.0 if high == low else (value - low) / (high - low)
            model_row[f"scaled_{col}"] = round(scaled, 6)

        for indicator_col in indicator_columns:
            model_row[indicator_col] = int(row.get(indicator_col, 0))

        for col, values in one_hot_map.items():
            row_value = normalize_text(row.get(col, "Unknown")) or "Unknown"
            for value in values:
                safe = value.lower().replace(" ", "_").replace("/", "_")
                safe = "".join(ch if (ch.isalnum() or ch == "_") else "" for ch in safe)
                if not safe:
                    safe = "unknown"
                key = f"{col}__{safe}"
                model_row[key] = 1 if row_value == value else 0

        model_rows.append(model_row)

    model_columns = list(model_rows[0].keys()) if model_rows else []
    return model_rows, model_columns


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def build_report(
    input_rows: int,
    cleaned_rows: int,
    duplicates_removed: int,
    subtype_counts: Dict[str, int],
    invalid_numeric: Dict[str, int],
    imputations: Dict[str, int],
    numeric_imputations_by_subtype: Dict[str, Dict[str, object]],
    numeric_imputation_counts_by_subtype: Dict[str, Dict[str, int]],
    winsor_summary: Dict[str, Dict[str, Dict[str, float]]],
    sparse_excluded_columns: List[str],
    missing_indicator_columns: List[str],
    sparse_missing_threshold: float,
    indicator_missing_threshold: float,
) -> Dict[str, object]:
    return {
        "input_rows": input_rows,
        "cleaned_rows": cleaned_rows,
        "duplicates_removed": duplicates_removed,
        "subtype_counts": subtype_counts,
        "invalid_numeric_values": invalid_numeric,
        "imputed_values": imputations,
        "numeric_imputations_by_subtype": numeric_imputations_by_subtype,
        "numeric_imputation_counts_by_subtype": numeric_imputation_counts_by_subtype,
        "winsorization_by_subtype": winsor_summary,
        "sparse_numeric_columns_excluded_from_model": sparse_excluded_columns,
        "missing_indicator_columns": missing_indicator_columns,
        "sparse_missing_threshold": sparse_missing_threshold,
        "indicator_missing_threshold": indicator_missing_threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the basketball player dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--clean-output", type=Path, default=DEFAULT_CLEAN_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--winsor-lower-quantile", type=float, default=0.01)
    parser.add_argument("--winsor-upper-quantile", type=float, default=0.99)
    parser.add_argument("--sparse-missing-threshold", type=float, default=DEFAULT_SPARSE_MISSING_THRESHOLD)
    parser.add_argument("--indicator-missing-threshold", type=float, default=DEFAULT_INDICATOR_MISSING_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.winsor_lower_quantile < args.winsor_upper_quantile <= 1:
        raise ValueError("Winsor quantiles must satisfy: 0 <= low < high <= 1")
    if not 0 <= args.sparse_missing_threshold <= 1:
        raise ValueError("Sparse missing threshold must satisfy: 0 <= threshold <= 1")
    if not 0 <= args.indicator_missing_threshold <= 1:
        raise ValueError("Indicator missing threshold must satisfy: 0 <= threshold <= 1")

    rows, original_columns = load_rows(args.input)
    input_rows = len(rows)

    rows, duplicates = deduplicate_rows(rows, original_columns)
    grouped_rows = group_rows_by_subtype(rows)
    subtype_counts = {key: len(group) for key, group in grouped_rows.items()}

    invalid_numeric = {col: 0 for col in NUMERIC_COLUMNS}
    numeric_impute_counts = {col: 0 for col in NUMERIC_COLUMNS}
    numeric_imputations_by_subtype: Dict[str, Dict[str, object]] = {}
    numeric_imputation_counts_by_subtype: Dict[str, Dict[str, int]] = {}
    winsor_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for key, group in grouped_rows.items():
        subtype_invalid = coerce_numeric(group)
        add_numeric_counts(invalid_numeric, subtype_invalid)

        subtype_imputations, subtype_impute_counts = impute_numeric(group)
        numeric_imputations_by_subtype[key] = subtype_imputations
        numeric_imputation_counts_by_subtype[key] = subtype_impute_counts
        add_numeric_counts(numeric_impute_counts, subtype_impute_counts)

        winsor_summary[key] = winsorize_numeric(group, args.winsor_lower_quantile, args.winsor_upper_quantile)

    text_imputations = impute_text(rows)
    imputations: Dict[str, int] = {}
    imputations.update(numeric_impute_counts)
    imputations.update(text_imputations)

    add_engineered_features(rows)
    missing_indicator_columns = add_missing_indicators(rows, numeric_impute_counts, args.indicator_missing_threshold)
    sparse_excluded_columns = sparse_numeric_columns(rows, numeric_impute_counts, args.sparse_missing_threshold)
    remove_missing_markers(rows)

    cleaned_columns = ordered_unique(list(original_columns) + ["bmi", "age_bucket"] + missing_indicator_columns)
    write_csv(args.clean_output, rows, cleaned_columns)

    scale_columns = [col for col in NUMERIC_COLUMNS if col not in sparse_excluded_columns] + ["bmi"]
    model_rows, model_columns = to_model_ready(rows, scale_columns=scale_columns, indicator_columns=missing_indicator_columns)
    write_csv(args.model_output, model_rows, model_columns)

    report = build_report(
        input_rows=input_rows,
        cleaned_rows=len(rows),
        duplicates_removed=duplicates,
        subtype_counts=subtype_counts,
        invalid_numeric=invalid_numeric,
        imputations=imputations,
        numeric_imputations_by_subtype=numeric_imputations_by_subtype,
        numeric_imputation_counts_by_subtype=numeric_imputation_counts_by_subtype,
        winsor_summary=winsor_summary,
        sparse_excluded_columns=sparse_excluded_columns,
        missing_indicator_columns=missing_indicator_columns,
        sparse_missing_threshold=args.sparse_missing_threshold,
        indicator_missing_threshold=args.indicator_missing_threshold,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Loaded rows: {input_rows}")
    print(f"Rows after dedupe: {len(rows)} (removed {duplicates})")
    print(f"Sparse numeric columns excluded from model: {len(sparse_excluded_columns)}")
    print(f"Missing indicators added: {len(missing_indicator_columns)}")
    print(f"Clean dataset: {args.clean_output}")
    print(f"Model-ready dataset: {args.model_output}")
    print(f"Report: {args.report_output}")


if __name__ == "__main__":
    main()
