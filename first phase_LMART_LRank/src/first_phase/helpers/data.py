from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
COLUMN_ALIASES = {
    "symbol": "symbol",
    "ticker": "symbol",
    "date": "timestamp",
    "datetime": "timestamp",
    "timestamp": "timestamp",
    "time": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj_close": "close",
    "volume": "volume",
}


@dataclass(frozen=True)
class DatasetManifest:
    source_path: str
    fingerprint: str
    row_count: int
    symbol_count: int
    timestamp_min: str
    timestamp_max: str
    duplicate_keys: int
    conflicting_duplicates: int
    inferred_timeframe: str
    inferred_cadence_seconds: int | None
    typical_price_divisor: float
    source_file_count: int = 1
    source_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _fingerprint_frame(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _map_columns(columns: list[str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for original in columns:
        key = original.strip().lower()
        if key in COLUMN_ALIASES:
            canonical = COLUMN_ALIASES[key]
            if canonical in mapped.values():
                raise ValueError(f"Ambiguous mapping for canonical column '{canonical}'")
            mapped[original] = canonical
    missing = [column for column in CANONICAL_COLUMNS if column not in mapped.values()]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return mapped


def normalize_market_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, DatasetManifest]:
    renamed = raw.rename(columns=_map_columns(list(raw.columns)))
    frame = renamed[CANONICAL_COLUMNS].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=CANONICAL_COLUMNS)
    frame = frame[(frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)]
    frame = frame[frame["volume"] >= 0]
    frame = frame.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)

    duplicated = frame.duplicated(subset=["symbol", "timestamp"], keep=False)
    conflict_count = 0
    if duplicated.any():
        duplicate_frame = frame.loc[duplicated].copy()
        grouped = duplicate_frame.groupby(["symbol", "timestamp"], sort=False)
        keep_rows: list[pd.Series] = []
        for _, group in grouped:
            if group[["open", "high", "low", "close", "volume"]].nunique().max() > 1:
                conflict_count += 1
            keep_rows.append(group.iloc[0])
        deduped = pd.DataFrame(keep_rows)
        non_dup = frame.loc[~duplicated]
        frame = pd.concat([non_dup, deduped], ignore_index=True).sort_values(
            ["symbol", "timestamp"], kind="stable"
        )
    if conflict_count:
        raise ValueError(
            f"Found {conflict_count} conflicting duplicate (symbol, timestamp) keys; "
            "configure an explicit source-precedence policy before training"
        )

    cadence_seconds = infer_cadence_seconds(frame)
    timeframe = infer_timeframe(cadence_seconds)
    manifest = DatasetManifest(
        source_path="in_memory",
        fingerprint=_fingerprint_frame(frame),
        row_count=len(frame),
        symbol_count=int(frame["symbol"].nunique()),
        timestamp_min=frame["timestamp"].min().isoformat(),
        timestamp_max=frame["timestamp"].max().isoformat(),
        duplicate_keys=int(duplicated.sum()),
        conflicting_duplicates=int(conflict_count),
        inferred_timeframe=timeframe,
        inferred_cadence_seconds=cadence_seconds,
        typical_price_divisor=3.0,
    )
    return frame.reset_index(drop=True), manifest


def infer_cadence_seconds(frame: pd.DataFrame) -> int | None:
    deltas: list[int] = []
    for _, group in frame.groupby("symbol", sort=False):
        seconds = group["timestamp"].sort_values().diff().dropna().dt.total_seconds()
        if not seconds.empty:
            deltas.extend(int(value) for value in seconds.head(100))
    if not deltas:
        return None
    return int(pd.Series(deltas).mode().iloc[0])


def infer_timeframe(cadence_seconds: int | None) -> str:
    if cadence_seconds is None:
        return "unknown"
    if cadence_seconds <= 90:
        return "one_minute"
    if cadence_seconds <= 3600:
        return "hourly"
    return "daily"


def load_market_frame(path: str | Path) -> tuple[pd.DataFrame, DatasetManifest]:
    source = Path(path)
    if source.is_dir():
        sources = sorted(
            candidate for candidate in source.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".parquet", ".pq", ".csv"}
        )
        if not sources:
            raise ValueError(f"No CSV/Parquet files found below: {source}")
    else:
        sources = [source]
    frames = []
    for item in sources:
        if item.suffix.lower() == ".csv":
            frames.append(pd.read_csv(item))
        elif item.suffix.lower() in {".parquet", ".pq"}:
            frames.append(pd.read_parquet(item))
        else:
            raise ValueError(f"Unsupported data source: {item}")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    frame, manifest = normalize_market_frame(raw)
    manifest = DatasetManifest(
        **{
            **manifest.to_dict(),
            "source_path": str(source),
            "source_file_count": len(sources),
            "source_files": tuple(str(item) for item in sources),
        }
    )
    return frame, manifest


def write_manifest(manifest: DatasetManifest, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
