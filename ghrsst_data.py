#!/usr/bin/env python3
"""Download and preprocess NOAA OISST v2.1 data for December 1.

Default output:
    seed_0  -> December 1, 2006
    ...
    seed_19 -> December 1, 2025

Each x.csv contains only longitude and latitude. Each y.csv contains raw SST.
Longitude and latitude are normalized using the training split only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


ERDDAP_DATASET_URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
    "ncdcOisst21Agg.nc"
)


@dataclass(frozen=True)
class Config:
    start_year: int
    end_year: int
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    train_fraction: float
    seed: int
    max_points: int
    raw_dir: str
    output_dir: str
    timeout_seconds: int
    retries: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Download NOAA OISST v2.1 data for December 1."
    )
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--lon-min", type=float, default=-75.0)
    parser.add_argument("--lon-max", type=float, default=-45.0)
    parser.add_argument("--lat-min", type=float, default=30.0)
    parser.add_argument("--lat-max", type=float, default=45.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Maximum ocean grid cells retained per year; 0 keeps all.",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/GHRSST_raw")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/GHRSST")
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year.")
    if not (-90 <= args.lat_min < args.lat_max <= 90):
        parser.error("Require -90 <= lat-min < lat-max <= 90.")
    if not (-180 <= args.lon_min < args.lon_max <= 180):
        parser.error("Require -180 <= lon-min < lon-max <= 180.")
    if not (0 < args.train_fraction < 1):
        parser.error("--train-fraction must lie strictly between 0 and 1.")
    if args.max_points < 0:
        parser.error("--max-points must be nonnegative.")

    return Config(
        start_year=args.start_year,
        end_year=args.end_year,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_points=args.max_points,
        raw_dir=str(args.raw_dir),
        output_dir=str(args.output_dir),
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )


def longitude_to_360(value: float) -> float:
    return value % 360.0


def longitude_to_180(values: pd.Series) -> pd.Series:
    return ((values + 180.0) % 360.0) - 180.0


def build_erddap_url(year: int, config: Config) -> str:
    date = f"{year}-12-01T12:00:00Z"
    lon_min_360 = longitude_to_360(config.lon_min)
    lon_max_360 = longitude_to_360(config.lon_max)

    if lon_min_360 > lon_max_360:
        raise ValueError("Dateline-crossing longitude ranges are unsupported.")

    selection = (
        f"[({date})]"
        f"[(0.0)]"
        f"[({config.lat_min}):1:({config.lat_max})]"
        f"[({lon_min_360}):1:({lon_max_360})]"
    )
    return f"{ERDDAP_DATASET_URL}?sst{selection}"


def download_file(
    url: str,
    destination: Path,
    timeout_seconds: int,
    retries: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Skipping existing file: {destination}")
        return

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            print(f"Downloading {destination.name} ({attempt}/{retries})...")
            with requests.get(
                url,
                stream=True,
                timeout=timeout_seconds,
                headers={"User-Agent": "nnkernels-GHRSST-downloader/1.0"},
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as file_handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            file_handle.write(chunk)
            if temporary.stat().st_size == 0:
                raise RuntimeError("NOAA returned an empty file.")
            temporary.replace(destination)
            return
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                delay = min(60, 2**attempt)
                print(f"Download failed: {error}; retrying in {delay}s.")
                time.sleep(delay)

    raise RuntimeError(f"Failed to download {destination}.") from last_error


def dataset_to_dataframe(path: Path, year: int) -> pd.DataFrame:
    with xr.open_dataset(path) as dataset:
        required = {"sst", "time", "latitude", "longitude"}
        missing = required.difference(dataset.variables)
        if missing:
            raise KeyError(f"{path} is missing variables: {sorted(missing)}")
        frame = dataset[["sst"]].to_dataframe().reset_index()

    if "zlev" in frame.columns:
        frame = frame.drop(columns="zlev")

    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["longitude"] = longitude_to_180(frame["longitude"].astype(float))
    frame["latitude"] = frame["latitude"].astype(float)
    frame["sst"] = pd.to_numeric(frame["sst"], errors="coerce")
    frame = frame.dropna(subset=["sst"])
    frame = frame[np.isfinite(frame["sst"])].copy()

    expected_date = f"{year}-12-01"
    dates = frame["time"].dt.strftime("%Y-%m-%d").unique().tolist()
    if dates != [expected_date]:
        raise ValueError(f"Expected only {expected_date}, found {dates}.")

    return frame[["longitude", "latitude", "sst"]].sort_values(
        ["latitude", "longitude"], ignore_index=True
    )


def subsample_points(
    frame: pd.DataFrame,
    max_points: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if max_points == 0 or len(frame) <= max_points:
        return frame.reset_index(drop=True)
    indices = rng.choice(len(frame), size=max_points, replace=False)
    return frame.iloc[np.sort(indices)].reset_index(drop=True)


def random_split(
    frame: pd.DataFrame,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    if n < 2:
        raise ValueError("At least two valid ocean grid cells are required.")
    permutation = rng.permutation(n)
    n_train = min(max(int(math.floor(train_fraction * n)), 1), n - 1)
    train = frame.iloc[permutation[:n_train]].reset_index(drop=True)
    test = frame.iloc[permutation[n_train:]].reset_index(drop=True)
    return train, test


def normalize_locations(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    train = train.copy()
    test = test.copy()
    metadata: dict[str, dict[str, float]] = {}

    for column in ("longitude", "latitude"):
        minimum = float(train[column].min())
        maximum = float(train[column].max())
        scale = maximum - minimum
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Cannot normalize column {column!r}.")
        train[column] = (train[column] - minimum) / scale
        test[column] = (test[column] - minimum) / scale
        metadata[column] = {"min": minimum, "max": maximum}

    return train, test, metadata


def save_replicate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    year: int,
    replicate_index: int,
    config: Config,
) -> None:
    root = Path(config.output_dir) / f"seed_{replicate_index}"
    train_dir = root / "train"
    test_dir = root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train, test, normalization = normalize_locations(train, test)
    x_columns = ["longitude", "latitude"]

    train[x_columns].to_csv(train_dir / "x.csv", index=False, header=False)
    train[["sst"]].to_csv(train_dir / "y.csv", index=False, header=False)
    test[x_columns].to_csv(test_dir / "x.csv", index=False, header=False)
    test[["sst"]].to_csv(test_dir / "y.csv", index=False, header=False)

    metadata = {
        "dataset": "NOAA OISST v2.1 AVHRR-only final",
        "date": f"{year}-12-01",
        "year": year,
        "replicate_index": replicate_index,
        "random_seed": config.seed + replicate_index,
        "train_fraction": config.train_fraction,
        "n_train": len(train),
        "n_test": len(test),
        "x_columns": x_columns,
        "response": "sst_degree_C",
        "normalization": normalization,
        "region": {
            "lon_min": config.lon_min,
            "lon_max": config.lon_max,
            "lat_min": config.lat_min,
            "lat_max": config.lat_max,
        },
        "source": ERDDAP_DATASET_URL,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(
        f"Prepared December 1, {year} as seed_{replicate_index}: "
        f"{len(train):,} train / {len(test):,} test."
    )


def main() -> None:
    config = parse_args()
    years = list(range(config.start_year, config.end_year + 1))
    Path(config.raw_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    for replicate_index, year in enumerate(years):
        raw_path = Path(config.raw_dir) / f"oisst_dec01_{year}.nc"
        download_file(
            build_erddap_url(year, config),
            raw_path,
            config.timeout_seconds,
            config.retries,
        )
        frame = dataset_to_dataframe(raw_path, year)
        rng = np.random.default_rng(config.seed + replicate_index)
        frame = subsample_points(frame, config.max_points, rng)
        train, test = random_split(frame, config.train_fraction, rng)
        save_replicate(train, test, year, replicate_index, config)

    print("All requested December 1 fields were processed successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
