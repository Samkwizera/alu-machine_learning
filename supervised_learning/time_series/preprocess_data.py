#!/usr/bin/env python3
"""Clean minute-level BTC exchange data and save an hourly data set."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = {
    "timestamp": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume_(btc)": "volume_btc",
    "volume_btc": "volume_btc",
    "volume_(currency)": "volume_currency",
    "volume_currency": "volume_currency",
    "weighted_price": "weighted_price",
    "vwap": "weighted_price",
}
PRICE_COLUMNS = ["open", "high", "low", "close", "weighted_price"]
BASE_FEATURES = PRICE_COLUMNS + ["volume_btc", "volume_currency"]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Coinbase and Bitstamp minute candles to hours."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more raw exchange CSV files.",
    )
    parser.add_argument(
        "--output",
        default="btc_hourly.npz",
        help="Output NPZ path (default: btc_hourly.npz).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Chronological fraction used for training (default: 0.70).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Chronological fraction used for validation (default: 0.15).",
    )
    return parser.parse_args()


def standardize_columns(frame):
    """Return a frame whose column labels use the project schema."""
    renamed = {}
    for column in frame.columns:
        key = column.strip().lower().replace(" ", "_")
        if key in RAW_COLUMNS:
            renamed[column] = RAW_COLUMNS[key]
    frame = frame.rename(columns=renamed)
    required = {"timestamp", *BASE_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Missing required columns: {}".format(
            ", ".join(sorted(missing))))
    return frame[["timestamp", *BASE_FEATURES]].copy()


def load_exchange(path):
    """Load one raw CSV, remove invalid rows, and aggregate it hourly."""
    frame = standardize_columns(pd.read_csv(path))
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).drop_duplicates("timestamp")
    frame["time"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame = frame.set_index("time").sort_index()

    # VWAP is correctly recomputed after changing from minute to hour.
    frame["weighted_value"] = (
        frame["weighted_price"] * frame["volume_btc"]
    )
    hourly = frame.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume_btc": "sum",
        "volume_currency": "sum",
        "weighted_value": "sum",
    })
    hourly["weighted_price"] = (
        hourly["weighted_value"] / hourly["volume_btc"].replace(0, np.nan)
    )
    hourly["weighted_price"] = hourly["weighted_price"].fillna(
        hourly["close"]
    )
    return hourly.drop(columns="weighted_value")


def combine_exchanges(frames):
    """Combine hourly exchange candles into one volume-weighted market."""
    combined = pd.concat(frames).sort_index()
    combined["weighted_value"] = (
        combined["weighted_price"] * combined["volume_btc"]
    )
    hourly = combined.groupby(level=0).agg({
        "open": "mean",
        "high": "max",
        "low": "min",
        "close": "mean",
        "volume_btc": "sum",
        "volume_currency": "sum",
        "weighted_value": "sum",
    })
    hourly["weighted_price"] = (
        hourly["weighted_value"] / hourly["volume_btc"].replace(0, np.nan)
    )
    hourly["weighted_price"] = hourly["weighted_price"].fillna(
        hourly["close"]
    )
    return hourly.drop(columns="weighted_value")


def clean_hourly(frame):
    """Create a regular hourly index and add cyclical calendar features."""
    frame = frame.replace([np.inf, -np.inf], np.nan).sort_index()
    full_index = pd.date_range(
        frame.index.min(), frame.index.max(), freq="1h", tz="UTC"
    )
    frame = frame.reindex(full_index)
    frame[PRICE_COLUMNS] = frame[PRICE_COLUMNS].interpolate(
        method="time", limit=3
    )
    frame[["volume_btc", "volume_currency"]] = frame[
        ["volume_btc", "volume_currency"]
    ].fillna(0.0)
    frame = frame.dropna(subset=PRICE_COLUMNS)
    segment_ids = frame.index.to_series().diff().ne(
        pd.Timedelta(hours=1)
    ).cumsum()
    longest_segment = segment_ids.value_counts().idxmax()
    frame = frame.loc[segment_ids == longest_segment]

    hour_angle = 2 * np.pi * frame.index.hour / 24
    week_angle = 2 * np.pi * frame.index.dayofweek / 7
    frame["hour_sin"] = np.sin(hour_angle)
    frame["hour_cos"] = np.cos(hour_angle)
    frame["weekday_sin"] = np.sin(week_angle)
    frame["weekday_cos"] = np.cos(week_angle)
    return frame


def preprocess(paths, output, train_fraction, validation_fraction):
    """Preprocess CSV paths and persist scaled data and scaler metadata."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction leaves no test partition")

    hourly = clean_hourly(combine_exchanges([
        load_exchange(path) for path in paths
    ]))
    if len(hourly) < 75:
        raise ValueError("At least 75 usable hourly rows are required")

    feature_names = list(hourly.columns)
    values = hourly.to_numpy(dtype=np.float32)
    train_end = int(len(values) * train_fraction)
    validation_end = int(
        len(values) * (train_fraction + validation_fraction)
    )
    means = values[:train_end].mean(axis=0)
    scales = values[:train_end].std(axis=0)
    scales[scales == 0] = 1.0
    scaled = (values - means) / scales
    timestamps = hourly.index.view("int64") // 10 ** 9

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        data=scaled.astype(np.float32),
        timestamps=timestamps,
        feature_names=np.asarray(feature_names),
        means=means,
        scales=scales,
        train_end=train_end,
        validation_end=validation_end,
    )
    summary = {
        "rows": len(values),
        "features": feature_names,
        "train_rows": train_end,
        "validation_rows": validation_end - train_end,
        "test_rows": len(values) - validation_end,
        "first_hour": hourly.index[0].isoformat(),
        "last_hour": hourly.index[-1].isoformat(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    arguments = parse_args()
    preprocess(
        arguments.inputs,
        arguments.output,
        arguments.train_fraction,
        arguments.validation_fraction,
    )
