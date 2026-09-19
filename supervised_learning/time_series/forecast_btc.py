#!/usr/bin/env python3
"""Train and evaluate a GRU that forecasts BTC one hour ahead."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


LOOKBACK = 24


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="NPZ created by preprocess_data.py")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="btc_forecast_results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_dataset(
    data, close_index, batch_size, shuffle=False, seed=42
) -> tf.data.Dataset:
    """Build (24 hourly observations, following close) pairs."""
    if len(data) <= LOOKBACK:
        raise ValueError("Each partition needs more than 24 hourly rows")
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=data[:-1],
        targets=data[LOOKBACK:, close_index],
        sequence_length=LOOKBACK,
        sequence_stride=1,
        shuffle=shuffle,
        seed=seed,
        batch_size=batch_size,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(feature_count):
    """Create and compile the recurrent forecasting model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, feature_count)),
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def plot_results(history, actual, predicted, output_dir):
    """Save training-history and test-forecast graphs."""
    plt.switch_backend("Agg")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history.history["loss"], label="training MSE")
    axis.plot(history.history["val_loss"], label="validation MSE")
    axis.set(xlabel="Epoch", ylabel="MSE (scaled)", title="Training history")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=160)
    plt.close(figure)

    shown = min(300, len(actual))
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(actual[-shown:], label="actual close", linewidth=1.5)
    axis.plot(predicted[-shown:], label="predicted close", linewidth=1.2)
    axis.set(
        xlabel="Test hour",
        ylabel="BTC close (USD)",
        title="One-hour-ahead BTC forecast",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "test_forecast.png", dpi=160)
    plt.close(figure)


def main(arguments):
    """Load preprocessed data, train the model, and save results."""
    tf.keras.utils.set_random_seed(arguments.seed)
    archive = np.load(arguments.data)
    data = archive["data"]
    feature_names = archive["feature_names"].tolist()
    train_end = int(archive["train_end"])
    validation_end = int(archive["validation_end"])
    close_index = feature_names.index("close")
    close_mean = float(archive["means"][close_index])
    close_scale = float(archive["scales"][close_index])

    train_data = data[:train_end]
    validation_data = data[train_end - LOOKBACK:validation_end]
    test_data = data[validation_end - LOOKBACK:]
    train_set = make_dataset(
        train_data, close_index, arguments.batch_size,
        shuffle=True, seed=arguments.seed
    )
    validation_set = make_dataset(
        validation_data, close_index, arguments.batch_size
    )
    test_set = make_dataset(test_data, close_index, arguments.batch_size)

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(data.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras", monitor="val_loss",
            save_best_only=True
        ),
    ]
    history = model.fit(
        train_set,
        validation_data=validation_set,
        epochs=arguments.epochs,
        callbacks=callbacks,
    )

    predictions = model.predict(test_set, verbose=0).reshape(-1)
    actual_scaled = np.concatenate([
        labels.numpy() for _, labels in test_set
    ])
    predicted = predictions * close_scale + close_mean
    actual = actual_scaled * close_scale + close_mean
    errors = predicted - actual
    baseline = (
        test_data[LOOKBACK - 1:-1, close_index] * close_scale
        + close_mean
    )
    baseline_errors = baseline - actual
    metrics = {
        "test_mse_usd": float(np.mean(errors ** 2)),
        "test_rmse_usd": float(np.sqrt(np.mean(errors ** 2))),
        "test_mae_usd": float(np.mean(np.abs(errors))),
        "baseline_rmse_usd": float(np.sqrt(np.mean(baseline_errors ** 2))),
        "baseline_mae_usd": float(np.mean(np.abs(baseline_errors))),
        "epochs_trained": len(history.history["loss"]),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({
        "actual_close_usd": actual,
        "predicted_close_usd": predicted,
        "baseline_close_usd": baseline,
        "error_usd": errors,
    }).to_csv(output_dir / "predictions.csv", index=False)
    plot_results(history, actual, predicted, output_dir)
    model.save(output_dir / "final_model.keras")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(parse_args())
