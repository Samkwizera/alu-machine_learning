# BTC Time-Series Forecasting

This project converts minute-level Coinbase and Bitstamp BTC/USD candles into
hourly observations, then trains a recurrent neural network to predict the
closing price one hour ahead from the preceding 24 hours.

## Files

- `preprocess_data.py`: validates and combines exchange CSVs, resamples them to
  one-hour candles, adds cyclical time features, makes chronological splits,
  standardizes using training statistics only, and saves a compressed NPZ.
- `forecast_btc.py`: creates `tf.data.Dataset` windows, trains a two-layer GRU
  with mean-squared-error loss, evaluates it in USD, and saves the model,
  metrics, predictions, and graphs.
- `BLOG_POST.md`: an English article draft ready for publication after a real
  training run. Replace its marked metric placeholders with `metrics.json`.

## Data schema

Each input CSV must contain `Timestamp`, `Open`, `High`, `Low`, `Close`,
`Volume_(BTC)`, `Volume_(Currency)`, and `Weighted_Price`. Column matching is
case-insensitive. The raw files are intentionally excluded because they are
large; pass their paths on the command line.

## Usage

```bash
python preprocess_data.py coinbase.csv bitstamp.csv --output btc_hourly.npz
python forecast_btc.py btc_hourly.npz --epochs 50
```

Outputs from training are written to `btc_forecast_results/`. TensorFlow,
pandas, NumPy, and Matplotlib are required.

## Method and leakage prevention

Hourly OHLC values use first/max/min/last aggregation. Volumes are summed and
VWAP is recomputed from BTC volume. Raw Unix time is discarded after ordering;
hour-of-day and weekday are represented with sine/cosine pairs. Data remains
chronological (70% train, 15% validation, 15% test). Means and standard
deviations are fitted only on the training partition. Validation and test
windows receive exactly 24 context rows from the preceding partition, while
their labels stay fully inside their own partition.

The network consists of a 64-unit GRU, 20% dropout, a 32-unit GRU, and two
dense layers. GRUs retain useful sequence memory while using fewer parameters
than an equivalent LSTM. The final unit is linear because price is continuous.
Training uses Adam and MSE, with early stopping and best-model checkpointing.

This is an educational forecasting project, not financial advice. Always
compare its performance with a naive “next close equals current close”
baseline before interpreting the results.
