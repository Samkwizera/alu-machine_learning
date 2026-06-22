#!/usr/bin/env python3
"""Visualize daily cryptocurrency exchange data."""

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=["Weighted_Price"])
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit="s")
df = df.set_index("Date")

df["Close"] = df["Close"].ffill()
for column in ["High", "Low", "Open"]:
    df[column] = df[column].fillna(df["Close"])
df[["Volume_(BTC)", "Volume_(Currency)"]] = df[
    ["Volume_(BTC)", "Volume_(Currency)"]
].fillna(0)

df = df[df.index.date >= date(2017, 1, 1)]
df = df.resample("D").agg(
    {
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
    }
)

df.plot()
plt.show()
