import glob
import pandas as pd
# Check intervals for GTFS RT data
file = glob.glob("data/Comparison Periods/GTFS-RT data/Feb/Week 1/*.parquet")[0]
df = pd.read_parquet(file)
print(df.columns.tolist())
print(df.head())

df["t"] = pd.to_datetime(df["vehicle.timestamp"], unit="s")
per_bus = df.sort_values("t").groupby("vehicle.vehicle.id")["t"].diff().dt.total_seconds()
print("Per-bus update intervals (seconds):")
print(per_bus.describe())
print(per_bus.quantile([0.5, 0.9, 0.99]))