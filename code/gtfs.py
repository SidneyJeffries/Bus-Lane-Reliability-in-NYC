import os
import pandas as pd

# Transitland snapshot used for each GTFS-RT comparison period.
# Each snapshot summarizes one "target week" of scheduled service (see notes).
DOWNLOADS = r"C:\Users\sidne\Downloads"
SNAPSHOTS = {
    "Feb": "tl-dataset-US-2025-03-14T04_00_27",   # schedule week Feb 10-16, 2025
    "April": "tl-dataset-US-2025-04-17T20_35_37", # schedule week Mar 30-Apr 5, 2025
    "July": "tl-dataset-US-2025-07-23T22_47_20",  # schedule week Jun 29-Jul 5, 2025
    "October": "tl-dataset-US-2025-10-13T11_14_54",
}
OUT_ROOT = "data/Comparison Periods/GTFS Static data"

# MTA bus feeds in Transitland: MTA Bus Company + the five NYCT borough bus feeds
MTA_BUS_FEEDS = [
    "f-dr5r-mtabc",
    "f-dr5r-mtanyctbusbrooklyn",
    "f-dr5r-mtanyctbusmanhattan",
    "f-dr5r-mtanyctbusstatenisland",
    "f-dr5x-mtanyctbusqueens",
    "f-dr72-mtanyctbusbronx",
]
DOW_COLS = [f"departure_count_dow{d}" for d in range(1, 8)]


def build_static(period, snapshot):
    prefix = os.path.join(DOWNLOADS, snapshot, snapshot + "-")
    out_dir = os.path.join(OUT_ROOT, period)
    os.makedirs(out_dir, exist_ok=True)

    stops = pd.read_csv(prefix + "stops.csv", low_memory=False, dtype={"stop_id": str})
    routes = pd.read_csv(prefix + "routes.csv", low_memory=False, dtype={"route_id": str})
    stops = stops[stops["feed_id"].isin(MTA_BUS_FEEDS)]
    routes = routes[routes["feed_id"].isin(MTA_BUS_FEEDS)]

    # Stop -> route lookup (long format). Transitland only lists up to 5 routes per stop.
    stop_routes = pd.concat(
        [
            stops[["feed_id", "stop_id", f"route_id_{i}", f"route_short_name_{i}", f"route_type_{i}"]]
            .rename(columns={f"route_id_{i}": "route_id",
                             f"route_short_name_{i}": "route_short_name",
                             f"route_type_{i}": "route_type"})
            for i in range(1, 6)
        ]
    ).dropna(subset=["route_id"]).drop_duplicates(["stop_id", "route_id"])

    # Stops: a stop served by both NYCT and MTA Bus Co appears once per feed (coords differ by
    # a few meters), with departures counting only that feed's routes. Collapse to one row per stop.
    stops_agg = stops.groupby("stop_id").agg(
        stop_name=("stop_name", "first"),
        stop_lat=("stop_lat", "mean"),
        stop_lon=("stop_lon", "mean"),
        feeds=("feed_id", lambda f: ";".join(sorted(f))),
        feed_version_fetched_at=("feed_version_fetched_at", "min"),
        **{c: (c, "sum") for c in DOW_COLS},
    ).reset_index()
    route_lists = stop_routes.groupby("stop_id")["route_short_name"].agg(lambda r: ";".join(sorted(set(r))))
    stops_agg["routes"] = stops_agg["stop_id"].map(route_lists)
    stops_agg["routes_maybe_truncated"] = stops_agg["stop_id"].isin(stops.loc[stops["route_id_5"].notna(), "stop_id"])

    # Routes: each NYCT borough feed repeats the full NYCT route list but only fills headways for its
    # own borough's routes, so keep the copy with the most headway data.
    # route_type 3 = bus, 711 = shuttle bus (extended GTFS type). Headways are in seconds.
    hw_cols = [c for c in routes.columns if c.startswith("hw_")]
    routes = (routes.assign(_filled=routes[hw_cols].notna().sum(axis=1))
              .sort_values("_filled", ascending=False)
              .drop_duplicates("route_id")
              .drop(columns="_filled")
              .sort_values("route_id"))

    dates = routes.filter(like="service_date").stack()
    print(f"{period}: {len(stops_agg)} stops, {len(routes)} routes "
          f"({routes['hw_best_headway_9am_4pm_median'].isna().sum()} without headways) | "
          f"fetched {stops['feed_version_fetched_at'].min()[:10]} to {stops['feed_version_fetched_at'].max()[:10]} | "
          f"schedule week {dates.min()} to {dates.max()}")

    stops_agg.to_csv(os.path.join(out_dir, "mta_bus_stops.csv"), index=False)
    stop_routes.to_csv(os.path.join(out_dir, "mta_bus_stop_routes.csv"), index=False)
    routes.to_csv(os.path.join(out_dir, "mta_bus_routes.csv"), index=False)


if __name__ == "__main__":
    for period, snapshot in SNAPSHOTS.items():
        if snapshot is None:
            print(f"{period}: no MTA snapshot available, skipped")
            continue
        build_static(period, snapshot)
