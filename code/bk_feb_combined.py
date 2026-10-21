# Brooklyn bus data, February comparison period: joined RT + static data, metrics, and schedule status.

import os
import sys

import numpy as np
import pandas as pd

try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive window
    CODE_DIR = os.getcwd() if os.path.basename(os.getcwd()) == "code" else os.path.join(os.getcwd(), "code")
os.chdir(os.path.dirname(CODE_DIR))
sys.path.insert(0, CODE_DIR)

from join_static_rt import METRICS_ROOT, STATIC_ROOT, load_brooklyn  # noqa: E402
from schedule_status import add_schedule_status  # noqa: E402

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
METRICS = os.path.join(METRICS_ROOT, "Feb")

#Joined data for Feb
feb_rt, feb_static = load_brooklyn(periods=["Feb"])
print(feb_rt.shape)
print(feb_rt.dtypes)
feb_rt.head()

#Static route data
print(feb_static.shape)
feb_static[["route_id", "route_long_name", "feed_id", "feed_version_fetched_at",
            "hw_dow1_dir0_selected_stop_name", "hw_dow1_dir0_headway_7am_9am_median",
            "hw_dow1_dir0_headway_9am_4pm_median"]].head(20)

#Metrics to calculate on time performance
arrivals = pd.read_parquet(os.path.join(METRICS, "brooklyn_arrivals.parquet"))   # bus passed stop at arrival_time
headways = pd.read_parquet(os.path.join(METRICS, "brooklyn_headways.parquet"))   # gap to the bus ahead, vs schedule
segments = pd.read_parquet(os.path.join(METRICS, "brooklyn_segments.parquet"))   # stop-to-stop travel times
trips = pd.read_parquet(os.path.join(METRICS, "brooklyn_trips.parquet"))         # first-to-last stop per trip
bunching = pd.read_parquet(os.path.join(METRICS, "brooklyn_bunching.parquet"))   # bunching episodes
for name, df in [("arrivals", arrivals), ("headways", headways), ("segments", segments),
                 ("trips", trips), ("bunching", bunching)]:
    print(f"{name}: {df.shape}")



static_routes = pd.read_csv(os.path.join(STATIC_ROOT, "Feb", "mta_bus_routes.csv"), dtype=str)
static_routes = static_routes[static_routes["route_id"].isin(feb_static["route_id"])]
arrivals, trips = add_schedule_status(arrivals, trips, static_routes)

# Trip-level status on every ping of that trip (a trip instance = trip_id + service date + vehicle)
trip_status = trips.assign(start_date=trips["service_date"].dt.strftime("%Y%m%d"))[
    ["trip_id", "start_date", "vehicle_id", "departure_status", "departure_schedule_deviation_s",
     "timepoint_status", "timepoint_schedule_deviation_s"]]
feb_rt = feb_rt.drop(columns=[c for c in trip_status.columns[3:] if c in feb_rt])
feb_rt = feb_rt.merge(trip_status, how="left",
                      left_on=["vehicle.trip.trip_id", "vehicle.trip.start_date", "vehicle.vehicle.id"],
                      right_on=["trip_id", "start_date", "vehicle_id"]).drop(columns=["trip_id", "start_date", "vehicle_id"])

print("trips with a status:", trips[["departure_status", "timepoint_status"]].notna().mean().round(3).to_dict())
arrivals.dropna(subset=["status"]).groupby("status_point")["status"].value_counts(normalize=True).unstack().round(3)

# %% One trip from pings to arrivals: raw pings vs estimated stop passages
example_trip = trips.loc[trips["route_id"].eq("B46+"), ["trip_uid", "n_stops"]].sort_values("n_stops").iloc[-1]["trip_uid"]
trip_arrivals = arrivals[arrivals["trip_uid"].eq(example_trip)].sort_values("arrival_time")
trip_id, start_date = trip_arrivals[["trip_id", "start_date"]].iloc[0]
trip_pings = feb_rt[feb_rt["vehicle.trip.trip_id"].eq(trip_id) & feb_rt["vehicle.trip.start_date"].eq(start_date)]
print(trip_pings[["timestamp_local", "vehicle.stop_id", "stop_name", "vehicle.vehicle.id"]].sort_values("timestamp_local").to_string())
trip_arrivals[["arrival_local", "stop_id", "stop_name", "filled", "gap_s"]]

# %% Headways at one stop over one weekday: what bunching and gaps look like
stop_day = headways[headways["route_id"].eq("B46+") & headways["direction_id"].eq(0)
                    & headways["service_date"].eq("2025-02-11")]
busiest_stop = stop_day["stop_name"].value_counts().index[0]
stop_day[stop_day["stop_name"].eq(busiest_stop)][
    ["arrival_local", "vehicle_id", "leader_vehicle_id", "headway_s", "sched_headway_s",
     "headway_ratio", "bunched", "gapped"]]

# %% Bunching and gaps by time of day (weekdays)
wk = headways[headways["day_type"].eq("weekday")]
wk.groupby("hour").agg(headways=("headway_s", "size"), bunched=("bunched", "mean"),
                       gapped=("gapped", "mean"), wait_assessment=("meets_wait_assessment", "mean")).round(3)

# %% Bunching by route (weekdays), worst first
by_route = wk.groupby("route_id").agg(
    headways=("headway_s", "size"), bunched=("bunched", "mean"), gapped=("gapped", "mean"),
    wait_assessment=("meets_wait_assessment", "mean"), median_ratio=("headway_ratio", "median"))
by_route.sort_values("bunched", ascending=False).round(3)

# %% Bunching episodes: how long do bunched pairs stay together?
print(bunching["n_stops"].describe())
print("persistent (3+ stops):", bunching["persistent"].mean().round(3))
bunching.sort_values("n_stops", ascending=False).head(20)

# %% Where does persistent bunching start? (weekday episodes of 3+ stops, by route and start stop)
(bunching[bunching["persistent"] & bunching["day_type"].eq("weekday")]
 .groupby(["route_id", "direction_id", "start_stop_name"]).size()
 .sort_values(ascending=False).head(25))

# %% Early / on time / late at the timepoint, by day type and hour
tp = arrivals[arrivals["status_point"].eq("timepoint")]
print(tp.groupby("day_type")["status"].value_counts(normalize=True).unstack().round(3))
print("deviation (min) quantiles:", (tp["schedule_deviation_s"] / 60).quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
tp[tp["day_type"].eq("weekday")].groupby("hour")["status"].value_counts(normalize=True).unstack().round(3)

# %% Early / on time / late at the timepoint by route (weekdays), most late first
(tp[tp["day_type"].eq("weekday")].groupby("route_id")["status"].value_counts(normalize=True)
 .unstack().round(3).sort_values("late", ascending=False))

# %% Does a late departure carry through? Timepoint status by departure status (trips with both)
both = trips.dropna(subset=["departure_status", "timepoint_status"])
pd.crosstab(both["departure_status"], both["timepoint_status"], normalize="index").round(3)
