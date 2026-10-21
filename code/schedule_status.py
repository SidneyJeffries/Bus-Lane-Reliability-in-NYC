# Early / on time / late status from the static schedule, shared by the analysis scripts.

import numpy as np
import pandas as pd

EARLY_S = 60
LATE_S = 300
TRIP_ID_PATTERN = r"-(Weekday|Saturday|Sunday)(?:-SDon|-SDoff)?-(\d{6})_"


def classify(deviation_s):
    return pd.Series(np.select([deviation_s < -EARLY_S, deviation_s > LATE_S, deviation_s.notna()],
                               ["early", "late", "on_time"], None), index=deviation_s.index, dtype="string")


def to_seconds(hhmmss):
    h, m, s = hhmmss.split(":")  # GTFS times can pass 24:00 for after-midnight trips
    return int(h) * 3600 + int(m) * 60 + int(s)


def scheduled_trips(arrivals, static):
    # Scheduled start: NYCT trip_ids encode it in hundredths of a minute after midnight,
    # e.g. UP_A5-Weekday-SDon-038300_B1_1 starts at 383.00 min = 06:23.
    seen = arrivals.drop_duplicates(["trip_id", "service_date"])[["trip_id", "route_id", "direction_id", "service_date"]]
    parts = seen["trip_id"].str.extract(TRIP_ID_PATTERN)
    seen = seen.assign(trip_day_type=parts[0].str.lower(), sched_start_s=parts[1].astype(float) / 100 * 60)
    seen = seen.dropna(subset=["sched_start_s"])

    # School holidays (e.g. Yom Kippur) run a no-school weekday timetable with its own trip_ids, which the static
    # snapshot doesn't list. Spot them as weekdays with no school-day (-SDon-) trips, and pair only the other
    # days' trips with the schedule below; holiday trips then borrow from a regular trip (see end).
    school_days = seen.loc[seen["trip_id"].str.contains("-SDon-"), "service_date"].unique()
    holiday = seen["trip_day_type"].eq("weekday") & ~seen["service_date"].isin(school_days) & (len(school_days) > 0)
    t = seen[~holiday].drop_duplicates("trip_id").drop(columns="service_date")
    holiday_trips = seen[holiday & ~seen["trip_id"].isin(t["trip_id"])].drop_duplicates("trip_id").drop(columns="service_date")

    rows = []
    for dow, day_type in [(1, "weekday"), (6, "saturday"), (7, "sunday")]:
        for d in (0, 1):
            times_col = f"hw_dow{dow}_dir{d}_departure_times"
            stop_col = f"hw_dow{dow}_dir{d}_selected_stop_id"
            for route_id, times, stop_id in zip(static["route_id"], static[times_col], static[stop_col]):
                if not isinstance(times, str):
                    continue
                tr = t[t["route_id"].eq(route_id) & t["direction_id"].eq(d)
                       & t["trip_day_type"].eq(day_type)].sort_values("sched_start_s")
                deps = np.array(sorted(to_seconds(x) for x in times.split()))
                if len(tr) != len(deps) or len(tr) < 2:
                    continue
                offset = deps - tr["sched_start_s"].to_numpy()
                if offset.min() < -60 or np.abs(np.diff(offset)).max() > 900:
                    continue
                rows.append(tr.assign(timepoint_stop_id=stop_id, sched_timepoint_s=deps))
    timepoints = pd.concat(rows)

    # Holiday trips: use the checkpoint time of the regular trip with the same route, direction and start time
    # (only where exactly one regular trip starts then, so the match is unambiguous)
    keys = ["route_id", "direction_id", "trip_day_type", "sched_start_s"]
    regular = timepoints.drop_duplicates(keys, keep=False)[keys + ["timepoint_stop_id", "sched_timepoint_s"]]
    borrowed = holiday_trips.merge(regular, on=keys)

    cols = ["trip_id", "timepoint_stop_id", "sched_timepoint_s"]
    timepoints = pd.concat([timepoints[cols], borrowed[cols]])
    return pd.concat([t, holiday_trips]).merge(timepoints, on="trip_id", how="left")


def add_schedule_status(arrivals, trips, static):
    sched = scheduled_trips(arrivals, static)
    a = arrivals.merge(sched[["trip_id", "sched_start_s", "timepoint_stop_id", "sched_timepoint_s"]],
                       on="trip_id", how="left")
    midnight = a["service_date"].dt.tz_localize("America/New_York")


    first = a.sort_values("arrival_time").drop_duplicates("trip_uid")
    origin = first.groupby(["route_id", "direction_id"])["stop_id"].agg(lambda s: s.mode().iloc[0]).rename("origin_stop_id")
    a = a.join(origin, on=["route_id", "direction_id"])
    at_departure = a.index.isin(first.index) & a["stop_id"].eq(a["origin_stop_id"]) & a["sched_start_s"].notna()
    at_timepoint = a["stop_id"].eq(a["timepoint_stop_id"])

    sched_s = np.where(at_timepoint, a["sched_timepoint_s"], np.where(at_departure, a["sched_start_s"], np.nan))
    a["status_point"] = pd.Series(np.select([at_timepoint, at_departure], ["timepoint", "departure"], None),
                                  index=a.index, dtype="string")
    a["sched_time"] = midnight + pd.to_timedelta(sched_s, unit="s")
    a["schedule_deviation_s"] = (a["arrival_time"] - a["sched_time"]).dt.total_seconds()
    a["status"] = classify(a["schedule_deviation_s"])
    a = a.drop(columns=["sched_start_s", "timepoint_stop_id", "sched_timepoint_s", "origin_stop_id"])

    #Parses out one row per trip
    per_trip = []
    for point in ["departure", "timepoint"]:
        p = a[a["status_point"].eq(point)].set_index("trip_uid")
        per_trip.append(p[["schedule_deviation_s", "status"]].add_prefix(f"{point}_"))
    trips = trips.drop(columns=[c for c in trips if c.startswith(("departure_", "timepoint_"))], errors="ignore")
    trips = trips.join(per_trip[0], on="trip_uid").join(per_trip[1], on="trip_uid")
    return a, trips
