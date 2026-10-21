import glob
import os
from collections import deque

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Joins each GTFS-RT vehicle-position file to the MTA bus static data
RT_ROOT = "data/Comparison Periods/GTFS-RT data"
STATIC_ROOT = "data/Comparison Periods/GTFS Static data"
OUT_ROOT = "data/Comparison Periods/Joined"
METRICS_ROOT = "data/Comparison Periods/Metrics"
PERIODS = ["Feb", "April", "July", "October"]
WEEKS = ["Week 1", "Week 2"]

# Service dates per period 
SERVICE_DATES = {
    "Feb": ("2025-02-01", "2025-02-14"),
    "April": ("2025-04-01", "2025-04-14"),
    "October": ("2025-10-01", "2025-10-14"),
}
# Dates to leave out of a period entirely. School holidays don't need to be here: schedule_status.py matches
# their trips to the regular timetable (Apr 14, Oct 2 and Oct 13 are kept that way).
SKIP_DATES = {}

# Headway bands per stop
HEADWAY_BANDS = ["7am_9am", "9am_4pm", "4pm_6pm", "6pm_7am"]

# Column names in RT data
TRIP = "vehicle.trip.trip_id"
DATE = "vehicle.trip.start_date"
ROUTE = "vehicle.trip.route_id"
DIR = "vehicle.trip.direction_id"
VEH = "vehicle.vehicle.id"
TS = "vehicle.timestamp"
STOP = "vehicle.stop_id"
LAT = "vehicle.position.latitude"
LON = "vehicle.position.longitude"

# Arrival inference settings
MAX_PING_GAP_S = 300       
MIN_EDGE_TRIPS = 5          
MIN_EDGE_SHARE = 0.05      
MAX_FILL_HOPS = 6           
TERMINAL_RADIUS_M = 100   

# Bunching settings (headway_ratio = observed / scheduled headway)
BUNCHED_RATIO = 0.25        
GAPPED_RATIO = 1.5         
PERSISTENT_BUNCH_STOPS = 3 

def load_static(period):
    folder = os.path.join(STATIC_ROOT, period)
    if not os.path.isdir(folder):
        return None, None

    routes = pd.read_csv(os.path.join(folder, "mta_bus_routes.csv"), dtype={"route_id": str})
    routes = routes[["route_id", "route_short_name", "route_long_name", "route_type"]
                    + [f"hw_best_headway_{b}_median" for b in HEADWAY_BANDS]]
    routes = routes.rename(columns={f"hw_best_headway_{b}_median": f"sched_headway_{b}_s"
                                    for b in HEADWAY_BANDS})

    stops = pd.read_csv(os.path.join(folder, "mta_bus_stops.csv"), dtype={"stop_id": str})
    stops["sched_departures_weekday"] = stops[[f"departure_count_dow{d}" for d in range(1, 6)]].mean(axis=1)
    stops = stops.rename(columns={"departure_count_dow6": "sched_departures_sat",
                                  "departure_count_dow7": "sched_departures_sun",
                                  "routes": "stop_routes"})
    stops = stops[["stop_id", "stop_name", "stop_lat", "stop_lon", "stop_routes",
                   "sched_departures_weekday", "sched_departures_sat", "sched_departures_sun"]]
    return routes, stops


def join_file(path, routes, stops, period, week):
    rt = pd.read_parquet(path)
    rt["timestamp_local"] = rt[TS].dt.tz_convert("America/New_York")
    rt["period"] = period
    rt["week"] = week

    joined = (rt.merge(routes, how="left", left_on=ROUTE, right_on="route_id")
                .drop(columns="route_id")
                .merge(stops, how="left", left_on=STOP, right_on="stop_id")
                .drop(columns="stop_id"))

    has_route = joined[ROUTE].notna()
    has_stop = joined[STOP].notna()
    route_match = joined.loc[has_route, "route_short_name"].notna().mean()
    stop_match = joined.loc[has_stop, "stop_name"].notna().mean()
    return joined, route_match, stop_match


def brooklyn_routes(period):
    # Brooklyn locals/SBS (B1..B103, B44+ ...) and BM express; route_type 3 excludes shuttles.
    # Case-sensitive so Bronx routes (BX1, BXM1 in GTFS-RT) are not picked up.
    routes = pd.read_csv(os.path.join(STATIC_ROOT, period, "mta_bus_routes.csv"), dtype={"route_id": str})
    bk = routes[routes["route_id"].str.match(r"^(B\d|BM\d)") & (routes["route_type"] == 3)]
    return bk


def load_brooklyn(periods=PERIODS, columns=None):
    # Load only Brooklyn rows from the joined files; repeated strings become categoricals to save memory.
    # All four periods take ~12 GB; pass periods=["Feb"] etc. to load less.
    static, rt = [], []
    for period in periods:
        bk = brooklyn_routes(period)
        static.append(bk.assign(period=period))
        files = sorted(glob.glob(os.path.join(OUT_ROOT, period, "*", "*.parquet")))
        for path in files:
            table = pq.read_table(path, columns=columns, filters=[(ROUTE, "in", bk["route_id"].tolist())])
            rt.append(table.to_pandas(strings_to_categorical=True))
    bk_static = pd.concat(static, ignore_index=True)
    bk_rt = pd.concat(rt, ignore_index=True)
    for col in ["period", "week", ROUTE, "route_short_name", "stop_name"]:
        if col in bk_rt:
            bk_rt[col] = bk_rt[col].astype("category")
    return bk_rt, bk_static


# Arrivals, headways and travel times

def dist_m(lat1, lon1, lat2, lon2):
    x = np.radians(lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
    y = np.radians(lat2 - lat1)
    return 6371000 * np.hypot(x, y)

def load_pings(period):
    cols = [TRIP, DATE, ROUTE, DIR, VEH, TS, STOP, LAT, LON]
    routes = brooklyn_routes(period)["route_id"].tolist()
    pings = pd.concat(
        [pq.read_table(f, columns=cols, filters=[(ROUTE, "in", routes)]).to_pandas()
         for f in sorted(glob.glob(os.path.join(OUT_ROOT, period, "*", "*.parquet")))],
        ignore_index=True)
    pings = pings.dropna(subset=[TRIP, DATE, STOP, DIR])
    start, end = SERVICE_DATES[period]
    pings["service_date"] = pd.to_datetime(pings[DATE], format="%Y%m%d")
    pings = pings[pings["service_date"].between(start, end)
                  & ~pings["service_date"].isin(pd.to_datetime(SKIP_DATES.get(period, [])))]
    pings = pings.drop_duplicates([TRIP, DATE, VEH, TS]).sort_values([TRIP, DATE, VEH, TS])
    pings["trip_uid"] = pings.groupby([TRIP, DATE, VEH], sort=False).ngroup()
    return pings.reset_index(drop=True)


def stop_transitions(pings):
    prev = pings.shift()
    changed = pings["trip_uid"].eq(prev["trip_uid"]) & pings[STOP].ne(prev[STOP])
    tr = pd.DataFrame({
        "trip_uid": pings["trip_uid"], "route_id": pings[ROUTE], "direction_id": pings[DIR],
        "from_stop": prev[STOP], "to_stop": pings[STOP],
        "t0": prev[TS], "t1": pings[TS],
        "lat0": prev[LAT], "lon0": prev[LON], "lat1": pings[LAT], "lon1": pings[LON],
    })[changed]
    tr["gap_s"] = (tr["t1"] - tr["t0"]).dt.total_seconds()
    return tr.reset_index(drop=True)


def stop_graph(tr):
    counts = tr.groupby(["route_id", "direction_id", "from_stop", "to_stop"]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby(["route_id", "direction_id", "from_stop"])["n"].transform("sum")
    edges = counts[(counts["n"] >= MIN_EDGE_TRIPS) & (counts["share"] >= MIN_EDGE_SHARE)]

    graphs = {}
    for key, g in edges.groupby(["route_id", "direction_id"]):
        adj = {}
        for a, b in zip(g["from_stop"], g["to_stop"]):
            adj.setdefault(a, set()).add(b)
        for a in list(adj):
            for c in sorted(adj[a]):
                if any(_path(adj, b, c, MAX_FILL_HOPS - 1, skip=(a, c)) for b in sorted(adj[a]) if b != c):
                    adj[a].discard(c)
        graphs[key] = adj
    return graphs


def _path(adj, start, end, max_hops, skip=None):
    # Shortest stop path start -> end (list of stops, inclusive) within max_hops, else None
    if start == end:
        return [start]
    queue, seen = deque([[start]]), {start}
    while queue:
        path = queue.popleft()
        if len(path) > max_hops:
            continue
        for nxt in sorted(adj.get(path[-1], ())):
            if skip and (path[-1], nxt) == skip:
                continue
            if nxt == end:
                return path + [nxt]
            if nxt not in seen:
                seen.add(nxt)
                queue.append(path + [nxt])
    return None


def estimate_arrivals(pings, stops):
    tr = stop_transitions(pings)
    tr = tr[tr["gap_s"] <= MAX_PING_GAP_S].reset_index(drop=True)
    graphs = stop_graph(tr)

    # Stops passed for each distinct (route, direction, from, to) move
    moves = tr[["route_id", "direction_id", "from_stop", "to_stop"]].drop_duplicates()
    passed = []
    for r, d, a, b in moves.itertuples(index=False):
        path = _path(graphs.get((r, d), {}), a, b, MAX_FILL_HOPS)
        passed.append(path[:-1] if path else [a])
    moves["passed_stop"] = passed
    moves = moves.explode("passed_stop")
    moves["filled"] = moves["passed_stop"].ne(moves["from_stop"])

    arr = tr.reset_index(names="move_id").merge(moves, on=["route_id", "direction_id", "from_stop", "to_stop"])
    arr = arr.merge(stops[["stop_id", "stop_lat", "stop_lon"]], left_on="passed_stop", right_on="stop_id", how="left")
    arr = arr.dropna(subset=["stop_lat"]).sort_values(["move_id", "filled"], kind="stable")

    # Distance along stops
    first = ~arr["move_id"].duplicated()
    prev_lat = arr["stop_lat"].shift().where(~first, arr["lat0"])
    prev_lon = arr["stop_lon"].shift().where(~first, arr["lon0"])
    arr["cum_m"] = dist_m(prev_lat, prev_lon, arr["stop_lat"], arr["stop_lon"]).groupby(arr["move_id"]).cumsum()
    last = ~arr["move_id"].duplicated(keep="last")
    tail = dist_m(arr["stop_lat"], arr["stop_lon"], arr["lat1"], arr["lon1"]).where(last)
    total = (arr["cum_m"] + tail).groupby(arr["move_id"]).transform("max")
    frac = (arr["cum_m"] / total).fillna(0.5).clip(0, 1)
    arr["arrival_time"] = arr["t0"] + pd.to_timedelta(frac * arr["gap_s"], unit="s")

    arrivals = arr[["trip_uid", "route_id", "direction_id", "passed_stop", "arrival_time", "filled", "gap_s"]]
    arrivals = arrivals.rename(columns={"passed_stop": "stop_id"})
    arrivals = pd.concat([arrivals, terminal_arrivals(pings, stops)], ignore_index=True)

    # A bus that briefly flips back to an earlier stop (GPS noise) is only counted at its first passage
    arrivals = arrivals.sort_values(["trip_uid", "arrival_time"]).drop_duplicates(["trip_uid", "stop_id"])

    trips = pings.drop_duplicates("trip_uid").set_index("trip_uid")[[TRIP, DATE, VEH, "service_date"]]
    arrivals = arrivals.join(trips, on="trip_uid").rename(
        columns={TRIP: "trip_id", DATE: "start_date", VEH: "vehicle_id"})
    arrivals = arrivals.merge(stops[["stop_id", "stop_name"]], on="stop_id", how="left")
    return add_time_bands(arrivals).reset_index(drop=True)


def terminal_arrivals(pings, stops):
    last_stop = pings.groupby("trip_uid")[STOP].transform("last")
    p = pings[pings[STOP].eq(last_stop)].merge(stops[["stop_id", "stop_lat", "stop_lon"]],
                                                left_on=STOP, right_on="stop_id")
    p["d"] = dist_m(p[LAT], p[LON], p["stop_lat"], p["stop_lon"])
    p = p.loc[p.groupby("trip_uid")["d"].idxmin()]
    p = p[p["d"] <= TERMINAL_RADIUS_M]
    return pd.DataFrame({"trip_uid": p["trip_uid"], "route_id": p[ROUTE], "direction_id": p[DIR],
                         "stop_id": p[STOP], "arrival_time": p[TS], "filled": False, "gap_s": np.nan})


def add_time_bands(df):
    local = df["arrival_time"].dt.tz_convert("America/New_York")
    df["arrival_local"] = local
    dow = df["service_date"].dt.dayofweek
    df["day_type"] = np.select([dow < 5, dow == 5], ["weekday", "saturday"], "sunday")
    hour = local.dt.hour
    df["hour"] = hour
    df["time_band"] = np.select([hour.between(7, 8), hour.between(9, 15), hour.between(16, 17)],
                                ["7am_9am", "9am_4pm", "4pm_6pm"], "6pm_7am")
    return df

# Headways by route and time period
def scheduled_headways(period):
    routes = pd.read_csv(os.path.join(STATIC_ROOT, period, "mta_bus_routes.csv"), dtype={"route_id": str})
    rows = []
    for dow, day_type in [(1, "weekday"), (6, "saturday"), (7, "sunday")]:
        for d in (0, 1):
            col = f"hw_dow{dow}_dir{d}_departure_times"
            for route_id, times in zip(routes["route_id"], routes[col]):
                if not isinstance(times, str):
                    continue
                # GTFS times can run past 24:00 for after-midnight trips
                secs = sorted(int(h) * 3600 + int(m) * 60 + int(s)
                              for h, m, s in (t.split(":") for t in times.split()))
                for start, end in zip(secs, secs[1:]):
                    for hour in range(start // 3600, (end - 1) // 3600 + 1):
                        rows.append((route_id, float(d), day_type, hour % 24, end - start))
    gaps = pd.DataFrame(rows, columns=["route_id", "direction_id", "day_type", "hour", "gap_s"])
    return (gaps.groupby(["route_id", "direction_id", "day_type", "hour"])["gap_s"]
                .agg(sched_headway_s="median", sched_gaps="size").reset_index())


def compute_headways(arrivals, period):
    h = arrivals.sort_values(["route_id", "direction_id", "stop_id", "service_date", "arrival_time"]).copy()
    grp = h.groupby(["route_id", "direction_id", "stop_id", "service_date"])
    h["headway_s"] = grp["arrival_time"].diff().dt.total_seconds()
    for col in ["trip_uid", "trip_id", "vehicle_id", "arrival_time"]:
        h[f"leader_{col}"] = grp[col].shift()
    # Same bus or same trip twice in a row (a bus ending one trip and starting the next at a terminal,
    # or a trip reassigned to another vehicle) is not a gap between two buses
    same_bus = h["vehicle_id"].eq(h["leader_vehicle_id"]) | h["trip_id"].eq(h["leader_trip_id"])
    h = h[~same_bus].dropna(subset=["headway_s"])
    h = h.merge(scheduled_headways(period), on=["route_id", "direction_id", "day_type", "hour"], how="left")
    h["headway_ratio"] = h["headway_s"] / h["sched_headway_s"]
    no_sched = h["sched_headway_s"].isna()
    h["bunched"] = (h["headway_ratio"] < BUNCHED_RATIO).astype("boolean").mask(no_sched)
    h["gapped"] = (h["headway_ratio"] > GAPPED_RATIO).astype("boolean").mask(no_sched)
    peak = h["day_type"].eq("weekday") & h["time_band"].isin(["7am_9am", "4pm_6pm"])
    allowance = np.where(peak, 180, 300)
    h["meets_wait_assessment"] = (h["headway_s"] <= h["sched_headway_s"] + allowance).astype("boolean").mask(no_sched)
    return h.reset_index(drop=True)

#Computes when buses are bunched on a particular route
def compute_bunching(headways):
    h = headways.dropna(subset=["bunched"]).sort_values(["leader_trip_uid", "trip_uid", "arrival_time"])
    pair = ["leader_trip_uid", "trip_uid"]
    bunched = h["bunched"].astype(bool)
    new_pair = h[pair].ne(h[pair].shift()).any(axis=1)
    starts = bunched & (new_pair | ~bunched.shift(fill_value=False).astype(bool))
    h = h.assign(episode_id=starts.cumsum())[bunched]

    g = h.groupby("episode_id")
    episodes = g.agg(
        route_id=("route_id", "first"), direction_id=("direction_id", "first"),
        service_date=("service_date", "first"), day_type=("day_type", "first"),
        time_band=("time_band", "first"), hour=("hour", "first"),
        leader_trip_id=("leader_trip_id", "first"), leader_vehicle_id=("leader_vehicle_id", "first"),
        follower_trip_id=("trip_id", "first"), follower_vehicle_id=("vehicle_id", "first"),
        start_stop_id=("stop_id", "first"), start_stop_name=("stop_name", "first"),
        end_stop_id=("stop_id", "last"), end_stop_name=("stop_name", "last"),
        start_time=("arrival_local", "first"), end_time=("arrival_local", "last"),
        n_stops=("stop_id", "size"),
        min_headway_s=("headway_s", "min"), median_headway_s=("headway_s", "median"),
        sched_headway_s=("sched_headway_s", "median"),
    ).reset_index()
    episodes["duration_s"] = (episodes["end_time"] - episodes["start_time"]).dt.total_seconds()
    episodes["persistent"] = episodes["n_stops"] >= PERSISTENT_BUNCH_STOPS
    return episodes


def compute_travel_times(arrivals):
    # Stop-to-stop segment times within each trip, and whole-trip times between first and last stop seen.
    # Segments where either end was filled in are interpolated within one ping gap, not directly observed.
    a = arrivals.sort_values(["trip_uid", "arrival_time"])
    nxt = a.groupby("trip_uid")[["stop_id", "stop_name", "arrival_time", "filled"]].shift(-1)
    segments = a.assign(to_stop_id=nxt["stop_id"], to_stop_name=nxt["stop_name"],
                        to_arrival_time=nxt["arrival_time"], interpolated=a["filled"] | nxt["filled"])
    segments = segments.dropna(subset=["to_stop_id"]).rename(
        columns={"stop_id": "from_stop_id", "stop_name": "from_stop_name", "arrival_time": "from_arrival_time"})
    segments["travel_s"] = (segments["to_arrival_time"] - segments["from_arrival_time"]).dt.total_seconds()
    segments = segments[["trip_uid", "trip_id", "service_date", "route_id", "direction_id", "day_type",
                         "time_band", "from_stop_id", "from_stop_name", "to_stop_id", "to_stop_name",
                         "from_arrival_time", "to_arrival_time", "travel_s", "interpolated"]]

    g = a.groupby("trip_uid")
    trips = g.agg(trip_id=("trip_id", "first"), service_date=("service_date", "first"),
                  route_id=("route_id", "first"), direction_id=("direction_id", "first"),
                  vehicle_id=("vehicle_id", "first"), day_type=("day_type", "first"),
                  time_band=("time_band", "first"),
                  first_stop_id=("stop_id", "first"), first_stop_name=("stop_name", "first"),
                  last_stop_id=("stop_id", "last"), last_stop_name=("stop_name", "last"),
                  start_time=("arrival_time", "first"), end_time=("arrival_time", "last"),
                  n_stops=("stop_id", "size")).reset_index()
    trips["runtime_s"] = (trips["end_time"] - trips["start_time"]).dt.total_seconds()
    return segments.reset_index(drop=True), trips


def build_metrics(period):
    _, stops = load_static(period)
    pings = load_pings(period)
    arrivals = estimate_arrivals(pings, stops)
    headways = compute_headways(arrivals, period)
    segments, trips = compute_travel_times(arrivals)
    bunching = compute_bunching(headways)

    out_dir = os.path.join(METRICS_ROOT, period)
    os.makedirs(out_dir, exist_ok=True)
    for name, df in [("arrivals", arrivals), ("headways", headways), ("segments", segments),
                     ("trips", trips), ("bunching", bunching)]:
        df.to_parquet(os.path.join(out_dir, f"brooklyn_{name}.parquet"), index=False)
    return pings, arrivals, headways, segments, trips, bunching


if __name__ == "__main__":
    for period in PERIODS:
        routes, stops = load_static(period)
        if routes is None:
            print(f"{period}: no static data in {STATIC_ROOT}/{period}, skipped (run gtfs.py)")
            continue

        for week in WEEKS:
            out_dir = os.path.join(OUT_ROOT, period, week)
            os.makedirs(out_dir, exist_ok=True)
            for path in sorted(glob.glob(os.path.join(RT_ROOT, period, week, "*.parquet"))):
                out_path = os.path.join(out_dir, os.path.basename(path))
                if os.path.exists(out_path):
                    continue
                joined, route_match, stop_match = join_file(path, routes, stops, period, week)
                joined.to_parquet(out_path, index=False)
                print(f"{period} {week} {os.path.basename(path)}: {len(joined):,} rows | "
                      f"{joined['timestamp_local'].min():%Y-%m-%d %H:%M} to "
                      f"{joined['timestamp_local'].max():%Y-%m-%d %H:%M} ET | "
                      f"route match {route_match:.1%}, stop match {stop_match:.1%}")

    # Comparison periods with service dates set; periods that already have metrics are skipped
    for period in SERVICE_DATES:
        if os.path.exists(os.path.join(METRICS_ROOT, period, "brooklyn_arrivals.parquet")):
            print(f"{period}: metrics already built, skipped")
            continue
        pings, arrivals, headways, segments, trips, bunching = build_metrics(period)
        print(f"{period}: {len(pings):,} pings -> {len(arrivals):,} arrivals "
              f"({arrivals['filled'].mean():.1%} filled in), {arrivals['trip_uid'].nunique():,} trips")
        print(f"{len(headways):,} headways, {len(segments):,} segments, {len(bunching):,} bunching episodes")

        wk = headways[headways["day_type"].eq("weekday")]
        summary = wk.groupby("time_band").agg(
            headways=("headway_s", "size"),
            median_min=("headway_s", lambda s: s.median() / 60),
            sched_median_min=("sched_headway_s", lambda s: s.median() / 60),
            wait_assessment=("meets_wait_assessment", "mean"),
            bunched=("bunched", "mean"),
            gapped=("gapped", "mean"))
        print("Weekday headways, all Brooklyn routes:")
        print(summary.round(3).to_string())
