# Flatbush Ave (B41, no bus lanes) vs Fulton St (B25, peak-hour bus lanes), for each comparison period in PERIODS:
# 1) table of % early / on time / late / bunched, plus a bus lane reference table
# 2) on-time performance by hour of day and by date
# 3) folium map of route segments and stops
# Outputs go to figures/, prefixed by period (feb_, apr_).

import os
import sys
import textwrap
import webbrowser

import folium
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive window
    CODE_DIR = os.getcwd() if os.path.basename(os.getcwd()) == "code" else os.path.join(os.getcwd(), "code")
os.chdir(os.path.dirname(CODE_DIR))
sys.path.insert(0, CODE_DIR)

from join_static_rt import METRICS_ROOT, SERVICE_DATES, SKIP_DATES, STATIC_ROOT  # noqa: E402
from schedule_status import add_schedule_status, classify  # noqa: E402

pd.set_option("display.width", 200)
PERIODS = ["Feb", "April", "October"]   # each needs its metrics built first (join_static_rt.py)
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

ROUTES = {"B41": "Flatbush Ave", "B25": "Fulton St"}   # add "B26": "Fulton St" etc. to compare more routes
CORRIDOR_COLORS = {"Flatbush Ave": "#2a78d6", "Fulton St": "#eb6834"}
STATUS_COLORS = {"early": "#eda100", "on_time": "#0ca30c", "late": "#d03b3b"}
STATUS_LABELS = {"early": "Early", "on_time": "On time", "late": "Late"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

sns.set_theme(style="whitegrid", rc={"grid.color": GRID, "axes.edgecolor": GRID, "text.color": INK,
                                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


# %% Load each period's metrics for the two routes and add early / on time / late
def load_period(period):
    start, end = (pd.Timestamp(d) for d in SERVICE_DATES[period])
    metrics = os.path.join(METRICS_ROOT, period)
    route_filter = [("route_id", "in", list(ROUTES))]
    arrivals = pd.read_parquet(os.path.join(metrics, "brooklyn_arrivals.parquet"), filters=route_filter)
    trips = pd.read_parquet(os.path.join(metrics, "brooklyn_trips.parquet"), filters=route_filter)
    headways = pd.read_parquet(os.path.join(metrics, "brooklyn_headways.parquet"), filters=route_filter,
                               columns=["route_id", "direction_id", "stop_id", "bunched"])
    static_routes = pd.read_csv(os.path.join(STATIC_ROOT, period, "mta_bus_routes.csv"), dtype=str)
    static_routes = static_routes[static_routes["route_id"].isin(ROUTES)]
    stops = pd.read_csv(os.path.join(STATIC_ROOT, period, "mta_bus_stops.csv"), dtype={"stop_id": str},
                        usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"])

    arrivals, trips = add_schedule_status(arrivals, trips, static_routes)
    arrivals["corridor"] = arrivals["route_id"].map(ROUTES)
    headways["corridor"] = headways["route_id"].map(ROUTES)
    label = f"{start:%b} {start.day}-{end.day} 2025"   # e.g. "Feb 1-14 2025"
    if SKIP_DATES.get(period):                         # e.g. "Oct 1-14 2025 (excl. Oct 2, 13)"
        label += f" (excl. {start:%b} " + ", ".join(str(pd.Timestamp(x).day) for x in SKIP_DATES[period]) + ")"
    print(f"{period}: share of trips with a checkpoint status:", trips.groupby("route_id")["timepoint_status"]
          .apply(lambda s: s.notna().mean()).round(2).to_dict())
    return {
        "period": period, "start": start, "end": end,
        "label": label,
        "prefix": period.lower()[:3],                         # output file prefix: feb_, apr_
        "arrivals": arrivals, "trips": trips, "headways": headways, "stops": stops,
        # Early / on time / late is measured where the schedule gives a time: the route's schedule checkpoint
        "tp": arrivals[arrivals["status_point"].eq("timepoint")],
    }


data = {period: load_period(period) for period in PERIODS}


# %% Tables: % early / on time / late / bunched by corridor, schedule checkpoints, bus lanes
def save_table(df, path, title, wrap=22):
    cells = [[textwrap.fill(str(v), wrap) for v in row] for row in df.to_numpy()]
    fig, ax = plt.subplots(figsize=(1.9 * len(df.columns), 0.75 * len(df) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=[textwrap.fill(c, 16) for c in df.columns],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 3)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0efec")
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()


def summary_table(d):
    tp, headways = d["tp"], d["headways"]
    status_pct = tp.groupby("corridor")["status"].value_counts(normalize=True).unstack().mul(100)
    return pd.DataFrame({
        "Route": tp.groupby("corridor")["route_id"].first(),
        "% Early": status_pct["early"],
        "% On time": status_pct["on_time"],
        "% Late": status_pct["late"],
        "% Bunched": headways.groupby("corridor")["bunched"].mean() * 100,
    }).round(1).reset_index().rename(columns={"corridor": "Corridor"})


def checkpoint_table(d):
    # Where early / on time / late is measured: each route's schedule checkpoint stop(s), by direction
    day_order = ["weekday", "saturday", "sunday"]
    toward = d["trips"].groupby(["route_id", "direction_id"])["last_stop_name"].agg(lambda s: s.mode().iloc[0])
    cp = (d["tp"].groupby(["corridor", "route_id", "direction_id", "stop_name"])
          .agg(days=("day_type", lambda s: ", ".join(x.title() for x in day_order if x in set(s))),
               arrivals=("status", "size"))
          .reset_index())
    cp["direction"] = "Toward " + cp.set_index(["route_id", "direction_id"]).index.map(toward).str.title()
    cp = cp[["corridor", "route_id", "direction", "stop_name", "days", "arrivals"]].set_axis(
        ["Corridor", "Route", "Direction", "Checkpoint stop", "Days measured", "Arrivals checked"], axis=1)
    cp["Checkpoint stop"] = cp["Checkpoint stop"].str.title()
    return cp


summaries = {}
for period, d in data.items():
    summaries[period] = summary_table(d)
    print(f"\n{period}:\n" + summaries[period].to_string(index=False))
    summaries[period].to_csv(os.path.join(OUT, f"{d['prefix']}_flatbush_vs_fulton_summary.csv"), index=False)
    save_table(summaries[period], os.path.join(OUT, f"{d['prefix']}_flatbush_vs_fulton_summary.png"),
               f"Flatbush Ave vs Fulton St, {d['label']}")
    save_table(checkpoint_table(d), os.path.join(OUT, f"{d['prefix']}_schedule_checkpoints.png"),
               f"Schedule checkpoints, {d['label']}: where early / on time / late is measured")

# All periods in one table
by_period = pd.concat([s.assign(Period=data[p]["label"]) for p, s in summaries.items()])
by_period = by_period[["Corridor", "Route", "Period", "% Early", "% On time", "% Late", "% Bunched"]]
by_period = by_period.sort_values("Corridor", kind="stable")
by_period.to_csv(os.path.join(OUT, "flatbush_vs_fulton_by_period.csv"), index=False)
save_table(by_period, os.path.join(OUT, "flatbush_vs_fulton_by_period.png"), "Flatbush Ave vs Fulton St by period")


# %% Chart: each measure across the comparison periods
trend = pd.concat([s.assign(period=f"{data[p]['start']:%b}") for p, s in summaries.items()])
measures = ["% Early", "% On time", "% Late", "% Bunched"]
fig, axes = plt.subplots(1, len(measures), figsize=(13, 4.2), sharey=True)
for ax, measure in zip(axes, measures):
    sns.lineplot(data=trend, x="period", y=measure, hue="Corridor", palette=CORRIDOR_COLORS, sort=False,
                 marker="o", markersize=7, linewidth=2, legend=False, ax=ax)
    ax.set_title(measure.removeprefix("% "), loc="left", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylim(0, 50)
axes[0].set_ylabel("%")
for corridor, g in trend.groupby("Corridor"):   # label the lines at the right end of the last panel
    axes[-1].annotate(corridor, (len(g) - 1, g[measures[-1]].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                      va="center", fontsize=9, color=INK)
axes[-1].set_xlim(right=len(PERIODS) - 0.4)
handles = [plt.Line2D([], [], color=c, marker="o", linewidth=2, label=k) for k, c in CORRIDOR_COLORS.items()]
fig.legend(handles=handles, loc="upper right", frameon=False, ncol=2)
fig.suptitle("Flatbush Ave vs Fulton St across 2025 comparison periods", x=0.01, ha="left", fontsize=13,
             weight="bold")
fig.text(0.01, -0.04, "Early / on time / late are measured at schedule checkpoints; February's B41 number uses fewer "
         "checkpoints than April and October.\nBunched = share of arrivals less than a quarter of the scheduled gap "
         "behind the bus ahead.", fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "flatbush_vs_fulton_trends.png"), dpi=200, bbox_inches="tight")
plt.show()

# Bus lane reference table (status during the 2025 study periods)
bus_lanes = pd.DataFrame([
    ["Fulton St (eastbound)", "B25, B26", "Flatbush Ave & Ft. Greene Pl", "May 2018", "Peak hour (4-7pm weekdays)"],
    ["Fulton St (westbound)", "B25, B26", "S. Oxford St & Hudson Ave", "May 2018", "Peak hour (7-10am weekdays)"],
    ["Fulton St", "B25, B26", "Lafayette Ave & Grand Ave", "Summer 2018",
     "Peak hour (N side 7-10am, S side 2-7pm weekdays)"],
    ["Flatbush Ave", "B41", "37 intersections", "2022", "No bus lane; transit signal priority only"],
    ["Flatbush Ave", "B41", "Livingston St & Grand Army Plaza", "Late 2026 (after study period)",
     "Center-running"],
], columns=["Street", "Routes", "Endpoints", "Installation Date", "Type of Bus Lane"])
bus_lanes.to_csv(os.path.join(OUT, "bus_lanes_flatbush_fulton.csv"), index=False)
save_table(bus_lanes, os.path.join(OUT, "bus_lanes_flatbush_fulton.png"), "Bus lanes on the study corridors")


# %% Charts: when is on-time performance highest? By hour of day (weekdays) and by date
def pct_on_time(df, by):
    return (df.groupby(["corridor", by])["status"].apply(lambda s: s.eq("on_time").mean() * 100)
              .rename("pct_on_time").reset_index())


def on_time_chart(d):
    tp, start, end = d["tp"], d["start"], d["end"]
    weekday = tp[tp["day_type"].eq("weekday")]
    by_hour = pct_on_time(weekday, "hour")
    by_hour = by_hour.merge(weekday.groupby(["corridor", "hour"]).size().rename("n").reset_index())
    by_hour = by_hour[by_hour["n"] >= 20]   # drop overnight hours with too few buses to trust
    by_date = pct_on_time(tp, "service_date")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, df, x in [(ax1, by_hour, "hour"), (ax2, by_date, "service_date")]:
        sns.lineplot(data=df, x=x, y="pct_on_time", hue="corridor", palette=CORRIDOR_COLORS,
                     marker="o", markersize=6, linewidth=2, ax=ax)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% on time at checkpoint")
        ax.get_legend().remove()
        ends = df.groupby("corridor").tail(1).sort_values("pct_on_time")   # label each line at its right end
        close = ends["pct_on_time"].diff().abs().min() < 6                  # nudge labels apart if they'd overlap
        for i, last in enumerate(ends.itertuples()):
            dy = (-6 if i == 0 else 6) if close else 0
            ax.annotate(last.corridor, (getattr(last, x), last.pct_on_time), xytext=(6, dy),
                        textcoords="offset points", va="center", fontsize=9, color=INK)

    ax1.set_title("Weekdays, by hour of day", loc="left", fontsize=11)
    ax1.set_xlabel("Time of day")
    ax1.set_xticks(range(0, 24, 3))
    ax1.xaxis.set_major_formatter(lambda h, _: f"{int(h) % 12 or 12}:00{'am' if h < 12 else 'pm'}")   # 6 -> 6:00am
    ax1.set_xlim(right=25)
    for day in pd.date_range(start, end):
        if day.dayofweek >= 5:   # shade weekends
            ax2.axvspan(day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=12), color="#f0efec", zorder=0)
    ax2.set_title("By date (weekends shaded)", loc="left", fontsize=11)
    ax2.set_xlabel("")
    ax2.set_xlim(right=end + pd.Timedelta(days=2))
    ax2.set_xticks(pd.date_range(start, end, freq="2D"))
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    handles = [plt.Line2D([], [], color=c, marker="o", linewidth=2, label=k) for k, c in CORRIDOR_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=2)
    fig.suptitle(f"On-time performance, {d['label']}", x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{d['prefix']}_on_time_by_hour_and_date.png"), dpi=200, bbox_inches="tight")
    plt.show()


for d in data.values():
    on_time_chart(d)


# %% Status at every stop (estimated), for the maps
# The schedule only gives a time at the checkpoint (and departure) stop. To color every segment, each trip's
# checkpoint deviation is carried along the route, adjusted by how much faster or slower that trip ran than
# the typical (median) trip for the same route, direction, day type and time band:
#   est. deviation at a stop = deviation at checkpoint + (trip's time from checkpoint - median time from checkpoint)
# This assumes scheduled running times match median observed running times.
STOP_KEYS = ["route_id", "direction_id", "stop_id"]


def stop_status(d):
    arrivals, stops = d["arrivals"], d["stops"]
    anchor = (arrivals.dropna(subset=["status"])
              .sort_values("status_point", ascending=False)   # prefer the timepoint over the departure
              .drop_duplicates("trip_uid")
              [["trip_uid", "stop_id", "arrival_time", "schedule_deviation_s"]]
              .set_axis(["trip_uid", "anchor_stop_id", "anchor_time", "anchor_deviation_s"], axis=1))
    along = arrivals.merge(anchor, on="trip_uid")
    along["since_anchor_s"] = (along["arrival_time"] - along["anchor_time"]).dt.total_seconds()
    typical = along.groupby(["route_id", "direction_id", "anchor_stop_id", "stop_id", "day_type", "time_band"])[
        "since_anchor_s"].transform("median")
    along["est_status"] = classify(along["anchor_deviation_s"] + along["since_anchor_s"] - typical)

    by_stop = along.groupby(STOP_KEYS)["est_status"].value_counts(normalize=True).unstack(fill_value=0)
    by_stop["most_common"] = by_stop[["early", "on_time", "late"]].idxmax(axis=1)
    by_stop["bunched"] = d["headways"].groupby(STOP_KEYS)["bunched"].mean()
    by_stop = by_stop.reset_index().merge(stops, on="stop_id")

    # Segments: each stop to the next stop most trips go to
    seq = arrivals.sort_values(["trip_uid", "arrival_time"])
    seq["next_stop_id"] = seq.groupby("trip_uid")["stop_id"].shift(-1)
    links = seq.dropna(subset=["next_stop_id"]).groupby(STOP_KEYS + ["next_stop_id"]).size().rename("n").reset_index()
    links = links[links["n"] >= 30].sort_values("n").drop_duplicates(STOP_KEYS, keep="last")
    segs = (links.merge(by_stop, on=STOP_KEYS)
            .merge(stops.add_prefix("next_"), on="next_stop_id"))   # colored by status at the segment's start stop
    return by_stop, segs


# %% Maps
BUNCH_BINS = [(0.10, "#9ec5f4", "Under 10%"), (0.15, "#3987e5", "10-15%"), (1.01, "#184f95", "15% or more")]
# Corridor labels: street name and the routes running on it, placed at the middle stop on that street
CORRIDOR_LABELS = {   # mapped route: (street in stop names, routes on the corridor, label offset in pixels)
    "B41": ("FLATBUSH AV", "B41", (-16, 22)),          # to the right of the line
    "B25": ("FULTON ST", "B25 (shown), B26", (75, 58)),  # above the line
}


def bunch_color(share):
    return next(color for upper, color, _ in BUNCH_BINS if share < upper)


def popup(row):
    return folium.Popup(
        f"<b>{row.stop_name}</b><br>{row.route_id} ({ROUTES[row.route_id]}), direction {int(row.direction_id)}"
        + f"<br>Early {row.early:.0%} &middot; On time {row.on_time:.0%} &middot; Late {row.late:.0%}"
        + f"<br>Bunched {row.bunched:.0%}", max_width=280)


def make_map(d):
    by_stop, segs = stop_status(d)
    m = folium.Map(location=[by_stop["stop_lat"].mean(), by_stop["stop_lon"].mean()], zoom_start=13, tiles=None)
    # Esri light gray basemap: no API key, and unlike OpenStreetMap's servers it doesn't block local HTML files
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
                     max_zoom=16, control=False).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", max_zoom=16, overlay=True, control=False).add_to(m)   # street and place names
    status_layer = folium.FeatureGroup("Early / on time / late")
    bunch_layer = folium.FeatureGroup("Bunching", show=False)
    stop_layer = folium.FeatureGroup("Stops")

    for row in segs.itertuples():
        line = [(row.stop_lat, row.stop_lon), (row.next_stop_lat, row.next_stop_lon)]
        folium.PolyLine(line, color=STATUS_COLORS[row.most_common], weight=5, opacity=0.9,
                        tooltip=f"{row.route_id}: {STATUS_LABELS[row.most_common]}").add_to(status_layer)
        if pd.notna(row.bunched):
            folium.PolyLine(line, color=bunch_color(row.bunched), weight=5, opacity=0.9,
                            tooltip=f"{row.route_id}: {row.bunched:.0%} bunched").add_to(bunch_layer)

    for row in by_stop.itertuples():
        folium.CircleMarker((row.stop_lat, row.stop_lon), radius=3, color=MUTED, weight=1, fill=True,
                            fill_color="white", fill_opacity=1, popup=popup(row)).add_to(stop_layer)

    for layer in [status_layer, bunch_layer, stop_layer]:
        layer.add_to(m)

    for route_id, (street, routes, offset) in CORRIDOR_LABELS.items():
        on_street = by_stop[by_stop["route_id"].eq(route_id) & by_stop["stop_name"].str.startswith(street)]
        dist = ((on_street["stop_lat"] - on_street["stop_lat"].mean()) ** 2
                + (on_street["stop_lon"] - on_street["stop_lon"].mean()) ** 2)
        mid = on_street.loc[dist.idxmin()]
        folium.Marker((mid["stop_lat"], mid["stop_lon"]), icon=folium.DivIcon(
            icon_size=(150, 44), icon_anchor=offset,
            html=f'<div style="display:inline-block;white-space:nowrap;background:white;padding:4px 8px;'
                 f'border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:12px system-ui,sans-serif;color:{INK}">'
                 f'<b style="font-size:13px">{ROUTES[route_id]}</b><br>Routes: {routes}</div>')).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    swatch = '<div><span style="display:inline-block;width:18px;height:5px;background:{};margin-right:6px"></span>{}</div>'
    legend = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:10px 12px;'
        'border:1px solid #ccc;border-radius:6px;font:12px system-ui,sans-serif;color:#0b0b0b">'
        f"<b>B41 Flatbush Ave &amp; B25 Fulton St, {d['label']}</b>"
        '<div style="margin-top:6px"><b>Most common status</b></div>'
        + "".join(swatch.format(STATUS_COLORS[k], STATUS_LABELS[k]) for k in STATUS_COLORS)
        + '<div style="margin-top:6px"><b>Bunched</b> (toggle layer)</div>'
        + "".join(swatch.format(color, label) for _, color, label in BUNCH_BINS)
        + '<div style="margin-top:6px;color:#52514e">Status at most stops is estimated.<br>'
          "Click a stop for details.</div></div>")
    m.get_root().html.add_child(folium.Element(legend))
    return m


for d in data.values():
    m = make_map(d)
    map_path = os.path.abspath(os.path.join(OUT, f"{d['prefix']}_flatbush_fulton_map.html"))
    m.save(map_path)
    print("Map saved to", map_path)
    try:
        get_ipython()  # noqa: F821 -- interactive window / Jupyter: show the map below the cell
        display(m)  # noqa: F821
    except NameError:  # plain `python flatbush_vs_fulton.py`: open the saved map in the browser
        webbrowser.open("file:///" + map_path.replace("\\", "/"))
