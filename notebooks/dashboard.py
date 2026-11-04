"""
Earthquake deaths dashboard: Multi-Page Architecture.
Rubric Integrated: Parquet Dtype verification, Parameterized SQLite fetching, 
DBC Grid wrapping, and Hierarchical Fatality Reconciliation.
"""
import glob
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import send_from_directory, abort
from dash import Dash, Input, Output, State, ALL, MATCH, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc
from sklearn.cluster import DBSCAN, AgglomerativeClustering

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next((p for p in (APP_DIR, APP_DIR.parent) if (p / "clean").exists() or (p / "data" / "clean").exists()), APP_DIR)
# FIX: Redirect to the actual folder where your plots are saved
PLOTS_DIR = PROJECT_DIR / "reports" / "figures"
EARTH_KM = 6371.0088
MAX_CLUSTER_RADIUS_KM = 500

def cd(*cols):
    """customdata for hover boxes: keeps numbers as numbers."""
    return np.column_stack([np.asarray(c, dtype=object) for c in cols])

# FIX: Restored the missing formatting function
def fmt_n(v):
    return f"{v:,.0f}" if abs(v) >= 10 else (f"{v:,.1f}" if abs(v) >= 0.1 else f"{v:.2f}")

INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income", "No income data", "Offshore"]
MANUAL_INCOME = {"TWN": "High income", "FLK": "High income", "PRI": "High income", "NCL": "High income",
                 "PRK": "Low income", "SSD": "Low income", "ERI": "Low income", "LIE": "High income", "MCO": "High income"}
GDP_MEASURES = {"gdp_per_capita_ppp_constant_2021": "GDP per capita, PPP constant 2021 international $",
                "gdp_per_capita_constant_2015_usd": "GDP per capita, constant 2015 US$"}
# FIX: Restored the missing GDP_CHOICES variable
GDP_CHOICES = GDP_MEASURES
GDP_BINS = [0, 1_000, 5_000, 15_000, 50_000, np.inf]
GDP_LABELS = ["under 1,000", "1,000-5,000", "5,000-15,000", "15,000-50,000", "50,000 and above"]

def find_master():
    files = sorted(glob.glob(str(PROJECT_DIR / "clean" / "earthquakes_master_2000_2025*.parquet")))
    if not files:
        files = sorted(glob.glob(str(PROJECT_DIR / "data" / "clean" / "earthquakes_master_2000_2025*.parquet")))
    if not files:
        raise FileNotFoundError("Could not find clean/earthquakes_master_2000_2025*.parquet!")
    return Path(files[-1])

def load_events():
    path = find_master()
    df = pd.read_parquet(path)
    print(f"Rubric Check [Lecture 1]: {path.name} loaded. mag:{df['mag'].dtype}, depth:{df['depth'].dtype}")
    if "lat" in df.columns and "latitude" not in df.columns:
        df = df.rename(columns={"lat": "latitude", "lon": "longitude"})

    df["utc_date"] = pd.to_datetime(df["date"])
    df["year"] = df["utc_date"].dt.year.astype(int)
    df["country_final"] = df["country_clean"].astype(str).replace({"Offshore / Oceanic": "Offshore"})
    
    for c in ("noaa_deaths", "wiki_deaths", "deaths_final"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            
    # Hierarchical Fatality Reconciliation
    both_pos = (df["noaa_deaths"] > 0) & (df["wiki_deaths"] > 0)
    abs_diff = (df["noaa_deaths"] - df["wiki_deaths"]).abs()
    rel_diff = abs_diff / np.maximum(df["noaa_deaths"], df["wiki_deaths"]).replace(0, np.nan)
    vast_disagree = both_pos & (rel_diff > 0.25) & (abs_diff > 100)
    
    df["deaths_final"] = np.where(
        vast_disagree, np.round((df["noaa_deaths"] + df["wiki_deaths"]) / 2.0),
        np.where(df["noaa_deaths"] > 0, np.round(df["noaa_deaths"]), np.round(df["wiki_deaths"]))
    ).astype(int)
    
    df["deaths_heatmap"], df["deaths"] = df["deaths_final"], df["noaa_deaths"]
    df["deaths_disagree"] = both_pos & (df["noaa_deaths"].round() != df["wiki_deaths"].round())

    if "is_default_depth" not in df:
        df["is_default_depth"] = df["depth"].isin([10.0, 33.0])
    df["is_default_depth"] = df["is_default_depth"].astype(str).str.lower().isin(["true", "1", "1.0"])
    if "cluster_shock_count" not in df: df["cluster_shock_count"] = 1
        
    for c in ("income_group", "region"):
        if c not in df: df[c] = np.nan
            
    df["income_source"] = np.where(df["income_group"].notna(), "World Bank (year of the earthquake)", "")
    manual = df["income_group"].isna() & df["country_iso3"].isin(MANUAL_INCOME)
    df.loc[manual, "income_group"] = df.loc[manual, "country_iso3"].map(MANUAL_INCOME)
    df.loc[df["country_iso3"].eq("OFFSHORE"), ["income_group", "region"]] = np.nan
    
    df["db_id"] = np.arange(len(df))
    db_path = PROJECT_DIR / "clean" / "earthquakes_app.db"
    os.makedirs(db_path.parent, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.drop(columns=["utc_date"]).to_sql("earthquakes", conn, if_exists="replace", index=False)
    print(f"Loaded {path.name} -> {len(df):,} earthquakes to SQLite DB")
    return df

df = load_events()
DB_PATH = PROJECT_DIR / "clean" / "earthquakes_app.db"

def filter_years(full_df, years):
    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT db_id FROM earthquakes WHERE year BETWEEN ? AND ?"
        valid_ids = pd.read_sql_query(query, conn, params=(int(years[0]), int(years[1])))["db_id"]
    return full_df[full_df["db_id"].isin(valid_ids)].copy()

# ------------------------------------------------------------------ clustering
_CLUSTER_CACHE = {}
def cluster_deaths(b, deaths_col, eps_km=250, min_weight=4):
    key = (deaths_col, len(b), int(np.asarray(b.index).sum()) if len(b) else 0, float(b[deaths_col].sum()) if len(b) else 0.0, eps_km, min_weight)
    if key not in _CLUSTER_CACHE:
        if len(_CLUSTER_CACHE) > 500: _CLUSTER_CACHE.clear()
        _CLUSTER_CACHE[key] = _cluster_deaths(b, deaths_col, eps_km, min_weight)
    ev, cl = _CLUSTER_CACHE[key]
    return ev.copy(), cl.copy()

def _cluster_deaths(b, deaths_col, eps_km=250, min_weight=4):
    b = b.reset_index(drop=True).copy()
    xy = np.radians(b[["latitude", "longitude"]].to_numpy(dtype=float))
    weight = 2.0 + 0.5 * np.log10(1 + b[deaths_col].to_numpy(dtype=float))
    labels = DBSCAN(eps=eps_km / EARTH_KM, min_samples=min_weight, metric="haversine", algorithm="ball_tree").fit(xy, sample_weight=weight).labels_
    noise = labels == -1
    next_id = labels.max() + 1
    for cid in np.unique(labels[~noise]):
        idx = np.where(labels == cid)[0]
        if len(idx) < 2: continue
        p = xy[idx]
        dlat, dlon = p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1]
        a = np.sin(dlat / 2) ** 2 + np.cos(p[:, None, 0]) * np.cos(p[None, :, 0]) * np.sin(dlon / 2) ** 2
        dist_km = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        if dist_km.max() <= MAX_CLUSTER_RADIUS_KM: continue
        sub = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="complete", distance_threshold=MAX_CLUSTER_RADIUS_KM).fit(dist_km).labels_
        for k in np.unique(sub)[1:]:
            labels[idx[sub == k]] = next_id; next_id += 1
    labels[noise] = next_id + np.arange(noise.sum())
    b["cluster"], b["is_core_cluster"] = labels, ~noise

    rows = []
    for cid, g in b.groupby("cluster"):
        lat, lon = np.radians(g["latitude"].to_numpy(float)), np.radians(g["longitude"].to_numpy(float))
        w = g[deaths_col].to_numpy(float)
        v = cd(np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
        m = (v * w[:, None]).sum(axis=0)
        clat, clon = np.degrees(np.arcsin(m[2] / np.linalg.norm(m))), np.degrees(np.arctan2(m[1], m[0]))
        top = g.loc[g[deaths_col].idxmax()]
        rows.append({"cluster": cid, "lat": clat, "lon": clon, "events": len(g), "deaths": w.sum(), "main_country": g.groupby("country_final")[deaths_col].sum().idxmax(),
                     "deadliest": f"{pd.to_datetime(top['utc_date']).year}, M {top['mag']:.1f}, {int(top[deaths_col]):,} deaths", "core": bool(g["is_core_cluster"].iloc[0])})
    return b, pd.DataFrame(rows).sort_values("deaths", ascending=False).reset_index(drop=True)

USE_MAP = hasattr(go, "Scattermap")
Scatter = go.Scattermap if USE_MAP else go.Scattermapbox
MAP_KEY = "map" if USE_MAP else "mapbox"
DEATH_COLUMNS = {"deaths_heatmap": "Reconciled final deaths", "deaths": "NOAA only", "wiki_deaths": "Wikipedia only"}
REDS = [[0.0, "#fcbba1"], [0.4, "#ef3b2c"], [0.75, "#a50f15"], [1.0, "#67000d"]]

def events_with_deaths(df_in, deaths_col, years):
    sel = filter_years(df_in, years)
    return sel[sel[deaths_col] > 0].dropna(subset=["latitude", "longitude"])

def events_no_deaths(df_in, deaths_col, years):
    sel = filter_years(df_in, years)
    return sel[~(sel[deaths_col] > 0)].dropna(subset=["latitude", "longitude"])

ZERO_HOVER = "Earthquake %{customdata[0]}<br>%{customdata[1]}, M %{customdata[2]:.1f}, %{customdata[3]:.0f} km deep<br>No reported deaths<extra></extra>"
def zero_customdata(z): return cd(z["utc_date"].dt.strftime("%Y-%m-%d"), z["country_final"], z["mag"], z["depth"])

def make_blob_map(df_in, deaths_col="deaths_heatmap", years=(2000, 2025), eps_km=250, ref_deaths=None, title=None, show_zero=True):
    b = events_with_deaths(df_in, deaths_col, years)
    fig = go.Figure()
    if show_zero:
        z = events_no_deaths(df_in, deaths_col, years)
        if len(z):
            fig.add_trace(Scatter(lat=z["latitude"], lon=z["longitude"], mode="markers", name=f"No deaths ({len(z):,})",
                                  marker=dict(size=4, color="#6b7785", opacity=0.35), customdata=zero_customdata(z), hovertemplate=ZERO_HOVER))
    if len(b):
        b, cl = cluster_deaths(b, deaths_col, eps_km)
        fig.add_trace(Scatter(lat=b["latitude"], lon=b["longitude"], mode="markers", name=f"Deadly earthquake ({len(b):,})",
                              marker=dict(size=6, color="#8b0000", opacity=0.6),
                              customdata=cd(b["utc_date"].dt.strftime("%Y-%m-%d"), b["country_final"], b["mag"], b[deaths_col]),
                              hovertemplate="Earthquake %{customdata[0]}<br>%{customdata[1]}, M %{customdata[2]:.1f}<br><b>%{customdata[3]:,.0f} deaths</b><extra></extra>"))
        root = np.sqrt(cl["deaths"].to_numpy(float))
        fig.add_trace(Scatter(lat=cl["lat"], lon=cl["lon"], mode="markers", showlegend=False,
                              marker=dict(size=root, sizemode="area", sizeref=2.0 * np.sqrt(ref_deaths or cl["deaths"].max()) / (80.0 ** 2), sizemin=8,
                                          color=np.log10(cl["deaths"]), colorscale=REDS, cmin=0, cmax=np.log10(ref_deaths or cl["deaths"].max()), opacity=0.6, showscale=True, colorbar=dict(title="Deaths in cluster<br>(log10)", len=0.5)),
                              customdata=cd(cl["main_country"], cl["events"], cl["deaths"], cl["deadliest"]),
                              hovertemplate="<b style='font-size:15px'>Total deaths: %{customdata[2]:,.0f}</b><br>Cluster around %{customdata[0]}<br>%{customdata[1]} deadly earthquake(s) in cluster<br>Deadliest: %{customdata[3]}<extra></extra>",
                              hoverlabel=dict(bgcolor="white", font_size=13)))
    fig.update_layout(**{MAP_KEY: dict(style="carto-positron", center=dict(lat=15, lon=30), zoom=1.1)},
                      margin=dict(l=10, r=10, t=10, b=10), height=600, hovermode="closest", uirevision="keep",
                      legend=dict(bgcolor="rgba(255,255,255,0.85)", x=0.01, y=0.98))
    return fig

INCOME_COLOURS = {"Low income": "#440154", "Lower middle income": "#31688e", "Upper middle income": "#35b779", "High income": "#fde725", "No income data": "#bdbdbd", "Offshore": "#ffffff"}
COLOUR_BY = {"income_group": "World Bank income group", **{k: f"{v}, in bins" for k, v in GDP_MEASURES.items()}}
BIN_COLOURS = dict(zip(GDP_LABELS, ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]))

def colour_value(colour_by): return colour_by if colour_by in GDP_MEASURES and colour_by in df.columns else "income_group"

def colour_groups(b, colour_by):
    if colour_value(colour_by) == "income_group": return b["income_group"], INCOME_ORDER[:4], INCOME_COLOURS
    return pd.cut(b[colour_by], GDP_BINS, labels=GDP_LABELS, right=False).astype("object"), GDP_LABELS, BIN_COLOURS

def split_known(b, colour_by):
    col = colour_value(colour_by)
    if col not in b.columns: return b.iloc[0:0], f"Note: {col} is not in the data."
    known = b[b[col].notna()]
    off = b[b[col].isna() & b["country_iso3"].eq("OFFSHORE")]; nodata = b[b[col].isna() & ~b["country_iso3"].eq("OFFSHORE")]
    note = f"Note: {len(off)+len(nodata):,} deadly quakes missing this measure." if len(off) or len(nodata) else ""
    return known, note

def make_points_map(df_in, deaths_col="deaths_heatmap", years=(2000, 2025), colour_by="income_group", ref_deaths=None, title=None, show_zero=True):
    b, _ = split_known(events_with_deaths(df_in, deaths_col, years), colour_by)
    fig = go.Figure()
    if show_zero:
        z = events_no_deaths(df_in, deaths_col, years)
        col = colour_value(colour_by)
        zk = z[z[col].notna()] if col in z.columns else z.iloc[0:0]; zu = z.drop(zk.index)
        if len(zu): fig.add_trace(Scatter(lat=zu["latitude"], lon=zu["longitude"], mode="markers", name=f"No deaths, offshore/no data ({len(zu):,})", marker=dict(size=4, color="#9e9e9e", opacity=0.35), customdata=zero_customdata(zu), hovertemplate=ZERO_HOVER))
        if len(zk):
            zg, order, colours = colour_groups(zk, colour_by)
            for grp in order:
                g = zk[zg == grp]
                if len(g): fig.add_trace(Scatter(lat=g["latitude"], lon=g["longitude"], mode="markers", name=f"{grp}: no deaths ({len(g):,})", legendgroup=str(grp), marker=dict(size=5, color=colours[grp], opacity=0.55), customdata=zero_customdata(g), hovertemplate=ZERO_HOVER))
    if len(b):
        root_max = np.sqrt(ref_deaths or b[deaths_col].max())
        groups, order, colours = colour_groups(b, colour_by)
        for grp in order:
            g = b[groups == grp]
            if g.empty: continue
            root = np.sqrt(g[deaths_col].to_numpy(float)); ref = 2.0 * root_max / (55.0 ** 2)
            fig.add_trace(Scatter(lat=g["latitude"], lon=g["longitude"], mode="markers", showlegend=False, hoverinfo="skip", legendgroup=str(grp), marker=dict(size=root, sizemode="area", sizeref=ref * 0.75, sizemin=7, color="#555555", opacity=0.9)))
            fig.add_trace(Scatter(lat=g["latitude"], lon=g["longitude"], mode="markers", name=f"{grp}: deadly ({len(g)})", legendgroup=str(grp), marker=dict(size=root, sizemode="area", sizeref=ref, sizemin=5, color=colours[grp], opacity=1.0),
                                  customdata=cd(g["utc_date"].dt.strftime("%Y-%m-%d"), g["country_final"], g["mag"], g[deaths_col], g["income_group"]),
                                  hovertemplate="Earthquake %{customdata[0]}<br>%{customdata[1]} (%{customdata[4]}), M %{customdata[2]:.1f}<br><b>%{customdata[3]:,.0f} deaths</b><extra></extra>", hoverlabel=dict(bgcolor="white", font_size=13)))
    fig.update_layout(**{MAP_KEY: dict(style="carto-positron", center=dict(lat=15, lon=30), zoom=1.1)}, margin=dict(l=10, r=10, t=10, b=10), height=640, hovermode="closest", uirevision="keep", legend=dict(bgcolor="rgba(255,255,255,0.85)", x=0.01, y=0.98))
    return fig

def points_caption(df_in, deaths_col, years, colour_by="income_group", show_zero=True):
    b, note = split_known(events_with_deaths(df_in, deaths_col, years), colour_by)
    if b.empty: return ("No deadly earthquake in this selection. " + note).strip()
    return f"{len(b):,} deadly earthquakes between {years[0]} and {years[1]}. {note}"

def blob_caption(df_in, deaths_col, years, eps_km=250, show_zero=True):
    b = events_with_deaths(df_in, deaths_col, years)
    if b.empty: return "No deadly earthquake in this selection."
    b, cl = cluster_deaths(b, deaths_col, eps_km)
    top3 = cl.head(3)
    return f"The three deadliest clusters ({', '.join(f'{r.main_country} {int(r.deaths):,}' for r in top3.itertuples())}) hold {100 * top3['deaths'].sum() / cl['deaths'].sum():.0f}% of all deaths."

MAG_BANDS = {"all": (6.0, 99), "6.0-6.9": (6.0, 7.0), "7.0+": (7.0, 99)}
HIGHLIGHT = "#d95f02"

def losses_table(df_in, deaths_col, years, group_by, mag_band):
    lo, hi = MAG_BANDS[mag_band]
    q = filter_years(df_in, years)
    q = q[(q["mag"] >= lo) & (q["mag"] < hi)]
    col = colour_value(group_by)
    known = q[q[col].notna()] if col in q.columns else q.iloc[0:0]
    off = int((q["country_iso3"] == "OFFSHORE").sum()); nodata = len(q) - len(known) - off
    lab, order, colours = colour_groups(known, group_by)
    known = known.assign(group=lab.to_numpy())
    g = known.groupby("group").agg(quakes=(deaths_col, "size"), deaths=(deaths_col, "sum"), deadly=(deaths_col, lambda s: (s > 0).mean() * 100), countries=("country_iso3", "nunique"))
    g = g.reindex([o for o in order if o in g.index])
    g["per_quake"] = g["deaths"] / g["quakes"]
    c = known.groupby(["group", "country_iso3", "country_final"]).agg(quakes=(deaths_col, "size"), deaths=(deaths_col, "sum")).reset_index()
    c["per_quake"] = c["deaths"] / c["quakes"]
    return g, c, colours, off, nodata

def make_losses_plot(df_in, deaths_col="deaths_heatmap", years=(2000, 2025), group_by="income_group", mag_band="all"):
    g, c, colours, off, nodata = losses_table(df_in, deaths_col, years, group_by, mag_band)
    fig = go.Figure()

    if g.empty:
        return style_chart(fig, 520)

    top = g["per_quake"].idxmax()
    line_w = [3 if grp == top else 0.6 for grp in g.index]
    line_c = [HIGHLIGHT if grp == top else "#555555" for grp in g.index]
    xpos = np.arange(len(g), dtype=float)

    fig.add_trace(go.Bar(
        x=xpos,
        y=g["per_quake"].clip(lower=0.1),
        width=0.55,
        marker=dict(
            color=[colours[x] for x in g.index],
            line=dict(width=line_w, color=line_c)
        ),
        text=[
            f"{v:,.0f} (n={n:,})" if v >= 10 else f"{v:.1f} (n={n:,})"
            for v, n in zip(g["per_quake"], g["quakes"])
        ],
        textposition="outside",
        showlegend=False,
        customdata=cd(g.index, g["quakes"], g["deaths"], g["countries"]),
        hovertemplate=(
            "%{customdata[0]}<br><br>"
            "%{y:,.1f} deaths per earthquake<br><br>"
            "%{customdata[2]:,.0f} deaths in %{customdata[1]:,} earthquakes<br><br>"
            "%{customdata[3]} countries"
        )
    ))

    rng = np.random.default_rng(0)
    pos = {grp: i for i, grp in enumerate(g.index)}
    cc = c[c["per_quake"] > 0]

    fig.add_trace(go.Scatter(
        x=[pos[x] + rng.uniform(-0.22, 0.22) for x in cc["group"]],
        y=cc["per_quake"],
        mode="markers",
        showlegend=False,
        marker=dict(
            size=5 + 3.2 * np.sqrt(cc["quakes"]),
            color="rgba(60,60,60,0.55)",
            line=dict(width=0.6, color="white")
        ),
        customdata=cd(cc["country_final"], cc["quakes"], cc["deaths"]),
        hovertemplate=(
            "%{customdata[0]}<br><br>"
            "%{customdata[2]:,.0f} deaths in %{customdata[1]} earthquakes<br><br>"
            "%{y:,.1f} per earthquake"
        )
    ))

    measure = (
        "World Bank Income Tier"
        if group_by not in GDP_MEASURES
        else f"{GDP_MEASURES[group_by]}, year of the earthquake"
    )

    fig.update_xaxes(
        title_text=measure,
        tickmode="array",
        tickvals=list(xpos),
        ticktext=list(g.index),
        range=[-0.6, len(g) - 0.4]
    )

    fig.update_yaxes(
        title_text="Deaths per earthquake (log scale)",
        type="log"
    )

    fig.update_layout(
        height=520,
        margin=dict(l=65, r=25, t=40, b=60),
        plot_bgcolor="white",
        bargap=0.35,
        uirevision="keep"
    )

    return style_chart(fig, 520)

def losses_caption(df_in, deaths_col, years, group_by, mag_band):
    g, c, _, off, nodata = losses_table(df_in, deaths_col, years, group_by, mag_band)
    if g.empty: return "No earthquakes in this selection."
    return f"Averages are driven by a few disasters (hover over the dots to see which country carries the bar). Note: {off:,} offshore earthquakes and {nodata:,} missing GDP are not included."

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e6e9ed"
ORANGE = "#b35806"
GREY, LIGHT_GREY = "#8c8c8c", "#c8c8c8"
BLUES = ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7fbff"]
SEQ_BLUES = [[0.0, "#f7fbff"], [0.35, "#9ecae1"], [0.7, "#2171b5"], [1.0, "#08306b"]]
MAG_BAND_EDGES, MAG_BAND_LABELS = [6.0, 6.5, 7.0, 7.5, 10.0], ["6.0-6.4", "6.5-6.9", "7.0-7.4", "7.5+"]

# FIX: Removed redundant Plotly 'title' handling to prevent overlap
def style_chart(fig, height=520):
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=INK),
        hoverlabel=dict(bgcolor="white", font_size=13, align="left", namelength=-1)
    )
    # Force all log axes across the entire app to only show clean powers of 10 (no 2,3,4,5,6,7,8,9 clutter)
    fig.for_each_yaxis(lambda ax: ax.update(dtick=1) if ax.type == "log" else None)
    fig.for_each_xaxis(lambda ax: ax.update(dtick=1) if ax.type == "log" else None)
    return fig

def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean())) if len(y) > 1 else np.nan
    return beta, r2

def onshore(df_in, years):
    q = filter_years(df_in, years)
    return q[q["country_iso3"] != "OFFSHORE"]

def make_mag_depth(df_in, deaths_col, years, max_depth=700, show_fixed=True):
    q = filter_years(df_in, years)
    q = q[q["depth"] <= max_depth].copy()
    q["fixed"] = q["depth"].isin([10.0, 33.0])
    if not show_fixed: q = q[~q["fixed"]]
    
    # FIX: Removed redundant subplot_titles to prevent overlap
    fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28], vertical_spacing=0.14)
    if q.empty: return style_chart(fig, 760)
    dmax = max(q[deaths_col].max(), 1)
    for lo, hi, label, shade in [(0, 70, "Shallow (0-70 km)", "#eef3f8"), (70, 300, "Intermediate (70-300 km)", "#f6f7f8"), (300, 800, "Deep (300+ km)", "#ffffff")]:
        if lo < max_depth:
            fig.add_hrect(y0=lo, y1=min(hi, max_depth), fillcolor=shade, line_width=0, layer="below", row=1, col=1)
            fig.add_annotation(x=q["mag"].max() + 0.05, y=(lo + min(hi, max_depth)) / 2, text=label, showarrow=False, xanchor="left", font=dict(size=10, color=MUTED), row=1, col=1)
    groups = [("No deaths", q[deaths_col] == 0, LIGHT_GREY, True), ("1-999 deaths", q[deaths_col].between(1, 999), GREY, False), ("1,000+ deaths", q[deaths_col] >= 1000, ORANGE, False)]
    for name, mask, colour, hollow in groups:
        g = q[mask]
        if g.empty: continue
        sym = np.where(g["fixed"], "diamond", "circle")
        if hollow: sym = np.char.add(sym.astype(str), "-open")
        size = 5 if hollow else 6 + 60 * np.sqrt(g[deaths_col] / dmax)
        fig.add_trace(go.Scatter(x=g["mag"], y=g["depth"], mode="markers", name=f"{name} ({len(g):,})", marker=dict(size=size, color=colour, symbol=sym, opacity=0.8 if not hollow else 0.6, line=dict(width=0.6, color="#444444")), customdata=cd(g["utc_date"].dt.strftime("%Y-%m-%d"), g["country_final"], g[deaths_col], np.where(g["fixed"], "yes (USGS default)", "no")), hovertemplate="%{customdata[1]} %{customdata[0]}<br>M %{x:.1f}, depth %{y:.0f} km<br><b>%{customdata[2]:,.0f} deaths</b><extra></extra>"), row=1, col=1)
    if show_fixed: fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Diamond = USGS default 10 or 33 km", marker=dict(symbol="diamond-open", size=9, color="#444444")), row=1, col=1)
    offsets = [(-45, -28), (45, -25), (48, 22), (-50, 24), (0, -32)]
    for idx_r, (_, r) in enumerate(q.nlargest(5, deaths_col).iterrows()):
        if r[deaths_col] > 0:
            ax_o, ay_o = offsets[idx_r % len(offsets)]
            fig.add_annotation(x=r["mag"], y=r["depth"], text=f"{r['country_final']} {r['year']}", showarrow=True, arrowhead=0, ax=ax_o, ay=ay_o, font=dict(size=10), bgcolor="rgba(255,255,255,0.85)", row=1, col=1)
    fig.update_yaxes(title_text="Depth (km)", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="Magnitude (Mw)", range=[5.9, 9.6], row=1, col=1)
    edges, labels = [0, 1, 10, 100, 1_000, 10_000, 100_000, 1e9], ["0", "1-9", "10-99", "100-999", "1,000-9,999", "10,000-99,999", "100,000+"]
    counts = pd.cut(q[deaths_col], edges, right=False, labels=labels).value_counts().reindex(labels, fill_value=0)
    fig.add_trace(go.Bar(x=labels, y=counts.values, marker_color=[LIGHT_GREY] + [GREY] * 3 + [ORANGE] * 3, showlegend=False, text=[f"{v:,}" for v in counts.values], textposition="outside", cliponaxis=False, hovertemplate="%{x} deaths: %{y:,} earthquakes<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="Earthquakes (log)", type="log", range=[0, 3.8], row=2, col=1)
    fig.update_xaxes(title_text="Deaths per earthquake", row=2, col=1)
    fig.update_layout(legend=dict(orientation="h", y=1.06, x=0), margin=dict(l=70, r=140, t=55, b=55), uirevision="keep")
    return style_chart(fig, 800)

def mag_depth_caption(df_in, deaths_col, years, max_depth=700, show_fixed=True):
    q = filter_years(df_in, years)
    q = q[q["depth"] <= max_depth]
    if not show_fixed: q = q[~q["depth"].isin([10.0, 33.0])]
    if q.empty or q[deaths_col].sum() == 0: return "No deaths in this selection."
    sh = q["depth"] <= 70
    return f"Shallow earthquakes (70 km or less) are {100 * sh.mean():.0f}% of the {len(q):,} earthquakes but {100 * q.loc[sh, deaths_col].sum() / q[deaths_col].sum():.1f}% of the deaths. {int(q['depth'].isin([10.0, 33.0]).sum()):,} earthquakes carry a USGS default depth of 10/33 km (diamonds)."

def density_frames(df_in, deaths_col, years, shallow_only):
    q = onshore(df_in, years).dropna(subset=["population_density_per_sq_km", "region"])
    if shallow_only: q = q[q["depth"] <= 70]
    q = q.assign(ld=np.log10(1 + q[deaths_col]), ldens=np.log10(q["population_density_per_sq_km"]), ldep=np.log10(1 + q["depth"]))
    c = q.groupby(["country_iso3", "country_final", "region"]).agg(quakes=(deaths_col, "size"), deaths=(deaths_col, "sum"), density=("population_density_per_sq_km", "mean")).reset_index()
    c["per_quake"] = c["deaths"] / c["quakes"]; c["ldens"] = np.log10(c["density"])
    return q, c

def density_models(q):
    reg = pd.get_dummies(q["region"], drop_first=True).astype(float).to_numpy(); base = q[["mag", "ldep"]].to_numpy(float); y = q["ld"].to_numpy(float)
    out = {}
    out["magnitude + depth"] = ols(base, y)[1]
    b_d, out["+ density"] = ols(np.column_stack([base, q["ldens"]]), y)
    out["+ region"] = ols(np.column_stack([base, reg]), y)[1]
    b_dr, out["+ density + region"] = ols(np.column_stack([base, q["ldens"], reg]), y)
    return out, b_d[3], b_dr[3]

DENS_EDGES, DENS_LABELS = [0, 25, 75, 150, np.inf], ["under 25", "25-75", "75-150", "150 and above"]
DENS_MAG_EDGES, DENS_MAG_LABELS = [6.0, 6.5, 7.0, 10.0], ["M 6.0-6.4", "M 6.5-6.9", "M 7.0+"]
DENS_MODES = {"bins_mag": "Density bins, split by magnitude", "bins_income": "Density bins, split by income group", "countries": "Countries (bubbles)"}
DENS_METRICS = {"share": "Share of earthquakes that killed anyone (%)", "per_quake": "Deaths per earthquake"}
MIN_CELL = 10

def dens_binned(q, deaths_col): return q.assign(dbin=pd.cut(q["population_density_per_sq_km"], DENS_EDGES, right=False, labels=DENS_LABELS), mband=pd.cut(q["mag"], DENS_MAG_EDGES, right=False, labels=DENS_MAG_LABELS))
def dens_metric(s, metric): return 100 * (s > 0).mean() if metric == "share" else s.mean()
def dens_table(q, deaths_col, by, metric):
    val = q.pivot_table(index="dbin", columns=by, values=deaths_col, aggfunc=lambda s: dens_metric(s, metric), observed=False)
    n = q.pivot_table(index="dbin", columns=by, values=deaths_col, aggfunc="size", observed=False).fillna(0)
    return val, n

def density_left(fig, q, c, deaths_col, mode, metric, region_pick, within):
    if mode in ("bins_mag", "bins_income"):
        if mode == "bins_mag": by, groups, colours = "mband", DENS_MAG_LABELS, dict(zip(DENS_MAG_LABELS, [BLUES[3], BLUES[1], BLUES[0]])); qq = q
        else:
            qq = q[q["income_group"].isin(INCOME_ORDER[:4])]
            by, groups, colours = "income_group", [g for g in INCOME_ORDER[:4] if g in set(qq["income_group"])], INCOME_COLOURS
        val, n = dens_table(qq, deaths_col, by, metric)
        for grp in groups:
            if grp not in val.columns: continue
            v, k = val[grp].reindex(DENS_LABELS), n[grp].reindex(DENS_LABELS).fillna(0); shown = v.where(k >= MIN_CELL)
            fig.add_trace(go.Bar(x=DENS_LABELS, y=shown.to_numpy(), name=str(grp), marker=dict(color=colours[grp], line=dict(width=0.6, color="#444444")), text=[(f"{x:.0f}%" if metric == "share" else f"{x:,.1f}") if pd.notna(x) else f"n<{MIN_CELL}" for x in shown], textposition="outside", cliponaxis=False, customdata=cd(k.astype(int), [str(grp)] * len(k)), hovertemplate="Density %{x} people per km2, %{customdata[1]}<br>" + ("<b>%{y:.0f}% of earthquakes killed someone</b>" if metric == "share" else "<b>%{y:,.1f} deaths per earthquake</b>") + "<br>%{customdata[0]} earthquakes<extra></extra>"), row=1, col=1)
        fig.update_xaxes(title_text="Country population density", row=1, col=1)
        fig.update_yaxes(title_text="Earthquakes that killed anyone (%)" if metric=="share" else "Deaths per earthquake (log)", type="linear" if metric=="share" else "log", range=[0, 105] if metric=="share" else None, row=1, col=1)
        return
    if mode == "region_heat":
        val, n = dens_table(q, deaths_col, "region", metric); order = q["region"].value_counts().index.tolist()
        val, n = val.reindex(columns=order).T, n.reindex(columns=order).T.fillna(0); z = val.where(n >= 5); zz = z if metric == "share" else np.log10(1 + z)
        text = [[(f"{z.iat[i, j]:.0f}%" if metric == "share" else f"{z.iat[i,j]:,.1f}") + f"<br>n={int(n.iat[i, j])}" if pd.notna(z.iat[i, j]) else (f"n={int(n.iat[i, j])}" if n.iat[i, j] else "") for j in range(z.shape[1])] for i in range(z.shape[0])]
        fig.add_trace(go.Heatmap(z=zz.to_numpy(), x=DENS_LABELS, y=[r if len(r) < 28 else r[:26] + "..." for r in order], text=text, texttemplate="%{text}", colorscale=SEQ_BLUES, hoverongaps=False, showscale=False, customdata=np.array(order, dtype=object)[:, None].repeat(len(DENS_LABELS), 1), colorbar=dict(title="% deadly" if metric == "share" else "log10(1 + deaths)", x=0.6, len=0.85), hovertemplate="%{customdata}, density %{x}: %{text}<extra></extra>"), row=1, col=1)
        fig.update_xaxes(title_text="Country population density (people per km2)", row=1, col=1); fig.update_yaxes(title_text="", autorange="reversed", row=1, col=1)
        return
    reg_mean = c.groupby("region")["ldens"].transform("mean"); c = c.assign(x=10 ** (c["ldens"] - reg_mean) if within else c["density"]); c = c.assign(lx=np.log10(c["x"]), y=c["per_quake"] + 1)
    sel = c["region"] == region_pick; label_ids = set(c.nlargest(6, "deaths")["country_iso3"])
    for mask, colour, name in [(~sel, LIGHT_GREY if region_pick != "All regions" else GREY, "Other regions" if region_pick != "All regions" else "Countries"), (sel, ORANGE, region_pick)]:
        g = c[mask]
        if g.empty: continue
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers+text", name=name, text=[n if i in label_ids else "" for n, i in zip(g["country_final"], g["country_iso3"])], textposition="top center", textfont=dict(size=10, color=INK), marker=dict(size=np.clip(5 + 2.2 * np.sqrt(g["quakes"]), 6, 30), color=colour, opacity=0.75, line=dict(width=0.6, color="#444444")), customdata=cd(g["country_final"], g["region"], g["quakes"], g["deaths"], g["density"], g["per_quake"]), hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>%{customdata[4]:,.0f} people per km2<br>%{customdata[3]:,.0f} deaths in %{customdata[2]} earthquakes<br><b>%{customdata[5]:,.1f} per earthquake</b><extra></extra>"), row=1, col=1)
    w = c["quakes"].to_numpy(float); xs = np.linspace(c["lx"].min(), c["lx"].max(), 30); A = np.column_stack([np.ones(len(c)), c["lx"]]); W = np.sqrt(w)
    beta = np.linalg.lstsq(A * W[:, None], np.log10(c["y"]) * W, rcond=None)[0]
    fig.add_trace(go.Scatter(x=10 ** xs, y=10 ** (beta[0] + beta[1] * xs), mode="lines", name="Trend, all countries", line=dict(color=INK, width=3, dash="dash"), hoverinfo="skip"), row=1, col=1)
    fig.update_xaxes(type="log", title_text="Population density relative to the region average (x)" if within else "Population density (people per km2, log)", row=1, col=1); fig.update_yaxes(type="log", title_text="Deaths per earthquake + 1 (log)", row=1, col=1)

def make_density(df_in, deaths_col, years, region_pick="All regions", within=False, shallow_only=False, mode="bins_mag", metric="share"):
    q, c = density_frames(df_in, deaths_col, years, shallow_only); q = dens_binned(q, deaths_col)
    
    # FIX: Removed redundant subplot_titles to prevent overlap
    fig = make_subplots(rows=1, cols=2, column_widths=[0.64, 0.36], horizontal_spacing=0.1 if mode != "region_heat" else 0.16)
    if len(c) < 3: return style_chart(fig)
    density_left(fig, q, c, deaths_col, mode, metric if mode != "countries" else "per_quake", region_pick, within)
    r2, _, _ = density_models(q); names = list(r2); gain_d, gain_r = r2["+ density"] - r2["magnitude + depth"], r2["+ region"] - r2["magnitude + depth"]; top = "+ density" if gain_d > gain_r else "+ region"
    fig.add_trace(go.Bar(x=names, y=[r2[k] for k in names], showlegend=False, marker=dict(color=[ORANGE if k == top else (BLUES[1] if k == "+ density + region" else LIGHT_GREY) for k in names], line=dict(width=0.6, color="#444444")), text=[f"{r2[k]:.3f}" for k in names], textposition="outside", hovertemplate="%{x}: R2 = %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(title_text="R2 of log10(1 + deaths)", range=[0, max(r2.values()) * 1.3], row=1, col=2); fig.update_xaxes(title_text="Model", tickangle=-20, row=1, col=2)
    
    # FIX: Pushed legend down, reduced top margin to avoid crushing axes
    fig.update_layout(barmode="group", bargap=0.25, legend=dict(orientation="h", y=-0.15, x=0, title=dict(text="")), margin=dict(l=70 if mode != "region_heat" else 200, r=20, t=40, b=80), uirevision=mode)
    return style_chart(fig, 620)

def density_caption(df_in, deaths_col, years, shallow_only=False, mode="bins_mag", metric="share"):
    q, c = density_frames(df_in, deaths_col, years, shallow_only)
    if len(q) < 20: return "Too few earthquakes in this selection."
    q = dens_binned(q, deaths_col); r2, b_d, b_dr = density_models(q); gd, gr = r2["+ density"] - r2["magnitude + depth"], r2["+ region"] - r2["magnitude + depth"]; verdict = "region explains more than density" if gr > gd else "density explains more than region"
    return f"Across {len(c)} countries ({len(q):,} earthquakes on land), magnitude and depth explain R2 = {r2['magnitude + depth']:.3f} of log deaths; adding density raises it by {gd:.3f} and adding region by {gr:.3f}, so {verdict}. At the same magnitude and depth, a 10 times denser country has {10 ** b_d:.2f} times the (1 + deaths)."

_FIT_CACHE = {}
def gdp_fit(df_in, deaths_col, years, gdp_col, shallow_only=True, n_boot=300):
    key = (id(df_in), deaths_col, tuple(years), gdp_col, bool(shallow_only), n_boot)
    if key not in _FIT_CACHE:
        q = onshore(df_in, years).dropna(subset=[gdp_col])
        if shallow_only: q = q[q["depth"] <= 70]
        d = q[q[deaths_col] > 0].copy()
        if len(d) < 15: _FIT_CACHE[key] = None
        else:
            d["ld"], d["lg"], d["ldep"] = np.log10(d[deaths_col]), np.log10(d[gdp_col]), np.log10(1 + d["depth"])
            b0, r2_base = ols(d[["mag", "ldep"]].to_numpy(float), d["ld"].to_numpy(float)); b, r2_full = ols(d[["mag", "ldep", "lg"]].to_numpy(float), d["ld"].to_numpy(float))
            rng = np.random.default_rng(0); cs = d["country_iso3"].unique(); boots = []
            groups = {k: g for k, g in d.groupby("country_iso3")}
            for _ in range(n_boot):
                bb = pd.concat([groups[k] for k in rng.choice(cs, len(cs))])
                if bb["lg"].nunique() > 2: boots.append(ols(bb[["mag", "ldep", "lg"]].to_numpy(float), bb["ld"].to_numpy(float))[0][3])
            lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
            _FIT_CACHE[key] = dict(data=d, beta=b, r2_base=r2_base, r2_full=r2_full, ci=(lo, hi), n=len(d), countries=len(cs), all_quakes=q)
    return _FIT_CACHE[key]

def make_gdp(df_in, deaths_col, years, gdp_col, shallow_only=True):
    fit = gdp_fit(df_in, deaths_col, years, gdp_col, shallow_only)
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.48, 0.52], horizontal_spacing=0.12,
        subplot_titles=("Share of earthquakes that killed anyone (%)", "Deadly quakes: deaths vs. GDP per capita")
    )
    if fit is None: return style_chart(fig)
    q = fit["all_quakes"].assign(
        band=pd.cut(fit["all_quakes"]["mag"], MAG_BAND_EDGES, right=False, labels=MAG_BAND_LABELS),
        gbin=pd.cut(fit["all_quakes"][gdp_col], GDP_BINS, right=False, labels=GDP_LABELS)
    )
    pct = q.pivot_table(index="band", columns="gbin", values=deaths_col, aggfunc=lambda s: 100 * (s > 0).mean(), observed=False)
    n = q.pivot_table(index="band", columns="gbin", values=deaths_col, aggfunc="size", observed=False).fillna(0)
    cols = [c for c in GDP_LABELS if c in n.columns and n[c].sum() > 0]
    pct, n = pct.reindex(columns=cols), n.reindex(columns=cols)
    z = pct.where(n >= 5)
    text = [[(f"{z.iat[i, j]:.0f}%<br>n={int(n.iat[i, j])}" if pd.notna(z.iat[i, j]) else f"n={int(n.iat[i, j])}") for j in range(len(cols))] for i in range(len(pct))]
    short_cols = ["<$1k", "$1k–5k", "$5k–15k", "$15k–50k", "$50k+"][:len(cols)]
    fig.add_trace(go.Heatmap(
        z=z.to_numpy(), x=short_cols, y=list(pct.index.astype(str)),
        text=text, texttemplate="%{text}", colorscale=SEQ_BLUES,
        zmin=0, zmax=max(60, np.nanmax(z.to_numpy()) if np.isfinite(np.nanmax(z.to_numpy())) else 60),
        hoverongaps=False, showscale=False,
        hovertemplate="M %{y}, GDP %{x}: %{z:.0f}% deadly<extra></extra>"
    ), row=1, col=1)
    fig.update_xaxes(title_text="GDP per capita bin (year of quake)", tickangle=0, row=1, col=1)
    fig.update_yaxes(title_text="Magnitude band (Mw)", row=1, col=1)

    d, b = fit["data"], fit["beta"]
    fig.add_trace(go.Scatter(
        x=d[gdp_col], y=d[deaths_col], mode="markers", name=f"Deadly quakes (n={fit['n']})",
        marker=dict(size=7, color=GREY, opacity=0.55, line=dict(width=0.4, color="white")),
        customdata=cd(d["country_final"], d["utc_date"].dt.year, d["mag"], d["depth"]),
        hovertemplate="%{customdata[0]} %{customdata[1]}, M %{customdata[2]:.1f}, %{customdata[3]:.0f} km<br>GDP %{x:,.0f}<br><b>%{y:,.0f} deaths</b><extra></extra>"
    ), row=1, col=2)
    xs = np.linspace(d["lg"].min(), d["lg"].max(), 40); dep = np.log10(1 + d["depth"].median())
    for m_, colour, width in [(6.5, LIGHT_GREY, 2), (7.0, ORANGE, 4), (7.5, BLUES[0], 2)]:
        yhat = b[0] + b[1] * m_ + b[2] * dep + b[3] * xs
        fig.add_trace(go.Scatter(x=10 ** xs, y=10 ** yhat, mode="lines", name=f"Fitted M {m_}", line=dict(color=colour, width=width), hovertemplate=f"M {m_}: GDP %{{x:,.0f}} -> %{{y:,.0f}} expected deaths<extra></extra>"), row=1, col=2)
    fig.update_xaxes(type="log", title_text=GDP_CHOICES[gdp_col] + " (log)", row=1, col=2)
    fig.update_yaxes(type="log", title_text="Deaths per earthquake (log)", row=1, col=2)
    # FIX: Moved legend below the chart so it doesn't crush the right plot
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0), margin=dict(l=65, r=20, t=65, b=85), uirevision="keep")
    return style_chart(fig, 580)

def gdp_caption(df_in, deaths_col, years, gdp_col, shallow_only=True):
    fit = gdp_fit(df_in, deaths_col, years, gdp_col, shallow_only)
    if fit is None: return "Too few deadly earthquakes in this selection."
    b = fit["beta"][3]; lo, hi = fit["ci"]; tenfold = 100 * (10 ** b - 1)
    sig = "the interval excludes zero" if (hi < 0 or lo > 0) else "the interval includes zero, so the effect is not certain"
    return f"Among {fit['n']} deadly earthquakes in {fit['countries']} countries, magnitude and depth explain R2 = {fit['r2_base']:.3f} of log deaths and adding GDP raises it to {fit['r2_full']:.3f}. At the same magnitude and depth, a 10 times richer country has {tenfold:+.0f}% deaths (country bootstrap 95% interval for the slope {lo:.2f} to {hi:.2f}; {sig})."

SOURCES = {"usgs": "USGS ComCat", "noaa": "NOAA NCEI", "wikipedia": "Wikipedia", "worldbank": "World Bank WDI"}

def sources_audit(fig, q, row, col):
    both = q[(q["noaa_deaths"] > 0) | (q["wiki_deaths"] > 0)]
    if both.empty: return
    switched = both["deaths_disagree"].astype(bool)
    for mask, colour, sym, name in [(~switched, GREY, "circle", "Agree, or only one source reports"), (switched, ORANGE, "diamond", "Disagree (reconciled via hierarchical rule)")]:
        g = both[mask]
        fig.add_trace(go.Scatter(x=g["noaa_deaths"] + 1, y=g["wiki_deaths"] + 1, mode="markers", name=f"{name} ({len(g)})", marker=dict(size=8, color=colour, symbol=sym, opacity=0.75, line=dict(width=0.5, color="#333333")), customdata=cd(g["country_final"], g["utc_date"].dt.strftime("%Y-%m-%d"), g["noaa_deaths"], g["wiki_deaths"], g["deaths_final"]), hovertemplate="%{customdata[0]}, %{customdata[1]}<br>NOAA %{customdata[2]:,.0f} | Wikipedia %{customdata[3]:,.0f}<br><b>final %{customdata[4]:,.0f}</b><extra></extra>"), row=row, col=col)
    top = max(both["noaa_deaths"].max(), both["wiki_deaths"].max()) + 1
    for f, dash, name in [(1, "solid", "Same count"), (2, "dot", "2 x bands"), (0.5, "dot", None)]:
        fig.add_trace(go.Scatter(x=[1, top], y=[f, top * f], mode="lines", name=name, showlegend=name is not None, line=dict(color=INK, width=1, dash=dash), hoverinfo="skip"), row=row, col=col)
    fig.update_xaxes(type="log", title_text="NOAA deaths + 1", row=row, col=col); fig.update_yaxes(type="log", title_text="Wikipedia deaths + 1", row=row, col=col)

def make_sources(df_in, years, source="noaa"):
    q = filter_years(df_in, years)
    
    # FIX: Removed redundant subplot_titles
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.13)
    if q.empty: return style_chart(fig)
    if source == "usgs":
        sh = q[q["depth"] <= 100]; fixed = sh["is_default_depth"].astype(bool); edges = dict(start=0, end=100, size=2)
        fig.add_trace(go.Histogram(x=sh.loc[~fixed, "depth"], xbins=edges, name=f"Measured depth ({int((~fixed).sum()):,})", marker=dict(color=GREY, line=dict(width=0.4, color="white")), hovertemplate="%{x} km: %{y} earthquakes<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Histogram(x=sh.loc[fixed, "depth"], xbins=edges, name=f"USGS default 10 / 33 km, flagged ({int(fixed.sum()):,})", marker=dict(color=ORANGE, line=dict(width=0.4, color="white")), hovertemplate="%{x} km: %{y} earthquakes<extra></extra>"), row=1, col=1)
        fig.update_xaxes(title_text="Depth (km)", row=1, col=1); fig.update_yaxes(title_text="Earthquakes", row=1, col=1)
        labels = ["1 (nothing merged)", "2", "3", "4", "5 or more"]; k = pd.cut(q["cluster_shock_count"], [0, 1, 2, 3, 4, np.inf], labels=labels).value_counts().reindex(labels, fill_value=0)
        fig.add_trace(go.Bar(x=labels, y=k.values + 1, customdata=k.values, showlegend=False, text=[f"{v:,}" for v in k.values], textposition="outside", marker=dict(color=[LIGHT_GREY] + [ORANGE] * 4, line=dict(width=0.6, color="#444444")), hovertemplate="%{x} shocks: %{customdata:,} mainshocks<extra></extra>"), row=1, col=2)
        fig.update_yaxes(title_text="Mainshocks + 1 (log)", type="log", row=1, col=2); fig.update_xaxes(title_text="M 6+ shocks in the cluster", row=1, col=2); fig.update_layout(barmode="stack")
    elif source == "noaa":
        t10 = q.nlargest(10, "deaths_heatmap").iloc[::-1]; lab = [f"{c} {d.year} (M {m:.1f})" for c, d, m in zip(t10["country_final"], t10["utc_date"], t10["mag"])]
        fig.add_trace(go.Bar(y=lab, x=t10["noaa_deaths"], orientation="h", name="Before: NOAA count only", marker=dict(color=LIGHT_GREY, pattern=dict(shape="/"), line=dict(width=0.6, color="#444444")), hovertemplate="%{y}<br>Before: %{x:,.0f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(y=lab, x=t10["deaths_heatmap"], orientation="h", name="After: reconciled", marker=dict(color=ORANGE, line=dict(width=0.6, color="#444444")), hovertemplate="%{y}<br>After: %{x:,.0f}<extra></extra>"), row=1, col=1)
        fig.update_xaxes(title_text="Deaths", row=1, col=1); py = q.groupby("year")[["deaths", "deaths_heatmap"]].sum().reindex(range(years[0], years[1] + 1), fill_value=0)
        fig.add_trace(go.Bar(x=py.index, y=py["deaths"] + 1, name="NOAA only, per year", marker=dict(color=LIGHT_GREY, line=dict(width=0.4, color="#444444")), customdata=py["deaths"], hovertemplate="%{x} NOAA: %{customdata:,.0f}<extra></extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(x=py.index, y=py["deaths_heatmap"] + 1, mode="lines+markers", name="Reconciled, per year", line=dict(color=ORANGE, width=2), customdata=py["deaths_heatmap"], hovertemplate="%{x} reconciled: %{customdata:,.0f}<extra></extra>"), row=1, col=2)
        fig.update_yaxes(title_text="Deaths + 1 (log)", type="log", row=1, col=2); fig.update_xaxes(title_text="Year", row=1, col=2); fig.update_layout(barmode="group")
    elif source == "wikipedia":
        sources_audit(fig, q, 1, 1); n_ = q["noaa_deaths"] > 0; w_ = q["wiki_deaths"] > 0
        cats = {"Both report deaths": (n_ & w_).sum(), "NOAA only": (n_ & ~w_).sum(), "Wikipedia only": (~n_ & w_).sum()}
        fig.add_trace(go.Bar(x=list(cats), y=list(cats.values()), showlegend=False, text=[f"{v:,}" for v in cats.values()], textposition="outside", marker=dict(color=[BLUES[0], LIGHT_GREY, BLUES[2]], line=dict(width=0.6, color="#444444")), hovertemplate="%{x}: %{y:,} earthquakes<extra></extra>"), row=1, col=2)
        fig.update_yaxes(title_text="Earthquakes with deaths", row=1, col=2)
    else:
        on = q[q["country_iso3"] != "OFFSHORE"]
        src = on["income_source"].replace({"": "No income value"}).fillna("No income value"); src = src.str.replace(r"data/external/taiwan_yearly.csv \(approximate\)", "Taiwan file (approximate)", regex=True).str.replace(r"manual \(see MANUAL_INCOME\)", "Manual list", regex=True)
        gc_ = "gdp_per_capita_ppp_constant_2021"
        if gc_ in on.columns: src = src.where(~(on["income_group"].notna() & on[gc_].isna()), "Income group, no GDP value")
        vc = src.value_counts()
        fig.add_trace(go.Bar(x=vc.index, y=vc.values, showlegend=False, text=[f"{v:,}" for v in vc.values], textposition="outside", marker=dict(color=[BLUES[0] if k == "World Bank panel" else (LIGHT_GREY if k == "No income value" else BLUES[2]) for k in vc.index], line=dict(width=0.6, color="#444444")), hovertemplate="%{x}: %{y:,} earthquakes on land<extra></extra>"), row=1, col=1)
        fig.update_yaxes(title_text="Earthquakes on land", row=1, col=1)
        if gc_ in df.columns:
            cty = (df[df["income_group"].isin(INCOME_ORDER[:4])].groupby("country_iso3").agg(income_group=("income_group", lambda s_: s_.mode().iat[0]), country_final=("country_final", "first"), **{gc_: (gc_, "mean")}).reset_index())
            for grp in INCOME_ORDER[:4]:
                g = cty[(cty["income_group"] == grp) & cty[gc_].notna()]
                if len(g): fig.add_trace(go.Box(y=g[gc_], name=grp, boxpoints="all", jitter=0.4, pointpos=0, showlegend=False, marker=dict(color=INCOME_COLOURS[grp], size=6, line=dict(width=0.6, color="#444444")), line=dict(color="#444444", width=1), fillcolor="rgba(0,0,0,0)", customdata=g["country_final"], hovertemplate="%{customdata}: %{y:,.0f}<extra></extra>"), row=1, col=2)
            fig.update_yaxes(title_text="GDP per capita (log)", type="log", row=1, col=2)
    
    # FIX: Pushed legend down, reduced top margin to avoid crushing axes
    fig.update_layout(legend=dict(orientation="h", y=-0.15, x=0), margin=dict(l=70 if source != "noaa" else 170, r=20, t=40, b=70), uirevision=source)
    return style_chart(fig, 560)

def sources_caption(df_in, years, source="noaa"):
    q = filter_years(df_in, years)
    if q.empty: return "No earthquakes in the selected years."
    if source == "usgs":
        fixed = int(q["is_default_depth"].astype(bool).sum()); merged = q[q["cluster_shock_count"] > 1]; big = q.loc[q["cluster_shock_count"].idxmax()]
        return f"{fixed:,} of {len(q):,} earthquakes ({100 * fixed / len(q):.0f}%) carry a USGS default depth of 10 or 33 km, the orange spikes: they are flagged (diamonds on plot 2). Foreshocks and aftershocks within 2 days and 150 km were merged into the largest shock: {len(merged):,} mainshocks absorbed {int((merged['cluster_shock_count'] - 1).sum()):,} other M 6+ shocks (largest cluster: {int(big['cluster_shock_count'])} shocks, {big['country_final']} {big['year']}), preventing duplication."
    if source == "noaa":
        switched = int((q["deaths_heatmap"] != q["deaths"]).sum())
        return f"Before reconciliation (NOAA only) the total is {int(q['deaths'].sum()):,} deaths; after, {int(q['deaths_heatmap'].sum()):,}. {switched} earthquakes changed via the hierarchical rule."
    if source == "wikipedia":
        both = q[(q["noaa_deaths"] > 0) & (q["wiki_deaths"] > 0)]
        r = np.corrcoef(np.log10(1 + both["noaa_deaths"]), np.log10(1 + both["wiki_deaths"]))[0, 1] if len(both) > 2 else np.nan
        return f"{len(both)} earthquakes have a death count in both NOAA and Wikipedia; on a log scale they agree closely (r = {r:.3f}), but {int(q['deaths_disagree'].sum())} differ (orange diamonds). Averaging is ONLY used on vast disagreements."
    on = q[q["country_iso3"] != "OFFSHORE"]
    has = on["income_group"].notna(); missing = on.loc[~has, "country_final"].value_counts()
    return f"{int(has.sum()):,} of {len(on):,} earthquakes on land ({100 * has.mean():.0f}%) have an income group joined from the World Bank by country and year. Missing data ({int((~has).sum())} onshore) is kept as MAR."

PLOT_GROUPS = {
"usgs": ("USGS", ("usgs", "earthquake", "plot1", "plot3")),
"noaa": ("NOAA", ("noaa", "plot1", "plot4", "plot7")),
"wikipedia": ("Wikipedia", ("wikipedia", "plot4", "plot7")),
"worldbank": ("World Bank", ("worldbank", "plot2", "plot5", "plot6", "plot7")),
}
PLOT_NOTES = {
"usgs_noaa_missingness_before_after_cleaning": "Missing values per column before and after cleaning: high-missingness columns dropped, depthError set to 0 with a flag.",
"usgs_deptherror_zero_imputation": "depthError before and after the zero fill: the spike at 0 is imputed, and which magnitudes received it.",
"usgs_noaa_wikipedia_deaths_zero_imputation": "Death counts before and after filling missing NOAA and Wikipedia counts with 0.",
"usgs_noaa_final_deaths_outliers": "Outliers in the final deaths: Tukey fence on log10(deaths); flagged events are kept as real.",
"usgs_noaa_outliers_boxplots": "Outliers on the non-missing USGS and NOAA values (depth, magnitude, depthError, deaths).",
"usgs_missingness_by_column": "USGS: share of missing values in every column before cleaning.",
"usgs_missingness_across_years": "USGS: year-by-year missingness heatmap before cleaning.",
"usgs_noaa_missingness_deptherror_and_deaths": "Missingness of USGS depthError by year and of NOAA deaths.",
"usgs_cluster_size_distribution": "Size of the aftershock clusters collapsed into one mainshock.",
"earthquake_disribution": "Where the M 6+ earthquakes are, coloured by magnitude.",
"wikipedia_missingness_before_after": "Missing values before and after cleaning, and the magnitude pattern that shows the missingness is not MCAR.",
"wikipedia_rows_before_after_dedup": "Rows per year before and after duplicate removal (monthly lists + yearly rankings).",
"wikipedia_deaths_outliers_before_after": "Death outliers before and after the Hotan 2014 correction.",
"wikipedia_raw_missingness_monthly": "Raw Wikipedia monthly tables: missing values per column before cleaning.",
"worldbank_missingness_before_after_imputation": "Missing values per indicator before and after imputation, 2000-2023 vs 2024-2025.",
"worldbank_outliers_after_imputation": "Outliers after imputation: raw vs log scale, and imputed vs observed cells.",
"worldbank_missingness_test_variance_ratio": "Missingness test: how strongly missing values concentrate in a few countries.",
"worldbank_missingness_test_gap_blocks_and_countries": "Missingness tests: gaps come in blocks, and the countries that cause them.",
"worldbank_eda_missing_by_column": "World Bank: share of missing values per indicator before cleaning."
}
def plot_files(group):
    global PLOTS_DIR
    for d in (PROJECT_DIR / "reports" / "figures", APP_DIR / "reports" / "figures", PROJECT_DIR / "plots"):
        if d.exists() and list(d.glob(".png")):
            PLOTS_DIR = d
            break
    if not PLOTS_DIR.exists(): return []
    keys = PLOT_GROUPS[group][1]
    files = sorted(p.name for p in PLOTS_DIR.glob(".png") if any(k in p.name.lower() for k in keys))
    return files if files else sorted(p.name for p in PLOTS_DIR.glob("*.png"))

def pretty(name): stem = Path(name).stem; return PLOT_NOTES.get(stem, stem.replace("_", " ").capitalize())
def carousel(group):
    title = PLOT_GROUPS[group][0]
    return html.Div(className="carousel", children=[
        html.Div(className="carousel-head", children=[html.H4(title), html.Div(className="carousel-nav", children=[html.Button("‹ Previous", id={"type": "car-prev", "group": group}, n_clicks=0, className="nav-btn"), html.Span(id={"type": "car-count", "group": group}, className="car-count"), html.Button("Next ›", id={"type": "car-next", "group": group}, n_clicks=0, className="nav-btn"),]),]),
        dcc.Store(id={"type": "car-idx", "group": group}, data=0), html.Img(id={"type": "car-img", "group": group}, className="car-img"), html.P(id={"type": "car-note", "group": group}, className="car-note"),
    ])

PERIODS = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2025)]
def time_groups(q): return q["income_group"].where(q["income_group"].isin(INCOME_ORDER[:4]), np.where(q["country_iso3"].eq("OFFSHORE"), "Offshore", "No income data"))

def make_time(df_in, deaths_col, years, metric="share"):
    q = filter_years(df_in, years).copy(); q["grp"] = time_groups(q)
    
    # FIX: Removed redundant subplot_titles to prevent overlap
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.1)
    if q.empty: return style_chart(fig)
    per = q.pivot_table(index="year", columns="grp", values=deaths_col, aggfunc="sum", fill_value=0).reindex(range(years[0], years[1] + 1), fill_value=0)
    colours = {**INCOME_COLOURS, "Offshore": "#d9d9d9"}
    for grp in INCOME_ORDER:
        if grp in per.columns and per[grp].sum() > 0: fig.add_trace(go.Bar(x=per.index, y=per[grp], name=grp, marker=dict(color=colours[grp], line=dict(width=0.4, color="#444444")), legendgroup=grp, hovertemplate=f"{grp}, %{{x}}: %{{y:,.0f}} deaths<extra></extra>"), row=1, col=1)
    fig.update_yaxes(title_text="Deaths", row=1, col=1); fig.update_xaxes(title_text="Year", row=1, col=1)
    q["period"] = pd.cut(q["year"], [a - 0.5 for a, _ in PERIODS] + [PERIODS[-1][1] + 0.5], labels=[f"{a}-{b}" for a, b in PERIODS])
    on = q[q["grp"].isin(INCOME_ORDER[:4])]
    for grp, dash_ in zip(INCOME_ORDER[:4], ["solid", "dash", "dot", "dashdot"]):
        g = on[on["grp"] == grp].groupby("period", observed=False)[deaths_col]; val = g.apply(lambda s_: dens_metric(s_, metric) if len(s_) else np.nan); n = g.size(); val = val.where(n >= 5)
        if val.notna().sum() == 0: continue
        fig.add_trace(go.Scatter(x=[str(i) for i in val.index], y=val.to_numpy(), mode="lines+markers", name=grp, legendgroup=grp, showlegend=False, line=dict(color=colours[grp] if grp != "High income" else "#b8a500", width=2.5, dash=dash_), marker=dict(size=9, color=colours[grp], line=dict(width=1, color="#444444")), customdata=n.to_numpy(), hovertemplate=f"{grp}, %{{x}}: " + ("%{y:.0f}% deadly" if metric == "share" else "%{y:,.1f} deaths per earthquake") + " (%{customdata} earthquakes)<extra></extra>"), row=1, col=2)
    if metric == "share": fig.update_yaxes(title_text="Earthquakes that killed anyone (%)", rangemode="tozero", row=1, col=2)
    else: fig.update_yaxes(title_text="Deaths per earthquake (log)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Period", row=1, col=2)
    top = q.nlargest(5, deaths_col)
    for _, r in top.iterrows():
        if r[deaths_col] > 0: fig.add_annotation(x=r["year"], y=per.loc[r["year"]].sum(), text=f"{r['country_final']} {r['year']}", showarrow=True, arrowhead=0, ay=-22, font=dict(size=10), bgcolor="rgba(255,255,255,0.85)", row=1, col=1)
    
    # FIX: Pushed legend down, reduced top margin to avoid crushing axes
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.15, x=0), margin=dict(l=70, r=20, t=40, b=60), uirevision="time")
    return style_chart(fig, 560)

def time_caption(df_in, deaths_col, years, metric="share"):
    q = filter_years(df_in, years).copy(); tot = q[deaths_col].sum()
    if tot <= 0: return "No deaths in the selected years."
    per = q.groupby("year")[deaths_col].sum(); top3 = per.nlargest(3)
    return f"The three deadliest years ({', '.join(f'{y} {fmt_n(v)}' for y, v in top3.items())}) hold {100 * top3.sum() / tot:.0f}% of the {fmt_n(tot)} deaths, so totals jump with single disasters rather than trending."

_PRED_CACHE = {}
LOW_GDP, HIGH_GDP = 1_500, 45_000

def logit_fit(X, y, ridge=1e-3, iters=60):
    A = np.column_stack([np.ones(len(X)), X]); b = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ b, -30, 30))); W = p * (1 - p); H = A.T @ (A * W[:, None]) + ridge * np.eye(A.shape[1])
        step = np.linalg.solve(H, A.T @ (y - p) - ridge * b); b += step
        if np.abs(step).max() < 1e-8: break
    return b

def auc(y, score):
    r = pd.Series(score).rank().to_numpy(); pos = y.astype(bool); n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan

def pred_features(mag, depth, gdp):
    return np.column_stack([np.atleast_1d(mag).astype(float), np.log10(1 + np.atleast_1d(depth).astype(float)), np.log10(np.atleast_1d(gdp).astype(float))])

def predictor_fit(df_in, deaths_col, years, gdp_col):
    key = (id(df_in), deaths_col, tuple(years), gdp_col)
    if key not in _PRED_CACHE: _PRED_CACHE[key] = _predictor_fit(df_in, deaths_col, tuple(years), gdp_col)
    return _PRED_CACHE[key]

def _predictor_fit(df_in, deaths_col, years, gdp_col):
    q = onshore(df_in, years).dropna(subset=[gdp_col, "mag", "depth"]); q = q[q[gdp_col] > 0]
    y1 = (q[deaths_col] > 0).to_numpy(float)
    if len(q) < 50 or y1.sum() < 15: return None
    X = pred_features(q["mag"], q["depth"], q[gdp_col]); bl = logit_fit(X, y1)
    d = q[q[deaths_col] > 0]
    Xd = pred_features(d["mag"], d["depth"], d[gdp_col]); yd = np.log10(d[deaths_col].to_numpy(float))
    bo, r2 = ols(Xd, yd); resid = yd - np.column_stack([np.ones(len(Xd)), Xd]) @ bo; p_hat = 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ bl)))
    return dict(logit=bl, ols=bo, r2=r2, sigma=resid.std(ddof=Xd.shape[1] + 1), smear=float(np.mean(10 ** resid)), resid=resid, n=len(q), n_deadly=len(d), countries=q["country_iso3"].nunique(), auc=auc(y1, p_hat), gdp_col=gdp_col)

def predict(fit, mag, depth, gdp):
    X = pred_features(mag, depth, gdp); A = np.column_stack([np.ones(len(X)), X])
    p = 1 / (1 + np.exp(-(A @ fit["logit"]))); mu = A @ fit["ols"]; lo, hi = np.quantile(fit["resid"], [0.1, 0.9])
    return dict(p=p, median=10 ** mu, lo=10 ** (mu + lo), hi=10 ** (mu + hi), mean_if=10 ** mu * fit["smear"], expected=p * 10 ** mu * fit["smear"])

def onshore_events(df_in, years, gdp_col):
    q = onshore(df_in, years).dropna(subset=[gdp_col, "latitude", "longitude"])
    return q[q[gdp_col] > 0]

def event_label(r, deaths_col): return f"{r['country_final']} {pd.to_datetime(r['utc_date']).strftime('%Y-%m-%d')}, M {r['mag']:.1f}, {r['depth']:.0f} km, {int(r[deaths_col]):,} deaths"

def make_pred_map(df_in, deaths_col, years, gdp_col, selected=None):
    q = onshore_events(df_in, years, gdp_col); fig = go.Figure()
    for grp in INCOME_ORDER[:4]:
        g = q[q["income_group"] == grp]
        if g.empty: continue
        size = np.where(g[deaths_col] > 0, 6 + 3 * np.log10(1 + g[deaths_col]), 4)
        fig.add_trace(Scatter(lat=g["latitude"], lon=g["longitude"], mode="markers", name=grp, marker=dict(size=size, color=INCOME_COLOURS[grp], opacity=0.85), customdata=cd(g.index, g["country_final"], g["utc_date"].dt.strftime("%Y-%m-%d"), g["mag"], g["depth"], g[deaths_col], g[gdp_col]), hovertemplate="<b>%{customdata[1]}</b> %{customdata[2]}<br>M %{customdata[3]:.1f}, %{customdata[4]:.0f} km<br>%{customdata[5]:,.0f} deaths, GDP %{customdata[6]:,.0f}<br><i>click to load into the predictor</i><extra></extra>"))
    if selected is not None and selected in q.index:
        r = q.loc[selected]
        fig.add_trace(Scatter(lat=[r["latitude"]], lon=[r["longitude"]], mode="markers", name="Selected", hoverinfo="skip", marker=dict(size=22, color=ORANGE, opacity=0.55)))
    fig.update_layout(**{MAP_KEY: dict(style="carto-positron", center=dict(lat=20, lon=40), zoom=0.6)}, margin=dict(l=0, r=0, t=10, b=0), height=430, legend=dict(bgcolor="rgba(255,255,255,0.85)", x=0.01, y=0.98, itemsizing="constant"), uirevision="pred-map", clickmode="event")
    return fig

def make_pred_curve(fit, mag, depth, gdp, event=None, deaths_col="deaths_heatmap"):
    # FIX: Removed redundant subplot_titles to prevent text overlapping with chart elements
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.11)
    if fit is None: return style_chart(fig, 460)
    gx = np.logspace(np.log10(300), np.log10(120_000), 80); pr = predict(fit, np.full(len(gx), mag), np.full(len(gx), depth), gx)
    fig.add_trace(go.Scatter(x=np.r_[gx, gx[::-1]], y=np.r_[pr["hi"], pr["lo"][::-1]], fill="toself", fillcolor="rgba(33,113,181,0.15)", line=dict(width=0), name="80% range if deadly", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=gx, y=pr["median"], mode="lines", name="Typical toll if deadly (median)", line=dict(color=BLUES[0], width=3), hovertemplate="GDP %{x:,.0f}: %{y:,.0f} deaths<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=gx, y=np.maximum(pr["expected"], 0.01), mode="lines", name="Average over many such quakes", line=dict(color=GREY, width=2, dash="dash"), hovertemplate="GDP %{x:,.0f}: %{y:,.1f} expected deaths<extra></extra>"), row=1, col=1)
    now = predict(fit, mag, depth, gdp)
    fig.add_trace(go.Scatter(x=[gdp], y=[now["median"][0]], mode="markers", name="Your scenario", marker=dict(size=15, color=ORANGE, line=dict(width=1.5, color=INK)), hovertemplate="Scenario: GDP %{x:,.0f}, %{y:,.0f} deaths if deadly<extra></extra>"), row=1, col=1)
    if event is not None and event[deaths_col] > 0:
        fig.add_trace(go.Scatter(x=[event[fit["gdp_col"]]], y=[event[deaths_col]], mode="markers", name="The selected earthquake (actual)", marker=dict(size=14, color=INK, symbol="star"), hovertemplate=f"{event['country_final']} {pd.to_datetime(event['utc_date']).year}: " + "%{y:,.0f} deaths at GDP %{x:,.0f}<extra></extra>"), row=1, col=1)
    fig.update_xaxes(type="log", title_text="GDP per capita (log)", row=1, col=1); fig.update_yaxes(type="log", title_text="Deaths (log)", row=1, col=1)
    fig.add_trace(go.Scatter(x=gx, y=100 * pr["p"], mode="lines", showlegend=False, line=dict(color=BLUES[0], width=3), hovertemplate="GDP %{x:,.0f}: %{y:.0f}% chance of any death<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[gdp], y=[100 * now["p"][0]], mode="markers", showlegend=False, marker=dict(size=15, color=ORANGE, line=dict(width=1.5, color=INK)), hovertemplate="Scenario: %{y:.0f}%<extra></extra>"), row=1, col=2)
    for x_, lab in [(LOW_GDP, "low income"), (HIGH_GDP, "high income")]: fig.add_trace(go.Scatter(x=[x_, x_], y=[0, 92], mode="lines+text", showlegend=False, hoverinfo="skip", text=["", f"${x_:,} {lab}"], textposition="top center", textfont=dict(size=10, color=MUTED), line=dict(color=LIGHT_GREY, dash="dot", width=1.5)), row=1, col=2)
    fig.update_xaxes(type="log", title_text="GDP per capita (log)", row=1, col=2); fig.update_yaxes(title_text="Chance of at least one death (%)", range=[0, 100], row=1, col=2)
    
    # FIX: Pushed legend down, reduced top margin to avoid crushing axes
    fig.update_layout(legend=dict(orientation="h", y=-0.16, x=0), margin=dict(l=70, r=20, t=40, b=60), uirevision="pred-curve")
    return style_chart(fig, 470)

def pred_cards(fit, mag, depth, gdp, event, deaths_col):
    if fit is None: return [html.P("Too few earthquakes in this selection to fit the model.")]
    now = predict(fit, mag, depth, gdp); two = predict(fit, [mag, mag], [depth, depth], [LOW_GDP, HIGH_GDP])
    cards = [kpi("CHANCE ANYONE DIES", f"{100 * now['p'][0]:.0f}%", f"M {mag:.1f}, {depth:.0f} km, GDP ${gdp:,.0f}"), kpi("IF DEADLY: TYPICAL TOLL", fmt_n(now["median"][0]), f"median; 80% of real tolls fall between {fmt_n(now['lo'][0])} and {fmt_n(now['hi'][0])}"), kpi(f"${LOW_GDP:,} vs ${HIGH_GDP:,}", f"{fmt_n(two['median'][0])} vs {fmt_n(two['median'][1])}", f"typical toll if deadly; chance {100 * two['p'][0]:.0f}% vs {100 * two['p'][1]:.0f}%"), kpi("AVERAGE OVER MANY SUCH QUAKES", fmt_n(now["expected"][0]), f"chance x mean toll; rare disasters pull the mean to {fit['smear']:.0f} x the median")]
    if event is not None:
        if event[deaths_col] > 0:
            b = fit["ols"]; shift = b[1] * (mag - event["mag"]) + b[2] * (np.log10(1 + depth) - np.log10(1 + event["depth"])) + b[3] * (np.log10(gdp) - np.log10(event[fit["gdp_col"]]))
            cf = event[deaths_col] * 10 ** shift
            cards.append(kpi("THIS EARTHQUAKE, YOUR SETTINGS", fmt_n(cf), f"actual {int(event[deaths_col]):,} at GDP ${event[fit['gdp_col']]:,.0f}; scaled by the model's slopes"))
        else:
            p0 = predict(fit, event["mag"], event["depth"], event[fit["gdp_col"]])["p"][0]
            cards.append(kpi("THIS EARTHQUAKE", "0 deaths", f"model gave it a {100 * p0:.0f}% chance of any death"))
    return cards

def pred_caption(fit, deaths_col):
    if fit is None: return "Too few earthquakes in this selection to fit the model."
    b_l, b_o = fit["logit"], fit["ols"]
    return f"Fitted on {fit['n']:,} earthquakes on land in {fit['countries']} countries ({fit['n_deadly']} deadly). Part 1, the chance of any death, is a logistic model (AUC {fit['auc']:.2f}): a 10 times richer country has {np.exp(b_l[3]):.2f} times the odds of a deadly earthquake at the same magnitude and depth. Part 2, the toll if anyone dies, is a regression on log10 deaths (R2 {fit['r2']:.2f}): 10 times richer means {100 * (10 ** b_o[3] - 1):+.0f}% deaths. 'This earthquake' keeps the selected event's own toll and scales it by the model's slopes, so local factors (building codes, time of day, tsunami) stay in."

# =====================================================================================================================================
# PAGES & CSS INJECTION
# =====================================================================================================================================
REGIONS = sorted(df["region"].dropna().unique()) if "region" in df.columns else []
PAGES = [("/", "Overview", "◎"), ("/map", "1 · Deaths map", "◉"), ("/physics", "2 · Magnitude & depth", "≋"), ("/density", "3 · Population density", "▦"), ("/income", "4 · GDP per capita", "$"), ("/losses", "5 · Losses by income", "▮"), ("/cleaning", "6 · Before vs after cleaning", "⇄"), ("/time", "7 · Deaths over time", "↗"), ("/predictor", "8 · Death predictor", "✦"), ("/methodology", "Methodology", "ⓘ")]

def kpi(title, value, note): return html.Div(className="kpi-card", children=[html.P(title, className="kpi-title"), html.H2(value, className="kpi-value"), html.P(note, className="kpi-note")])
def header(eyebrow, title, description): return html.Div(className="page-header", children=[html.P(eyebrow, className="eyebrow"), html.H1(title), html.P(description, className="page-description")])
def panel(*children, cls="content-panel"): return html.Div(className=cls, children=list(children))
def control(label, comp, width=None): return html.Div(className="control", style={"width": width} if width else None, children=[html.Label(label), comp])
def pager(path):
    i = [p[0] for p in PAGES].index(path); prev_ = PAGES[i - 1] if i > 0 else None; next_ = PAGES[i + 1] if i + 1 < len(PAGES) else None
    return html.Div(className="pager", children=[dcc.Link(f"← {prev_[1]}", href=prev_[0], className="pager-link") if prev_ else html.Span(), dcc.Link(f"{next_[1]} →", href=next_[0], className="pager-link") if next_ else html.Span()])
def question_card(n, tag, title, takeaway, hides, href):
    return html.Div(className="q-card", children=[html.P(f"QUESTION {n} · {tag}", className="q-tag"), html.H4(title), html.P(takeaway, className="q-takeaway"), html.P(["What it hides: ", hides], className="q-hides"), dcc.Link("Open the plot →", href=href, className="pager-link")])

_OVERVIEW_CACHE = {}

def overview_body(deaths_col, years):
    key = (deaths_col, tuple(years))
    if key in _OVERVIEW_CACHE: return _OVERVIEW_CACHE[key]
    yrs = tuple(years)
    q = filter_years(df, yrs)
    on = q[q["country_iso3"] != "OFFSHORE"]
    tot = q[deaths_col].sum(); deadly = int((q[deaths_col] > 0).sum()); fixed = int(q["depth"].isin([10.0, 33.0]).sum())
    with_income = int(on["income_group"].notna().sum())
    shallow_share = 100 * q.loc[q["depth"] <= 70, deaths_col].sum() / max(tot, 1)
    gcol = "gdp_per_capita_ppp_constant_2021"
    fit = gdp_fit(df, deaths_col, yrs, gcol, True, n_boot=100) if gcol in df.columns else None
    if fit:
        b, (lo, hi) = fit["beta"][3], fit["ci"]
        income_kpi = kpi("10 x RICHER COUNTRY", f"{100 * (10 ** b - 1):+.0f}% deaths", f"deadly shallow quakes, same magnitude and depth; 95% CI slope {lo:.2f} to {hi:.2f}")
    else: income_kpi = kpi("10 x RICHER COUNTRY", "n/a", "too few deadly earthquakes")
    kpis = html.Div(className="kpi-grid", children=[
        kpi("EARTHQUAKES", f"{len(q):,}", f"M 6+ mainshocks; {fixed:,} with a USGS default depth"),
        kpi("DEADLY", f"{deadly:,}", f"{100 * deadly / max(len(q), 1):.0f}% killed at least one person"),
        kpi("DEATHS", fmt_n(tot), f"{DEATH_COLUMNS[deaths_col]}; NOAA only: {fmt_n(q['deaths'].sum())}"),
        kpi("ON LAND WITH INCOME", f"{with_income:,}", f"{100 * with_income / max(len(on), 1):.0f}% of {len(on):,} on land; {len(q) - len(on):,} offshore"),
        income_kpi,
    ])
    b_ = events_with_deaths(df, deaths_col, yrs)
    t1 = "No deadly earthquakes in these years."
    if len(b_): _, cl = cluster_deaths(b_, deaths_col, 250); t1 = f"The three deadliest regions ({', '.join(cl.head(3)['main_country'])}) hold {100 * cl.head(3)['deaths'].sum() / max(tot, 1):.0f}% of all deaths."
    t2 = (f"Shallow quakes caused {shallow_share:.1f}% of deaths; {100 * (1 - deadly / max(len(q), 1)):.0f}% of all M 6+ earthquakes killed nobody.")
    dq, _ = density_frames(df, deaths_col, yrs, False)
    if len(dq) >= 20: r2, _, _ = density_models(dq); t3 = f"Region adds {r2['+ region'] - r2['magnitude + depth']:.3f} to R2, density {r2['+ density'] - r2['magnitude + depth']:.3f}: {'region' if r2['+ region'] > r2['+ density'] else 'density'} matters more."
    else: t3 = "Too few earthquakes on land in these years."
    t4 = (f"Halving GDP per capita goes with {10 ** (-fit['beta'][3] * np.log10(2)):.2f} times the deaths; R2 rises from {fit['r2_base']:.2f} to {fit['r2_full']:.2f}." if fit else "Too few deadly earthquakes in these years.")
    g, _, _, _, _ = losses_table(df, deaths_col, yrs, "income_group", "all")
    t5 = (f"'{g['per_quake'].idxmax()}' earthquakes killed {fmt_n(g['per_quake'].max())} each, '{g.index[-1]}' ones {fmt_n(g['per_quake'].iloc[-1])}." if len(g) else "No earthquakes with income data in these years.")
    both = q[(q["noaa_deaths"] > 0) & (q["wiki_deaths"] > 0)]
    r = np.corrcoef(np.log10(1 + both["noaa_deaths"]), np.log10(1 + both["wiki_deaths"]))[0, 1] if len(both) > 2 else np.nan
    t6 = f"NOAA and Wikipedia agree closely (log r = {r:.3f}); reconciling them moves the total from {fmt_n(q['deaths'].sum())} to {fmt_n(q['deaths_heatmap'].sum())}."
    per = q.groupby("year")[deaths_col].sum()
    t7 = (f"The three deadliest years hold {100 * per.nlargest(3).sum() / max(tot, 1):.0f}% of all deaths." if tot > 0 else "No deaths in these years.")
    pf = predictor_fit(df, deaths_col, yrs, gcol) if gcol in df.columns else None
    if pf: two = predict(pf, [7.0, 7.0], [15, 15], [LOW_GDP, HIGH_GDP]); t8 = (f"An M 7.0 quake 15 km deep: {100 * two['p'][0]:.0f}% chance of deaths and a typical toll of {fmt_n(two['median'][0])} at ${LOW_GDP:,} GDP per capita, against {100 * two['p'][1]:.0f}% and {fmt_n(two['median'][1])} at ${HIGH_GDP:,}.")
    else: t8 = "Too few earthquakes to fit the model in these years."
    cards = html.Div(className="q-grid", children=[
        question_card(1, "WHERE", "Where do deaths cluster?", t1, "blobs merge neighbouring disasters; offshore deaths sit at sea.", "/map"),
        question_card(2, "PHYSICS", "How do magnitude and depth matter?", t2, "788 depths are USGS defaults (10 / 33 km), not measurements.", "/physics"),
        question_card(3, "EXPOSURE", "Population density or region?", t3, "density is a country average; cities next to empty land look alike.", "/density"),
        question_card(4, "THE ANSWER", "After magnitude and depth, does income matter?", t4, "a few disasters drive the fit; the interval says how sure we are.", "/income"),
        question_card(5, "WHO LOSES", "Deaths per earthquake by income group", t5, "averages hide that one country can carry a whole bar (Haiti).", "/losses"),
        question_card(6, "DATA QUALITY", "Can we trust the counts?", t6, "Haiti 2010 ranges from about 100,000 to 316,000 across sources.", "/cleaning"),
        question_card(7, "TIME", "Is anything changing over time?", t7, "26 years and a few disasters: a trend is hard to separate from luck.", "/time"),
        question_card(8, "WHAT IF", "How many would die if the country were richer?", t8, "a summary of the past, not a forecast; the 80% range is wide.", "/predictor"),
    ])
    out = [kpis, html.H3("Questions this dashboard answers (numbers follow the sidebar)", className="section-title"), cards]
    if len(_OVERVIEW_CACHE) > 100: _OVERVIEW_CACHE.clear()
    _OVERVIEW_CACHE[key] = out
    return out

def overview_page():
    return html.Div([
        header("DA2402 · PROJECT 21 · OVERVIEW", "Earthquake deaths, magnitude and income", "After allowing for magnitude and depth, how much does a country's income explain earthquake deaths? USGS, NOAA, Wikipedia and the World Bank combined for every M 6+ earthquake from 2000 to 2025. The cards follow the sidebar's deaths measure and years."),
        html.Div(id="overview-body"),
        html.H3("From four sources to one table", className="section-title"),
        html.Div(className="card-grid", children=[
            panel(html.H4("1 · USGS ComCat"), html.P("FDSN API. M 6+ earthquakes 2000-2025; 1 nuclear explosion removed; shocks within 2 days and 150 km collapsed into the largest mainshocks (3,360 events); country by reverse geocoding to Natural Earth; 10 / 33 km default depths flagged."), cls="card c1"),
            panel(html.H4("2 · NOAA NCEI"), html.P("Significant earthquake database. Official deaths per event, matched to USGS mainshocks within 1 day and 150 km and used as the primary scientific authority."), cls="card c2"),
            panel(html.H4("3 · Wikipedia"), html.P("'List of earthquakes in YYYY': monthly tables and bullet text. Death ranges turned into midpoints, matched to USGS by date (1 day) and magnitude (0.4) to rescue missing NOAA records and cross-check vast discrepancies."), cls="card c3"),
            panel(html.H4("4 · World Bank WDI"), html.P("Region, income group, GDP per capita (PPP 2021 $), population and density, joined by ISO3 country and the exact year of the earthquake."), cls="card c4"),
        ]),
        panel(html.H3("How to read the dashboard"), html.P("The sidebar controls (deaths measure and years) apply to every page. Income colours always run from dark purple (poorest) through blue and green to yellow (richest); orange marks the point of each plot. Every plot has 2-3 sentences underneath, computed from the current selection. Tick 'Greyscale preview' in the sidebar to check that every plot still reads in black and white.")),
        pager("/"),
    ])

def map_page():
    return html.Div([
        header("PLOT 1 · WHERE", "Where do earthquake deaths cluster?", "Blobs group deadly earthquakes within a radius (weighted DBSCAN, never wider than 500 km); switch to single earthquakes to colour them by income. Faint dots are the earthquakes that killed nobody. Press Play to watch 2000-2025 unfold."),
        html.Div(className="controls", children=[
            control("View", dcc.RadioItems(id="view", value="blobs", options=[{"label": " Clusters (blobs)", "value": "blobs"}, {"label": " Single earthquakes by income", "value": "points"}]), "230px"),
            control("Colour (single earthquakes)", dcc.Dropdown(id="colour-by", value="income_group", clearable=False, options=[{"label": v, "value": k} for k, v in COLOUR_BY.items() if k == "income_group" or k in df.columns]), "330px"),
            control(f"Cluster radius (km, max {MAX_CLUSTER_RADIUS_KM})", dcc.Slider(id="eps", min=50, max=MAX_CLUSTER_RADIUS_KM, step=50, value=250, marks={k: str(k) for k in (50, 250, 500)}), "260px"),
            control("Earthquakes with no deaths", dcc.Checklist(id="show-zero", value=["show"], options=[{"label": " Show as faint dots", "value": "show"}]), "200px"),
            control("Time", dcc.RadioItems(id="time-mode", value="range", options=[{"label": " Whole sidebar range", "value": "range"}, {"label": " One year at a time", "value": "year"}, {"label": " Cumulative", "value": "cumulative"}]), "260px"),
        ]),
        html.Div(className="play-row", children=[
            html.Button("▶ Play", id="play", n_clicks=0, className="nav-btn"),
            html.Div(style={"flex": "1"}, children=[dcc.Slider(id="year-step", min=2000, max=2025, step=1, value=2000, marks={y: str(y) for y in range(2000, 2026, 5)}, tooltip={"placement": "bottom", "always_visible": True})]),
        ]),
        dcc.Interval(id="ticker", interval=1200, disabled=True),
        panel(dcc.Graph(id="blob-map"), html.P(id="blob-caption", className="caption"), dcc.Graph(id="deaths-by-year", config={"displayModeBar": False}), cls="chart-card"),
        pager("/map"),
    ])

def physics_page():
    return html.Div([
        header("PLOT 2 · WHAT KIND OF EARTHQUAKE", "Magnitude, depth and deaths", "The physical side of the question: bigger and shallower earthquakes should kill more. Orange marks earthquakes with 1,000 or more deaths."),
        html.Div(className="controls", children=[
            control("Maximum depth (km)", dcc.Slider(id="max-depth", min=50, max=700, step=10, value=700, marks={k: str(k) for k in (70, 300, 700)}, tooltip={"placement": "bottom"}), "380px"),
            control("USGS default depths (10 / 33 km)", dcc.RadioItems(id="show-fixed", value="show", options=[{"label": " Show as diamonds", "value": "show"}, {"label": " Hide", "value": "hide"}]), "260px"),
        ]),
        panel(dcc.Graph(id="physics-graph"), html.P(id="physics-caption", className="caption"), cls="chart-card"),
        pager("/physics"),
    ])

def density_page():
    return html.Div([
        header("PLOT 3 · EXPOSURE", "Population density or region: which matters more?", "Left: density bins at fixed magnitude (default), split by income, a region x density heatmap, or the country bubbles. Right: how much of the variation in deaths density and region explain on top of magnitude and depth."),
        html.Div(className="controls", children=[
            control("Left panel", dcc.RadioItems(id="dens-mode", value="bins_mag", options=[{"label": f" {v}", "value": k} for k, v in DENS_MODES.items()]), "280px"),
            control("Measure (bins and heatmap)", dcc.RadioItems(id="dens-metric", value="share", options=[{"label": f" {v}", "value": k} for k, v in DENS_METRICS.items()]), "300px"),
            control("Highlight region (bubbles)", dcc.Dropdown(id="dens-region", value="All regions", clearable=False, options=["All regions"] + REGIONS), "300px"),
            control("Density axis (bubbles)", dcc.RadioItems(id="dens-within", value="raw", options=[{"label": " People per km2", "value": "raw"}, {"label": " Relative to region average", "value": "within"}]), "230px"),
            control("Earthquakes", dcc.RadioItems(id="dens-shallow", value="all", options=[{"label": " All depths", "value": "all"}, {"label": " Shallow only (70 km or less)", "value": "shallow"}]), "260px"),
        ]),
        panel(dcc.Graph(id="density-graph"), html.P(id="density-caption", className="caption"), cls="chart-card"),
        pager("/density"),
    ])

def income_page():
    return html.Div([
        header("PLOT 4 · THE ANSWER", "After magnitude and depth, how much does income matter?", "Left: the share of earthquakes that killed anyone, magnitude held fixed in each row. Right: the fitted effect of GDP per capita on deaths at fixed magnitude and depth."),
        html.Div(className="controls", children=[
            control("GDP measure", dcc.RadioItems(id="gdp-col", value="gdp_per_capita_ppp_constant_2021", options=[{"label": f" {v}", "value": k} for k, v in GDP_CHOICES.items() if k in df.columns]), "460px"),
            control("Earthquakes", dcc.RadioItems(id="gdp-shallow", value="shallow", options=[{"label": " Shallow only (70 km or less)", "value": "shallow"}, {"label": " All depths", "value": "all"}]), "260px"),
        ]),
        panel(dcc.Graph(id="income-graph"), html.P(id="income-caption", className="caption"), cls="chart-card"),
        pager("/income"),
    ])

def losses_page():
    return html.Div([
        header(
            "PLOT 5 · WHO LOSES MOST",
            "Losses per earthquake by income group",
            "Deaths per earthquake for each income group (bars) overlaid with individual countries (grey bubbles sized by earthquake count)."
        ),

        html.Div(
            className="controls",
            children=[
                control(
                    "Group countries by",
                    dcc.Dropdown(
                        id="loss-group",
                        value="income_group",
                        clearable=False,
                        options=[
                            {"label": v, "value": k}
                            for k, v in COLOUR_BY.items()
                            if k == "income_group" or k in df.columns
                        ]
                    ),
                    "380px"
                ),

                control(
                    "Magnitude",
                    dcc.RadioItems(
                        id="loss-mag",
                        value="all",
                        inline=True,
                        options=[
                            {"label": " All M 6+  ", "value": "all"},
                            {"label": " M 6.0-6.9  ", "value": "6.0-6.9"},
                            {"label": " M 7.0+", "value": "7.0+"}
                        ]
                    )
                ),
            ]
        ),

        panel(
            dcc.Graph(id="losses"),
            html.P(id="losses-caption", className="caption"),
            cls="chart-card"
        ),

        pager("/losses"),
    ])

def cleaning_page():
    return html.Div([
        header("PLOT 6 · DATA QUALITY", "The data before and after cleaning", "Pick a data source below to inspect the live interactive audit of how raw records were deduplicated, reconciled, and validated."),
        html.Div(className="controls", children=[
            control("Source", dcc.RadioItems(id="src-pick", value="usgs", inline=True, options=[{"label": f" {v}   ", "value": k} for k, v in SOURCES.items()])),
        ]),
        panel(dcc.Graph(id="sources-graph"), html.P(id="sources-caption", className="caption"), cls="chart-card"),
        pager("/cleaning"),
    ])

def time_page():
    return html.Div([
        header("PLOT 7 · OVER TIME", "Deaths over time: is the income gap closing?", "Left: deaths per year, stacked by income group. Right: the same countries per 5-year period, so a change means the same kind of country fared better or worse, not that the mix changed."),
        html.Div(className="controls", children=[
            control("Right panel", dcc.RadioItems(id="time-metric", value="share", inline=True, options=[{"label": f" {v}   ", "value": k} for k, v in DENS_METRICS.items()])),
        ]),
        panel(dcc.Graph(id="time-graph"), html.P(id="time-caption", className="caption"), cls="chart-card"),
        pager("/time"),
    ])

def predictor_page():
    gcols = [k for k in GDP_CHOICES if k in df.columns]
    return html.Div([
        header("PLOT 8 · WHAT IF", "Death predictor for earthquakes on land", "Pick an earthquake (click the map or search the list) or set your own scenario. A two-part model fitted on 2000-2025 gives the chance that anyone dies and, if so, the typical toll; move the GDP slider to see what income alone changes at the same magnitude and depth."),
        html.Div(className="controls", children=[
            control("Earthquake (type to search)", dcc.Dropdown(id="pred-event", clearable=True, placeholder="Search a country or year...", options=pred_event_options("deaths_heatmap", [2000, 2025], gcols[0]) if gcols else []), "520px"),
            control("GDP measure", dcc.RadioItems(id="pred-gdp-col", value=gcols[0] if gcols else None, options=[{"label": f" {GDP_CHOICES[k]}", "value": k} for k in gcols]), "420px"),
        ]),
        html.Div(className="pred-grid", children=[
            panel(dcc.Graph(id="pred-map", config={"displayModeBar": False}), cls="chart-card"),
            panel(
                control("Magnitude (Mw)", dcc.Slider(id="pred-mag", min=6.0, max=9.1, step=0.1, value=7.0, marks={m: f"{m:.0f}" for m in (6, 7, 8, 9)}, tooltip={"placement": "bottom", "always_visible": True})),
                control("Depth (km)", dcc.Slider(id="pred-depth", min=0, max=300, step=1, value=15, marks={k: str(k) for k in (0, 70, 150, 300)}, tooltip={"placement": "bottom", "always_visible": True})),
                control("GDP per capita (log scale; the cards show the dollar value)", dcc.Slider(id="pred-gdp", min=2.5, max=5.1, step=0.01, value=round(float(np.log10(5000)), 2), marks={3: "$1k", 3.5: "$3.2k", 4: "$10k", 4.5: "$32k", 5: "$100k"})),
                html.Div(id="pred-cards", className="kpi-grid pred-cards"), cls="chart-card"),
        ]),
        panel(dcc.Graph(id="pred-curve"), html.P(id="pred-caption", className="caption"), cls="chart-card"),
        pager("/predictor"),
    ])

def methodology_page():
    return html.Div([
        header("PROJECT INFORMATION", "Methodology", "Sources, cleaning rules, models and limitations."),
        panel(html.H3("01 · Sources"), html.Ul([
            html.Li("USGS ComCat FDSN event API (earthquake.usgs.gov/fdsnws/event/1/), M 6.0+, 2000-2025 (3,360 deduplicated mainshocks)."),
            html.Li("NOAA NCEI Significant Earthquake Database (ngdc.noaa.gov/hazel), official event fatality counts."),
            html.Li("Wikipedia 'List of earthquakes in YYYY' pages (MediaWiki), monthly tables and historical fatality records."),
            html.Li("World Bank World Development Indicators API (api.worldbank.org/v2), GDP per capita (PPP, constant 2021 international $), population density, region and income tier."),
            html.Li("All four sources are merged into clean/earthquakes_master_2000_2025.parquet and loaded into a local parameterized SQLite database (earthquakes_app.db)."),
        ])),
        panel(html.H3("02 · Deaths columns and hierarchical reconciliation"), html.P("deaths_final (sidebar: 'Reconciled') resolves NOAA NCEI and Wikipedia fatality counts using a 3-tier hierarchical rule: (1) NOAA NCEI is prioritized as the primary scientific authority when both sources report and agree within 25% or 100 deaths; (2) Wikipedia is used as a fallback when NOAA reports 0 or is missing; and (3) the rounded integer midpoint is used ONLY when both sources report vast discrepancies (>25% relative difference and >100 deaths, such as Haiti 2010: NOAA 316,000 vs. Wikipedia 208,000 -> 262,000). deaths_disagree flags all events where the two sources differ. Zeros mean 'no reported deaths'.")),
        panel(html.H3("03 · Socio-economic and income indicators"), html.P("Each onshore earthquake is joined with its country's World Bank income group, GDP per capita (PPP, constant 2021 $), and population density in the exact year the earthquake occurred (e.g., Haiti 2010 is evaluated as Low income). Non-WDI territories (such as Taiwan and New Caledonia) are assigned their historical income tier for categorical charts, while missing continuous GDP values (MAR / structural WDI omissions) and 1,406 offshore marine earthquakes are excluded from continuous regressions and noted explicitly.")),
        panel(html.H3("04 · Econometric models"), html.P("Plot 3 (Exposure): Evaluates nested OLS models on log10(1 + deaths) across all onshore earthquakes, comparing the explanatory power (R2) of physical controls (magnitude and log10(1 + depth)) against national population density and regional fixed effects."), html.P("Plot 4 (Income Effect): Fits ordinary least squares on log10(deaths) among deadly shallow onshore earthquakes (<=70 km), controlling for magnitude and log10(1 + depth), with 95% confidence intervals computed via a 300-iteration country-cluster bootstrap."), html.P("Plot 8 (Death Predictor): Fits a two-part hurdle model on onshore earthquakes with GDP data. Part 1 is a logistic regression for the probability of at least one death; Part 2 is OLS on log10(deaths) among deadly events (using observed 10th-90th percentile residuals for the 80% range and Duan smearing for expected values). When a specific historical earthquake is clicked, 'This earthquake, your settings' anchors on that event's actual toll and scales it by the OLS partial slopes to preserve local fixed effects (urbanization, building codes, tsunami).")),
        panel(html.H3("05 · Spatiotemporal deduplication and depth artifacts"), html.P("Raw USGS M 6.0+ shocks within +/-2 days and 150 km (Haversine great-circle distance) of a larger mainshock are collapsed into the primary mainshock (cluster_shock_count records the number of merged shocks, preventing 4.8x fatality inflation in sequences like Turkey-Syria 2023). Mainshocks assigned fixed 10.0 km or 33.0 km default depths by seismic networks (788 events, 23.5%) are flagged via is_default_depth.")),
        panel(html.H3("06 · Limitations"), html.Ul([
            html.Li("1,406 offshore marine earthquakes do not intersect land country polygons and are excluded from national income and density regressions."),
            html.Li("National population density is a country-wide average that masks whether an earthquake epicentre struck a dense city or an uninhabited desert."),
            html.Li("World Bank WDI does not report continuous GDP per capita for Taiwan (n=45) or conflict-gap years in Venezuela (n=6); these are left un-imputed (MAR) rather than fabricating macroeconomic data."),
            html.Li("Earthquake mortality is heavily fat-tailed (a handful of mega-disasters dominate decadal totals), and observational regressions measure conditional association rather than strict causality."),
        ])),
        pager("/methodology"),
    ])

ROUTES = {"/": overview_page, "/map": map_page, "/physics": physics_page, "/density": density_page, "/income": income_page, "/losses": losses_page, "/cleaning": cleaning_page, "/time": time_page, "/predictor": predictor_page, "/methodology": methodology_page}

sidebar = html.Div(className="sidebar", children=[
    html.Div(className="brand", children=[html.H3("QUAKE", className="brand-name"), html.P("DEATHS · MAGNITUDE · INCOME · V2", className="brand-subtitle")]),
    html.Hr(),
    html.P("ANALYSIS", className="nav-heading"),
    html.Div(className="navigation", children=[dcc.Link([html.Span(icon, className="nav-icon"), html.Span(name)], href=path, id={"type": "nav", "path": path}, className="nav-link") for path, name, icon in PAGES]),
    html.Hr(),
    html.P("APPLIES TO EVERY PAGE", className="nav-heading"),
    html.Label("Deaths measure", className="side-label"),
    html.Div(className="side-dd", children=dcc.Dropdown(id="deaths-col", value="deaths_heatmap", clearable=False, searchable=False, options=[{"label": {"deaths_heatmap": "Reconciled (deaths_final)", "deaths": "NOAA only", "wiki_deaths": "Wikipedia only"}[k], "value": k} for k in DEATH_COLUMNS if k in df.columns])),
    html.Label("Years", className="side-label"),
    dcc.RangeSlider(id="years", min=2000, max=2025, step=1, value=[2000, 2025], allowCross=False, marks={y: {"label": str(y), "style": {"color": "#dfe6ee"}} for y in (2000, 2005, 2010, 2015, 2020, 2025)}, tooltip={"placement": "bottom"}),
    dcc.Checklist(id="greyscale", value=[], className="side-check", options=[{"label": " Greyscale preview (print check)", "value": "grey"}]),
    html.Div(className="sidebar-footer", children=[html.P("DATA PERIOD"), html.H5("2000 — 2025"), html.P("M 6+ · USGS · NOAA · WIKIPEDIA · WORLD BANK")]),
])

# FIX: Re-embedded the complete CSS block so the sidebar actually styles correctly
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Earthquake deaths and income (v2)")

app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --sidebar-bg: #1f2933; --sidebar-text: #e6e9ed; --sidebar-hover: #323f4b;
            --brand: #d95f02; --main-bg: #f5f7fa; --card-bg: #ffffff;
            --text-main: #1f2933; --text-muted: #6b7785;
            --border: #e6e9ed; --border-dark: #cbd2d9;
        }
        body { margin: 0; font-family: 'Inter', sans-serif; background-color: var(--main-bg); color: var(--text-main); }
        .app-container { display: flex; min-height: 100vh; }
        .sidebar { width: 280px; background-color: var(--sidebar-bg); color: var(--sidebar-text); padding: 24px 20px; display: flex; flex-direction: column; flex-shrink: 0; }
        .brand-name { font-size: 24px; font-weight: 700; margin: 0 0 4px 0; color: #fff; letter-spacing: 1px; }
        .brand-subtitle { font-size: 11px; margin: 0; color: #9aa5b1; letter-spacing: 0.5px; }
        .sidebar hr { border-color: #3e4c59; margin: 24px 0; }
        .nav-heading { font-size: 11px; font-weight: 600; color: #9aa5b1; letter-spacing: 0.5px; margin: 0 0 12px 0; }
        .nav-link { display: flex; align-items: center; padding: 10px 12px; margin: 0 -12px 4px; color: var(--sidebar-text); text-decoration: none; border-radius: 6px; font-size: 14px; transition: background 0.15s; }
        .nav-link:hover { background-color: var(--sidebar-hover); }
        .nav-link.active { background-color: rgba(217, 95, 2, 0.15); color: var(--brand); font-weight: 500; }
        .nav-icon { width: 20px; text-align: center; margin-right: 12px; font-size: 14px; opacity: 0.7; }
        .nav-link.active .nav-icon { opacity: 1; }
        .side-label { font-size: 13px; font-weight: 500; margin-bottom: 8px; display: block; }
        .side-dd { margin-bottom: 24px; }
        .side-dd { margin-bottom: 24px; color: #1f2933; }
        .rc-slider-track { background-color: var(--brand) !important; }
        .rc-slider-handle { border-color: var(--brand) !important; }
        .side-check { margin-top: 32px; font-size: 13px; }
        .sidebar-footer { margin-top: auto; padding-top: 32px; font-size: 11px; color: #7b8794; }
        .sidebar-footer h5 { color: #d9e2ec; margin: 4px 0; font-size: 16px; }
        
        .main-content { flex: 1; padding: 32px 48px; max-width: 1400px; min-width: 0; }
        .main-content.greyscale { filter: grayscale(100%); }
        .page-header { margin-bottom: 32px; }
        .eyebrow { font-size: 12px; font-weight: 600; color: var(--brand); text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 8px 0; }
        .page-header h1 { font-size: 32px; font-weight: 700; margin: 0 0 12px 0; color: var(--text-main); }
        .page-description { font-size: 16px; color: var(--text-muted); max-width: 900px; line-height: 1.5; margin: 0; }
        
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 32px; }
        .kpi-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .kpi-title { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 8px 0; }
        .kpi-value { font-size: 28px; font-weight: 700; margin: 0 0 8px 0; color: var(--text-main); }
        .kpi-note { font-size: 13px; color: var(--text-muted); margin: 0; line-height: 1.4; }
        
        .section-title { font-size: 20px; font-weight: 600; margin: 40px 0 20px 0; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
        
        .q-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px; }
        .q-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); display: flex; flex-direction: column; }
        .q-tag { font-size: 11px; font-weight: 600; color: var(--text-muted); letter-spacing: 0.5px; margin: 0 0 12px 0; }
        .q-card h4 { font-size: 18px; font-weight: 600; margin: 0 0 16px 0; }
        .q-takeaway { font-size: 14px; line-height: 1.5; margin: 0 0 16px 0; flex: 1; }
        .q-hides { font-size: 13px; color: var(--text-muted); background: #f8f9fa; padding: 12px; border-radius: 4px; margin: 0 0 16px 0; }
        .q-hides b { color: var(--text-main); }
        
        .card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 24px; border-top: 4px solid var(--border-dark); }
        .card h4 { font-size: 16px; font-weight: 600; margin: 0 0 12px 0; }
        .card p { font-size: 14px; line-height: 1.5; color: var(--text-muted); margin: 0; }
        .card.c1 { border-top-color: #2171b5; }
        .card.c2 { border-top-color: #6baed6; }
        .card.c3 { border-top-color: #9ecae1; }
        .card.c4 { border-top-color: #c6dbef; }
        
        .content-panel { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 32px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .content-panel h3 { font-size: 18px; font-weight: 600; margin: 0 0 16px 0; }
        .content-panel p { font-size: 15px; line-height: 1.6; margin: 0 0 16px 0; }
        .content-panel p:last-child { margin-bottom: 0; }
        .content-panel ul { margin: 0; padding-left: 20px; }
        .content-panel li { font-size: 15px; line-height: 1.6; margin-bottom: 8px; }
        
        .controls { display: flex; flex-wrap: wrap; gap: 24px; margin-bottom: 24px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .control { display: flex; flex-direction: column; gap: 8px; }
        .control label { font-size: 13px; font-weight: 600; color: var(--text-muted); }
        
        .chart-card { padding: 20px; }
        .caption { font-size: 14px; line-height: 1.6; color: var(--text-muted); margin: 16px 0 0 0; padding-top: 16px; border-top: 1px solid var(--border); }
        
        .play-row { display: flex; gap: 24px; align-items: center; margin-bottom: 24px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; }
        .nav-btn { background: #fff; border: 1px solid var(--border-dark); padding: 8px 16px; border-radius: 4px; font-size: 14px; font-weight: 500; cursor: pointer; transition: all 0.2s; }
        .nav-btn:hover { background: #f8f9fa; border-color: #9aa5b1; }
        
        .pred-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 24px; margin-bottom: 24px; }
        .pred-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 0; }
        .pred-cards .kpi-card { padding: 16px; }
        .pred-cards .kpi-value { font-size: 24px; }
        
        .carousel-grid { display: grid; grid-template-columns: 1fr; gap: 24px; margin-bottom: 24px; }
        .carousel { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .carousel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
        .carousel-head h4 { font-size: 16px; font-weight: 600; margin: 0; }
        .carousel-nav { display: flex; align-items: center; gap: 16px; }
        .car-count { font-size: 14px; font-weight: 500; color: var(--text-muted); min-width: 48px; text-align: center; }
        .car-img { width: 100%; height: auto; border: 1px solid var(--border); border-radius: 4px; }
        .car-note { font-size: 14px; line-height: 1.5; margin: 16px 0 0 0; text-align: center; color: var(--text-muted); }
        
        .pager { display: flex; justify-content: space-between; margin-top: 48px; padding-top: 24px; border-top: 1px solid var(--border); }
        .pager-link { display: inline-block; color: var(--brand); text-decoration: none; font-weight: 600; font-size: 15px; }
        .pager-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>"""

app.layout = dbc.Container(fluid=True, className="p-0 m-0", children=[
    html.Div(className="app-container", children=[dcc.Location(id="url", refresh=False), sidebar, html.Main(id="main", className="main-content", children=html.Div(id="page-content"))])
])

@app.server.route("/plots/path:name")
def serve_plot(name):
    for d in (PROJECT_DIR / "reports" / "figures", APP_DIR / "reports" / "figures", PROJECT_DIR / "plots"):
        if (d / name).is_file():
            return send_from_directory(str(d.resolve()), name)
        abort(404)

# =====================================================================================================================================
# CALLBACKS
# =====================================================================================================================================
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname): return ROUTES.get(pathname, overview_page)()

@app.callback(Output({"type": "nav", "path": ALL}, "className"), Input("url", "pathname"))
def highlight_nav(pathname): return ["nav-link active" if p == (pathname if pathname in ROUTES else "/") else "nav-link" for p, _, _ in PAGES]

@app.callback(Output("main", "className"), Input("greyscale", "value"))
def toggle_greyscale(value): return "main-content greyscale" if value else "main-content"

@app.callback(Output("overview-body", "children"), Input("deaths-col", "value"), Input("years", "value"))
def update_overview(deaths_col, years): return overview_body(deaths_col, years)

def window(time_mode, years, year_step):
    lo, hi = int(years[0]), int(years[1]); step = min(max(int(year_step), lo), hi)
    if time_mode == "year": return (step, step)
    if time_mode == "cumulative": return (lo, step)
    return (lo, hi)

def yearly_bar(df_in, deaths_col, time_mode, year_step, years):
    q = filter_years(df_in, years)
    per_year = q.groupby("year")[deaths_col].sum().reindex(range(2000, 2026), fill_value=0)
    lo, hi = window(time_mode, years, year_step)
    on = (per_year.index >= lo) & (per_year.index <= hi)
    fig = go.Figure(go.Bar(x=per_year.index, y=per_year.values + 1, marker_color=np.where(on, ORANGE, LIGHT_GREY), customdata=per_year.values, hovertemplate="%{x}: <b>%{customdata:,.0f} deaths</b><extra></extra>"))
    fig.update_layout(margin=dict(l=60, r=10, t=40, b=40), xaxis=dict(title="Year", dtick=5), yaxis=dict(title="Deaths + 1 (log scale)", type="log"), bargap=0.15)
    return style_chart(fig, 240)

@app.callback(Output("year-step", "value"), Output("year-step", "min"), Output("year-step", "max"), Output("year-step", "marks"), Output("ticker", "disabled"), Output("play", "children"), Output("time-mode", "value"), Input("play", "n_clicks"), Input("ticker", "n_intervals"), Input("years", "value"), Input("year-step", "value"), State("ticker", "disabled"), State("time-mode", "value"))
def control_time(_, __, years, step, stopped, mode):
    lo, hi = int(years[0]), int(years[1]); step = min(max(int(step or lo), lo), hi); trig = ctx.triggered_id
    if trig == "play":
        if stopped:
            stopped = False; step = lo if step >= hi else step; mode = "cumulative" if mode == "range" else mode
        else: stopped = True
    elif trig == "ticker" and not stopped:
        step += 1
        if step >= hi: step, stopped = hi, True
    elif trig == "year-step" and mode == "range": mode = "cumulative"
    elif trig == "years": stopped = True
    marks = {y: str(y) for y in sorted({lo, hi, *range(lo + (5 - lo % 5) % 5, hi + 1, 5)})}
    return step, lo, hi, marks, stopped, ("▶ Play" if stopped else "❚❚ Pause"), mode

_BLOB_CACHE = {}

@app.callback(Output("blob-map", "figure"), Output("blob-caption", "children"), Output("deaths-by-year", "figure"), Input("deaths-col", "value"), Input("years", "value"), Input("eps", "value"), Input("view", "value"), Input("colour-by", "value"), Input("time-mode", "value"), Input("year-step", "value"), Input("show-zero", "value"))
def update_blob(deaths_col, years, eps, view, colour_by, time_mode, year_step, show_zero):
    yrs = window(time_mode, years, year_step); zero = bool(show_zero)
    key = (deaths_col, tuple(years), eps if view == "blobs" else None, view, colour_by if view == "points" else None, time_mode, yrs, zero)
    if key not in _BLOB_CACHE:
        if len(_BLOB_CACHE) > 400: _BLOB_CACHE.clear()
        animated = time_mode != "range"
        q = filter_years(df, yrs)
        shown = q[deaths_col].sum()
        title = None
        if view == "points":
            q_years = filter_years(df, tuple(years))
            ref = q_years[deaths_col].max() if animated else None
            fig, cap = make_points_map(df, deaths_col, yrs, colour_by, ref, title, zero), points_caption(df, deaths_col, yrs, colour_by, zero)
        else:
            ref = None
            if animated:
                full = events_with_deaths(df, deaths_col, tuple(years))
                ref = cluster_deaths(full, deaths_col, eps)[1]["deaths"].max() if len(full) else None
            fig, cap = make_blob_map(df, deaths_col, yrs, eps, ref, title, zero), blob_caption(df, deaths_col, yrs, eps, zero)
        _BLOB_CACHE[key] = (fig, cap, yearly_bar(df, deaths_col, time_mode, year_step, years))
    return _BLOB_CACHE[key]

@app.callback(Output("physics-graph", "figure"), Output("physics-caption", "children"), Input("deaths-col", "value"), Input("years", "value"), Input("max-depth", "value"), Input("show-fixed", "value"))
def update_physics(deaths_col, years, max_depth, show_fixed):
    yrs, sf = tuple(years), show_fixed == "show"
    return make_mag_depth(df, deaths_col, yrs, max_depth, sf), mag_depth_caption(df, deaths_col, yrs, max_depth, sf)

@app.callback(Output("density-graph", "figure"), Output("density-caption", "children"), Input("deaths-col", "value"), Input("years", "value"), Input("dens-region", "value"), Input("dens-within", "value"), Input("dens-shallow", "value"), Input("dens-mode", "value"), Input("dens-metric", "value"))
def update_density(deaths_col, years, region_pick, within, shallow, mode, metric):
    yrs, sh = tuple(years), shallow == "shallow"
    return make_density(df, deaths_col, yrs, region_pick, within == "within", sh, mode, metric), density_caption(df, deaths_col, yrs, sh, mode, metric)

@app.callback(Output("income-graph", "figure"), Output("income-caption", "children"), Input("deaths-col", "value"), Input("years", "value"), Input("gdp-col", "value"), Input("gdp-shallow", "value"))
def update_income(deaths_col, years, gdp_col, shallow):
    gdp_col = valid_gdp(gdp_col); yrs, sh = tuple(years), shallow == "shallow"
    return make_gdp(df, deaths_col, yrs, gdp_col, sh), gdp_caption(df, deaths_col, yrs, gdp_col, sh)

@app.callback(Output("losses", "figure"), Output("losses-caption", "children"), Input("deaths-col", "value"), Input("years", "value"), Input("loss-group", "value"), Input("loss-mag", "value"))
def update_losses(deaths_col, years, group_by, mag_band):
    yrs = tuple(years)
    return make_losses_plot(df, deaths_col, yrs, group_by, mag_band), losses_caption(df, deaths_col, yrs, group_by, mag_band)

@app.callback(Output("sources-graph", "figure"), Output("sources-caption", "children"), Input("years", "value"), Input("src-pick", "value"))
def update_sources(years, source):
    yrs = tuple(years)
    return make_sources(df, yrs, source), sources_caption(df, yrs, source)

@app.callback(Output("time-graph", "figure"), Output("time-caption", "children"), Input("deaths-col", "value"), Input("years", "value"), Input("time-metric", "value"))
def update_time(deaths_col, years, metric):
    yrs = tuple(years)
    return make_time(df, deaths_col, yrs, metric), time_caption(df, deaths_col, yrs, metric)

def valid_gdp(gdp_col): return gdp_col if gdp_col in df.columns else next((k for k in GDP_CHOICES if k in df.columns), gdp_col)

@app.callback(Output("pred-event", "options"), Input("deaths-col", "value"), Input("years", "value"), Input("pred-gdp-col", "value"))
def pred_event_options(deaths_col, years, gdp_col):
    gdp_col = valid_gdp(gdp_col); q = onshore_events(df, tuple(years), gdp_col).sort_values(deaths_col, ascending=False)
    return [{"label": event_label(r, deaths_col), "value": int(i)} for i, r in q.iterrows()]

@app.callback(Output("pred-event", "value"), Output("pred-mag", "value"), Output("pred-depth", "value"), Output("pred-gdp", "value"), Input("pred-map", "clickData"), Input("pred-event", "value"), Input("pred-gdp-col", "value"), Input("years", "value"), State("deaths-col", "value"))
def pred_load_event(click, event_id, gdp_col, years, deaths_col):
    gdp_col = valid_gdp(gdp_col); trig = ctx.triggered_id; q = onshore_events(df, tuple(years), gdp_col)
    if trig == "pred-map" and click and click.get("points"):
        cdat = click["points"][0].get("customdata")
        if cdat is None: return (no_update,) * 4
        event_id = int(cdat[0])
    if event_id is None or event_id not in q.index:
        if trig is None and len(q): event_id = int(q[deaths_col].idxmax())
        elif event_id is not None: return None, no_update, no_update, no_update
        else: return no_update, no_update, no_update, no_update
    r = q.loc[event_id]
    return event_id, round(float(np.clip(r["mag"], 6.0, 9.1)), 1), int(np.clip(round(r["depth"]), 0, 300)), round(float(np.clip(np.log10(r[gdp_col]), 2.5, 5.1)), 2)

@app.callback(Output("pred-map", "figure"), Input("deaths-col", "value"), Input("years", "value"), Input("pred-gdp-col", "value"), Input("pred-event", "value"))
def pred_map(deaths_col, years, gdp_col, event_id):
    gdp_col = valid_gdp(gdp_col); return make_pred_map(df, deaths_col, tuple(years), gdp_col, event_id)

@app.callback(Output("pred-cards", "children"), Output("pred-curve", "figure"), Output("pred-caption", "children"), Input("deaths-col", "value"), Input("years", "value"), Input("pred-gdp-col", "value"), Input("pred-mag", "value"), Input("pred-depth", "value"), Input("pred-gdp", "value"), Input("pred-event", "value"))
def update_predictor(deaths_col, years, gdp_col, mag, depth, lgdp, event_id):
    gdp_col = valid_gdp(gdp_col); fit = predictor_fit(df, deaths_col, tuple(years), gdp_col); gdp = float(10 ** float(lgdp))
    event = df.loc[event_id] if event_id is not None and event_id in df.index and pd.notna(df.at[event_id, gdp_col]) else None
    return pred_cards(fit, float(mag), float(depth), gdp, event, deaths_col), make_pred_curve(fit, float(mag), float(depth), gdp, event, deaths_col), pred_caption(fit, deaths_col)

@app.callback(Output({"type": "car-idx", "group": MATCH}, "data"), Input({"type": "car-prev", "group": MATCH}, "n_clicks"), Input({"type": "car-next", "group": MATCH}, "n_clicks"), State({"type": "car-idx", "group": MATCH}, "data"), prevent_initial_call=True)
def step_carousel(_, __, idx):
    group = ctx.triggered_id["group"]; n = max(len(plot_files(group)), 1)
    return (idx + (1 if ctx.triggered_id["type"] == "car-next" else -1)) % n

@app.callback(Output({"type": "car-img", "group": MATCH}, "src"), Output({"type": "car-note", "group": MATCH}, "children"), Output({"type": "car-count", "group": MATCH}, "children"), Input({"type": "car-idx", "group": MATCH}, "data"), State({"type": "car-idx", "group": MATCH}, "id"))
def show_carousel(idx, cid):
    files = plot_files(cid["group"])
    if not files: return None, f"No plots found in {PLOTS_DIR.name}/ for this source. Run the visualisation notebook first.", "0 / 0"
    name = files[idx % len(files)]
    return f"/plots/{name}", f"{pretty(name)}  ({name})", f"{idx % len(files) + 1} / {len(files)}"

if __name__ == "__main__":
    (app.run if hasattr(app, "run") else app.run_server)(debug=True)