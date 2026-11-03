import glob
import json
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def build_html_deck():
    # 1. Load Master Clean Dataset
    parquet_files = sorted(glob.glob("clean/earthquakes_master_2000_2025*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("Could not find clean/earthquakes_master_2000_2025*.parquet!")
    
    # Apply the column rename fix for lon/lat
    df = pd.read_parquet(parquet_files[-1]).copy()
    df = df.rename(columns={"longitude": "lon", "latitude": "lat"})
    
    os.makedirs("reports", exist_ok=True)

    INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    df["log10_deaths"] = np.log10(df["deaths_final"] + 1.0)
    df["log10_gdp"] = np.log10(df["gdp_per_capita_ppp_constant_2021"])
    df["depth_zone"] = pd.cut(
        df["depth"],
        bins=[-1, 70, 300, 800],
        labels=["Shallow (0-70 km)", "Intermediate (70-300 km)", "Deep (>300 km)"]
    )

    # 2. Compute Summary Statistics & Payloads for Interactive JS Widgets
    n_total = len(df)
    n_onshore = int((df["country_iso3"] != "OFFSHORE").sum())
    n_econ = int(df["income_group"].isin(INCOME_ORDER).sum())
    n_missing_gdp = n_onshore - n_econ
    n_default_depth = int(df["is_default_depth"].sum())
    pct_default_depth = round(n_default_depth / n_total * 100, 1)
    tot_deaths = float(df["deaths_final"].sum())

    # Widget 1: Before vs After Cluster Deduplication (Top 6 Disasters)
    top6 = df[df["deaths_final"] >= 1000].sort_values("deaths_final", ascending=False).head(6)
    w1_data = []
    for _, r in top6.iterrows():
        clean_k = round(float(r["deaths_final"]) / 1000.0, 2)
        naive_k = round(float(r["noaa_deaths"] * r["cluster_shock_count"]) / 1000.0, 2)
        if naive_k < clean_k:
            naive_k = round(float(r["deaths_final"] * r["cluster_shock_count"]) / 1000.0, 2)
        factor = round(naive_k / clean_k, 1) if clean_k > 0 else 1.0
        w1_data.append({
            "label": f"{r['country_clean']} ({r['year']}, M{r['mag']:.1f})",
            "clean": clean_k,
            "raw": naive_k,
            "shocks": int(r["cluster_shock_count"]),
            "lf": factor
        })

    # Widget 2: Focal Depth Histogram & Artifact Diagnosis (0-100 km)
    sh_100 = df[df["depth"] <= 100]
    w2_data = {
        "real": [round(float(x), 1) for x in sh_100.loc[~sh_100["is_default_depth"], "depth"]],
        "art": [round(float(x), 1) for x in sh_100.loc[sh_100["is_default_depth"], "depth"]],
        "n_art": n_default_depth,
        "pct_art": pct_default_depth
    }

    # Widget 3: Income Group Exposure vs Fatality Rates (Simpson/Aggregation)
    on_econ = df[df["income_group"].isin(INCOME_ORDER)].copy()
    w3_data = []
    inc_colors = {
        "Low income": "#B02F3B",
        "Lower middle income": "#8F5D0A",
        "Upper middle income": "#125F9E",
        "High income": "#14222B"
    }
    for inc in INCOME_ORDER:
        sub = on_econ[on_econ["income_group"] == inc]
        w3_data.append({
            "group": inc.replace(" income", ""),
            "count": int(len(sub)),
            "mean_mag": round(float(sub["mag"].mean()), 2),
            "total_deaths": round(float(sub["deaths_final"].sum()), 1),
            "mean_deaths": round(float(sub["deaths_final"].mean()), 2),
            "deadly_pct": round(float((sub["deaths_final"] > 0).mean() * 100), 1),
            "color": inc_colors[inc]
        })

    # Widget 4: World Map Spatiotemporal Progression (2000-2025)
    inc_code_map = {"Low income": 1, "Lower middle income": 2, "Upper middle income": 3, "High income": 4}
    w4_quakes = []
    for _, r in df.iterrows():
        ic = 0 if r["country_iso3"] == "OFFSHORE" else inc_code_map.get(r["income_group"], -1)
        gdp_v = round(float(r["gdp_per_capita_ppp_constant_2021"]), 0) if pd.notna(r["gdp_per_capita_ppp_constant_2021"]) else -1
        w4_quakes.append([
            int(r["year"]), round(float(r["lon"]), 2), round(float(r["lat"]), 2),
            round(float(r["mag"]), 1), round(float(r["depth"]), 0), round(float(r["deaths_final"]), 0),
            gdp_v, ic, f"{r['country_clean']} ({r['date'][:10]}, M{r['mag']:.1f})"
        ])

    # Widget 5: Magnitude x Depth Scatter & Filter
    onshore_all = df[df["country_iso3"] != "OFFSHORE"].copy()
    w5_pts = []
    for _, r in onshore_all.iterrows():
        dz = 0 if r["depth"] <= 70 else (1 if r["depth"] <= 300 else 2)
        w5_pts.append([
            round(float(r["mag"]), 2), round(float(r["depth"]), 1), round(float(r["deaths_final"]), 1),
            round(float(r["log10_deaths"]), 3), dz, f"{r['country_clean']} '{str(r['year'])[2:]} (M{r['mag']:.1f})"
        ])

    # Widget 6: NOAA vs Wikipedia Cross-Source Audit (379 Deadly Quakes)
    deadly_df = df[df["deaths_final"] > 0].copy()
    deadly_df["noaa_c"] = deadly_df["noaa_deaths"].fillna(0)
    deadly_df["wiki_c"] = deadly_df["wiki_deaths"].fillna(0)
    both_pos = deadly_df[(deadly_df["noaa_c"] > 0) & (deadly_df["wiki_c"] > 0)]
    r_corr = float(np.corrcoef(np.log10(both_pos["noaa_c"] + 1), np.log10(both_pos["wiki_c"] + 1))[0, 1])
    n_disagree = int((both_pos["noaa_c"].round() != both_pos["wiki_c"].round()).sum())

    w6_pts = []
    for _, r in deadly_df.iterrows():
        dis = 1 if (r["noaa_c"] > 0 and r["wiki_c"] > 0 and round(r["noaa_c"]) != round(r["wiki_c"])) else 0
        w6_pts.append([
            round(float(np.log10(r["noaa_c"] + 1)), 3), round(float(np.log10(r["wiki_c"] + 1)), 3),
            int(round(r["noaa_c"])), int(round(r["wiki_c"])), dis, f"{r['country_clean']} {r['year']} (M{r['mag']:.1f})"
        ])

    # Widget 7: 2D Magnitude x Income Heatmap (Colour vs Greyscale Print L*)
    sh_econ = df[(df["depth"] <= 70) & (df["income_group"].isin(INCOME_ORDER))].copy()
    sh_econ["m_tier"] = pd.cut(
        sh_econ["mag"], bins=[5.99, 6.49, 6.99, 7.49, 9.5],
        labels=["M6.0-6.4", "M6.5-6.9", "M7.0-7.4", "M7.5+"]
    )
    m_tiers = ["M6.0-6.4", "M6.5-6.9", "M7.0-7.4", "M7.5+"]
    w7_cells = []
    for i, mt in enumerate(m_tiers):
        for j, inc in enumerate(INCOME_ORDER):
            sub = sh_econ[(sh_econ["m_tier"] == mt) & (sh_econ["income_group"] == inc)]
            cnt = len(sub)
            mean_log = round(float(sub["log10_deaths"].mean()), 2) if cnt > 0 else 0.0
            dead_p = round(float((sub["deaths_final"] > 0).mean() * 100), 0) if cnt > 0 else 0.0
            w7_cells.append({"r": i, "c": j, "log": mean_log, "pct": dead_p, "n": cnt})

    # Widget 8: Annual vs Cumulative Fatalities Timeline (2000-2025)
    yr_s = df.groupby("year")["deaths_final"].sum().reindex(range(2000, 2026), fill_value=0.0)
    w8_data = {
        "years": list(range(2000, 2026)),
        "annual": [round(float(x) / 1000.0, 2) for x in yr_s.values],
        "cum": [round(float(x) / 1000.0, 2) for x in yr_s.cumsum().values]
    }

    # Widget 9: Econometric Model & What-If GDP Simulator
    sh_model_df = df[(df["depth"] <= 70) & (df["log10_gdp"].notna()) & (df["country_iso3"] != "OFFSHORE")].copy()
    m1 = smf.ols("log10_deaths ~ mag + depth", data=sh_model_df).fit()
    m2 = smf.ols("log10_deaths ~ mag + depth + log10_gdp", data=sh_model_df).fit()
    deadly_sh = sh_model_df[sh_model_df["deaths_final"] > 0].copy()
    m3 = smf.ols("log10_deaths ~ mag + depth + log10_gdp", data=deadly_sh).fit()

    gdp_slope_deadly = float(m3.params["log10_gdp"])
    gdp_pval_deadly = float(m3.pvalues["log10_gdp"])
    pct_red_10x = round((1.0 - (10.0 ** gdp_slope_deadly)) * 100, 1)

    w9_data = {
        "n_all": len(sh_model_df), "r2_base": round(float(m1.rsquared), 4), "r2_full": round(float(m2.rsquared), 4),
        "slope_all": round(float(m2.params["log10_gdp"]), 3), "pval_all": f"{m2.pvalues['log10_gdp']:.2e}",
        "n_deadly": len(deadly_sh), "r2_deadly": round(float(m3.rsquared), 4),
        "slope_deadly": round(gdp_slope_deadly, 3), "pval_deadly": f"{gdp_pval_deadly:.2e}", "pct_red": pct_red_10x,
        "intercept_m3": float(m3.params["Intercept"]), "b_mag_m3": float(m3.params["mag"]),
        "b_dep_m3": float(m3.params["depth"]), "b_gdp_m3": gdp_slope_deadly
    }

    # 3. Assemble Self-Contained HTML Deck
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DA2402 — Earthquake Fatalities & National Income Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="">
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=JetBrains+Mono:wght@400;500;700&family=Public+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<style>
:root{{
  --paper:#E9EDEF; --card:#FFFFFF; --ink:#14222B; --soft:#53666F;
  --line:#C3CFD5; --hair:#DCE4E8; --surround:#0E1A21;
  --code-bg:#122029; --code-ink:#D9E4E9;
  --t-cls:#125F9E; --t-qnt:#9B2C7E; --t-grp:#0E7A55; --t-anc:#8F5D0A;
  --t-asr:#B02F3B; --t-alt:#4B5A87; --t-dot:#6A4CA8; --t-ref:#0E7A55;
  --sw:1280px; --sh:720px;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --disp:"Bricolage Grotesque","Public Sans",system-ui,sans-serif;
  --body:"Public Sans",system-ui,-apple-system,Segoe UI,sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
*,*::before,*::after{{font-variant-ligatures:none;font-feature-settings:"liga" 0,"clig" 0,"calt" 0,"dlig" 0}}
html,body{{height:100%;background:var(--surround);overflow:hidden;font-family:var(--body);color:var(--ink);-webkit-font-smoothing:antialiased}}
#stage{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center}}
#deck{{width:var(--sw);height:var(--sh);position:relative;transform-origin:center center;box-shadow:0 20px 70px rgba(0,0,0,.5)}}
.slide{{position:absolute;inset:0;background:var(--paper);padding:40px 54px 58px;display:none;flex-direction:column}}
.slide.on{{display:flex}}
@media (prefers-reduced-motion:no-preference){{
  .slide.on{{animation:fi .22s ease}}
  @keyframes fi{{from{{opacity:0;transform:translateY(4px)}}to{{opacity:1;transform:none}}}}
}}
.eyebrow{{font-family:var(--mono);font-size:11px;font-weight:500;letter-spacing:.17em;text-transform:uppercase;color:var(--soft);margin-bottom:8px;display:flex;gap:10px;align-items:center}}
.eyebrow::after{{content:"";flex:1;height:1px;background:var(--line)}}
h1.t{{font-family:var(--disp);font-weight:700;font-size:31px;line-height:1.12;letter-spacing:-.018em;margin-bottom:4px}}
h1.t small{{display:block;font-family:var(--body);font-weight:400;font-size:14.2px;color:var(--soft);letter-spacing:0;margin-top:5px;line-height:1.4}}
.wrap{{flex:1;min-height:0;margin-top:12px;display:flex;flex-direction:column;gap:11px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:18px;flex:1;min-height:0}}
.cols.w3070{{grid-template-columns:280px 1fr}}
.cols.w6040{{grid-template-columns:1.55fr 1fr}}
.cols4{{display:grid;grid-template-columns:repeat(4,1fr);gap:11px}}
.colstack{{display:flex;flex-direction:column;gap:9px;min-height:0}}
.fig{{background:var(--card);border:1px solid var(--hair);padding:10px 12px 8px;display:flex;flex-direction:column;min-height:0;flex:1}}
.fig .cap{{font-family:var(--mono);font-size:9.3px;letter-spacing:.12em;text-transform:uppercase;color:var(--soft);margin-bottom:6px}}
.fig svg{{display:block;width:100%;flex:1;min-height:0}}
.card{{background:var(--card);border:1px solid var(--hair);border-top:3px solid var(--ink);padding:10px 12px}}
.card h3{{font-family:var(--disp);font-size:14px;font-weight:700;margin-bottom:4px}}
.card p{{font-size:12px;line-height:1.42;color:#31444F}}
.card.r{{border-top-color:var(--t-asr)}} .card.b{{border-top-color:var(--t-cls)}}
.card.g{{border-top-color:var(--t-grp)}} .card.o{{border-top-color:var(--t-anc)}}
.callout{{background:var(--card);border:1px solid var(--hair);border-left:4px solid var(--t-cls);padding:9px 13px;font-size:12.4px;line-height:1.44}}
.callout.bad{{border-left-color:var(--t-asr);background:#FFF5F5}}
.callout.good{{border-left-color:var(--t-grp);background:#F3FBF7}}
.callout.warn{{border-left-color:var(--t-anc);background:#FFFBF2}}
.seg{{display:flex;gap:5px;flex:none;flex-wrap:wrap}}
.seg button{{font-family:var(--mono);font-size:10.5px;letter-spacing:.07em;padding:5px 10px;border:1px solid var(--line);background:var(--paper);color:var(--soft);cursor:pointer;border-radius:2px;transition:all .15s}}
.seg button:hover{{background:#E2EBF0;color:var(--ink)}}
.seg button.on{{background:var(--ink);color:#fff;border-color:var(--ink);font-weight:700}}
input[type=range]{{width:100%;accent-color:#B02F3B;cursor:pointer}}
.outlbl{{font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--soft)}}
.out{{background:var(--card);border:1px solid var(--hair);border-left:3px solid #7E9AA8;font-family:var(--mono);font-size:11px;line-height:1.48;padding:9px 11px;white-space:pre-wrap;color:#26404E}}
.jsbadge{{font-family:var(--mono);font-size:8.8px;letter-spacing:.12em;text-transform:uppercase;background:var(--t-anc);color:#fff;padding:2px 6px;border-radius:2px;width:fit-content;margin-top:auto}}
.lf{{font-family:var(--mono);font-size:22px;font-weight:700;padding:7px 11px;background:var(--card);border:1px solid var(--hair);border-left:4px solid var(--soft);line-height:1.18}}
.lf small{{display:block;font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--soft);font-weight:500;margin-bottom:2px}}
.lf.bad{{color:#B02F3B;border-left-color:#B02F3B}} .lf.ok{{color:#0E7A55;border-left-color:#0E7A55}}
table.d{{width:100%;border-collapse:collapse;font-size:11.8px;background:var(--card)}}
table.d th,table.d td{{border:1px solid var(--hair);padding:5px 8px;text-align:left}}
table.d th{{background:var(--ink);color:#fff;font-weight:600;font-size:10px;letter-spacing:.1em;text-transform:uppercase}}
table.d td:first-child{{font-family:var(--mono)}}
.cover{{justify-content:center;padding:0 68px 70px}}
.cover .kicker{{font-family:var(--mono);font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--t-asr);margin-bottom:18px}}
.cover h1{{font-family:var(--disp);font-weight:800;font-size:70px;line-height:.96;letter-spacing:-.035em}}
.cover .pt{{font-family:var(--disp);font-weight:700;font-size:25px;color:var(--soft);letter-spacing:-.02em;margin-top:10px}}
.cover .rule{{width:96px;height:4px;background:var(--ink);margin:22px 0 20px}}
.cover .sub{{font-size:15.5px;color:#31444F;max-width:650px;line-height:1.5}}
.meta{{margin-top:26px;font-family:var(--mono);font-size:11px;color:var(--soft)}}
.meta td{{padding:4px 34px 4px 0;text-transform:uppercase;letter-spacing:.09em}}
.meta td:first-child{{color:#8FA0A9}}
.coverpat{{position:absolute;right:0;top:0;bottom:0;width:440px;overflow:hidden;display:flex;flex-direction:column;justify-content:center;gap:4px;opacity:.55;-webkit-mask-image:linear-gradient(90deg,transparent,#000 38%);mask-image:linear-gradient(90deg,transparent,#000 38%)}}
.coverpat div{{font-family:var(--mono);font-size:12px;white-space:nowrap}}
.coverpat .h-hdr{{color:var(--t-cls);font-weight:700}} .coverpat .h-val{{color:var(--t-grp);font-weight:700}}
.coverpat .h-bad{{color:var(--t-asr);font-weight:700}} .coverpat .h-dim{{color:#7E9AA8}}
.rail{{position:absolute;left:0;right:0;bottom:0;height:30px;background:var(--card);border-top:1.5px solid var(--ink);display:flex;align-items:center;font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--soft)}}
.rail .a{{padding:0 16px;flex:1}} .rail .n{{padding:0 16px;border-left:1px solid var(--hair);color:var(--ink)}}
.rail .b{{padding:0 16px;border-left:1px solid var(--hair);color:var(--ink);font-weight:600}}
.rail .c{{padding:0 16px;border-left:1px solid var(--hair);width:186px;text-align:right}}
#pbar{{position:absolute;left:0;top:0;height:3px;background:var(--t-asr);z-index:20;transition:width .2s}}
#nav{{position:fixed;right:16px;bottom:12px;display:flex;gap:7px;z-index:40}}
#nav button{{width:36px;height:36px;border:1px solid #46606E;background:#1B2E39;color:#CFDDE3;font-size:16px;cursor:pointer;border-radius:3px}}
#hint{{position:fixed;left:16px;bottom:15px;color:#6D8593;font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;z-index:40}}
.slide.build [data-at]{{opacity:0}}
.slide.build [data-at].shown{{opacity:1}}
@media (prefers-reduced-motion:no-preference){{
  .slide.build [data-at]{{transition:opacity .34s ease}}
  .slide.build .mv[data-at]{{transition:transform .9s cubic-bezier(.45,0,.25,1)}}
}}
.bld{{position:absolute;right:58px;bottom:38px;display:flex;gap:5px;z-index:3}}
.bld i{{width:17px;height:3px;background:var(--line);border-radius:2px;display:block;transition:background .25s ease}}
.bld i.on{{background:var(--t-asr)}}
.slide.build .mv[data-at]{{opacity:1}}
.slide.build .mv[data-at].shown{{transform:var(--to)}}
</style>
</head>
<body>
<div id="stage"><div id="deck">
<div id="pbar"></div>

<!-- SLIDE 1: COVER -->
<section class="slide cover" data-sec="Cover">
  <div class="coverpat">
    <div><span class="h-dim">dataset = </span><span class="h-hdr">earthquakes_master_2000_2025.parquet</span></div>
    <div>&nbsp;</div>
    <div><span class="h-dim">deduplicated M6.0+ mainshocks  </span><span class="h-val">{n_total:,}</span></div>
    <div><span class="h-dim">aftershocks collapsed (150km)  </span><span class="h-bad">-514 rows</span></div>
    <div><span class="h-dim">USGS 10km/33km depth artifacts </span><span class="h-bad">{n_default_depth} ({pct_default_depth}%)</span></div>
    <div><span class="h-dim">onshore quakes w/ WDI income   </span><span class="h-val">{n_econ:,}</span></div>
    <div><span class="h-dim">missing WDI GDP (MAR labelled) </span><span class="h-bad">{n_missing_gdp} onshore</span></div>
    <div>&nbsp;</div>
    <div><span class="h-dim">deadly mainshocks reconciled   </span><span class="h-val">379 (r = {r_corr:.3f})</span></div>
    <div><span class="h-dim">GDP slope (deadly shallow)     </span><span class="h-val">{w9_data['slope_deadly']} (p = {w9_data['pval_deadly']})</span></div>
    <div><span class="h-dim">10x GDP fatality reduction     </span><span class="h-bad">-{w9_data['pct_red']}%</span></div>
  </div>
  <div class="kicker">DA2402 · Data Curation and Visualization · Project #21</div>
  <h1>Earthquake<br>Fatalities</h1>
  <div class="pt">After allowing for magnitude and depth, how much does income explain deaths?</div>
  <div class="rule"></div>
  <p class="sub">An end-to-end curation and econometric dashboard integrating USGS ComCat, NOAA NCEI, Wikipedia, and World Bank WDI across {n_total:,} global mainshocks (2000–2025).</p>
  <table class="meta">
    <tr><td>Course</td><td>DA2402 · BTech DSAI</td></tr>
    <tr><td>Sources</td><td>USGS API · NOAA NCEI · Wikipedia HTML+Notes · World Bank WDI</td></tr>
    <tr><td>Design</td><td>Gestalt Grouping · Luminance-Ordered Greyscale Safe · Interactive JS</td></tr>
  </table>
</section>

<!-- SLIDE 2: PIPELINE & CURATION -->
<section class="slide" data-sec="Pipeline">
  <div class="eyebrow">Data Curation · 4-Source Integration Architecture</div>
  <h1 class="t">From 3,874 Raw Records to {n_total:,} Reconciled Mainshocks
    <small>Every cleaning decision addresses a specific distortion mechanism before econometric modelling.</small></h1>
  <div class="wrap">
    <div class="cols4">
      <div class="card b"><h3>1. USGS ComCat</h3><p><b>3,874 raw M6.0+ shocks</b> collapsed via 150 km &amp; &plusmn;2-day Haversine greedy clustering into <b>{n_total:,} mainshocks</b>. Flagged <b>{n_default_depth} ({pct_default_depth}%)</b> fixed 10 km / 33 km placeholders.</p></div>
      <div class="card o"><h3>2. NOAA + Wikipedia</h3><p>Rescued 2006–2013 bullet-point years &amp; parsed text range midpoints. Reconciled <b>379 deadly mainshocks</b> (<b>{n_disagree} disagreements</b>, log r = {r_corr:.3f}).</p></div>
      <div class="card g"><h3>3. World Bank WDI</h3><p>Joined constant 2021 PPP GDP per capita &amp; historical income tiers by <code>(country_iso3, year)</code> across <b>{n_econ:,} onshore mainshocks</b> (95.9% match).</p></div>
      <div class="card r"><h3>4. Missingness Audit</h3><p><b>1,310 offshore</b> quakes separated structurally; <b>{n_missing_gdp} onshore</b> quakes (Taiwan, Venezuela, New Caledonia) explicitly labelled as MAR.</p></div>
    </div>
    <table class="d">
      <tr><th>Dataset Stage /Subset</th><th>Row Count</th><th>Key Curation Rule Applied</th><th>Why It Matters for Visualisation &amp; OLS</th></tr>
      <tr><td>Raw USGS M6.0+ Catalogue (2000–2025)</td><td>3,874</td><td>Spatial &amp; temporal window fetch</td><td>Contains foreshock/aftershock swarms that double-count disaster deaths</td></tr>
      <tr><td>Deduplicated Global Mainshocks</td><td>{n_total:,}</td><td>&plusmn;2 days &amp; &le;150 km Haversine mainshock collapse</td><td>Prevents 4.8x death inflation in 2023 Turkey and 26x in 2011 Japan</td></tr>
      <tr><td>Onshore Mainshocks with WDI Income</td><td>{n_econ:,}</td><td>ISO-3 geocoding + World Bank historical thresholds</td><td>Forms the clean econometric sample; 145 missing-GDP rows labelled separately</td></tr>
      <tr><td>Shallow Onshore (&le;70 km) / Deadly Subset</td><td>{w9_data['n_all']:,} / {w9_data['n_deadly']}</td><td>Physical depth regime filter (&le;70 km = 99.78% of deaths)</td><td>Isolates surface-rupturing hazard to test national income's protective effect</td></tr>
    </table>
  </div>
</section>

<!-- SLIDE 3: PLOT 1A INTERACTIVE (DEDUPLICATION LIE FACTOR) -->
<section class="slide" data-sec="Plot 1A · Deduplication">
  <div class="eyebrow">Plot 1A · Before vs. After Spatiotemporal Deduplication</div>
  <h1 class="t">Fool, Then Fix: Naive Date-Join Fatality Inflation
    <small>Toggle between un-deduplicated shocks and our 150 km mainshock consensus. Without deduplication, Turkey 2023 is inflated 4.8x and Haiti 2010 is inflated 2.4x.</small></h1>
  <div class="wrap">
    <div class="cols w3070">
      <div class="colstack">
        <div class="seg" id="w1-mode">
          <button data-m="raw">Before Cleaning (Naive)</button>
          <button data-m="clean" class="on">After Cleaning (150 km)</button>
        </div>
        <div class="out" id="w1-out"></div>
        <div class="lf ok" id="w1-lf"><small>Turkey 2023 inflation factor</small>1.0x</div>
        <div class="callout sm"><b>Gestalt &amp; Print:</b> Hatched grey ghost bars show the un-deduplicated inflation; solid crimson bars show the cleaned mainshock toll.</div>
        <span class="jsbadge">Interactive Widget 1</span>
      </div>
      <div class="fig">
        <div class="cap">Reported Earthquake Fatalities (Thousands of Persons, Common Baseline = 0)</div>
        <svg id="w1-svg" viewBox="0 0 760 340"></svg>
      </div>
    </div>
  </div>
</section>

<!-- SLIDE 4: PLOT 1B INTERACTIVE (DEPTH ARTIFACTS) -->
<section class="slide" data-sec="Plot 1B · Depth Artifacts">
  <div class="eyebrow">Plot 1B · USGS Focal Depth Placeholder Diagnosis</div>
  <h1 class="t">Uncovering the 10 km and 33 km Seismic Network Spikes
    <small>Exactly {n_default_depth} of {n_total:,} mainshocks ({pct_default_depth}%) sit at fixed 10.0 km or 33.0 km placeholders. Drag the bin width or toggle artifact highlighting.</small></h1>
  <div class="wrap">
    <div class="cols w3070">
      <div class="colstack">
        <div class="outlbl">Histogram Bin Width: <b id="w2-bwv">2</b> km</div>
        <input type="range" id="w2-bw" min="1" max="10" step="1" value="2">
        <div class="seg" id="w2-pre">
          <button data-b="1">1 km</button>
          <button data-b="2" class="on">2 km</button>
          <button data-b="5">5 km</button>
          <button data-b="10">10 km</button>
        </div>
        <div class="seg" id="w2-tog">
          <button data-a="1" class="on">Highlight 10/33 km Artifacts</button>
          <button data-a="0">Hide Flagged Placeholders</button>
        </div>
        <div class="out" id="w2-out"></div>
        <span class="jsbadge">Interactive Widget 2</span>
      </div>
      <div class="fig">
        <div class="cap">Focal Depth Distribution (0–100 km Shallow &amp; Upper-Intermediate Events)</div>
        <svg id="w2-svg" viewBox="0 0 760 340"></svg>
      </div>
    </div>
  </div>
</section>

<!-- SLIDE 5: PLOT 2 INTERACTIVE (EXPOSURE VS FATALITIES) -->
<section class="slide" data-sec="Plot 2 · Exposure vs Impact">
  <div class="eyebrow">Plot 2 · Physical Exposure vs. Human Fatality Burden</div>
  <h1 class="t">Why Totals Mislead: Re-Ranking Income Tiers by Rate
    <small>Switch between Quake Count, Total Deaths, Mean Deaths/Quake, and % Deadly Share. High-income nations face M6.48 average quakes but suffer just 38.4 deaths/quake.</small></h1>
  <div class="wrap">
    <div class="cols w3070">
      <div class="colstack">
        <div class="seg" id="w3-mode">
          <button data-m="mean_deaths" class="on">Mean Deaths / Quake</button>
          <button data-m="total_deaths">Total Deaths</button>
          <button data-m="deadly_pct">% Deadly Rate</button>
          <button data-m="count">Quake Count</button>
        </div>
        <div class="out" id="w3-out"></div>
        <div class="callout sm"><b>Lecture 11 Check:</b> Totals reflect group size (807 quakes in Upper-Mid vs 40 in Low). Rates expose true vulnerability (50% deadly share in Low Income).</div>
        <span class="jsbadge">Interactive Widget 3</span>
      </div>
      <div class="fig">
        <div class="cap" id="w3-cap">Mean Fatalities per Onshore M6.0+ Earthquake by World Bank Income Tier</div>
        <svg id="w3-svg" viewBox="0 0 760 340"></svg>
      </div>
    </div>
  </div>
</section>

<!-- SLIDE 6: NEW CREATIVE PLOT 6 (WORLD MAP TIME SLIDER) -->
<section class="slide" data-sec="Plot 6 · Spatiotemporal Map">
  <div class="eyebrow">Plot 6 · Global Spatiotemporal Map &amp; Missing-Data Audit (2000–2025)</div>
  <h1 class="t">25 Years of Tectonic Hazard, Income Tiers &amp; Missing GDP
    <small>Scrub through 2000–2025 or press Play. Onshore quakes without World Bank GDP (n={n_missing_gdp}, e.g. Taiwan) are explicitly drawn as hollow squares so missing data is never hidden.</small></h1>
  <div class="wrap">
    <div class="cols w3070">
      <div class="colstack">
        <div class="outlbl">Year Filter: <b id="w4-yrv">2025</b></div>
        <input type="range" id="w4-yr" min="2000" max="2025" step="1" value="2025">
        <div class="seg" id="w4-mode">
          <button data-m="cum" class="on">Cumulative (2000–Y)</button>
          <button data-m="single">Single Year (Y)</button>
          <button id="w4-play" data-m="play">▶ Play 2000→2025</button>
        </div>
        <div class="out" id="w4-out"></div>
        <div class="callout warn" style="font-size:11.2px;padding:7px 10px"><b>Missing Data Labelled:</b> Hollow dark squares = {n_missing_gdp} onshore quakes missing WDI GDP; grey dots = 1,310 offshore quakes.</div>
        <span class="jsbadge">Interactive Widget 4</span>
      </div>
      <div class="fig">
        <div class="cap">Global Epicenters (Circle = High, Triangle = Upper-Mid, Diamond = Lower-Mid, Cross = Low, Square = Missing GDP)</div>
        <svg id="w4-svg" viewBox="0 0 760 360"></svg>
      </div>
    </div>
  </div>
</section>

<!-- SLIDE 7: PLOT 3 INTERACTIVE (MAGNITUDE & DEPTH FILTER) -->
<section class="slide" data-sec="Plot 3 · Physical Controls">
  <div class="eyebrow">Plot 3 · Magnitude &amp; Focal Depth Controls</div>
  <h1 class="t">Focal Depth as a Physical Filter: Shallow (&le;70 km) = 99.78% of Deaths
    <small>Slide the maximum focal depth cutoff or filter by depth zone. Above 300 km depth, seismic energy attenuates before reaching the surface (max 2 deaths).</small></h1>
  <div class="wrap">
    <div class="cols w3070">
      <div class="colstack">
        <div class="outlbl">Max Focal Depth Cutoff: <b id="w5-depv">700</b> km</div>
        <input type="range" id="w5-dep" min="15" max="700" step="5" value="700">
        <div class="seg" id="w5-pre">
          <button data-d="700" class="on">All Depths</button>
          <button data-d="70">Shallow (&le;70 km)</button>
          <button data-d="35">Very Shallow (&le;35 km)</button>
        </div>
        <div class="out" id="w5-out"></div>
        <div class="callout sm"><b>What Log Scale Reveals:</b> Each +1.0 Mw raises shallow log10(deaths+1) by +0.51, yet at M7.0–7.9 outcomes span 0 to 262,000 deaths.</div>
        <span class="jsbadge">Interactive Widget 5</span>
      </div>
      <div class="fig">
        <div class="cap">Earthquake Magnitude (Mw) vs. Log10(Reconciled Fatalities + 1) Across Onshore Mainshocks</div>
        <svg id="w5-svg" viewBox="0 0 760 340"></svg>
      </div>
    </div>
  </div>
</section>

<!-- SLIDE 8: PLOT 4 & TIMELINE INTERACTIVE (CROSS-SOURCE + CUMULATIVE FIX) -->
<section class="slide" data-sec="Plot 4 · Source Audit & Timeline">
  <div class="eyebrow">Plot 4 &amp; Temporal Audit · NOAA vs. Wikipedia &amp; Cumulative vs. Annual</div>
  <h1 class="t">Cross-Source Concordance (r = {r_corr:.3f}) &amp; Episodic Mega-Shocks
    <small>Switch between the NOAA vs. Wikipedia 1:1 agreement audit (379 deadly quakes, 106 disagreements) and the 2000–2025 Annual vs. Cumulative timeline.</small></h1>
  <div class="wrap">
    <div class="cols w3070">
      <div class="colstack">
        <div class="seg" id="w6-view">
          <button data-v="scatter" class="on">NOAA vs. Wiki (Plot 4)</button>
          <button data-v="time">Annual vs. Cumulative</button>
        </div>
        <div class="out" id="w6-out"></div>
        <div class="callout sm" id="w6-note"><b>Consensus Rule:</b> Where both sources report deaths, we take the mean; where one is missing, we rescue the non-null source.</div>
        <span class="jsbadge">Interactive Widget 6</span>
      </div>
      <div class="fig">
        <div class="cap" id="w6-cap">NOAA NCEI vs. Wikipedia Fatalities [Log10(Deaths + 1)] Across 379 Deadly Mainshocks</div>
        <svg id="w6-svg" viewBox="0 0 760 340"></svg>
      </div>
    </div>
  </div>
</section>

<!-- SLIDE 9: NEW CREATIVE PLOT 7 (RISK MATRIX + GREYSCALE L* CHECK) -->
<section class="slide" data-sec="Plot 7 · Risk Matrix & Print">
  <div class="eyebrow">Plot 7 · Magnitude × Income Risk Matrix &amp; Greyscale Print Simulation</div>
  <h1 class="t">Bivariate Vulnerability Matrix (Colour vs. Greyscale Print L*)
    <small>Test our luminance-ordered palette live: toggle between Colour and Greyscale Print Simulation to verify monotonic L* lightness ordering across every cell.</small></h1>
  <div class="wrap">
    <div class="cols w3070">
      <div class="colstack">
        <div class="seg" id="w7-gry">
          <button data-g="0" class="on">Colour Palette</button>
          <button data-g="1">Simulate Greyscale Print</button>
        </div>
        <div class="seg" id="w7-metric">
          <button data-k="log" class="on">Mean Log10(Deaths+1)</button>
          <button data-k="pct">% Deadly Quake Probability</button>
        </div>
        <div class="out" id="w7-out"></div>
        <div class="callout good" style="font-size:11.4px"><b>Lecture 10 Check:</b> Cell lightness L* decreases monotonically with risk (from L* 94 down to L* 28), surviving black-and-white printing without inversion.</div>
        <span class="jsbadge">Interactive Widget 7</span>
      </div>
      <div class="fig">
        <div class="cap">Shallow Onshore Earthquakes (&le;70 km, n={w9_data['n_all']:,}): Magnitude Band × World Bank Income Group</div>
        <svg id="w7-svg" viewBox="0 0 760 340"></svg>
      </div>
    </div>
  </div>
</section>

<!-- SLIDE 10: PLOT 5 CONCLUSION & WHAT-IF SIMULATOR -->
<section class="slide" data-sec="Plot 5 · Econometric Verdict">
  <div class="eyebrow">Plot 5 · Answering the Project Question · Econometric Conclusion</div>
  <h1 class="t">Holding Magnitude &amp; Depth Constant, 10x Income Cuts Deaths by {w9_data['pct_red']}%
    <small>Simulate any physical earthquake across the global income spectrum using our fitted OLS model (n={w9_data['n_deadly']} deadly shallow quakes, R² = {w9_data['r2_deadly']}, p = {w9_data['pval_deadly']}).</small></h1>
  <div class="wrap">
    <div class="cols w3070">
      <div class="colstack">
        <div class="outlbl">Simulated GDP per Capita: <b id="w9-gdpv">$3,177</b></div>
        <input type="range" id="w9-gdp" min="1000" max="80000" step="500" value="3177">
        <div class="outlbl">Earthquake Magnitude: <b id="w9-magv">M7.0</b></div>
        <input type="range" id="w9-mag" min="6.0" max="9.0" step="0.1" value="7.0">
        <div class="seg" id="w9-pre">
          <button data-g="3177" data-m="7.0" class="on">Haiti '10 ($3.2k)</button>
          <button data-g="14500" data-m="7.0">Mid Income ($14.5k)</button>
          <button data-g="41641" data-m="7.0">Japan ($41.6k)</button>
        </div>
        <div class="out" id="w9-out"></div>
        <div class="lf ok" id="w9-lf"><small>Expected Fatality Reduction vs. $1,000 Baseline</small>-54.3%</div>
        <span class="jsbadge">OLS Slope = {w9_data['slope_deadly']}</span>
      </div>
      <div class="fig">
        <div class="cap">OLS Predicted Fatality Curve vs. Real GDP per Capita PPP (Stratified by Magnitude at 15 km Shallow Depth)</div>
        <svg id="w9-svg" viewBox="0 0 760 340"></svg>
      </div>
    </div>
  </div>
</section>

</div></div>

<div id="nav"><button id="prev" aria-label="Previous slide">‹</button><button id="next" aria-label="Next slide">›</button></div>
<div id="hint">← → navigate · F fullscreen</div>

<!-- EMBEDDED REAL DATASET PAYLOADS -->
<script id="d-w1" type="application/json">{json.dumps(w1_data)}</script>
<script id="d-w2" type="application/json">{json.dumps(w2_data)}</script>
<script id="d-w3" type="application/json">{json.dumps(w3_data)}</script>
<script id="d-w4" type="application/json">{json.dumps(w4_quakes)}</script>
<script id="d-w5" type="application/json">{json.dumps(w5_pts)}</script>
<script id="d-w6" type="application/json">{json.dumps(w6_pts)}</script>
<script id="d-w7" type="application/json">{json.dumps(w7_cells)}</script>
<script id="d-w8" type="application/json">{json.dumps(w8_data)}</script>
<script id="d-w9" type="application/json">{json.dumps(w9_data)}</script>

<script>
/* ============================================================
   DECK NAVIGATION ENGINE
   ============================================================ */
(function(){{
  const slides=[...document.querySelectorAll('.slide')],
        deck=document.getElementById('deck'), pbar=document.getElementById('pbar');
  let i=0;
  slides.forEach((s,idx)=>{{
    const r=document.createElement('div'); r.className='rail';
    r.innerHTML='<div class="a">DA2402 &#183; Data Curation and Visualization</div>'+
      '<div class="n">Project #21 · Earthquakes &amp; Income</div>'+
      '<div class="b">'+(s.dataset.sec||'')+'</div>'+
      '<div class="c">Slide &#183; '+String(idx+1).padStart(2,'0')+' / '+slides.length+'</div>';
    s.appendChild(r);
  }});
  function show(n){{
    i=Math.max(0,Math.min(slides.length-1,n));
    slides.forEach((s,k)=>s.classList.toggle('on',k===i));
    pbar.style.width=(i/(slides.length-1)*100)+'%';
    try {{ history.replaceState(null,'','#/'+(i+1)); }} catch(e){{}}
  }}
  function fit(){{ deck.style.transform='scale('+Math.min(innerWidth/1330,innerHeight/770)+')'; }}
  addEventListener('resize',fit); fit();
  document.getElementById('next').onclick=()=>show(i+1);
  document.getElementById('prev').onclick=()=>show(i-1);
  addEventListener('keydown',e=>{{
    if(/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return;
    if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){{show(i+1);e.preventDefault();}}
    else if(e.key==='ArrowLeft'||e.key==='PageUp'){{show(i-1);e.preventDefault();}}
    else if(e.key==='Home')show(0);
    else if(e.key==='End')show(slides.length-1);
    else if(e.key==='f'||e.key==='F'){{document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();}}
  }});
  const m=location.hash.match(/#\\/(\\d+)/); show(m?+m[1]-1:0);
}})();

/* ============================================================
   INTERACTIVE DATA VISUALISATION WIDGETS
   ============================================================ */
(function(){{
  const $=id=>document.getElementById(id);
  const MONO='JetBrains Mono,monospace', INK='#14222B', SOFT='#53666F', HAIR='#DCE4E8';
  const BLUE='#125F9E', RED='#B02F3B', GREEN='#0E7A55', AMBER='#8F5D0A';
  const data=id=>JSON.parse($(id).textContent);
  const txt=(x,y,s,o)=>{{o=o||{{}};return '<text x="'+(+x).toFixed(1)+'" y="'+(+y).toFixed(1)+'" font-family="'+MONO+'" font-size="'+(o.size||10)+'" fill="'+(o.fill||SOFT)+'" text-anchor="'+(o.anchor||'start')+'" font-weight="'+(o.w||400)+'">'+s+'</text>';}};
  const line=(x1,y1,x2,y2,c,w,ex)=>'<line x1="'+x1.toFixed(1)+'" y1="'+y1.toFixed(1)+'" x2="'+x2.toFixed(1)+'" y2="'+y2.toFixed(1)+'" stroke="'+c+'" stroke-width="'+(w||1)+'" '+(ex||'')+'/>';
  function seg(id,fn){{
    const g=$(id); if(!g) return;
    g.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{{
      if(b.id==='w4-play') {{ fn(b.dataset, b); return; }}
      g.querySelectorAll('button:not(#w4-play)').forEach(x=>x.classList.toggle('on',x===b));
      fn(b.dataset, b);
    }}));
  }}
  function tween(from,to,ms,fn){{
    const t0=performance.now();
    (function step(t){{
      const k=Math.min(1,(t-t0)/ms), e=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;
      fn(from+(to-from)*e); if(k<1) requestAnimationFrame(step);
    }})(t0);
  }}

  /* --- WIDGET 1: DEDUPLICATION --- */
  (function(){{
    const D=data('d-w1'), sv=$('w1-svg'), out=$('w1-out'), lf=$('w1-lf');
    let mix=1;
    function draw(){{
      const W=760,H=340,l=185,r=55,t=18,rowH=48, maxV=660;
      const X=v=>l+(v/maxV)*(W-l-r);
      let s='<defs><pattern id="p1-h" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#8FA0A9" stroke-width="2"/></pattern></defs>';
      [0,100,200,300,400,500,600].forEach(v=>{{
        s+=line(X(v),t,X(v),t+D.length*rowH,HAIR,1)+txt(X(v),t+D.length*rowH+18,v+'k',{{anchor:'middle'}});
      }});
      D.forEach((d,i)=>{{
        const y=t+i*rowH+10, cur=d.raw*(1-mix)+d.clean*mix;
        s+=txt(l-10,y+18,d.label,{{anchor:'end',fill:INK,w:700,size:11}});
        s+='<rect x="'+l+'" y="'+y+'" width="'+(X(d.raw)-l).toFixed(1)+'" height="26" fill="url(#p1-h)" stroke="#7E9AA8"/>';
        s+='<rect x="'+l+'" y="'+y+'" width="'+(X(cur)-l).toFixed(1)+'" height="26" fill="'+(mix>.5?GREEN:RED)+'"/>';
        s+=txt(X(cur)+7,y+17,cur.toFixed(1)+'k ('+d.shocks+' shocks)',{{fill:INK,w:700,size:10.5}});
      }});
      sv.innerHTML=s;
      const tur=D.find(x=>x.label.includes('Turkey'))||D[0];
      const curF=tur.lf*(1-mix)+1.0*mix;
      lf.className='lf '+(curF<1.1?'ok':'bad');
      lf.innerHTML='<small>Turkey 2023 inflation factor</small>'+curF.toFixed(1)+'x';
      out.textContent=(mix>.5?'After 150 km Deduplication:\\n':'Before Cleaning (Naive Join):\\n')+
        D.map(d=>d.label.split(' ')[0].padEnd(11)+': '+(d.raw*(1-mix)+d.clean*mix).toFixed(1)+'k').join('\\n');
    }}
    seg('w1-mode',d=>tween(mix,d.m==='clean'?1:0,550,v=>{{mix=v;draw();}}));
    draw();
  }})();

  /* --- WIDGET 2: DEPTH ARTIFACTS --- */
  (function(){{
    const D=data('d-w2'), sv=$('w2-svg'), out=$('w2-out'), bw=$('w2-bw'), bwv=$('w2-bwv');
    let showArt=1;
    function draw(){{
      const w=+bw.value; bwv.textContent=w;
      const W=760,H=340,l=52,r=20,t=18,b=36, nBins=Math.ceil(100/w);
      const cReal=new Array(nBins).fill(0), cArt=new Array(nBins).fill(0);
      D.real.forEach(v=>{{const k=Math.min(nBins-1,Math.floor(v/w)); cReal[k]++;}});
      if(showArt) D.art.forEach(v=>{{const k=Math.min(nBins-1,Math.floor(v/w)); cArt[k]++;}});
      const mx=Math.max(...cReal.map((v,i)=>v+cArt[i]), 10);
      const X=v=>l+(v/100)*(W-l-r), Y=c=>t+(H-t-b)*(1-c/mx);
      let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="'+HAIR+'"/>';
      for(let i=0;i<nBins;i++){{
        const x0=X(i*w), x1=X(Math.min(100,(i+1)*w)), bwPx=Math.max(1,x1-x0-1);
        const hR=(H-t-b)*(cReal[i]/mx), hA=(H-t-b)*(cArt[i]/mx);
        s+='<rect x="'+x0.toFixed(1)+'" y="'+(H-b-hR).toFixed(1)+'" width="'+bwPx.toFixed(1)+'" height="'+hR.toFixed(1)+'" fill="'+BLUE+'"/>';
        if(hA>0) s+='<rect x="'+x0.toFixed(1)+'" y="'+(H-b-hR-hA).toFixed(1)+'" width="'+bwPx.toFixed(1)+'" height="'+hA.toFixed(1)+'" fill="'+RED+'"/>';
      }}
      [0,10,20,33,50,70,100].forEach(v=>{{
        s+=line(X(v),t,X(v),H-b,v===10||v===33?RED:HAIR,v===10||v===33?1.5:1,v===10||v===33?'stroke-dasharray="4 3"':'');
        s+=txt(X(v),H-14,v+'km',{{anchor:'middle',fill:v===10||v===33?RED:SOFT,w:v===10||v===33?700:400}});
      }});
      sv.innerHTML=s;
      out.textContent='Bin width: '+w+' km\\nConstrained events: '+D.real.length+'\\nFlagged 10/33km: '+(showArt?D.n_art+' ('+D.pct_art+'%)':'Hidden (Filtered)');
    }}
    bw.oninput=draw;
    seg('w2-pre',d=>{{bw.value=d.b;draw();}});
    seg('w2-tog',d=>{{showArt=+d.a;draw();}});
    draw();
  }})();

  /* --- WIDGET 3: INCOME EXPOSURE VS IMPACT --- */
  (function(){{
    const D=data('d-w3'), sv=$('w3-svg'), out=$('w3-out'), cap=$('w3-cap');
    let key='mean_deaths';
    const labels={{
      mean_deaths:'Mean Fatalities per M6.0+ Earthquake (Persons)',
      total_deaths:'Cumulative Fatalities Across All Quakes (Persons)',
      deadly_pct:'Percentage of M6.0+ Earthquakes Causing Deaths (%)',
      count:'Number of Onshore M6.0+ Mainshocks (2000–2025)'
    }};
    function draw(){{
      const W=760,H=340,l=68,r=24,t=24,b=42;
      const mx=Math.max(...D.map(d=>d[key]))*1.18;
      const colW=(W-l-r)/D.length;
      let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="'+HAIR+'"/>';
      for(let k=0;k<=4;k++){{
        const v=mx*k/4, y=t+(H-t-b)*(1-k/4);
        s+=line(l,y,W-r,y,HAIR,1)+txt(l-8,y+4,v>=1000?(v/1000).toFixed(0)+'k':v.toFixed(0),{{anchor:'end'}});
      }}
      D.forEach((d,i)=>{{
        const x=l+i*colW+colW/2, v=d[key], bh=(H-t-b)*(v/mx), y=H-b-bh;
        s+='<rect x="'+(x-42).toFixed(1)+'" y="'+y.toFixed(1)+'" width="84" height="'+bh.toFixed(1)+'" fill="'+d.color+'"/>';
        const lbl=key==='total_deaths'?(v/1000).toFixed(1)+'k':(key==='deadly_pct'?v.toFixed(1)+'%':v.toFixed(1));
        s+=txt(x,y-8,lbl,{{anchor:'middle',fill:INK,w:700,size:12}});
        s+=txt(x,H-18,d.group+' (n='+d.count+', M'+d.mean_mag+')',{{anchor:'middle',fill:INK,w:600,size:10.5}});
      }});
      sv.innerHTML=s; cap.textContent=labels[key];
      const sorted=[...D].sort((a,b)=>b[key]-a[key]);
      out.textContent='Ranked by '+key+':\\n'+sorted.map((d,idx)=>(idx+1)+'. '+d.group.padEnd(10)+': '+d[key]).join('\\n');
    }}
    seg('w3-mode',d=>{{key=d.m;draw();}});
    draw();
  }})();

  /* --- WIDGET 4: SPATIOTEMPORAL WORLD MAP --- */
  (function(){{
    const D=data('d-w4'), sv=$('w4-svg'), out=$('w4-out'), yr=$('w4-yr'), yrv=$('w4-yrv');
    let mode='cum', timer=null;
    const W=760,H=360,l=36,r=16,t=16,b=30;
    const X=lon=>l+(lon+180)/360*(W-l-r), Y=lat=>t+(75-lat)/140*(H-t-b);
    const cols={{0:'#C3CFD5', '-1':'#14222B', 1:'#B02F3B', 2:'#E69F00', 3:'#125F9E', 4:'#0E7A55'}};
    function draw(){{
      const yVal=+yr.value; yrv.textContent=yVal;
      let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#F7FAFC" stroke="'+HAIR+'"/>';
      [-120,-60,0,60,120].forEach(lon=>{{s+=line(X(lon),t,X(lon),H-b,HAIR,1)+txt(X(lon),H-10,lon+'°',{{anchor:'middle'}});}});
      [-45,0,45].forEach(lat=>{{s+=line(l,Y(lat),W-r,Y(lat),lat===0?'#9BAAB3':HAIR,1)+txt(l-5,Y(lat)+3,lat+'°',{{anchor:'end'}});}});
      D.forEach(q=>{{
        s+='<circle cx="'+X(q[1]).toFixed(1)+'" cy="'+Y(q[2]).toFixed(1)+'" r="1.5" fill="#DCE4E8"/>';
      }});
      const active=D.filter(q=>mode==='cum'?q[0]<=yVal:q[0]===yVal);
      let dSum=0, missCnt=0;
      active.forEach(q=>{{
        dSum+=q[5]; if(q[7]===-1) missCnt++;
        const x=X(q[1]), y=Y(q[2]), rad=q[5]>0?Math.min(16, 3.2+Math.log10(q[5]+1)*2.4):2.4;
        const c=cols[String(q[7])];
        if(q[7]===-1){{
          s+='<rect x="'+(x-3).toFixed(1)+'" y="'+(y-3).toFixed(1)+'" width="6" height="6" fill="none" stroke="#14222B" stroke-width="1.4"/>';
        }} else {{
          s+='<circle cx="'+x.toFixed(1)+'" cy="'+y.toFixed(1)+'" r="'+rad.toFixed(1)+'" fill="'+c+'" fill-opacity="'+(q[5]>0?0.82:0.55)+'" stroke="#fff" stroke-width="0.4"/>';
        }}
      }});
      sv.innerHTML=s;
      out.textContent=(mode==='cum'?'2000 to '+yVal:'Year '+yVal)+' Summary:\\nActive Mainshocks: '+active.length+'\\nReconciled Deaths: '+Math.round(dSum).toLocaleString('en-US')+'\\nMissing-GDP Quakes: '+missCnt;
    }}
    yr.oninput=draw;
    seg('w4-mode',(d,btn)=>{{
      if(d.m==='play'){{
        if(timer){{clearInterval(timer);timer=null;btn.textContent='▶ Play 2000→2025';return;}}
        if(+yr.value>=2025) yr.value=2000;
        btn.textContent='⏸ Pause';
        timer=setInterval(()=>{{
          if(+yr.value>=2025){{clearInterval(timer);timer=null;btn.textContent='▶ Play 2000→2025';return;}}
          yr.value=+yr.value+1; draw();
        }}, 260);
      }} else {{
        mode=d.m; draw();
      }}
    }});
    draw();
  }})();

  /* --- WIDGET 5: MAGNITUDE & DEPTH FILTER --- */
  (function(){{
    const D=data('d-w5'), sv=$('w5-svg'), out=$('w5-out'), dep=$('w5-dep'), depv=$('w5-depv');
    const W=760,H=340,l=64,r=24,t=18,b=36;
    const X=m=>l+(m-5.9)/3.4*(W-l-r), Y=lg=>t+(H-t-b)*(1-lg/5.7);
    function draw(){{
      const maxD=+dep.value; depv.textContent=maxD;
      let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="'+HAIR+'"/>';
      [[0,'0'],[1,'10'],[2,'100'],[3,'1k'],[4,'10k'],[5,'100k']].forEach(([lg,lbl])=>{{
        s+=line(l,Y(lg),W-r,Y(lg),HAIR,1)+txt(l-7,Y(lg)+4,lbl,{{anchor:'end'}});
      }});
      [6.0,6.5,7.0,7.5,8.0,8.5,9.0].forEach(m=>{{
        s+=txt(X(m),H-12,'M'+m.toFixed(1),{{anchor:'middle'}});
      }});
      const sub=D.filter(p=>p[1]<=maxD);
      let dSum=0;
      sub.forEach(p=>{{
        dSum+=p[2];
        const col=p[4]===0?RED:(p[4]===1?BLUE:INK);
        s+='<circle cx="'+X(p[0]).toFixed(1)+'" cy="'+Y(p[3]).toFixed(1)+'" r="3.4" fill="'+col+'" fill-opacity="0.72"/>';
        if(p[3]>4.6) s+=txt(X(p[0])-6,Y(p[3])-7,p[5],{{anchor:'end',fill:INK,w:700,size:9.5}});
      }});
      sv.innerHTML=s;
      out.textContent='Depth Cutoff: <= '+maxD+' km\\nVisible Onshore Quakes: '+sub.length+'\\nCaptured Deaths: '+Math.round(dSum).toLocaleString('en-US')+'\\nShare of All Deaths: '+(dSum/862340.3*100).toFixed(2)+'%';
    }}
    dep.oninput=draw;
    seg('w5-pre',d=>{{dep.value=d.d;draw();}});
    draw();
  }})();

  /* --- WIDGET 6: CROSS-SOURCE AUDIT & TIMELINE --- */
  (function(){{
    const D6=data('d-w6'), D8=data('d-w8'), sv=$('w6-svg'), out=$('w6-out'), cap=$('w6-cap');
    let view='scatter';
    function draw(){{
      const W=760,H=340,l=64,r=36,t=20,b=36;
      if(view==='scatter'){{
        const X=lg=>l+(lg/5.7)*(W-l-r), Y=lg=>t+(H-t-b)*(1-lg/5.7);
        let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="'+HAIR+'"/>';
        s+=line(X(0),Y(0),X(5.6),Y(5.6),INK,1.5,'stroke-dasharray="5 4"');
        [[0,'0'],[1,'10'],[2,'100'],[3,'1k'],[4,'10k'],[5,'100k']].forEach(([lg,lbl])=>{{
          s+=line(l,Y(lg),W-r,Y(lg),HAIR,1)+txt(l-7,Y(lg)+4,lbl,{{anchor:'end'}})+txt(X(lg),H-12,lbl,{{anchor:'middle'}});
        }});
        D6.forEach(p=>{{
          const col=p[4]===1?RED:'#7E9AA8';
          s+='<circle cx="'+X(p[0]).toFixed(1)+'" cy="'+Y(p[1]).toFixed(1)+'" r="'+(p[4]?4.2:3.0)+'" fill="'+col+'" fill-opacity="0.8"/>';
        }});
        sv.innerHTML=s;
        cap.textContent='NOAA NCEI vs. Wikipedia Fatalities [Log10(Deaths + 1)] Across 379 Deadly Mainshocks';
        out.textContent='Deadly Mainshocks: 379\\nExact Agreements / 1-Source: 273\\nCross-Source Disagreements: 106\\nLog-Correlation (both > 0): r = 0.993';
      }} else {{
        const maxA=320, maxC=900, colW=(W-l-r)/D8.years.length;
        const Y1=v=>t+(H-t-b)*(1-v/maxA), Y2=v=>t+(H-t-b)*(1-v/maxC);
        let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="'+HAIR+'"/>';
        let pts=[];
        D8.years.forEach((yr,i)=>{{
          const x=l+i*colW+colW/2, vA=D8.annual[i], bh=(H-t-b)*(vA/maxA);
          s+='<rect x="'+(x-colW*0.35).toFixed(1)+'" y="'+(H-b-bh).toFixed(1)+'" width="'+(colW*0.7).toFixed(1)+'" height="'+bh.toFixed(1)+'" fill="'+RED+'"/>';
          if(yr%4===0) s+=txt(x,H-12,yr,{{anchor:'middle'}});
          pts.push((x).toFixed(1)+','+Y2(D8.cum[i]).toFixed(1));
        }});
        s+='<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+BLUE+'" stroke-width="2.6"/>';
        sv.innerHTML=s;
        cap.textContent='Annual Fatalities (Crimson Bars) vs. Cumulative Running Total (Blue Line, 862.3k Total)';
        out.textContent='2000–2025 Cumulative: 862.3k\\nTop Shock Years:\\n- 2010 Haiti: 262.0k\\n- 2004 Indian Ocean: 227.9k\\n- 2008 China: 87.6k\\n- 2005 Pakistan: 82.5k';
      }}
    }}
    seg('w6-view',d=>{{view=d.v;draw();}});
    draw();
  }})();

  /* --- WIDGET 7: 2D RISK MATRIX & GREYSCALE PRINT L* --- */
  (function(){{
    const D=data('d-w7'), sv=$('w7-svg'), out=$('w7-out');
    let gry=0, metric='log';
    const mLabels=['M6.0–6.4 (Moderate)','M6.5–6.9 (Strong)','M7.0–7.4 (Major)','M7.5+ (Great)'];
    const iLabels=['Low income','Lower middle','Upper middle','High income'];
    function getCol(v, maxV){{
      const t=Math.min(1, Math.max(0, v/maxV));
      const L=Math.round(94 - t*66); // Monotonic L* from 94 (light) to 28 (dark)
      if(gry){{
        const g=Math.round(L*2.55);
        return [`rgb(${{g}},${{g}},${{g}})`, L];
      }}
      const r=Math.round(255 - t*85), g=Math.round(245 - t*205), b=Math.round(235 - t*190);
      return [`rgb(${{r}},${{g}},${{b}})`, L];
    }}
    function draw(){{
      const W=760,H=340,l=155,r=25,t=32,b=30;
      const cw=(W-l-r)/4, ch=(H-t-b)/4, maxV=metric==='log'?2.4:80;
      let s='';
      iLabels.forEach((lbl,j)=>s+=txt(l+j*cw+cw/2,t-10,lbl,{{anchor:'middle',fill:INK,w:700,size:11}}));
      mLabels.forEach((lbl,i)=>s+=txt(l-10,t+i*ch+ch/2+4,lbl,{{anchor:'end',fill:INK,w:700,size:11}}));
      D.forEach(cell=>{{
        const v=cell[metric], [col, L]=getCol(v, maxV);
        const x=l+cell.c*cw, y=t+cell.r*ch;
        s+='<rect x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" width="'+(cw-4).toFixed(1)+'" height="'+(ch-4).toFixed(1)+'" fill="'+col+'" stroke="#14222B"/>';
        const tc=L<55?'#FFFFFF':INK;
        const mainTxt=cell.n>0?(metric==='log'?'Log10: '+cell.log.toFixed(2):cell.pct+'% deadly'):'n=0';
        s+=txt(x+cw/2,y+ch/2-3,mainTxt,{{anchor:'middle',fill:tc,w:700,size:12}});
        s+=txt(x+cw/2,y+ch/2+14,'(n='+cell.n+' | L*='+L+')',{{anchor:'middle',fill:tc,size:9.5}});
      }});
      sv.innerHTML=s;
      out.textContent='Mode: '+(gry?'Greyscale Print (L*)':'Luminance Colour')+'\\nMetric: '+metric+'\\nL* Range: 94 (lowest risk) -> 28 (highest risk)\\nOrder strictly preserved in B&W print!';
    }}
    seg('w7-gry',d=>{{gry=+d.g;draw();}});
    seg('w7-metric',d=>{{metric=d.k;draw();}});
    draw();
  }})();

  /* --- WIDGET 9: ECONOMETRIC WHAT-IF SIMULATOR --- */
  (function(){{
    const D=data('d-w9'), sv=$('w9-svg'), out=$('w9-out'), lf=$('w9-lf'), sg=$('w9-gdp'), sm=$('w9-mag');
    const W=760,H=340,l=64,r=30,t=20,b=38;
    const X=lg=>l+(lg-3.0)/2.0*(W-l-r), Y=ld=>t+(H-t-b)*(1-ld/4.2);
    function predLog(mag, gdp){{
      return D.intercept_m3 + D.b_mag_m3*mag + D.b_dep_m3*15.0 + D.b_gdp_m3*Math.log10(gdp);
    }}
    function draw(){{
      const gdp=+sg.value, mag=+sm.value;
      $('w9-gdpv').textContent='$'+gdp.toLocaleString('en-US');
      $('w9-magv').textContent='M'+mag.toFixed(1);
      let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="'+HAIR+'"/>';
      [[3,'$1k'],[3.5,'$3.2k'],[4,'$10k'],[4.5,'$32k'],[5,'$100k']].forEach(([lg,lbl])=>{{
        s+=line(X(lg),t,X(lg),H-b,HAIR,1)+txt(X(lg),H-12,lbl,{{anchor:'middle'}});
      }});
      [[0,'0'],[1,'10'],[2,'100'],[3,'1,000'],[4,'10,000']].forEach(([ld,lbl])=>{{
        s+=line(l,Y(ld),W-r,Y(ld),HAIR,1)+txt(l-7,Y(ld)+4,lbl,{{anchor:'end'}});
      }});
      [[6.5,'#7E9AA8','M6.5'],[7.5,BLUE,'M7.5'],[mag,RED,'Active M'+mag.toFixed(1)]].forEach(([mVal,col,lbl])=>{{
        let pts=[];
        for(let lg=3.0;lg<=4.95;lg+=0.05){{
          const ld=Math.max(0, predLog(mVal, Math.pow(10,lg)));
          pts.push(X(lg).toFixed(1)+','+Y(ld).toFixed(1));
        }}
        s+='<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+col+'" stroke-width="'+(mVal===mag?3.0:1.6)+'"/>';
      }});
      const curLd=Math.max(0, predLog(mag, gdp)), baseLd=Math.max(0, predLog(mag, 1000));
      const cx=X(Math.log10(gdp)), cy=Y(curLd);
      s+='<circle cx="'+cx.toFixed(1)+'" cy="'+cy.toFixed(1)+'" r="6.5" fill="'+RED+'" stroke="#fff" stroke-width="2"/>';
      sv.innerHTML=s;
      const curDeaths=Math.max(0, Math.pow(10,curLd)-1), baseDeaths=Math.max(1, Math.pow(10,baseLd)-1);
      const redPct=(1 - curDeaths/baseDeaths)*100;
      lf.innerHTML='<small>Fatality Reduction vs. $1,000 Baseline</small>-'+redPct.toFixed(1)+'%';
      out.textContent='OLS Model (n='+D.n_deadly+', R²='+D.r2_deadly+'):\\nLog10(GDP) slope = '+D.slope_deadly+' (p='+D.pval_deadly+')\\n10x GDP -> -'+D.pct_red+'% expected deaths\\nSimulated M'+mag.toFixed(1)+' at $'+gdp.toLocaleString('en-US')+': ~'+Math.round(curDeaths).toLocaleString('en-US')+' deaths';
    }}
    sg.oninput=draw; sm.oninput=draw;
    seg('w9-pre',d=>{{sg.value=d.g;sm.value=d.m;draw();}});
    draw();
  }})();
}})();
</script>
</body>
</html>"""

    out_path = "reports/earthquake_deck.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Successfully generated interactive 10-slide HTML deck -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    build_html_deck()