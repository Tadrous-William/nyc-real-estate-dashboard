"""NYC Real Estate Investor Dashboard (Streamlit).

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import PROCESSED, MODELS, REPORTS, HOME_TYPES, mix_adjusted_change  # noqa: E402
from train_model import prepare  # noqa: E402

st.set_page_config(page_title="NYC Real Estate Investor Dashboard", page_icon="🏙️", layout="wide")

BOROUGH_COLORS = {"Manhattan": "#2563EB", "Brooklyn": "#DC2626", "Queens": "#059669",
                  "Bronx": "#D97706", "Staten Island": "#7C3AED"}
MAIN_TYPES = ["One-family home", "Two-family home", "Three-family home", "Condo apartment", "Co-op apartment"]
ALL_TYPES = MAIN_TYPES + ["Rental building (walk-up)", "Rental building (elevator)"]


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading sales...")
def load_sales():
    return pd.read_parquet(PROCESSED / "sales.parquet")


@st.cache_data
def load_rents():
    return pd.read_parquet(PROCESSED / "rents.parquet")


@st.cache_data
def load_yield():
    return pd.read_parquet(PROCESSED / "zip_yield.parquet")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODELS / "home_price_model.joblib")
    meta = json.loads((MODELS / "model_metrics.json").read_text())
    return bundle, meta


def money(x):
    if pd.isna(x):
        return "-"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    return f"${x / 1e3:,.0f}K" if abs(x) >= 1e4 else f"${x:,.0f}"


def pct(x):
    return "-" if pd.isna(x) else f"{x:+.1%}"


sales = load_sales()
last_day = sales["sale_date"].max()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Filters")
boroughs = st.sidebar.multiselect("Borough", list(BOROUGH_COLORS), default=list(BOROUGH_COLORS))
types = st.sidebar.multiselect("Property type", ALL_TYPES, default=MAIN_TYPES)
years = st.sidebar.slider("Years", 2019, int(last_day.year), (2019, int(last_day.year)))
st.sidebar.caption(f"Data: NYC Department of Finance sales, Jan 2019 to {last_day:%b %Y}. "
                   "Rents: Zillow Observed Rent Index.")

f = sales[sales["borough_name"].isin(boroughs) & sales["property_type"].isin(types)
          & sales["year"].between(*years)]

st.title("NYC Real Estate Investor Dashboard")
st.caption(f"{len(sales):,} market sales across New York City, updated to {last_day:%B %Y} · "
           "market trends, neighborhood prices, a home price estimator and rental yields")

tab_market, tab_hoods, tab_value, tab_invest, tab_report = st.tabs(
    ["Market overview", "Neighborhoods", "Price estimator", "Investor view", "Monthly report"])

# ---------------------------------------------------------------- market overview
with tab_market:
    end = f["sale_date"].max() if len(f) else last_day
    cur = f[f["sale_date"] > end - pd.DateOffset(months=12)]
    prev = f[(f["sale_date"] <= end - pd.DateOffset(months=12)) & (f["sale_date"] > end - pd.DateOffset(months=24))]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sales, last 12 months", f"{len(cur):,}", pct(len(cur) / len(prev) - 1) if len(prev) else None)
    mp, pp = cur["sale_price"].median(), prev["sale_price"].median()
    c2.metric("Median sale price", money(mp), pct(mp / pp - 1) if len(prev) else None)
    ms, ps = cur["price_per_sqft"].median(), prev["price_per_sqft"].median()
    c3.metric("Median price per sq ft (homes)", money(ms), pct(ms / ps - 1) if not pd.isna(ps) else None)
    c4.metric("Sales over $1M", f"{(cur['sale_price'] >= 1e6).mean():.0%}")

    monthly = (f.groupby(["month", "borough_name"])["sale_price"].median().rename("median_price").reset_index())
    monthly["median_price"] = monthly.groupby("borough_name")["median_price"].transform(
        lambda s: s.rolling(3, min_periods=1).median())
    fig = px.line(monthly, x="month", y="median_price", color="borough_name", color_discrete_map=BOROUGH_COLORS,
                  labels={"month": "", "median_price": "Median sale price (3-month rolling)", "borough_name": ""},
                  title="Median sale price by borough")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.08), height=430)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    vol = f.groupby(["month", "property_type"]).size().rename("sales").reset_index()
    fig = px.bar(vol, x="month", y="sales", color="property_type", title="Monthly sales by property type",
                 labels={"month": "", "sales": "Sales", "property_type": ""})
    fig.update_layout(legend=dict(orientation="h", y=-0.2), height=420, bargap=0.05)
    left.plotly_chart(fig, use_container_width=True)
    by_type = cur.groupby("property_type")["sale_price"].median().sort_values().reset_index()
    fig = px.bar(by_type, x="sale_price", y="property_type", orientation="h", text=by_type["sale_price"].map(money),
                 title="Median price by property type (last 12 months)",
                 labels={"sale_price": "Median sale price", "property_type": ""})
    fig.update_traces(marker_color="#2563EB", textposition="outside")
    fig.update_layout(height=420, xaxis=dict(showgrid=False))
    right.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- neighborhoods
with tab_hoods:
    end = f["sale_date"].max() if len(f) else last_day
    cur = f[f["sale_date"] > end - pd.DateOffset(months=12)]
    prev = f[(f["sale_date"] <= end - pd.DateOffset(months=12)) & (f["sale_date"] > end - pd.DateOffset(months=24))]
    a = cur.groupby(["borough_name", "neighborhood"]).agg(sales=("sale_price", "size"),
                                                          median_price=("sale_price", "median"),
                                                          price_per_sqft=("price_per_sqft", "median"))
    hood = a.join(mix_adjusted_change(cur, prev, ["borough_name", "neighborhood"], min_type_sales=15, min_total=50)[["price_change_1y"]], how="left")
    hood = hood[hood["sales"] >= 20].reset_index()

    left, right = st.columns([3, 2])
    growth = hood.dropna(subset=["price_change_1y"]).nlargest(15, "price_change_1y").sort_values("price_change_1y")
    fig = px.bar(growth, x="price_change_1y", y="neighborhood", color="borough_name", orientation="h",
                 color_discrete_map=BOROUGH_COLORS,
                 title="Fastest-rising neighborhoods (like-for-like, last 12 months)",
                 labels={"price_change_1y": "Change in median price", "neighborhood": "", "borough_name": ""})
    fig.update_layout(xaxis_tickformat="+.0%", height=520, legend=dict(orientation="h", y=-0.12))
    fig.update_yaxes(categoryorder="total ascending")
    left.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**All neighborhoods (20+ sales in the last 12 months)**")
        st.dataframe(hood.sort_values("median_price", ascending=False)[
            ["neighborhood", "borough_name", "sales", "median_price", "price_per_sqft", "price_change_1y"]],
            hide_index=True, height=480,
            column_config={"neighborhood": "Neighborhood", "borough_name": "Borough", "sales": "Sales",
                           "median_price": st.column_config.NumberColumn("Median price ($)", format="localized"),
                           "price_per_sqft": st.column_config.NumberColumn("$ per sq ft", format="%.0f"),
                           "price_change_1y": st.column_config.NumberColumn("1-year change", format="percent")})

    pts = cur.dropna(subset=["latitude", "longitude"])
    pts = pts.sample(min(len(pts), 12000), random_state=1)
    pts = pts.assign(price_label=pts["sale_price"].map(money), log_price=np.log10(pts["sale_price"]))
    fig = px.scatter_map(pts, lat="latitude", lon="longitude", color="log_price", color_continuous_scale="Turbo",
                         hover_name="address", hover_data={"price_label": True, "property_type": True,
                                                           "neighborhood": True, "latitude": False,
                                                           "longitude": False, "log_price": False},
                         zoom=9.6, center={"lat": 40.70, "lon": -73.94}, height=620, map_style="carto-positron",
                         title="Where homes sold in the last 12 months (color = sale price)")
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    ticks = [np.log10(v) for v in (250e3, 500e3, 1e6, 2e6, 5e6)]
    fig.update_layout(coloraxis_colorbar=dict(title="Price", tickvals=ticks,
                                              ticktext=[money(10 ** t) for t in ticks]))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- price estimator
with tab_value:
    bundle, meta = load_model()
    t = meta["test"]
    st.markdown("#### Estimate the value of a 1-3 family home")
    st.caption(f"XGBoost model trained on {meta['trained_rows']:,} home sales. Tested on "
               f"{t['n']:,} sales from {meta['test_period']} that it had never seen: median error "
               f"{t['median_abs_pct_error']:.1%}, {t['within_20pct']:.0%} of estimates within ±20% of the real price.")
    homes = sales[sales["property_type"].isin(HOME_TYPES)]
    c1, c2, c3 = st.columns(3)
    b = c1.selectbox("Borough", ["Brooklyn", "Queens", "Staten Island", "Bronx", "Manhattan"])
    hoods = homes[homes["borough_name"] == b]["neighborhood"].value_counts()
    hoods = sorted(hoods[hoods >= 30].index)
    h = c2.selectbox("Neighborhood", hoods)
    ptype = c3.selectbox("Property type", HOME_TYPES)
    ref = homes[(homes["neighborhood"] == h) & (homes["property_type"] == ptype)]
    if ref.empty:
        ref = homes[homes["neighborhood"] == h]
    c4, c5, c6 = st.columns(3)
    sqft = c4.number_input("Building size (sq ft)", 400, 10000, int(ref["gross_square_feet"].median() or 1800), 50)
    land = c5.number_input("Lot size (sq ft)", 500, 20000, int(ref["land_square_feet"].median() or 2000), 50)
    built = c6.number_input("Year built", 1850, 2026, int(ref["year_built"].median() or 1930), 1)

    zips = ref["zip_code"].value_counts().index.tolist() or homes[homes["neighborhood"] == h]["zip_code"].unique().tolist()
    row = pd.DataFrame([{"borough_name": b, "neighborhood": h, "zip_code": zips[0], "property_type": ptype,
                         "gross_square_feet": sqft, "land_square_feet": land, "year_built": built,
                         "residential_units": HOME_TYPES.index(ptype) + 1,
                         "latitude": ref["latitude"].median(), "longitude": ref["longitude"].median(),
                         "sale_date": last_day}])
    X, _ = prepare(row, bundle["levels"])
    est = float(np.exp(bundle["model"].predict(X))[0])
    err = t["median_abs_pct_error"]

    k1, k2, k3 = st.columns(3)
    k1.metric("Estimated value", money(est))
    k2.metric("Likely range", f"{money(est * (1 - err))} to {money(est * (1 + err))}".replace("$", "\\$"))
    k3.metric("Estimated price per sq ft", money(est / sqft))

    recent = homes[(homes["neighborhood"] == h) & (homes["sale_date"] > last_day - pd.DateOffset(months=24))]
    recent = recent.dropna(subset=["gross_square_feet", "year_built"])
    if len(recent):
        d = (np.abs(np.log(recent["gross_square_feet"] / sqft)) * 2
             + np.abs(recent["year_built"] - built) / 40
             + (recent["property_type"] != ptype) * 0.5)
        comps = recent.assign(score=d).nsmallest(6, "score")
        st.markdown("**Comparable sales (same neighborhood, last 24 months)**")
        st.dataframe(comps[["sale_date", "address", "property_type", "gross_square_feet", "year_built",
                            "sale_price", "price_per_sqft"]], hide_index=True,
                     column_config={"sale_date": st.column_config.DateColumn("Sold", format="MMM YYYY"),
                                    "address": "Address", "property_type": "Type",
                                    "gross_square_feet": st.column_config.NumberColumn("Sq ft", format="%d"),
                                    "year_built": st.column_config.NumberColumn("Built", format="%d"),
                                    "sale_price": st.column_config.NumberColumn("Price ($)", format="localized"),
                                    "price_per_sqft": st.column_config.NumberColumn("$ per sq ft", format="%.0f")})
        fig = go.Figure(go.Bar(x=comps["address"], y=comps["sale_price"], marker_color="#94A3B8",
                               text=comps["sale_price"].map(money), textposition="outside", name="Comparable sales"))
        fig.add_hline(y=est, line_dash="dash", line_color="#2563EB", annotation_text=f"Estimate {money(est)}")
        fig.update_layout(height=380, title="Estimate vs comparable sales", yaxis_title="Sale price", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    imp = pd.Series(meta["feature_importance"]).sort_values()
    names = {"gross_square_feet": "Building size", "land_square_feet": "Lot size", "year_built": "Year built",
             "residential_units": "Units", "latitude": "Location (lat)", "longitude": "Location (lon)",
             "t": "Sale date (market trend)", "borough_name": "Borough", "neighborhood": "Neighborhood",
             "zip_code": "Zip code", "property_type": "Property type"}
    with st.expander("What drives the estimate? (model feature importance)"):
        fig = px.bar(x=imp.values, y=[names.get(i, i) for i in imp.index], orientation="h",
                     labels={"x": "Importance", "y": ""})
        fig.update_traces(marker_color="#2563EB")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        base = meta["baseline_test"]["median_abs_pct_error"]
        st.caption(f"For comparison, a simple 'neighborhood price per sq ft x size' estimate has a median error "
                   f"of {base:.1%} on the same 2026 sales, versus {err:.1%} for this model.")

# ---------------------------------------------------------------- investor view
with tab_invest:
    zy = load_yield()
    zy = zy[zy["borough_name"].isin(boroughs)]
    rents = load_rents()
    st.markdown("#### Where does renting out pay the most?")
    st.caption("Gross rental yield = 12 x typical monthly rent (Zillow ZORI) / median sale price of condos and "
               "one-family homes in the last 12 months, before taxes, fees and vacancy. Co-ops are excluded because "
               "their price does not include the monthly maintenance fee.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Zip codes analysed", f"{len(zy)}")
    c2.metric("Median gross yield", f"{zy['gross_yield'].median():.1%}")
    top = zy.iloc[0] if len(zy) else None
    if top is not None:
        c3.metric("Highest yield", f"{top['gross_yield']:.1%}")
        c3.caption(f"Zip {top['zip_code']} · {top['neighborhood']}")

    left, right = st.columns(2)
    t15 = zy.nlargest(15, "gross_yield").sort_values("gross_yield")
    fig = px.bar(t15, x="gross_yield", y=t15["zip_code"] + " · " + t15["neighborhood"], color="borough_name",
                 color_discrete_map=BOROUGH_COLORS, orientation="h", title="Top 15 zip codes by gross rental yield",
                 labels={"gross_yield": "Gross yield", "y": "", "borough_name": ""})
    fig.update_layout(xaxis_tickformat=".1%", height=520, legend=dict(orientation="h", y=-0.12))
    fig.update_yaxes(categoryorder="total ascending")
    left.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(zy, x="median_price", y="rent", size="sales", color="gross_yield",
                     color_continuous_scale="Tealgrn", hover_name="neighborhood",
                     hover_data={"zip_code": True, "gross_yield": ":.1%", "median_price": ":$,.0f", "rent": ":$,.0f"},
                     title="Price vs rent by zip code (bigger = more sales)",
                     labels={"median_price": "Median sale price", "rent": "Typical monthly rent", "gross_yield": "Yield"})
    fig.update_layout(height=520, coloraxis_colorbar_tickformat=".1%")
    right.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([2, 3])
    zsel = left.selectbox("Rent trend for zip code", zy["zip_code"] + " · " + zy["neighborhood"])
    zc = zsel.split(" · ")[0]
    rz = rents[rents["zip_code"] == zc]
    zrow = zy[zy["zip_code"] == zc].iloc[0]
    left.metric("Typical rent now", money(zrow["rent"]), pct(zrow["rent_change_1y"]))
    left.metric("Median price (last 12 months)", money(zrow["median_price"]))
    left.metric("Gross yield", f"{zrow['gross_yield']:.1%}")
    fig = px.line(rz, x="date", y="rent", title=f"Typical rent in {zsel}", labels={"date": "", "rent": "Monthly rent"})
    fig.update_traces(line_color="#059669")
    fig.update_layout(height=360)
    right.plotly_chart(fig, use_container_width=True)

    st.dataframe(zy[["zip_code", "neighborhood", "borough_name", "sales", "median_price", "rent", "rent_change_1y",
                     "gross_yield"]], hide_index=True, height=360,
                 column_config={"zip_code": "Zip", "neighborhood": "Main neighborhood", "borough_name": "Borough",
                                "sales": "Sales", "median_price": st.column_config.NumberColumn("Median price ($)", format="localized"),
                                "rent": st.column_config.NumberColumn("Monthly rent ($)", format="%.0f"),
                                "rent_change_1y": st.column_config.NumberColumn("Rent 1-yr change", format="percent"),
                                "gross_yield": st.column_config.NumberColumn("Gross yield", format="percent")})

# ---------------------------------------------------------------- report
with tab_report:
    st.markdown("#### Monthly market report (PDF)")
    st.write("A ready-to-send PDF with the month's key numbers, price trends, top neighborhoods and rental yields. "
             "It is generated automatically from the same data.")
    pdfs = sorted(REPORTS.glob("nyc_market_report_*.pdf"))
    if pdfs:
        st.download_button(f"Download {pdfs[-1].name}", pdfs[-1].read_bytes(), file_name=pdfs[-1].name,
                           mime="application/pdf")
    else:
        st.info("Run `python src/report.py` to create the latest report.")

st.divider()
st.caption("Sources: NYC Department of Finance Rolling & Annualized Sales (NYC Open Data), Zillow Observed Rent Index, "
           "US Census ZCTA gazetteer. Built by Tadrous William, data scientist.")
