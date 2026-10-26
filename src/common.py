"""Shared paths and lookups for the NYC real estate dashboard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
ASSETS = ROOT / "assets"
for p in (RAW, PROCESSED, MODELS, REPORTS, ASSETS):
    p.mkdir(parents=True, exist_ok=True)

BOROUGHS = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}

# NYC Department of Finance building class categories -> simple property types
PROPERTY_TYPE_RULES = [
    ("01 ", "One-family home"),
    ("02 ", "Two-family home"),
    ("03 ", "Three-family home"),
    ("04 ", "Condo apartment"),
    ("12 ", "Condo apartment"),
    ("13 ", "Condo apartment"),
    ("15 ", "Condo apartment"),
    ("17 ", "Condo apartment"),
    ("09 ", "Co-op apartment"),
    ("10 ", "Co-op apartment"),
    ("07 ", "Rental building (walk-up)"),
    ("08 ", "Rental building (elevator)"),
    ("14 ", "Rental building (walk-up)"),
]


def property_type(category: str) -> str:
    c = (category or "").upper()
    for prefix, name in PROPERTY_TYPE_RULES:
        if c.startswith(prefix):
            return name
    return "Other"


HOME_TYPES = ["One-family home", "Two-family home", "Three-family home"]


def mix_adjusted_change(cur, prev, by, min_type_sales=10, min_total=30):
    """Year-over-year price change that compares like with like.

    A plain median can jump just because a neighborhood sold more expensive property types this year.
    Here the change is computed inside each property type, then averaged with this year's sales as weights.
    """
    import pandas as pd
    a = cur.groupby(by + ["property_type"])["sale_price"].agg(["median", "size"])
    b = prev.groupby(by + ["property_type"])["sale_price"].agg(["median", "size"])
    j = a.join(b, lsuffix="_now", rsuffix="_ago", how="inner")
    j = j[(j["size_now"] >= min_type_sales) & (j["size_ago"] >= min_type_sales)]
    j["chg"] = j["median_now"] / j["median_ago"] - 1
    j["w"] = j["size_now"]
    out = j.groupby(level=by).apply(lambda g: pd.Series({
        "price_change_1y": (g["chg"] * g["w"]).sum() / g["w"].sum(), "compared_sales": g["w"].sum()}))
    return out[out["compared_sales"] >= min_total]
