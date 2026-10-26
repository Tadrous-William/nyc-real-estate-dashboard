"""Clean the raw downloads into analysis-ready tables.

Outputs (data/processed):
- sales.parquet     one row per market sale, 2019 to today
- rents.parquet     Zillow typical rent by NYC zip code and month
- zip_yield.parquet gross rental yield by zip code (last 12 months)
"""
import numpy as np
import pandas as pd

from common import RAW, PROCESSED, BOROUGHS, property_type

NYC_COUNTIES = {"New York County": "Manhattan", "Kings County": "Brooklyn", "Queens County": "Queens",
                "Bronx County": "Bronx", "Richmond County": "Staten Island"}


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def load_sales() -> pd.DataFrame:
    hist = pd.read_csv(RAW / "nyc_sales_2019_2025.csv", dtype=str)
    new = pd.read_csv(RAW / "nyc_sales_2026.csv", dtype=str)
    df = pd.concat([hist, new], ignore_index=True)

    for c in ["sale_price", "land_square_feet", "gross_square_feet", "year_built", "residential_units",
              "commercial_units", "total_units", "block", "lot", "borough", "latitude", "longitude"]:
        df[c] = to_num(df[c])
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df["zip_code"] = df["zip_code"].astype(str).str.extract(r"(\d{5})")[0]
    df["borough_name"] = df["borough"].map(BOROUGHS)
    df["neighborhood"] = df["neighborhood"].str.strip().str.title()
    df["address"] = df["address"].str.strip().str.title()
    df["property_type"] = df["building_class_category"].map(property_type)
    df["bbl"] = (df["borough"] * 1_000_000_000 + df["block"] * 10_000 + df["lot"]).astype("Int64")
    return df


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["sale_date", "sale_price", "borough_name", "zip_code"])
    df = df[df["sale_price"] >= 50_000]                      # drop transfers / non-market deeds
    df = df[df["property_type"] != "Other"]

    # Portfolio sales: one deed price repeated over many lots on the same day -> not a single-property price
    grp = df.groupby(["sale_date", "sale_price", "block", "borough"])["lot"].transform("count")
    df = df[grp == 1]

    df.loc[df["gross_square_feet"] <= 0, "gross_square_feet"] = np.nan
    df.loc[df["land_square_feet"] <= 0, "land_square_feet"] = np.nan
    df.loc[(df["year_built"] < 1800) | (df["year_built"] > 2026), "year_built"] = np.nan

    # Price per square foot is only meaningful where the building area is known (1-3 family homes, rentals)
    df["price_per_sqft"] = df["sale_price"] / df["gross_square_feet"]
    df.loc[(df["gross_square_feet"] < 300) | (df["price_per_sqft"] < 50) | (df["price_per_sqft"] > 5000),
           "price_per_sqft"] = np.nan

    # Remove extreme prices inside each property type (top 0.5% / bottom 1%)
    lo = df.groupby("property_type")["sale_price"].transform(lambda s: s.quantile(0.01))
    hi = df.groupby("property_type")["sale_price"].transform(lambda s: s.quantile(0.995))
    df = df[(df["sale_price"] >= lo) & (df["sale_price"] <= hi)]

    # Fill missing coordinates: same property sold before, otherwise the zip code centre
    known = df.dropna(subset=["latitude"]).drop_duplicates("bbl", keep="last").set_index("bbl")
    miss = df["latitude"].isna()
    df.loc[miss, "latitude"] = df.loc[miss, "bbl"].map(known["latitude"])
    df.loc[miss, "longitude"] = df.loc[miss, "bbl"].map(known["longitude"])
    cent = pd.read_csv(RAW / "zip_centroids.csv", dtype={"GEOID": str}).set_index("GEOID")
    miss = df["latitude"].isna()
    df.loc[miss, "latitude"] = df.loc[miss, "zip_code"].map(cent["INTPTLAT"])
    df.loc[miss, "longitude"] = df.loc[miss, "zip_code"].map(cent["INTPTLONG"])

    df["month"] = df["sale_date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["sale_date"].dt.year
    keep = ["sale_date", "month", "year", "borough_name", "neighborhood", "zip_code", "address",
            "property_type", "building_class_category", "residential_units", "total_units",
            "gross_square_feet", "land_square_feet", "year_built", "sale_price", "price_per_sqft",
            "latitude", "longitude", "bbl"]
    df = df[keep].sort_values("sale_date").reset_index(drop=True)
    print(f"sales: {n0:,} raw rows -> {len(df):,} clean market sales")
    return df


def clean_rents() -> pd.DataFrame:
    z = pd.read_csv(RAW / "zillow_zori_zip_ny.csv", dtype={"RegionName": str})
    z = z[z["CountyName"].isin(NYC_COUNTIES)]
    date_cols = [c for c in z.columns if c[:2] == "20"]
    long = z.melt(id_vars=["RegionName", "CountyName"], value_vars=date_cols, var_name="date", value_name="rent")
    long = long.dropna(subset=["rent"])
    long["date"] = pd.to_datetime(long["date"]).dt.to_period("M").dt.to_timestamp()
    long["zip_code"] = long["RegionName"].str.zfill(5)
    long["borough_name"] = long["CountyName"].map(NYC_COUNTIES)
    out = long[["zip_code", "borough_name", "date", "rent"]].reset_index(drop=True)
    print(f"rents: {out['zip_code'].nunique()} zip codes, {out['date'].min():%Y-%m} to {out['date'].max():%Y-%m}")
    return out


def zip_yield(sales: pd.DataFrame, rents: pd.DataFrame) -> pd.DataFrame:
    """Gross rental yield = 12 x typical monthly rent / median price of condos and one-family homes (last 12 months).

    Co-ops are excluded: their price does not include the monthly maintenance fee, which would overstate the yield.
    """
    end = sales["sale_date"].max()
    recent = sales[(sales["sale_date"] > end - pd.DateOffset(months=12))
                   & sales["property_type"].isin(["Condo apartment", "One-family home"])]
    price = recent.groupby("zip_code").agg(median_price=("sale_price", "median"), sales=("sale_price", "size"),
                                           borough_name=("borough_name", "first"),
                                           neighborhood=("neighborhood", lambda s: s.mode().iat[0]))
    price = price[price["sales"] >= 15]
    latest = rents.sort_values("date").groupby("zip_code").tail(1).set_index("zip_code")
    year_ago = rents.merge(latest[["date"]].rename(columns={"date": "latest"}), left_on="zip_code", right_index=True)
    year_ago = year_ago[year_ago["date"] == year_ago["latest"] - pd.DateOffset(years=1)].set_index("zip_code")
    out = price.join(latest[["rent", "date"]], how="inner")
    out["rent_change_1y"] = out["rent"] / year_ago["rent"] - 1
    out["gross_yield"] = out["rent"] * 12 / out["median_price"]
    cent = pd.read_csv(RAW / "zip_centroids.csv", dtype={"GEOID": str}).set_index("GEOID")
    out["latitude"] = cent["INTPTLAT"].reindex(out.index).values
    out["longitude"] = cent["INTPTLONG"].reindex(out.index).values
    out = out.rename(columns={"date": "rent_month"}).reset_index().sort_values("gross_yield", ascending=False)
    print(f"yield: {len(out)} zip codes with enough sales and rent data")
    return out


def main():
    sales = clean_sales(load_sales())
    rents = clean_rents()
    zy = zip_yield(sales, rents)
    sales.to_parquet(PROCESSED / "sales.parquet", index=False)
    rents.to_parquet(PROCESSED / "rents.parquet", index=False)
    zy.to_parquet(PROCESSED / "zip_yield.parquet", index=False)
    print(sales.groupby("property_type")["sale_price"].agg(["size", "median"]).round(0))


if __name__ == "__main__":
    main()
