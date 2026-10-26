"""Download the public data used by the dashboard.

Sources (all free and public):
- NYC Department of Finance, annualized sales 2016-2025 (NYC Open Data w2pb-icbu)
- NYC Department of Finance, rolling sales, last 12 months (NYC Open Data usep-8jbt)
- Zillow Observed Rent Index (ZORI) by zip code, monthly
- US Census ZCTA gazetteer (zip code centroids)
"""
import io
import time
import zipfile

import pandas as pd
import requests

from common import RAW

SOCRATA = "https://data.cityofnewyork.us/resource/{}.json"
FIELDS = ("borough,neighborhood,building_class_category,address,zip_code,residential_units,"
          "commercial_units,total_units,land_square_feet,gross_square_feet,year_built,"
          "building_class_at_time_of,sale_price,sale_date,block,lot")
PAGE = 50_000


def fetch_socrata(dataset: str, where: str, extra_fields: str = "") -> pd.DataFrame:
    fields = FIELDS + ("," + extra_fields if extra_fields else "")
    frames, offset = [], 0
    while True:
        params = {"$select": fields, "$where": where, "$limit": PAGE, "$offset": offset, "$order": ":id"}
        for attempt in range(5):
            try:
                r = requests.get(SOCRATA.format(dataset), params=params, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"  retry {attempt + 1}: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed to download {dataset} at offset {offset}")
        rows = r.json()
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        offset += len(rows)
        print(f"  {dataset}: {offset:,} rows")
        if len(rows) < PAGE:
            break
    return pd.concat(frames, ignore_index=True)


def main():
    print("NYC sales 2019-2025 ...")
    hist = fetch_socrata("w2pb-icbu", "sale_date >= '2019-01-01' AND sale_price > 10000", "latitude,longitude")
    hist.to_csv(RAW / "nyc_sales_2019_2025.csv", index=False)

    print("NYC sales, rolling last 12 months (keeping 2026 only) ...")
    roll = fetch_socrata("usep-8jbt", "sale_date >= '2026-01-01' AND sale_price > 10000")
    roll.to_csv(RAW / "nyc_sales_2026.csv", index=False)

    print("Zillow rents by zip ...")
    url = "https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_uc_sfrcondomfr_sm_month.csv"
    zori = pd.read_csv(url)
    zori[zori["State"] == "NY"].to_csv(RAW / "zillow_zori_zip_ny.csv", index=False)

    print("Census zip code centroids ...")
    gz = requests.get("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/"
                      "2023_Gaz_zcta_national.zip", timeout=120)
    with zipfile.ZipFile(io.BytesIO(gz.content)) as z:
        name = [n for n in z.namelist() if n.endswith(".txt")][0]
        cent = pd.read_csv(z.open(name), sep="\t", dtype={"GEOID": str})
    cent.columns = [c.strip() for c in cent.columns]
    cent[["GEOID", "INTPTLAT", "INTPTLONG"]].to_csv(RAW / "zip_centroids.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
