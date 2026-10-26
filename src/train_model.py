"""Train the home price estimator (automated valuation model) for NYC 1-3 family homes.

Honest evaluation: the model is trained on 2019-2025 sales and tested on 2026 sales it has never seen.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from common import PROCESSED, MODELS, HOME_TYPES

CAT = ["borough_name", "neighborhood", "zip_code", "property_type"]
NUM = ["gross_square_feet", "land_square_feet", "year_built", "residential_units", "latitude", "longitude", "t"]
FEATURES = CAT + NUM
START = pd.Timestamp("2019-01-01")


def prepare(df: pd.DataFrame, levels: dict | None = None):
    X = df.copy()
    X["t"] = (X["sale_date"].dt.year - START.year) * 12 + X["sale_date"].dt.month - 1
    if levels is None:
        levels = {c: sorted(X[c].dropna().astype(str).unique()) for c in CAT}
    for c in CAT:
        vals = X[c].astype(str)
        X[c] = pd.Categorical(vals.where(vals.isin(levels[c])), categories=levels[c])  # unseen -> missing
    return X[FEATURES], levels


def evaluate(y_true_price, y_pred_price) -> dict:
    ape = np.abs(y_pred_price - y_true_price) / y_true_price
    return {"median_abs_pct_error": float(np.median(ape)), "mean_abs_pct_error": float(np.mean(ape)),
            "within_10pct": float(np.mean(ape <= 0.10)), "within_20pct": float(np.mean(ape <= 0.20)),
            "r2_log_price": float(r2_score(np.log(y_true_price), np.log(y_pred_price))), "n": int(len(ape))}


def make_model(n_estimators=4000):
    return XGBRegressor(n_estimators=n_estimators, learning_rate=0.03, max_depth=8, min_child_weight=5,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, tree_method="hist",
                        enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1, random_state=42)


def main():
    sales = pd.read_parquet(PROCESSED / "sales.parquet")
    homes = sales[sales["property_type"].isin(HOME_TYPES)].dropna(
        subset=["gross_square_feet", "land_square_feet", "year_built"]).copy()
    homes = homes[homes["sale_date"] >= START]

    train = homes[homes["sale_date"] < "2025-07-01"]
    valid = homes[(homes["sale_date"] >= "2025-07-01") & (homes["sale_date"] < "2026-01-01")]
    test = homes[homes["sale_date"] >= "2026-01-01"]
    Xtr, levels = prepare(train)
    Xva, _ = prepare(valid, levels)
    Xte, _ = prepare(test, levels)

    m = make_model()
    m.set_params(early_stopping_rounds=150, eval_metric="mae")
    m.fit(Xtr, np.log(train["sale_price"]), eval_set=[(Xva, np.log(valid["sale_price"]))], verbose=False)
    best = int(m.best_iteration) + 1
    test_metrics = evaluate(test["sale_price"].values, np.exp(m.predict(Xte)))
    print(f"best trees: {best}; 2026 test: {json.dumps(test_metrics, indent=1)}")

    # Baseline to show the model adds value: neighborhood median price per sqft x size
    ppsf = train.groupby("neighborhood")["price_per_sqft"].median()
    base = test["neighborhood"].map(ppsf).fillna(train["price_per_sqft"].median()) * test["gross_square_feet"]
    base_metrics = evaluate(test["sale_price"].values, base.values)
    print(f"baseline (neighborhood $/sqft x size): median error {base_metrics['median_abs_pct_error']:.1%}")

    # Final model on all data up to today, same number of trees
    Xall, levels = prepare(homes)
    final = make_model(best)
    final.fit(Xall, np.log(homes["sale_price"]))
    imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False)

    joblib.dump({"model": final, "levels": levels, "features": FEATURES, "start": str(START.date())},
                MODELS / "home_price_model.joblib")
    meta = {"trained_rows": int(len(homes)), "train_period": "2019-01 to 2025-12",
            "test_period": "2026-01 to " + test["sale_date"].max().strftime("%Y-%m"),
            "test": test_metrics, "baseline_test": base_metrics, "trees": best,
            "feature_importance": imp.round(4).to_dict()}
    (MODELS / "model_metrics.json").write_text(json.dumps(meta, indent=2))
    print(imp.round(3))


if __name__ == "__main__":
    main()
