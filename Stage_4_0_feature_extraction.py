# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

MERGED_CSV = "data/index70_with_sdg17.csv"  
OUTDIR = Path("cluster_outputs")
OUTDIR.mkdir(exist_ok=True, parents=True)

YEAR_MIN, YEAR_MAX = 2003, 2022
RECENT_FROM = 2018
SEED = 2025
np.random.seed(SEED)

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.replace(r"[^0-9A-Za-z]+", "_", regex=True)
                  .str.strip("_")
                  .str.lower()
    )
    return df

def linear_trend(years: np.ndarray, values: np.ndarray):
    mask = np.isfinite(years) & np.isfinite(values)
    if mask.sum() < 3:
        return np.nan, np.nan
    x = years[mask].astype(float)
    y = values[mask].astype(float)
    a, b = np.polyfit(x, y, 1)  
    y_hat = a * x + b
    resid_std = np.std(y - y_hat, ddof=1) if len(y) > 1 else np.nan
    return a, resid_std

def series_features(df_country: pd.DataFrame, value_col: str, recent_from: int = RECENT_FROM):
    d = df_country.sort_values("year")
    y = d[value_col].to_numpy(dtype=float)
    t = d["year"].to_numpy(dtype=float)

    slope_all, residstd_all = linear_trend(t, y)
    mean_all = np.nanmean(y)

    m_recent = d["year"] >= recent_from
    slope_recent, _ = linear_trend(t[m_recent], y[m_recent])
    mean_recent = np.nanmean(y[m_recent])

    dy = np.diff(y)
    max_rise = np.nanmax(dy) if dy.size else np.nan
    max_drop = np.nanmin(dy) if dy.size else np.nan

    p10 = np.nanpercentile(y, 10)
    p50 = np.nanpercentile(y, 50)
    p90 = np.nanpercentile(y, 90)

    return {
        f"{value_col}__mean_all": mean_all,
        f"{value_col}__slope_all": slope_all,
        f"{value_col}__residstd_all": residstd_all,
        f"{value_col}__mean_recent": mean_recent,
        f"{value_col}__slope_recent": slope_recent,
        f"{value_col}__max_rise": max_rise,
        f"{value_col}__max_drop": max_drop,
        f"{value_col}__p10": p10,
        f"{value_col}__p50": p50,
        f"{value_col}__p90": p90,
    }

df = pd.read_csv(MERGED_CSV)
df = norm_cols(df)

need = {"countrycode","countryname","year","index_70"}
assert need.issubset(df.columns), f"missing col：{need - set(df.columns)}"
sdg_cols = [f"sdg_{i}" for i in range(1,18)]
for c in sdg_cols:
    if c not in df.columns:
        raise KeyError(f"missing col：{c}")

df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

# =====  Index_70 cty features =====
rows_idx = []
for code, d in df.groupby("countrycode"):
    feats = series_features(d, "index_70", recent_from=RECENT_FROM)
    feats.update({
        "countrycode": code,
        "countryname": d["countryname"].dropna().iloc[0] if d["countryname"].notna().any() else code
    })
    rows_idx.append(feats)
idx_feat = pd.DataFrame(rows_idx)
idx_feat.to_csv(OUTDIR / "features_index70.csv", index=False)

print("output：features_index70.csv shape：", idx_feat.shape)
print(idx_feat.head(3))

# =====  SDG cty features=====
rows_sdg = []
for code, d in df.groupby("countrycode"):
    row = {
        "countrycode": code,
        "countryname": d["countryname"].dropna().iloc[0] if d["countryname"].notna().any() else code
    }
    for i in range(1, 18):
        col = f"sdg_{i}"
        fi = series_features(d, col, recent_from=RECENT_FROM)
        row[f"{col}__mean_all"]     = fi[f"{col}__mean_all"]
        row[f"{col}__slope_all"]    = fi[f"{col}__slope_all"]
        row[f"{col}__mean_recent"]  = fi[f"{col}__mean_recent"]
        row[f"{col}__slope_recent"] = fi[f"{col}__slope_recent"]
    rows_sdg.append(row)

sdg_feat = pd.DataFrame(rows_sdg)
sdg_feat.to_csv(OUTDIR / "features_sdg.csv", index=False)

print("output：features_sdg.csv shape：", sdg_feat.shape)
print(sdg_feat.iloc[:2, :10])  
