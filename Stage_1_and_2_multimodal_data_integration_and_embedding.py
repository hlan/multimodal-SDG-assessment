# -*- coding: utf-8 -*-

import os, re, json, random
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from autogluon.multimodal import MultiModalPredictor

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

ALL_CSV       = r"training_data/all_data.csv"
BENCHMARK_CSV = r"sdg_hdi_training_data/all_data_with_sdg_hdi.csv"  

ID_COL    = "Id"
CODE_COL  = "CountryCode"
NAME_COL  = "CountryName"
TEXT_COL  = "description"
IMG_COL   = "image"

TABULAR_NUMERIC_COLS: Tuple[str, ...] = tuple()

SEED = 42
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ================== Dim setup ==================
# e.g., DIM_IMAGE=256, DIM_TEXT=512, DIM_NUMERIC=1024
DIM_IMAGE: Optional[int]   = 16
DIM_TEXT: Optional[int]    = 64
DIM_NUMERIC: Optional[int] = 1028

OUT_DIR   = Path(f"_new/training_data/run_{timestamp}"); OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR   = Path(f"_new/artifacts/embeddings/run_{timestamp}"); EMB_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(f"_new/artifacts/mm_embedding_model_{timestamp}")

ALL_INDEX_OUT = OUT_DIR / f"index_all_with_benchmarks_{timestamp}.csv"
PCA_CSV_OUT   = OUT_DIR / f"pca_explained_variance_all_{timestamp}.csv"
MAPPING_JSON  = EMB_DIR / f"pca_scaler_mapping_{timestamp}.json"
TRAIN_OUT     = OUT_DIR / f"training_0317_{timestamp}.csv"
TEST_OUT      = OUT_DIR / f"testing_1822_{timestamp}.csv"

TARGETS: Dict[str, float] = {"Index_70": 0.70, "Index_80": 0.80, "Index_90": 0.90}

def detect_year_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["Year", "year", "YEAR"]:
        if cand in df.columns:
            return cand
    return None

YEAR_REGEX = re.compile(r"(19\d{2}|20\d{2})")
def extract_year_from_path(path_str: str) -> Optional[int]:
    if not isinstance(path_str, str):
        return None
    m = YEAR_REGEX.search(path_str or "")
    if m:
        y = int(m.group(1))
        if 2000 <= y <= 2030:
            return y
    return None

def ensure_year_column(df: pd.DataFrame, img_col: str) -> Optional[str]:
    ycol = detect_year_col(df)
    if ycol:
        print(f"detected year column -> '{ycol}'")
        return ycol
    print(f"[WARN] no 'Year' column; try extracting from '{img_col}' by regex ...")
    years = df[img_col].apply(extract_year_from_path) if img_col in df.columns else pd.Series([None]*len(df))
    if years.notna().mean() > 0:
        yname = "_Year_inferred"
        df[yname] = years
        print(f"inferred year for {years.notna().mean()*100:.1f}% rows as '{yname}'")
        return yname
    print(f"failed to infer year; proceeding without year")
    return None

def infer_numeric_cols(df: pd.DataFrame, year_col: Optional[str]) -> list:
    excluded = {ID_COL, CODE_COL, NAME_COL, TEXT_COL, IMG_COL}
    if year_col:
        excluded.add(year_col)
    return [c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

def robust_minmax_scale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = 100.0 * (x - lo) / max(hi - lo, 1e-9)
    return np.clip(scaled, 0.0, 100.0)

def _rankdata(a: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    n = a.shape[0]
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    unique, first_idx = np.unique(a[order], return_index=True)
    for i in range(len(unique)):
        start = first_idx[i]
        end = first_idx[i+1] if i+1 < len(unique) else n
        if end - start > 1:
            avg = (start + end - 1) / 2.0
            ranks[order[start:end]] = avg
    return ranks / max(n - 1, 1.0)

def spearmanr_no_scipy(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x_m = x[mask]; y_m = y[mask]
    if x_m.size < 3:
        return np.nan
    rx = _rankdata(x_m); ry = _rankdata(y_m)
    return float(np.corrcoef(rx, ry)[0, 1])

df = pd.read_csv(ALL_CSV)
for col in [CODE_COL, NAME_COL, TEXT_COL, IMG_COL]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")
YEAR_COL = ensure_year_column(df, IMG_COL)

if not TABULAR_NUMERIC_COLS:
    TABULAR_NUMERIC_COLS = tuple(infer_numeric_cols(df, YEAR_COL))
print(f"Numeric feature columns detected: {len(TABULAR_NUMERIC_COLS)}")

random.seed(SEED); np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda_manual_seed_all(SEED)
        print("CUDA available")
    else:
        print("CPU mode")
except Exception:
    torch = None

DUMMY_LABEL = "_dummy_y"
if DUMMY_LABEL in df.columns:
    raise ValueError(f"Column {DUMMY_LABEL} already exists in data")

def _fit_and_extract(df_in: pd.DataFrame, use_cols: list, tag: str) -> np.ndarray:
    df_tmp = df_in[[*use_cols]].copy()
    df_tmp["_dummy_y"] = np.random.randint(0, 2, size=len(df_tmp))

    MAX_ROWS = 5000  
    if len(df_tmp) > MAX_ROWS:
        df_tmp = df_tmp.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)

    predictor = MultiModalPredictor(
        label="_dummy_y",
        problem_type="binary",
        path=str(MODEL_DIR) + f"_{tag}",
    )
    predictor.fit(
        train_data=df_tmp,
        hyperparameters=None,
        time_limit=90,
        holdout_frac=0.1,
    )
    emb = predictor.extract_embedding(df_tmp.drop(columns=["_dummy_y"]))
    print(f" [{tag}] emb shape: {emb.shape}")
    return emb

img_cols = [IMG_COL, CODE_COL, NAME_COL] + ([YEAR_COL] if YEAR_COL else [])
txt_cols = [TEXT_COL, CODE_COL, NAME_COL] + ([YEAR_COL] if YEAR_COL else [])
num_cols = list(TABULAR_NUMERIC_COLS) + [CODE_COL, NAME_COL] + ([YEAR_COL] if YEAR_COL else [])

emb_img = _fit_and_extract(df, img_cols, "image")
emb_txt = _fit_and_extract(df, txt_cols, "text")
emb_num = _fit_and_extract(df, num_cols, "numeric")

from sklearn.decomposition import PCA as _PCA_local
def _project_to_dim(X: np.ndarray, target_dim: Optional[int], seed: int, tag: str):
    X = np.asarray(X, float)
    n, d = X.shape
    info = {
        "tag": tag,
        "orig_dim": d,
        "target_dim": target_dim if target_dim is not None else d,
        "mode": "identity",
        "pca_components": None,
        "pca_mean": None,
        "pca_explained_variance_ratio": None,
        "random_matrix_shape": None,
        "random_matrix_seed": None,
    }
    if (target_dim is None) or (target_dim == d):
        return X, info
    if target_dim < d:
        pca_local = _PCA_local(n_components=target_dim, random_state=seed)
        Xc = X - X.mean(axis=0, keepdims=True)
        Xp = pca_local.fit_transform(Xc)
        info.update({
            "mode": "pca_reduce",
            "pca_components": pca_local.components_.tolist(),
            "pca_mean": pca_local.mean_.tolist(),
            "pca_explained_variance_ratio": pca_local.explained_variance_ratio_.tolist(),
        })
        return Xp, info
    else:
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((d, target_dim))
        Xc = X - X.mean(axis=0, keepdims=True)
        Xp = Xc @ R
        info.update({
            "mode": "random_expand",
            "random_matrix_shape": [d, target_dim],
            "random_matrix_seed": seed,
        })
        return Xp, info

emb_img_proj, proj_info_img = _project_to_dim(emb_img, DIM_IMAGE, SEED+11, "image")
emb_txt_proj, proj_info_txt = _project_to_dim(emb_txt, DIM_TEXT, SEED+22, "text")
emb_num_proj, proj_info_num = _project_to_dim(emb_num, DIM_NUMERIC, SEED+33, "numeric")

# ================== z-score ==================
def _zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    return (X - mu) / sd

emb_img_z = _zscore(emb_img_proj)
emb_txt_z = _zscore(emb_txt_proj)
emb_num_z = _zscore(emb_num_proj)

all_emb = np.concatenate([emb_img_z, emb_txt_z, emb_num_z], axis=1)

EMB_SAVE = EMB_DIR / f"all_embeddings_{timestamp}.npy"
np.save(EMB_SAVE, all_emb)
print(f"embeddings (zscored & concatenated, no weights) -> {EMB_SAVE}")

# ================== PCA  ==================
scaler = StandardScaler(with_mean=True, with_std=True)
all_emb_scaled = scaler.fit_transform(all_emb)

pca = PCA(n_components=min(128, all_emb_scaled.shape[1]), random_state=SEED)
all_pca = pca.fit_transform(all_emb_scaled)

var_ratio = pca.explained_variance_ratio_
eigs = pca.explained_variance_
cum_ratio = np.cumsum(var_ratio)
pc1_var_ratio_pct = round(var_ratio[0] * 100.0, 2)
print(f"[PCA] PC1 explained variance ratio: {pc1_var_ratio_pct}%")
pd.DataFrame({
    "PC": np.arange(1, len(var_ratio) + 1),
    "explained_variance_ratio": var_ratio,
    "explained_variance_ratio_%": var_ratio * 100.0,
    "cumulative_ratio": cum_ratio,
    "cumulative_ratio_%": cum_ratio * 100.0,
}).to_csv(PCA_CSV_OUT, index=False)
print(f"PCA contributions -> {PCA_CSV_OUT}")

# ================== Index（70/80/90 + PC1） ==================
def pick_k_for(target=0.80) -> int:
    return int(np.searchsorted(cum_ratio, target) + 1)

def make_index_by_threshold(thr: float, S_full: np.ndarray, eigs_all: np.ndarray, var_ratio_all: np.ndarray):
    k = pick_k_for(thr)
    S = S_full[:, :k]
    stds = np.sqrt(eigs_all[:k])
    Z = S / stds
    w = var_ratio_all[:k] / var_ratio_all[:k].sum()
    idx_raw = Z @ w
    q_lo, q_hi = np.percentile(idx_raw, [1, 99])
    idx_0_100 = robust_minmax_scale(idx_raw, q_lo, q_hi)
    params = {
        "k": int(k),
        "weights": w.tolist(),
        "stds": stds.tolist(),
        "q_lo": float(q_lo), "q_hi": float(q_hi),
        "direction_flipped": False,
        "target_cum_var": float(thr),
        "spearman_rho_with_proxy": None,
    }
    return idx_0_100, params

indices_all = {}; params_all = {}
for name, thr in TARGETS.items():
    idx_vec, params = make_index_by_threshold(thr, all_pca, eigs, var_ratio)
    indices_all[name] = idx_vec
    params_all[name]  = params
    print(f"[INDEX] {name}: k={params['k']}")

pc1_raw = all_pca[:, 0]
q1_lo, q1_hi = np.percentile(pc1_raw, [1, 99])
Index_PC1_all = robust_minmax_scale(pc1_raw, q1_lo, q1_hi)

def build_out_df(df_in: pd.DataFrame, year_col_opt: Optional[str]) -> pd.DataFrame:
    base_cols = [c for c in [ID_COL, CODE_COL, NAME_COL, year_col_opt] if c and c in df_in.columns]
    out = df_in[base_cols].copy()
    if year_col_opt and year_col_opt in out.columns and year_col_opt != "Year":
        out.rename(columns={year_col_opt: "Year"}, inplace=True)
    if "Year" not in out.columns:
        out["Year"] = np.nan
    return out

out_df = build_out_df(df, YEAR_COL)
for name in TARGETS.keys(): out_df[name] = indices_all[name]
out_df["Index_PC1"] = Index_PC1_all
out_df["PC1_explained_variance_ratio_%"] = pc1_var_ratio_pct

def _normalize_benchmark_columns(bench_df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in bench_df.columns:
        lc = c.strip().lower()
        if lc in ("year", "yr"): ren[c] = "Year"
        elif lc in ("countrycode", "code", "country", "ccode"): ren[c] = "CountryCode"
    return bench_df.rename(columns=ren)

bench = pd.read_csv(BENCHMARK_CSV)
bench = _normalize_benchmark_columns(bench)

need_cols = {"CountryCode", "Year", "sdgi_s", "hdi"}
missing = need_cols - set(bench.columns)
if missing:
    raise ValueError(f" miss: {missing}；exist: {list(bench.columns)}")

bench["Year"] = pd.to_numeric(bench["Year"], errors="coerce").astype("Int64")

out_df = out_df.merge(
    bench[["CountryCode", "Year", "sdgi_s", "hdi"]],
    on=["CountryCode", "Year"], how="left", validate="m:1"
)
out_df["hdi"] = out_df["hdi"].astype(float) * 100.0  # ×100 for vis

out_df.to_csv(ALL_INDEX_OUT, index=False)
print(f"[SAVE] All indices + sdgi_s/hdi -> {ALL_INDEX_OUT}")

mapping = {
    "timestamp": timestamp,
    "seed": SEED,
    "modality_weights": {
        "rule": "no weights (concat after per-modality z-score)",
        "applied": False
    },
    "projection": {
        "image": proj_info_img,
        "text": proj_info_txt,
        "numeric": proj_info_num,
    },
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "pca_components": pca.components_.tolist(),
    "pca_mean": pca.mean_.tolist(),
    "explained_variance_ratio": var_ratio.tolist(),
    "explained_variance": eigs.tolist(),
    "targets": TARGETS,
    "target_params": params_all,
    "text_col": TEXT_COL,
    "img_col": IMG_COL,
    "numeric_cols": list(TABULAR_NUMERIC_COLS),
    "year_col_used": YEAR_COL,
    "pc1_var_ratio": float(var_ratio[0]),
    "pc1_q_lo": float(q1_lo),
    "pc1_q_hi": float(q1_hi),
    "output_columns": [
        ID_COL, CODE_COL, NAME_COL, "Year",
        "Index_70", "Index_80", "Index_90", "Index_PC1",
        "PC1_explained_variance_ratio_%", "sdgi_s", "hdi"
    ],
}
with open(MAPPING_JSON, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print(f"[SAVE] Mapping & params -> {MAPPING_JSON}")

# ================== split train/test ==================
def split_by_year_windows(df_all: pd.DataFrame):
    if "Year" not in df_all.columns:
        raise ValueError(" miss Year column，cannot split by year")
    train_keys = df_all[(df_all["Year"] >= 2003) & (df_all["Year"] <= 2017)].copy()
    test_keys  = df_all[(df_all["Year"] >= 2018) & (df_all["Year"] <= 2022)].copy()
    return train_keys, test_keys

train_keys, test_keys = split_by_year_windows(out_df)

df_year = df.copy()
if YEAR_COL and YEAR_COL != "Year":
    df_year = df_year.rename(columns={YEAR_COL: "Year"})
if "Year" not in df_year.columns:
    raise ValueError("miss Year column")

train_full = df_year[(df_year["Year"] >= 2003) & (df_year["Year"] <= 2017)].copy()
index_cols = ["Index_70","Index_80","Index_90","Index_PC1","PC1_explained_variance_ratio_%","sdgi_s","hdi"]
train_full = train_full.merge(
    train_keys[[CODE_COL, "Year"] + index_cols],
    on=[CODE_COL, "Year"], how="left", validate="m:1"
)
train_full.to_csv(TRAIN_OUT, index=False)
print(f"Training (full features) -> {TRAIN_OUT}")

test_full = df_year[(df_year["Year"] >= 2018) & (df_year["Year"] <= 2022)].copy()
test_full = test_full.merge(
    test_keys[[CODE_COL, "Year"] + index_cols],
    on=[CODE_COL, "Year"], how="left", validate="m:1"
)
test_full.to_csv(TEST_OUT, index=False)
print(f"Testing (full features)  -> {TEST_OUT}")

def print_stats(name: str, arr: np.ndarray):
    arr = np.asarray(arr, float)
    print(f"\n[STATS] {name}: count={arr.size}, >=90:{(arr>=90).sum()}, >=80:{(arr>=80).sum()}, >=50:{(arr>=50).sum()}, <20:{(arr<20).sum()}")

for name in list(TARGETS.keys()) + ["Index_PC1"]:
    if name in out_df.columns:
        print_stats(name, out_df[name].values)

print(f"\n timestamp = {timestamp}")
