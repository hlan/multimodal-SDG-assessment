import os, re, json, random
from pathlib import Path
from typing import Optional, Dict, Tuple
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

DIM_IMAGE: Optional[int]   = 16
DIM_TEXT: Optional[int]    = 64
DIM_NUMERIC: Optional[int] = 749

TRAIN_YEARS = (2003, 2017)
TEST_YEARS  = (2018, 2022)

EMB_FIT_TIME_LIMIT = 90
EMB_FIT_MAX_ROWS   = 5000

INCLUDE_ID_COLS_IN_EMBEDDING = True

SEED = 42
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

TARGETS: Dict[str, float] = {"Index_70": 0.70, "Index_80": 0.80, "Index_90": 0.90}

OUT_DIR   = Path(f"_new/training_data/run_{timestamp}_{DIM_IMAGE}_{DIM_TEXT}_{DIM_NUMERIC}")
EMB_DIR   = Path(f"_new/artifacts/embeddings/run_{timestamp}")
MODEL_DIR = Path(f"_new/artifacts/mm_embedding_model_{timestamp}")
for d in [OUT_DIR, EMB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_OUT     = OUT_DIR / f"training_0317_{timestamp}.csv"
TEST_OUT      = OUT_DIR / f"testing_1822_{timestamp}.csv"
ALL_INDEX_OUT = OUT_DIR / f"index_all_with_benchmarks_{timestamp}.csv"
PCA_CSV_OUT   = OUT_DIR / f"pca_explained_variance_train_{timestamp}.csv"
MAPPING_JSON  = EMB_DIR / f"pca_scaler_mapping_{timestamp}.json"

INDEX_COLS = ["Index_70", "Index_80", "Index_90", "Index_PC1",
              "PC1_explained_variance_ratio_%", "sdgi_s", "hdi"]

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            print("[INFO] CUDA available")
        else:
            print("[INFO] CPU mode")
    except Exception:
        pass

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

def ensure_year_column(df: pd.DataFrame, img_col: str) -> str:
    ycol = detect_year_col(df)
    if ycol:
        if ycol != "Year":
            df.rename(columns={ycol: "Year"}, inplace=True)
        print("[INFO] year column -> 'Year'")
        return "Year"
    print(f"[WARN] no 'Year' column; extracting from '{img_col}' by regex ...")
    years = df[img_col].apply(extract_year_from_path) if img_col in df.columns else pd.Series([None]*len(df))
    if years.notna().mean() > 0:
        df["Year"] = years
        print(f"[INFO] inferred year for {years.notna().mean()*100:.1f}% rows")
        return "Year"
    raise ValueError("Cannot determine the Year column; unable to split train/test by year")

def infer_numeric_cols(df: pd.DataFrame) -> list:
    excluded = {ID_COL, CODE_COL, NAME_COL, TEXT_COL, IMG_COL, "Year"}
    return [c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

def robust_minmax_scale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = 100.0 * (x - lo) / max(hi - lo, 1e-9)
    return np.clip(scaled, 0.0, 100.0)

def fit_encoder_on_train(train_df: pd.DataFrame, use_cols: list, tag: str) -> MultiModalPredictor:
    df_tmp = train_df[use_cols].copy()
    dummy = "_dummy_y"
    if dummy in df_tmp.columns:
        raise ValueError(f"Column {dummy} already exists")
    rng = np.random.default_rng(SEED)
    df_tmp[dummy] = rng.integers(0, 2, size=len(df_tmp))
    if len(df_tmp) > EMB_FIT_MAX_ROWS:
        df_tmp = df_tmp.sample(n=EMB_FIT_MAX_ROWS, random_state=SEED).reset_index(drop=True)

    predictor = MultiModalPredictor(
        label=dummy,
        problem_type="binary",
        path=str(MODEL_DIR) + f"_{tag}",
    )
    predictor.fit(
        train_data=df_tmp,
        hyperparameters=None,
        time_limit=EMB_FIT_TIME_LIMIT,
        holdout_frac=0.1,
    )
    return predictor

def extract_embeddings(predictor: MultiModalPredictor, df: pd.DataFrame, use_cols: list, tag: str) -> np.ndarray:
    emb = predictor.extract_embedding(df[use_cols].copy())
    emb = np.asarray(emb, float)
    print(f"[INFO] [{tag}] emb shape: {emb.shape}")
    return emb

def fit_projection(X_train: np.ndarray, target_dim: Optional[int], seed: int, tag: str):
    X = np.asarray(X_train, float)
    n, d = X.shape
    state = {
        "tag": tag, "orig_dim": int(d),
        "target_dim": int(target_dim) if target_dim is not None else int(d),
        "mode": "identity",
        "train_mean": X.mean(axis=0).tolist(),
        "pca_components": None,
        "pca_explained_variance_ratio": None,
        "random_matrix_seed": None,
    }
    if (target_dim is None) or (target_dim == d):
        return X, state
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    if target_dim < d:
        p = PCA(n_components=target_dim, random_state=seed)
        Xp = p.fit_transform(Xc)
        state.update({
            "mode": "pca_reduce",
            "pca_components": p.components_.tolist(),
            "pca_explained_variance_ratio": p.explained_variance_ratio_.tolist(),
        })
        return Xp, state
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((d, target_dim))
    state.update({"mode": "random_expand", "random_matrix_seed": int(seed)})
    return Xc @ R, state

def apply_projection(X_test: np.ndarray, state: dict) -> np.ndarray:
    X = np.asarray(X_test, float)
    mode = state["mode"]
    if mode == "identity":
        return X
    mu = np.asarray(state["train_mean"], float)[None, :]
    Xc = X - mu
    if mode == "pca_reduce":
        comps = np.asarray(state["pca_components"], float)
        return Xc @ comps.T
    if mode == "random_expand":
        rng = np.random.default_rng(state["random_matrix_seed"])
        R = rng.standard_normal((state["orig_dim"], state["target_dim"]))
        return Xc @ R
    raise ValueError(f"unknown projection mode: {mode}")

def fit_zscore(X_train: np.ndarray):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-9
    return mu, sd

def apply_zscore(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(X, float) - mu) / sd

def normalize_benchmark_columns(bench_df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in bench_df.columns:
        lc = c.strip().lower()
        if lc in ("year", "yr"): ren[c] = "Year"
        elif lc in ("countrycode", "code", "country", "ccode"): ren[c] = "CountryCode"
    return bench_df.rename(columns=ren)

def merge_benchmarks(df_out: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    out = df_out.merge(
        bench[["CountryCode", "Year", "sdgi_s", "hdi"]],
        on=["CountryCode", "Year"], how="left", validate="m:1"
    )
    out["hdi"] = out["hdi"].astype(float) * 100.0
    return out

def main():
    set_global_seed(SEED)

    df = pd.read_csv(ALL_CSV)
    for col in [CODE_COL, NAME_COL, TEXT_COL, IMG_COL]:
        if col not in df.columns:
            raise ValueError(f"[data] Missing column: {col}")
    ensure_year_column(df, IMG_COL)

    global TABULAR_NUMERIC_COLS
    if not TABULAR_NUMERIC_COLS:
        TABULAR_NUMERIC_COLS = tuple(infer_numeric_cols(df))
    print(f"[INFO] Numeric feature columns detected: {len(TABULAR_NUMERIC_COLS)}")

    train_df = df[(df["Year"] >= TRAIN_YEARS[0]) & (df["Year"] <= TRAIN_YEARS[1])].reset_index(drop=True)
    test_df  = df[(df["Year"] >= TEST_YEARS[0])  & (df["Year"] <= TEST_YEARS[1])].reset_index(drop=True)
    print(f"[INFO] train rows: {len(train_df)} ({TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}), "
          f"test rows: {len(test_df)} ({TEST_YEARS[0]}-{TEST_YEARS[1]})")
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty; check the Year column")

    id_extra = [CODE_COL, NAME_COL, "Year"] if INCLUDE_ID_COLS_IN_EMBEDDING else []
    img_cols = [IMG_COL] + id_extra
    txt_cols = [TEXT_COL] + id_extra
    num_cols = list(TABULAR_NUMERIC_COLS) + id_extra

    enc_img = fit_encoder_on_train(train_df, img_cols, "image")
    enc_txt = fit_encoder_on_train(train_df, txt_cols, "text")
    enc_num = fit_encoder_on_train(train_df, num_cols, "numeric")

    emb_img_tr = extract_embeddings(enc_img, train_df, img_cols, "image/train")
    emb_img_te = extract_embeddings(enc_img, test_df,  img_cols, "image/test")
    emb_txt_tr = extract_embeddings(enc_txt, train_df, txt_cols, "text/train")
    emb_txt_te = extract_embeddings(enc_txt, test_df,  txt_cols, "text/test")
    emb_num_tr = extract_embeddings(enc_num, train_df, num_cols, "numeric/train")
    emb_num_te = extract_embeddings(enc_num, test_df,  num_cols, "numeric/test")

    emb_img_tr_p, st_img = fit_projection(emb_img_tr, DIM_IMAGE,   SEED + 11, "image")
    emb_txt_tr_p, st_txt = fit_projection(emb_txt_tr, DIM_TEXT,    SEED + 22, "text")
    emb_num_tr_p, st_num = fit_projection(emb_num_tr, DIM_NUMERIC, SEED + 33, "numeric")
    emb_img_te_p = apply_projection(emb_img_te, st_img)
    emb_txt_te_p = apply_projection(emb_txt_te, st_txt)
    emb_num_te_p = apply_projection(emb_num_te, st_num)

    mu_i, sd_i = fit_zscore(emb_img_tr_p)
    mu_t, sd_t = fit_zscore(emb_txt_tr_p)
    mu_n, sd_n = fit_zscore(emb_num_tr_p)

    train_emb = np.concatenate([
        apply_zscore(emb_img_tr_p, mu_i, sd_i),
        apply_zscore(emb_txt_tr_p, mu_t, sd_t),
        apply_zscore(emb_num_tr_p, mu_n, sd_n),
    ], axis=1)
    test_emb = np.concatenate([
        apply_zscore(emb_img_te_p, mu_i, sd_i),
        apply_zscore(emb_txt_te_p, mu_t, sd_t),
        apply_zscore(emb_num_te_p, mu_n, sd_n),
    ], axis=1)

    np.save(EMB_DIR / f"train_embeddings_{timestamp}.npy", train_emb)
    np.save(EMB_DIR / f"test_embeddings_{timestamp}.npy",  test_emb)
    print(f"[SAVE] embeddings -> {EMB_DIR} (train {train_emb.shape}, test {test_emb.shape})")

    scaler = StandardScaler(with_mean=True, with_std=True)
    train_scaled = scaler.fit_transform(train_emb)
    test_scaled  = scaler.transform(test_emb)

    pca = PCA(n_components=min(128, train_scaled.shape[1]), random_state=SEED)
    train_pca = pca.fit_transform(train_scaled)
    test_pca  = pca.transform(test_scaled)

    var_ratio = pca.explained_variance_ratio_
    eigs      = pca.explained_variance_
    cum_ratio = np.cumsum(var_ratio)
    pc1_var_ratio_pct = round(var_ratio[0] * 100.0, 2)
    print(f"[PCA] PC1 explained variance ratio (train): {pc1_var_ratio_pct}%")

    pd.DataFrame({
        "PC": np.arange(1, len(var_ratio) + 1),
        "explained_variance_ratio": var_ratio,
        "explained_variance_ratio_%": var_ratio * 100.0,
        "cumulative_ratio": cum_ratio,
        "cumulative_ratio_%": cum_ratio * 100.0,
    }).to_csv(PCA_CSV_OUT, index=False)
    print(f"[SAVE] PCA contributions (train) -> {PCA_CSV_OUT}")

    def pick_k_for(target: float) -> int:
        return int(np.searchsorted(cum_ratio, target) + 1)

    def build_index(thr: float):
        k = pick_k_for(thr)
        stds = np.sqrt(eigs[:k])
        w = var_ratio[:k] / var_ratio[:k].sum()
        raw_tr = (train_pca[:, :k] / stds) @ w
        raw_te = (test_pca[:, :k]  / stds) @ w
        q_lo, q_hi = np.percentile(raw_tr, [1, 99])
        idx_tr = robust_minmax_scale(raw_tr, q_lo, q_hi)
        idx_te = robust_minmax_scale(raw_te, q_lo, q_hi)
        params = {
            "k": int(k),
            "weights": w.tolist(),
            "stds": stds.tolist(),
            "q_lo": float(q_lo), "q_hi": float(q_hi),
            "direction_flipped": False,
            "target_cum_var": float(thr),
        }
        return idx_tr, idx_te, params

    idx_tr_all, idx_te_all, params_all = {}, {}, {}
    for name, thr in TARGETS.items():
        itr, ite, prm = build_index(thr)
        idx_tr_all[name], idx_te_all[name], params_all[name] = itr, ite, prm
        print(f"[INDEX] {name}: k={prm['k']}")

    q1_lo, q1_hi = np.percentile(train_pca[:, 0], [1, 99])
    pc1_tr = robust_minmax_scale(train_pca[:, 0], q1_lo, q1_hi)
    pc1_te = robust_minmax_scale(test_pca[:, 0],  q1_lo, q1_hi)

    bench = normalize_benchmark_columns(pd.read_csv(BENCHMARK_CSV))
    need = {"CountryCode", "Year", "sdgi_s", "hdi"}
    missing = need - set(bench.columns)
    if missing:
        raise ValueError(f"[benchmark] Missing required columns: {missing}")
    bench["Year"] = pd.to_numeric(bench["Year"], errors="coerce").astype("Int64")

    def attach_indices(base_df: pd.DataFrame, idx_map: dict, pc1: np.ndarray) -> pd.DataFrame:
        out = base_df.copy()
        for name in TARGETS.keys():
            out[name] = idx_map[name]
        out["Index_PC1"] = pc1
        out["PC1_explained_variance_ratio_%"] = pc1_var_ratio_pct
        return merge_benchmarks(out, bench)

    train_full = attach_indices(train_df, idx_tr_all, pc1_tr)
    test_full  = attach_indices(test_df,  idx_te_all, pc1_te)

    train_full.to_csv(TRAIN_OUT, index=False)
    test_full.to_csv(TEST_OUT, index=False)
    print(f"[SAVE] Training (full features + indices + benchmarks) -> {TRAIN_OUT}")
    print(f"[SAVE] Testing  (full features + indices + benchmarks) -> {TEST_OUT}")

    lean_cols = [c for c in [ID_COL, CODE_COL, NAME_COL, "Year"] if c in df.columns] + INDEX_COLS
    lean = pd.concat([train_full, test_full], axis=0, ignore_index=True)[lean_cols]
    lean.to_csv(ALL_INDEX_OUT, index=False)
    print(f"[SAVE] All indices + benchmarks (lean) -> {ALL_INDEX_OUT}")

    mapping = {
        "timestamp": timestamp,
        "seed": SEED,
        "train_years": list(TRAIN_YEARS),
        "test_years": list(TEST_YEARS),
        "fitting_discipline": "all parameters (encoders, projections, z-score, scaler, PCA, "
                              "index weights, percentile bounds) fitted on training data only",
        "include_id_cols_in_embedding": INCLUDE_ID_COLS_IN_EMBEDDING,
        "emb_fit_time_limit_s": EMB_FIT_TIME_LIMIT,
        "modality_weights": {"rule": "no weights (concat after per-modality z-score)",
                             "applied": False},
        "projection": {"image": st_img, "text": st_txt, "numeric": st_num},
        "zscore": {
            "image":   {"mean": mu_i.tolist(), "std": sd_i.tolist()},
            "text":    {"mean": mu_t.tolist(), "std": sd_t.tolist()},
            "numeric": {"mean": mu_n.tolist(), "std": sd_n.tolist()},
        },
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "explained_variance_ratio": var_ratio.tolist(),
        "explained_variance": eigs.tolist(),
        "targets": TARGETS,
        "target_params": params_all,
        "pc1_q_lo": float(q1_lo), "pc1_q_hi": float(q1_hi),
        "text_col": TEXT_COL, "img_col": IMG_COL,
        "numeric_cols": list(TABULAR_NUMERIC_COLS),
        "output_files": {"train": str(TRAIN_OUT), "test": str(TEST_OUT)},
    }
    with open(MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Mapping & params -> {MAPPING_JSON}")

    def print_stats(name: str, arr: np.ndarray, split: str):
        arr = np.asarray(arr, float)
        print(f"[STATS] {split}/{name}: count={arr.size}, >=90:{(arr>=90).sum()}, "
              f">=80:{(arr>=80).sum()}, >=50:{(arr>=50).sum()}, <20:{(arr<20).sum()}")
    for name in list(TARGETS.keys()) + ["Index_PC1"]:
        print_stats(name, train_full[name].values, "train")
        print_stats(name, test_full[name].values, "test")

    print(f"\n[RUN] timestamp = {timestamp}")
    print(f"[DONE] Output directory: {OUT_DIR}")

if __name__ == "__main__":
    main()
