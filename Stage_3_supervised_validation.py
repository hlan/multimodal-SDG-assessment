# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import sys

from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    import torch
except Exception:
    torch = None

TRAIN_RAW = Path("data/training.csv")
TEST_RAW  = Path("data/testing.csv")

OUT_ROOT  = Path("results/supervised_validation/16_64_749")

TARGET_LABELS = ["Index_70", "sdgi_s", "hdi", "Index_80", "Index_90"]

TIME_LIMIT = 30000   

TEXT_COL = "description"
IMG_COL  = "image"

ALL_TARGET_LIKE_COLS = [
    "Index_70",
    "Index_80",
    "Index_90",
    "Index_PC1",
    "PC1_explained_variance_ratio_%",
    "sdgi_s",
    "hdi",
]

NON_FEATURE_COLS = [
    "Id",
    "CountryCode",
    "CountryName",
    "Year",
]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("year", "yr"):
            ren[c] = "Year"
        elif cl in ("countrycode", "code", "country", "ccode"):
            ren[c] = "CountryCode"
    return df.rename(columns=ren) if ren else df

def spearman_no_scipy(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return np.nan

    def rankdata(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        uniq, first = np.unique(a[order], return_index=True)
        for i in range(len(uniq)):
            s, e = first[i], first[i+1] if i+1 < len(uniq) else len(a)
            if e - s > 1:
                ranks[order[s:e]] = (s + e - 1) / 2.0
        return ranks / max(len(a) - 1, 1.0)

    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

def metrics_summary(y_true, y_pred):
    y_true = np.nan_to_num(np.asarray(y_true, float), nan=0.0)
    y_pred = np.nan_to_num(np.asarray(y_pred, float), nan=0.0)

    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "SpearmanRho": spearman_no_scipy(y_true, y_pred),
    }

def print_device_info():
    print("========== Device Info ==========")
    if torch is None:
        print(" torch not run on CPU")
    else:
        cuda = torch.cuda.is_available()
        print(f"CUDA available: {cuda}")
        if cuda:
            try:
                n = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                print(f"GPU Count: {n}, Name: {name}")
            except Exception as e:
                print(f"cannot read GPU: {e}")
    print("=================================")

 # ---------- start ----------
def main():
    print_device_info()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    OUT_DIR = OUT_ROOT / f"run_{ts}"
    MODELS_DIR = OUT_DIR / "models"
    PRED_DIR   = OUT_DIR / "predictions"
    EVAL_DIR   = OUT_DIR / "eval"
    LOG_DIR    = OUT_DIR / "logs"
    for d in [OUT_DIR, MODELS_DIR, PRED_DIR, EVAL_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    train_raw = normalize_columns(pd.read_csv(TRAIN_RAW))
    test_raw  = normalize_columns(pd.read_csv(TEST_RAW))

    for df, name in [(train_raw, "train"), (test_raw, "test")]:
        for col in ["CountryCode", "Year"]:
            if col not in df.columns:
                raise ValueError(f"[{name}] missing col: {col}")

    summary_rows = []

    for label in TARGET_LABELS:
        if label not in train_raw.columns or label not in test_raw.columns:
            print(f"{label} not in train/test, skip")
            continue

        print(f"\n=== start {label} ===")

        exclude_cols = [c for c in ALL_TARGET_LIKE_COLS if c != label]
        existing_exclude = [c for c in exclude_cols if c in train_raw.columns]

        candidate_cols = [
            c for c in train_raw.columns
            if c not in existing_exclude and c not in NON_FEATURE_COLS
        ]

        feature_cols = candidate_cols.copy()
        if label not in feature_cols:
            feature_cols.append(label)

        train_cols = feature_cols  
        train_df = train_raw[train_cols].dropna(subset=[label]).copy()

        if train_df.empty:
            print(f"{label} training data empty, skip")
            continue

        test_features = test_raw[[c for c in feature_cols if c != label]].copy()

        save_path = MODELS_DIR / label

        predictor = MultiModalPredictor(
            label=label,
            problem_type="regression",
            path=str(save_path),
        )

        log_file = LOG_DIR / f"train_{label}.txt"
        with open(log_file, "w", encoding="utf-8") as lf:
            bak = sys.stdout
            sys.stdout = lf
            try:
                print(f"[PROGRESS] training {label} ...（up to {TIME_LIMIT//60} mins）")
                predictor.fit(train_data=train_df, time_limit=TIME_LIMIT)
            finally:
                sys.stdout = bak
        print(f"model saved：{save_path}")
        print(f" log saved：{log_file}")

        # ---------- evaluation process with R²、MAE、RMSE、Spearman ρ ----------
        yhat = predictor.predict(test_features)

        pred_df = test_raw[["CountryCode", "Year"]].copy()
        pred_df[f"Supervised_{label}"] = yhat
        pred_file = PRED_DIR / f"test_supervised_{label}_{ts}.csv"
        pred_df.to_csv(pred_file, index=False)

        merged_eval = pred_df.merge(
            test_raw[["CountryCode", "Year", label]],
            on=["CountryCode", "Year"],
            how="inner",
            validate="m:1",
        )
        if not merged_eval.empty:
            m = metrics_summary(
                merged_eval[label].values,
                merged_eval[f"Supervised_{label}"].values,
            )
            eval_file = EVAL_DIR / f"eval_detail_{label}_{ts}.csv"
            merged_eval.to_csv(eval_file, index=False)
            print(
                f"[EVAL] {label} -> "
                f"R2={m['R2']:.4f}, MAE={m['MAE']:.4f}, "
                f"RMSE={m['RMSE']:.4f}, Spearman={m['SpearmanRho']:.4f}"
            )
            summary_rows.append({
                "Label": label,
                "ModelPath": str(save_path),
                "PredPath": str(pred_file),
                "EvalDetailPath": str(eval_file),
                "LogFile": str(log_file),
                "TrainRows": len(train_df),
                "TestRows": len(merged_eval),
                **m,
            })

        print(f"=== {label} done ===")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = EVAL_DIR / f"metrics_summary_{ts}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n summary：{summary_file}")
        print(summary_df.to_string(index=False))

    print(f"\n all output：{OUT_DIR}")

if __name__ == "__main__":
    main()
