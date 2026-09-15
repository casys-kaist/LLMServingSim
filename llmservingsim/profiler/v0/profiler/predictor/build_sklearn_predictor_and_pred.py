#!/usr/bin/env python3
import argparse, hashlib, os, pickle
from itertools import product
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def _mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    zero = y_true == 0
    err = np.zeros_like(y_true, dtype=float)
    err[~zero] = np.abs((y_true[~zero] - y_pred[~zero]) / y_true[~zero]) * 100
    err[zero] = np.where(y_pred[zero] == 0, 0, 100)
    return np.mean(err)

SCORER = make_scorer(_mape, greater_is_better=False)

def train_model(name, df, X_cols, y_col, rf_grid, cv, n_jobs, cache_dir):
    mdl_path = os.path.join(cache_dir, f"{name}.pkl")
    if os.path.isfile(mdl_path):
        with open(mdl_path, "rb") as f:
            return pickle.load(f)
    gs = GridSearchCV(RandomForestRegressor(random_state=42), rf_grid, scoring=SCORER,
                      cv=min(cv, len(df)), n_jobs=n_jobs)
    X, y = df[X_cols], df[y_col]
    gs.fit(X, y)
    with open(mdl_path, "wb") as f:
        pickle.dump(gs.best_estimator_, f, protocol=pickle.HIGHEST_PROTOCOL)
    return gs.best_estimator_

def predict_and_save(name, model, X, cache_dir, overhead):
    preds = model.predict(X)
    # TODO(hmchoi): for TPU profiled result, we don't need to scale it
    X_out = X.copy(); X_out["prediction"] = (preds * 1e6 * overhead).astype(np.int64) # to ns
    csv_path = os.path.join(cache_dir, f"{name}_predictions.csv")
    X_out.to_csv(csv_path, index=False)
    return csv_path

def load_and_split_attention_csv(path):
    df = pd.read_csv(path).drop_duplicates()

    if "time_stats.attn_kv_cache_save.median" not in df.columns:
        df["time_stats.attn_kv_cache_save.median"] = 0

    df.fillna({"time_stats.attn_kv_cache_save.median": 0}, inplace=True)

    df["num_tokens"] = df[["prefill_chunk_size", "batch_size"]].max(axis=1)
    df["is_decode"] = df["prefill_chunk_size"] == 0
    df["prefill_chunk_size"] = df["prefill_chunk_size"]

    return df[~df["is_decode"]], df[df["is_decode"]]

def build_grids(max_tokens, kv_granularity, chunk_granularity,
                max_prefill_chunk, max_batch):
    kv_range = np.arange(0, max_tokens + 1, kv_granularity)

    chunk_range = np.arange(chunk_granularity, max_prefill_chunk + 1, chunk_granularity)
    pf_kv, pf_chunk = zip(*product(kv_range, chunk_range))
    prefill_X = pd.DataFrame({
        "kv_cache_size": pf_kv,
        "prefill_chunk_size": np.array(pf_chunk),
    })

    dec_batch = np.arange(1, max_batch + 1)
    dec_kv = kv_range
    dc_batch, dc_kv = zip(*product(dec_batch, dec_kv))
    decode_X = pd.DataFrame({"batch_size": dc_batch, "kv_cache_size": dc_kv})
    return prefill_X, decode_X
