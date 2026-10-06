"""Weekly-anchored forecasts and purged out-of-sample baseline comparisons.

Research-only: main continuous returns are not executable futures P&L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.train_xgboost_compare import add_features, make_model, purged_training_rows, product_configs, target_column
from src.research_storage import ResearchStore


def week_anchor(dates: pd.Series, asof: pd.Timestamp) -> pd.Timestamp:
    week = asof.isocalendar()[:2]
    same = dates.map(lambda value: value.isocalendar()[:2] == week)
    return dates.loc[same].min()


def predict_block(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> dict[str, np.ndarray]:
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    xgb = make_model()
    ridge.fit(train[features], train[target])
    xgb.fit(train[features], train[target])
    return {"zero": np.zeros(len(test)), "momentum": test["return_5d_pct"].to_numpy(),
            "ridge": ridge.predict(test[features]), "xgboost": xgb.predict(test[features])}


def frozen_model(train: pd.DataFrame, features: list[str], target: str) -> dict:
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(train[features], train[target])
    scaler, regression = ridge.steps[0][1], ridge.steps[1][1]
    xgb = make_model()
    xgb.fit(train[features], train[target])
    return {"features": features, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "ridge_coef": regression.coef_.tolist(), "ridge_intercept": float(regression.intercept_),
            "xgboost_json": xgb.get_booster().save_raw(raw_format="json").decode(), "train_rows": len(train),
            "train_label_end_date": train["label_end_date"].max().date().isoformat(),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="microseconds")}


def frozen_predict(model: dict, test: pd.DataFrame) -> dict:
    values = test[model["features"]].to_numpy(dtype=float)
    scaled = (values - np.asarray(model["scaler_mean"])) / np.asarray(model["scaler_scale"])
    xgb = make_model()
    xgb.load_model(bytearray(model["xgboost_json"], "utf-8"))
    return {"zero": 0.0, "momentum": float(test["return_5d_pct"].iloc[0]),
            "ridge": float((scaled @ np.asarray(model["ridge_coef"]) + model["ridge_intercept"])[0]),
            "xgboost": float(xgb.predict(test[model["features"]])[0])}


def forecast_product(frame: pd.DataFrame, code: str, expected_date: str, store: ResearchStore | None = None) -> dict:
    asof = pd.Timestamp(expected_date)
    # Truncate before targets/features for reproducible historical research.
    frame = frame.loc[pd.to_datetime(frame["date"]) <= asof].copy()
    forecasts = {}
    for horizon in (5, 10):
        feature_data, model_data, features = add_features(frame, product_configs(code)[0], horizon)
        if feature_data.empty or feature_data["date"].iloc[-1] != asof:
            raise ValueError(f"{code}: latest features unavailable on {expected_date}")
        anchor = week_anchor(pd.to_datetime(frame["date"]), asof)
        train = purged_training_rows(model_data.loc[model_data["date"] < anchor], anchor)
        if len(train) < 252:
            raise ValueError(f"{code}: insufficient training history")
        key = f"models/{code}/anchor={anchor.date().isoformat()}/{horizon}d.json"
        model = store.read_json(key) if store else None
        if model is None:
            model = frozen_model(train, features, target_column(horizon))
            if store:
                store.immutable_json(model, key)
        if model["features"] != features or pd.Timestamp(model["train_label_end_date"]) >= anchor:
            raise ValueError(f"{code}: frozen model metadata violates the research boundary")
        predictions = frozen_predict(model, feature_data.iloc[[-1]])
        if not all(np.isfinite(value) for value in predictions.values()):
            raise ValueError(f"{code}: non-finite research forecast")
        forecasts[str(horizon)] = {"predicted_return_pct": predictions["xgboost"],
                                   "baselines_pct": predictions,
                                   "model_anchor_date": anchor.date().isoformat(),
                                   "train_label_end_date": model["train_label_end_date"],
                                   "model_trained_at": model["trained_at"],
                                   "train_rows": model["train_rows"], "target_units": "log_return_pct"}
    prices = frame.sort_values("date")["close"].astype(float)
    returns = np.log(prices / prices.shift(1))
    historical = float(returns.tail(20).std() * np.sqrt(252) * 100)
    ewma = float(np.sqrt(returns.pow(2).ewm(alpha=0.06, adjust=False).mean().iloc[-1] * 252) * 100)
    return {"source_date": expected_date, "close": float(prices.iloc[-1]), "forecasts": forecasts,
            "volatility": {"historical_20d_annualized_pct": historical, "ewma_annualized_pct": ewma,
                           "status": "baselines_only", "future_horizons": [5, 20]},
            "price_series": "vendor_main_continuous_not_executable_pnl"}


def evaluate_product(frame: pd.DataFrame, code: str, horizon: int, max_test_rows: int = 252) -> tuple[dict, pd.DataFrame]:
    _, data, features = add_features(frame, product_configs(code)[0], horizon)
    target = target_column(horizon)
    # Whole-week groups match the live weekly anchor (including holiday-short weeks).
    grouped = data.assign(week=data["date"].dt.strftime("%G-%V"))
    earliest = grouped["date"].iloc[max(len(grouped) - max_test_rows, 0)]
    blocks = []
    for _, test in grouped.groupby("week", sort=True):
        if test["date"].max() < earliest:
            continue
        anchor = week_anchor(pd.to_datetime(frame["date"]), test["date"].iloc[0])
        train = purged_training_rows(data.loc[data["date"] < anchor], anchor)
        if len(train) < 252:
            continue
        block = test[["date", "label_end_date", target]].copy()
        predictions = predict_block(train, test, features, target)
        for model, values in predictions.items():
            block[model] = values
        block["model_anchor_date"] = anchor
        block["train_label_end_date"] = train["label_end_date"].max()
        blocks.append(block)
    if not blocks:
        raise ValueError(f"{code}: insufficient out-of-sample history")
    predictions = pd.concat(blocks, ignore_index=True)
    scores = {}
    for model in ("zero", "momentum", "ridge", "xgboost"):
        actual, pred = predictions[target], predictions[model]
        scores[model] = {"mae": float(mean_absolute_error(actual, pred)),
                         "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
                         "direction_accuracy": None if model == "zero" else float((np.sign(actual) == np.sign(pred)).mean()), "rows": len(actual)}
    return {"product": code, "horizon": horizon, "scores": scores,
            "start": predictions["date"].min().date().isoformat(), "end": predictions["date"].max().date().isoformat(),
            "interpretation": "overlapping_labels_not_independent; no_trade_profit_claim"}, predictions
