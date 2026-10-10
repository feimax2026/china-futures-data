"""Product-specific training windows; select parameters on past purged data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

MODEL_VERSION = "product-purged-v2"
PROFILES = {
    "SF": {"train_rows": 1260, "depths": (2, 3), "lambdas": (10.0, 20.0)},
    "AU": {"train_rows": 1890, "depths": (2, 3), "lambdas": (5.0, 10.0)},
    "AG": {"train_rows": 1260, "depths": (2, 3), "lambdas": (10.0, 20.0)},
}


def neutral_threshold(daily_volatility_pct, horizon):
    return np.maximum(0.05, np.asarray(daily_volatility_pct) * np.sqrt(horizon) * 0.25)


def select_model(train: pd.DataFrame, features: list[str], target: str, code: str):
    profile = PROFILES.get(code, {"train_rows": len(train), "depths": (3,), "lambdas": (5.0,)})
    train = train.tail(profile["train_rows"])
    candidates = [(d, r) for d in profile["depths"] for r in profile["lambdas"]]
    def create(depth, regularization):
        return XGBRegressor(n_estimators=250, max_depth=depth, learning_rate=0.03,
                            subsample=0.85, colsample_bytree=0.85, reg_lambda=regularization,
                            reg_alpha=0.1, objective="reg:squarederror", random_state=42, n_jobs=4)
    scores = []
    zero_mse = None
    # Validation is entirely inside the outer training set. Purge labels that
    # extend into validation, then refit the chosen model on all known labels.
    if len(candidates) > 1 and len(train) >= 504:
        validation = train.iloc[-126:]
        zero_mse = float(np.mean(validation[target].to_numpy() ** 2))
        inner = train.iloc[:-126]
        inner = inner.loc[inner.label_end_date < validation.date.iloc[0]]
        for depth, regularization in candidates:
            model = create(depth, regularization)
            model.fit(inner[features], inner[target])
            scores.append((float(mean_squared_error(validation[target], model.predict(validation[features]))), depth, regularization))
        _, depth, regularization = min(scores)
    else:
        depth, regularization = candidates[0]
    model = create(depth, regularization)
    model.fit(train[features], train[target])
    return model, {"version": MODEL_VERSION, "product": code, "train_rows": len(train),
                   "max_depth": depth, "reg_lambda": regularization,
                   "selection": "past_126_rows_purged_mse" if scores else "fixed_insufficient_validation",
                   "validation_mse": min(scores)[0] if scores else None,
                   "validation_zero_mse": zero_mse,
                   "beats_zero_validation": min(scores)[0] < zero_mse if scores else None}
