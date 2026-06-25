from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.product_config import PROJECT_ROOT, ProductConfig, get_product


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ML_DIR = PROJECT_ROOT / "analysis" / "ml"
ML_REPORT_DIR = PROJECT_ROOT / "reports" / "ml"
ML_FIGURE_DIR = ML_REPORT_DIR / "figures"

DEFAULT_FORWARD_DAYS = 5
BACKTEST_START_DATE = pd.Timestamp("2021-05-11")
INITIAL_TRAIN_RATIO = 0.60
MIN_TRAIN_ROWS = 252
TEST_WINDOW_DAYS = 63
TRANSACTION_COST_BPS = 2.0

BASE_COLUMNS = {"date", "open", "high", "low", "close", "volume", "settle"}
WEIGHTED_FEATURE_CANDIDATES = [
    "active_contracts",
    "dominant_volume_share",
    "second_volume_share",
    "dominant_oi_share",
    "second_oi_share",
    "volume_roll_pressure",
    "oi_roll_pressure",
    "dominant_contract_changed",
    "dominant_volume_contract_changed",
]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    label: str
    path: str | Path
    output_prefix: str


@dataclass(frozen=True)
class BacktestMetrics:
    dataset: str
    horizon_days: int
    rows: int
    trainable_rows: int
    model_start_date: str
    model_end_date: str
    backtest_start_date: str
    backtest_end_date: str
    mae: float
    rmse: float
    r2: float
    directional_accuracy: float
    strategy_total_return_pct: float
    buy_hold_total_return_pct: float
    strategy_annual_return_pct: float
    strategy_annual_vol_pct: float
    strategy_sharpe: float
    max_drawdown_pct: float
    trade_count: int


@dataclass(frozen=True)
class DatasetResult:
    config: DatasetConfig
    feature_data: pd.DataFrame
    model_data: pd.DataFrame
    predictions: pd.DataFrame
    importance: pd.DataFrame
    latest_signal: pd.DataFrame
    metrics: BacktestMetrics
    feature_cols: list[str]


def ensure_dirs() -> None:
    ML_DIR.mkdir(parents=True, exist_ok=True)
    ML_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def product_configs(product_code: str) -> list[DatasetConfig]:
    product = get_product(product_code)
    lower = product.lower_code
    return [
        DatasetConfig(
            name=f"{lower}_main",
            label=f"{product.code}0 主力连续",
            path=PROJECT_ROOT / "data" / "raw" / product.main_csv,
            output_prefix=f"{lower}_main",
        ),
        DatasetConfig(
            name=f"{lower}_weighted",
            label=f"{product.code} 全月份持仓加权收益指数",
            path=PROJECT_ROOT / "data" / "processed" / f"{lower}_weighted_index.csv",
            output_prefix=f"{lower}_weighted",
        ),
    ]


def target_column(horizon: int) -> str:
    return f"future_{horizon}d_return_pct"


def direction_column(horizon: int) -> str:
    return f"future_{horizon}d_up"


def prediction_column(horizon: int) -> str:
    return f"pred_future_{horizon}d_return_pct"


def artifact_prefix(config: DatasetConfig, horizon: int) -> str:
    if horizon == DEFAULT_FORWARD_DAYS:
        return config.output_prefix
    return f"{config.output_prefix}_{horizon}d"


def product_artifact_name(product: ProductConfig, horizon: int, suffix: str) -> str:
    if horizon == DEFAULT_FORWARD_DAYS:
        return f"{product.lower_code}_{suffix}"
    return f"{product.lower_code}_{horizon}d_{suffix}"


def purged_train_end(test_start: int, horizon: int) -> int:
    return max(test_start - horizon, 0)


def validate_horizon(horizon: int) -> int:
    if horizon < 1:
        raise ValueError("Prediction horizon must be at least 1 trading day.")
    return horizon


def plot_label(config: DatasetConfig) -> str:
    return config.label.replace("主力连续", "Main Continuous").replace("全月份持仓加权收益指数", "Weighted Return Index")


def normalize_market_frame(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    if "open_interest" not in out.columns and "hold" in out.columns:
        out = out.rename(columns={"hold": "open_interest"})

    required = BASE_COLUMNS | {"open_interest"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {sorted(missing)}")

    out["date"] = pd.to_datetime(out["date"])
    for column in ["open", "high", "low", "close", "volume", "settle", "open_interest"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in WEIGHTED_FEATURE_CANDIDATES + ["oi_weighted_close"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["dataset"] = dataset_name
    return out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return normalize_market_frame(pd.read_csv(path), config.name)


def add_features(
    df: pd.DataFrame,
    config: DatasetConfig,
    horizon: int = DEFAULT_FORWARD_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    horizon = validate_horizon(horizon)
    out = normalize_market_frame(df, config.name)
    close = out["close"].astype(float)
    volume = out["volume"].replace(0, np.nan).astype(float)
    open_interest = out["open_interest"].replace(0, np.nan).astype(float)

    out["daily_return_pct"] = np.log(close / close.shift(1)) * 100
    out["intraday_range_pct"] = (out["high"] - out["low"]) / close * 100
    out["settle_close_spread_pct"] = (out["settle"] - close) / close * 100

    feature_cols: list[str] = []
    for window in [1, 2, 3, 5, 10, 20]:
        column = f"return_{window}d_pct"
        out[column] = np.log(close / close.shift(window)) * 100
        feature_cols.append(column)

    for window in [5, 10, 20, 60]:
        vol_column = f"volatility_{window}d_pct"
        ma_column = f"ma_gap_{window}d_pct"
        out[vol_column] = out["daily_return_pct"].rolling(window).std()
        out[ma_column] = (close / close.rolling(window).mean() - 1) * 100
        feature_cols.extend([vol_column, ma_column])

    for window in [1, 5, 20]:
        volume_column = f"volume_change_{window}d_pct"
        oi_column = f"open_interest_change_{window}d_pct"
        out[volume_column] = np.log(volume / volume.shift(window)) * 100
        out[oi_column] = np.log(open_interest / open_interest.shift(window)) * 100
        feature_cols.extend([volume_column, oi_column])

    for window in [20, 60]:
        oi_z_column = f"open_interest_zscore_{window}d"
        out[oi_z_column] = (
            (open_interest - open_interest.rolling(window).mean())
            / open_interest.rolling(window).std()
        )
        feature_cols.append(oi_z_column)

    feature_cols.extend(["intraday_range_pct", "settle_close_spread_pct"])
    if "oi_weighted_close" in out.columns:
        out["oi_volume_weighted_close_gap_pct"] = (out["oi_weighted_close"] - close) / close * 100
        feature_cols.append("oi_volume_weighted_close_gap_pct")

    for column in WEIGHTED_FEATURE_CANDIDATES:
        if column in out.columns:
            feature_cols.append(column)

    target_col = target_column(horizon)
    up_col = direction_column(horizon)
    out[target_col] = np.log(close.shift(-horizon) / close) * 100
    out[up_col] = (out[target_col] > 0).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan)
    feature_data = out.dropna(subset=feature_cols).copy()
    model_data = out.dropna(subset=feature_cols + [target_col]).copy()
    return feature_data, model_data, feature_cols


def make_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )


def prediction_start_position(model_data: pd.DataFrame) -> int:
    after_start = np.flatnonzero(model_data["date"].to_numpy() >= BACKTEST_START_DATE.to_datetime64())
    if len(after_start) > 0 and after_start[0] >= MIN_TRAIN_ROWS:
        return int(after_start[0])
    return max(int(len(model_data) * INITIAL_TRAIN_RATIO), MIN_TRAIN_ROWS)


def walk_forward_predict(
    model_data: pd.DataFrame,
    feature_cols: list[str],
    config: DatasetConfig,
    horizon: int = DEFAULT_FORWARD_DAYS,
) -> pd.DataFrame:
    horizon = validate_horizon(horizon)
    target_col = target_column(horizon)
    up_col = direction_column(horizon)
    pred_col = prediction_column(horizon)
    predictions: list[pd.DataFrame] = []

    start = prediction_start_position(model_data)
    while start < len(model_data):
        end = min(start + TEST_WINDOW_DAYS, len(model_data))
        train_end = purged_train_end(start, horizon)
        train = model_data.iloc[:train_end]
        test = model_data.iloc[start:end]
        if len(train) < MIN_TRAIN_ROWS:
            start = end
            continue

        model = make_model()
        model.fit(train[feature_cols], train[target_col])

        block = test[["date", "close", "daily_return_pct", target_col, up_col]].copy()
        block["dataset"] = config.name
        block[pred_col] = model.predict(test[feature_cols])
        block["train_end_date"] = train["date"].iloc[-1]
        block["test_window_start"] = test["date"].iloc[0]
        block["test_window_end"] = test["date"].iloc[-1]
        predictions.append(block)
        start = end

    if not predictions:
        raise ValueError(f"{config.name} does not have enough rows for walk-forward prediction.")

    result = pd.concat(predictions, ignore_index=True)
    result["signal"] = np.where(result[pred_col] > 0, 1, -1)
    result["position"] = result["signal"].shift(1).fillna(0)
    cost_pct = TRANSACTION_COST_BPS / 100.0
    result["turnover"] = result["position"].diff().abs().fillna(result["position"].abs())
    result["strategy_daily_return_pct"] = (
        result["position"] * result["daily_return_pct"] - result["turnover"] * cost_pct
    )
    result["buy_hold_daily_return_pct"] = result["daily_return_pct"]
    result["strategy_equity"] = np.exp(result["strategy_daily_return_pct"].cumsum() / 100)
    result["buy_hold_equity"] = np.exp(result["buy_hold_daily_return_pct"].cumsum() / 100)
    return result


def train_final_model(
    feature_data: pd.DataFrame,
    model_data: pd.DataFrame,
    feature_cols: list[str],
    horizon: int = DEFAULT_FORWARD_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = validate_horizon(horizon)
    target_col = target_column(horizon)
    pred_col = prediction_column(horizon)
    model = make_model()
    model.fit(model_data[feature_cols], model_data[target_col])

    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    latest = feature_data.iloc[[-1]][["date", "close"] + feature_cols].copy()
    latest[pred_col] = model.predict(latest[feature_cols])
    latest["signal"] = np.where(latest[pred_col] > 0, 1, -1)
    return importance, latest[["date", "close", pred_col, "signal"]]


def compute_metrics(
    model_data: pd.DataFrame,
    predictions: pd.DataFrame,
    config: DatasetConfig,
    horizon: int = DEFAULT_FORWARD_DAYS,
) -> BacktestMetrics:
    horizon = validate_horizon(horizon)
    target_col = target_column(horizon)
    pred_col = prediction_column(horizon)
    y_true = predictions[target_col]
    y_pred = predictions[pred_col]
    daily = predictions["strategy_daily_return_pct"]
    annual_return = daily.mean() * 252
    annual_vol = daily.std(ddof=0) * np.sqrt(252)
    equity = predictions["strategy_equity"]
    drawdown = equity / equity.cummax() - 1

    return BacktestMetrics(
        dataset=config.label,
        horizon_days=horizon,
        rows=len(predictions),
        trainable_rows=len(model_data),
        model_start_date=model_data["date"].iloc[0].date().isoformat(),
        model_end_date=model_data["date"].iloc[-1].date().isoformat(),
        backtest_start_date=predictions["date"].iloc[0].date().isoformat(),
        backtest_end_date=predictions["date"].iloc[-1].date().isoformat(),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        directional_accuracy=float((np.sign(y_true) == np.sign(y_pred)).mean()),
        strategy_total_return_pct=float((equity.iloc[-1] - 1) * 100),
        buy_hold_total_return_pct=float((predictions["buy_hold_equity"].iloc[-1] - 1) * 100),
        strategy_annual_return_pct=float(annual_return),
        strategy_annual_vol_pct=float(annual_vol),
        strategy_sharpe=float(annual_return / annual_vol) if annual_vol else np.nan,
        max_drawdown_pct=float(drawdown.min() * 100),
        trade_count=int((predictions["turnover"] > 0).sum()),
    )


def evaluate_forecasts(
    predictions: pd.DataFrame,
    horizon: int = DEFAULT_FORWARD_DAYS,
) -> pd.DataFrame:
    horizon = validate_horizon(horizon)
    target_col = target_column(horizon)
    pred_col = prediction_column(horizon)
    rows: list[dict[str, float | int | str]] = []

    for forecast, mask, correct_direction in [
        ("bullish", predictions[pred_col] > 0, predictions[target_col] > 0),
        ("bearish", predictions[pred_col] <= 0, predictions[target_col] < 0),
    ]:
        sample = predictions.loc[mask]
        if sample.empty:
            continue
        rows.append(
            {
                "forecast": forecast,
                "horizon_days": horizon,
                "samples": len(sample),
                "sample_share": float(len(sample) / len(predictions)),
                "avg_predicted_return_pct": float(sample[pred_col].mean()),
                "avg_realized_return_pct": float(sample[target_col].mean()),
                "median_realized_return_pct": float(sample[target_col].median()),
                "direction_accuracy": float(correct_direction.loc[mask].mean()),
            }
        )

    return pd.DataFrame(rows)


def plot_feature_importance(
    importance: pd.DataFrame,
    config: DatasetConfig,
    horizon: int = DEFAULT_FORWARD_DAYS,
) -> Path:
    prefix = artifact_prefix(config, horizon)
    path = ML_FIGURE_DIR / f"{prefix}_xgboost_feature_importance.png"
    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"], color="#376996")
    ax.set_title(f"{plot_label(config)} XGBoost {horizon}-day feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_comparison_equity(
    results: list[DatasetResult],
    product: ProductConfig,
    horizon: int = DEFAULT_FORWARD_DAYS,
) -> Path:
    path = ML_FIGURE_DIR / product_artifact_name(
        product, horizon, "xgboost_compare_equity_curve.png"
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#174A7C", "#C65D2E", "#6B7D3A", "#8A5A83"]

    for idx, result in enumerate(results):
        color = colors[idx % len(colors)]
        predictions = result.predictions
        ax.plot(
            predictions["date"],
            predictions["strategy_equity"],
            label=f"{plot_label(result.config)} strategy",
            linewidth=1.6,
            color=color,
        )
        ax.plot(
            predictions["date"],
            predictions["buy_hold_equity"],
            label=f"{plot_label(result.config)} buy & hold",
            linewidth=1.0,
            linestyle="--",
            color=color,
            alpha=0.65,
        )

    ax.set_title(f"{product.code} XGBoost {horizon}-day walk-forward comparison")
    ax.set_ylabel("Equity, start = 1")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    formatted = df.copy()
    for column in formatted.select_dtypes(include=[float]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{float_digits}f}")
    return formatted.to_markdown(index=False)


def run_dataset(
    config: DatasetConfig,
    horizon: int = DEFAULT_FORWARD_DAYS,
) -> DatasetResult:
    horizon = validate_horizon(horizon)
    raw = load_dataset(config)
    feature_data, model_data, feature_cols = add_features(raw, config, horizon)
    predictions = walk_forward_predict(model_data, feature_cols, config, horizon)
    importance, latest_signal = train_final_model(
        feature_data, model_data, feature_cols, horizon
    )
    metrics = compute_metrics(model_data, predictions, config, horizon)
    forecast_quality = evaluate_forecasts(predictions, horizon)
    prefix = artifact_prefix(config, horizon)

    feature_data.to_csv(ML_DIR / f"{prefix}_xgboost_all_features.csv", index=False)
    model_data.to_csv(ML_DIR / f"{prefix}_xgboost_features.csv", index=False)
    predictions.to_csv(ML_DIR / f"{prefix}_xgboost_predictions.csv", index=False)
    importance.to_csv(ML_DIR / f"{prefix}_xgboost_feature_importance.csv", index=False)
    latest_signal.to_csv(ML_DIR / f"{prefix}_xgboost_latest_signal.csv", index=False)
    pd.DataFrame([metrics.__dict__]).to_csv(
        ML_DIR / f"{prefix}_xgboost_backtest_metrics.csv", index=False
    )
    forecast_quality.to_csv(
        ML_DIR / f"{prefix}_xgboost_forecast_quality.csv", index=False
    )
    plot_feature_importance(importance, config, horizon)

    return DatasetResult(
        config=config,
        feature_data=feature_data,
        model_data=model_data,
        predictions=predictions,
        importance=importance,
        latest_signal=latest_signal,
        metrics=metrics,
        feature_cols=feature_cols,
    )


def write_report(
    results: list[DatasetResult],
    product: ProductConfig,
    comparison_figure: Path,
    horizon: int = DEFAULT_FORWARD_DAYS,
) -> Path:
    horizon = validate_horizon(horizon)
    pred_col = prediction_column(horizon)
    report_path = ML_REPORT_DIR / product_artifact_name(
        product, horizon, "xgboost_compare_report.md"
    )
    metrics = pd.DataFrame([result.metrics.__dict__ for result in results])
    forecast_quality_rows = []
    for result in results:
        quality = evaluate_forecasts(result.predictions, horizon).copy()
        quality.insert(0, "dataset", result.config.label)
        forecast_quality_rows.append(quality)
    forecast_quality = pd.concat(forecast_quality_rows, ignore_index=True)
    latest_rows = []
    for result in results:
        latest_live_signal = result.latest_signal.iloc[-1]
        latest_backtest_signal = result.predictions.iloc[-1]
        latest_rows.append(
            {
                "dataset": result.config.label,
                "latest_feature_date": latest_live_signal["date"].date().isoformat(),
                "latest_close": latest_live_signal["close"],
                pred_col: latest_live_signal[pred_col],
                "latest_signal": int(latest_live_signal["signal"]),
                "last_verified_date": latest_backtest_signal["date"].date().isoformat(),
                "last_verified_pred_pct": latest_backtest_signal[pred_col],
                "last_verified_signal": int(latest_backtest_signal["signal"]),
            }
        )

    top_feature_sections = []
    for result in results:
        top_feature_sections.append(
            f"### {result.config.label}\n\n"
            f"{markdown_table(result.importance.head(12), 4)}\n\n"
            f"![{result.config.label} feature importance]"
            f"(figures/{artifact_prefix(result.config, horizon)}_xgboost_feature_importance.png)"
        )

    report = f"""# {product.name_zh} {product.code} XGBoost {horizon}日预测：主力连续 vs 全月份加权指数

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 为什么做这个对照

主力连续合约 `{product.code}0` 在移仓换月附近容易出现成交量和持仓量的机械变化。XGBoost 可以学习这种模式，但不能自动区分“真实供需信息”和“合约切换噪音”。所以这里新增一个全月份持仓加权收益指数：先用所有可用 `{product.code}YYMM` 合约的上一日持仓量合成当日收益率，再累乘成连续指数，同时保留成交量加权价格、持仓量加权价格、主力份额、次主力份额和换月压力等字段作为额外特征。

## 2. 数据与训练设置

- 主力连续：`data/raw/{product.main_csv}`
- 全月份持仓加权收益指数：`data/processed/{product.lower_code}_weighted_index.csv`
- 预测目标：未来 `{horizon}` 个交易日对数收益率
- 回测起点：`{BACKTEST_START_DATE.date().isoformat()}`
- 训练方式：purged expanding walk-forward，训练集和测试集之间隔离 `{horizon}` 行，每次向前测试 `{TEST_WINDOW_DAYS}` 个交易日
- 信号规则：预测值大于 0 做多，否则做空
- 交易成本假设：单次换仓 `{TRANSACTION_COST_BPS}` bps

## 3. 核心结果

{markdown_table(metrics, 4)}

![Comparison equity](figures/{comparison_figure.name})

## 4. 预测能力

这里直接检查历史预测发生后，未来 `{horizon}` 个交易日的真实收益。`direction_accuracy` 表示看多后实际上涨、或看空后实际下跌的比例；它不涉及仓位、止损或每日换仓。

{markdown_table(forecast_quality, 4)}

## 5. 最近信号

{markdown_table(pd.DataFrame(latest_rows), 4)}

## 6. 特征重要性

{chr(10).join(top_feature_sections)}

## 7. 初步解读

这份报告的重点不是直接给交易信号，而是检查“数据构造方式”会如何改变模型表现。如果加权收益指数的回撤更低、方向准确率更稳，说明它更适合作为研究底座；如果主力连续短期收益更高但依赖换月特征，那就要小心它可能学到了合约切换噪音。
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_product(
    product_code: str,
    horizon: int = DEFAULT_FORWARD_DAYS,
) -> list[DatasetResult]:
    horizon = validate_horizon(horizon)
    ensure_dirs()
    product = get_product(product_code)
    results = [run_dataset(config, horizon) for config in product_configs(product.code)]
    comparison_figure = plot_comparison_equity(results, product, horizon)
    report_path = write_report(results, product, comparison_figure, horizon)

    metrics = pd.DataFrame([result.metrics.__dict__ for result in results])
    metrics_path = ML_DIR / product_artifact_name(
        product, horizon, "xgboost_compare_metrics.csv"
    )
    metrics.to_csv(metrics_path, index=False)

    print(f"comparison metrics -> {metrics_path}")
    print(f"comparison figure -> {comparison_figure}")
    print(f"report -> {report_path}")
    for result in results:
        metric = result.metrics
        print(
            "summary -> "
            f"{result.config.label}: rows={metric.rows}, "
            f"directional_accuracy={metric.directional_accuracy:.3f}, "
            f"strategy_total_return_pct={metric.strategy_total_return_pct:.2f}, "
            f"max_drawdown_pct={metric.max_drawdown_pct:.2f}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare main-continuous and weighted-index XGBoost models.")
    parser.add_argument("--product", default="JM", help="Product code, e.g. JM or I.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_FORWARD_DAYS,
        help="Forward return horizon in trading days.",
    )
    args = parser.parse_args()
    run_product(args.product, args.horizon)


if __name__ == "__main__":
    main()
