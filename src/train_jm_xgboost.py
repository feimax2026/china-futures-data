from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "coking_coal_JM0.csv"
ML_DIR = PROJECT_ROOT / "analysis" / "ml"
ML_REPORT_DIR = PROJECT_ROOT / "reports" / "ml"
ML_FIGURE_DIR = ML_REPORT_DIR / "figures"

SYMBOL = "JM0"
FORWARD_DAYS = 5
INITIAL_TRAIN_RATIO = 0.60
TEST_WINDOW_DAYS = 63
TRANSACTION_COST_BPS = 2.0


@dataclass(frozen=True)
class BacktestMetrics:
    rows: int
    start_date: str
    end_date: str
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


def ensure_dirs() -> None:
    ML_DIR.mkdir(parents=True, exist_ok=True)
    ML_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_jm_csv() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing {RAW_CSV}. Run src/download_futures.py first.")

    df = pd.read_csv(RAW_CSV)
    required = {"date", "open", "high", "low", "close", "volume", "hold", "settle"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {RAW_CSV}: {sorted(missing)}")

    df = df.rename(columns={"hold": "open_interest"})
    df["symbol"] = SYMBOL
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].replace(0, np.nan).astype(float)
    open_interest = out["open_interest"].replace(0, np.nan).astype(float)

    out["daily_return_pct"] = np.log(close / close.shift(1)) * 100
    out["intraday_range_pct"] = (out["high"] - out["low"]) / close * 100
    out["settle_close_spread_pct"] = (out["settle"] - close) / close * 100

    feature_cols: list[str] = []
    for window in [1, 2, 3, 5, 10, 20]:
        col = f"return_{window}d_pct"
        out[col] = np.log(close / close.shift(window)) * 100
        feature_cols.append(col)

    for window in [5, 10, 20, 60]:
        vol_col = f"volatility_{window}d_pct"
        ma_col = f"ma_gap_{window}d_pct"
        out[vol_col] = out["daily_return_pct"].rolling(window).std()
        out[ma_col] = (close / close.rolling(window).mean() - 1) * 100
        feature_cols.extend([vol_col, ma_col])

    for window in [1, 5, 20]:
        volume_col = f"volume_change_{window}d_pct"
        oi_col = f"open_interest_change_{window}d_pct"
        out[volume_col] = np.log(volume / volume.shift(window)) * 100
        out[oi_col] = np.log(open_interest / open_interest.shift(window)) * 100
        feature_cols.extend([volume_col, oi_col])

    for window in [20, 60]:
        oi_z_col = f"open_interest_zscore_{window}d"
        out[oi_z_col] = (
            (open_interest - open_interest.rolling(window).mean())
            / open_interest.rolling(window).std()
        )
        feature_cols.append(oi_z_col)

    feature_cols.extend(["intraday_range_pct", "settle_close_spread_pct"])
    out[f"future_{FORWARD_DAYS}d_return_pct"] = np.log(close.shift(-FORWARD_DAYS) / close) * 100
    out[f"future_{FORWARD_DAYS}d_up"] = (
        out[f"future_{FORWARD_DAYS}d_return_pct"] > 0
    ).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan)
    feature_data = out.dropna(subset=feature_cols).copy()
    model_data = out.dropna(subset=feature_cols + [f"future_{FORWARD_DAYS}d_return_pct"]).copy()
    feature_data.to_csv(ML_DIR / "jm_xgboost_all_features.csv", index=False)
    model_data.to_csv(ML_DIR / "jm_xgboost_features.csv", index=False)
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


def walk_forward_predict(model_data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    target_col = f"future_{FORWARD_DAYS}d_return_pct"
    initial_train_size = int(len(model_data) * INITIAL_TRAIN_RATIO)
    predictions: list[pd.DataFrame] = []

    start = initial_train_size
    while start < len(model_data):
        end = min(start + TEST_WINDOW_DAYS, len(model_data))
        train = model_data.iloc[:start]
        test = model_data.iloc[start:end]

        model = make_model()
        model.fit(train[feature_cols], train[target_col])

        block = test[["date", "close", "daily_return_pct", target_col, f"future_{FORWARD_DAYS}d_up"]].copy()
        block["pred_future_5d_return_pct"] = model.predict(test[feature_cols])
        block["train_end_date"] = train["date"].iloc[-1]
        block["test_window_start"] = test["date"].iloc[0]
        block["test_window_end"] = test["date"].iloc[-1]
        predictions.append(block)

        start = end

    if not predictions:
        raise ValueError("Not enough rows for walk-forward prediction.")

    result = pd.concat(predictions, ignore_index=True)
    result["signal"] = np.where(result["pred_future_5d_return_pct"] > 0, 1, -1)
    result["position"] = result["signal"].shift(1).fillna(0)
    cost_pct = TRANSACTION_COST_BPS / 100.0
    result["turnover"] = result["position"].diff().abs().fillna(result["position"].abs())
    result["strategy_daily_return_pct"] = (
        result["position"] * result["daily_return_pct"] - result["turnover"] * cost_pct
    )
    result["buy_hold_daily_return_pct"] = result["daily_return_pct"]
    result["strategy_equity"] = np.exp(result["strategy_daily_return_pct"].cumsum() / 100)
    result["buy_hold_equity"] = np.exp(result["buy_hold_daily_return_pct"].cumsum() / 100)
    result.to_csv(ML_DIR / "jm_xgboost_predictions.csv", index=False)
    return result


def train_final_model(
    feature_data: pd.DataFrame, model_data: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_col = f"future_{FORWARD_DAYS}d_return_pct"
    model = make_model()
    model.fit(model_data[feature_cols], model_data[target_col])
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(ML_DIR / "jm_xgboost_feature_importance.csv", index=False)

    latest = feature_data.iloc[[-1]][["date", "close"] + feature_cols].copy()
    latest["pred_future_5d_return_pct"] = model.predict(latest[feature_cols])
    latest["signal"] = np.where(latest["pred_future_5d_return_pct"] > 0, 1, -1)
    latest_signal = latest[["date", "close", "pred_future_5d_return_pct", "signal"]]
    latest_signal.to_csv(ML_DIR / "jm_xgboost_latest_signal.csv", index=False)
    return importance, latest_signal


def compute_metrics(predictions: pd.DataFrame) -> BacktestMetrics:
    target_col = f"future_{FORWARD_DAYS}d_return_pct"
    y_true = predictions[target_col]
    y_pred = predictions["pred_future_5d_return_pct"]
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)

    daily = predictions["strategy_daily_return_pct"]
    annual_return = daily.mean() * 252
    annual_vol = daily.std(ddof=0) * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol else np.nan
    equity = predictions["strategy_equity"]
    drawdown = equity / equity.cummax() - 1

    metrics = BacktestMetrics(
        rows=len(predictions),
        start_date=predictions["date"].iloc[0].date().isoformat(),
        end_date=predictions["date"].iloc[-1].date().isoformat(),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        directional_accuracy=float((direction_true == direction_pred).mean()),
        strategy_total_return_pct=float((equity.iloc[-1] - 1) * 100),
        buy_hold_total_return_pct=float((predictions["buy_hold_equity"].iloc[-1] - 1) * 100),
        strategy_annual_return_pct=float(annual_return),
        strategy_annual_vol_pct=float(annual_vol),
        strategy_sharpe=float(sharpe),
        max_drawdown_pct=float(drawdown.min() * 100),
        trade_count=int((predictions["turnover"] > 0).sum()),
    )
    pd.DataFrame([metrics.__dict__]).to_csv(ML_DIR / "jm_xgboost_backtest_metrics.csv", index=False)
    return metrics


def plot_equity_curve(predictions: pd.DataFrame) -> Path:
    path = ML_FIGURE_DIR / "jm_xgboost_equity_curve.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(predictions["date"], predictions["strategy_equity"], label="XGBoost signal", linewidth=1.5)
    ax.plot(predictions["date"], predictions["buy_hold_equity"], label="Buy and hold", linewidth=1.2)
    ax.set_title("JM0 XGBoost walk-forward backtest")
    ax.set_ylabel("Equity, start = 1")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_feature_importance(importance: pd.DataFrame) -> Path:
    path = ML_FIGURE_DIR / "jm_xgboost_feature_importance.png"
    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"], color="#376996")
    ax.set_title("JM0 XGBoost feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    formatted = df.copy()
    for col in formatted.select_dtypes(include=[float]).columns:
        formatted[col] = formatted[col].map(lambda value: f"{value:.{float_digits}f}")
    return formatted.to_markdown(index=False)


def write_report(
    model_data: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    latest_signal: pd.DataFrame,
    metrics: BacktestMetrics,
    feature_cols: list[str],
) -> Path:
    report_path = ML_REPORT_DIR / "jm_xgboost_report.md"
    metrics_table = pd.DataFrame([metrics.__dict__])
    latest_backtest_signal = predictions.iloc[-1]
    latest_live_signal = latest_signal.iloc[-1]
    top_features = importance.head(15)

    report = f"""# 焦煤 JM0 XGBoost 未来 5 日收益预测

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 目标

这份报告用焦煤连续合约 `JM0` 的日线数据训练 XGBoost 回归模型，预测未来 `{FORWARD_DAYS}` 个交易日收益率。当前版本读取本地 CSV，不再依赖 Yahoo Finance 或股票数据源。

## 2. 数据字段

- 输入文件：`data/raw/coking_coal_JM0.csv`
- 使用字段：`date/open/high/low/close/volume/settle/hold`
- `hold` 已映射为 `open_interest`
- 样本区间：`{model_data["date"].iloc[0].date().isoformat()}` 到 `{model_data["date"].iloc[-1].date().isoformat()}`
- 特征数量：`{len(feature_cols)}`
- 有效样本：`{len(model_data):,}`

## 3. 特征工程

保留的特征方向：

- 过去 1、2、3、5、10、20 日收益率
- 5、10、20、60 日历史波动率
- 5、10、20、60 日均线偏离
- 成交量变化率
- 持仓量变化率
- 持仓量 z-score
- 日内振幅
- 结算价与收盘价偏离

## 4. Walk-forward 回测设置

- 初始训练比例：`{INITIAL_TRAIN_RATIO:.0%}`
- 每次向前测试窗口：`{TEST_WINDOW_DAYS}` 个交易日
- 预测目标：未来 `{FORWARD_DAYS}` 日对数收益率
- 信号规则：预测值大于 0 做多，否则做空
- 交易成本假设：单次换仓 `{TRANSACTION_COST_BPS}` bps

## 5. 核心结果

{markdown_table(metrics_table, 4)}

## 6. 最近一条预测

下面有两种“最近”：第一行是当前数据可用的最新特征日期，第二行是回测中最后一个已经能验证未来 5 日真实收益的日期。

| date | close | pred_future_5d_return_pct | signal |
|---|---:|---:|---:|
| {latest_live_signal["date"].date().isoformat()} | {latest_live_signal["close"]:.2f} | {latest_live_signal["pred_future_5d_return_pct"]:.4f} | {int(latest_live_signal["signal"])} |
| {latest_backtest_signal["date"].date().isoformat()} | {latest_backtest_signal["close"]:.2f} | {latest_backtest_signal["pred_future_5d_return_pct"]:.4f} | {int(latest_backtest_signal["signal"])} |

## 7. 特征重要性

{markdown_table(top_features, 4)}

![Equity curve](figures/jm_xgboost_equity_curve.png)

![Feature importance](figures/jm_xgboost_feature_importance.png)

## 8. 如何阅读

这不是正式交易策略，而是机器学习研究样板。重点看三件事：

1. 时间序列切分是否避免未来数据泄露。
2. 预测未来 5 日收益是否比随机方向更稳定。
3. 回测收益是否来自少数极端行情，还是较稳定的信号贡献。

下一步应该加入 `I0/SM0` 的跨品种特征，并把做多/做空阈值从 0 改成更保守的分位数阈值。
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    ensure_dirs()
    raw = load_jm_csv()
    feature_data, model_data, feature_cols = add_features(raw)
    predictions = walk_forward_predict(model_data, feature_cols)
    importance, latest_signal = train_final_model(feature_data, model_data, feature_cols)
    metrics = compute_metrics(predictions)
    plot_equity_curve(predictions)
    plot_feature_importance(importance)
    report_path = write_report(model_data, predictions, importance, latest_signal, metrics, feature_cols)

    print(f"all features -> {ML_DIR / 'jm_xgboost_all_features.csv'}")
    print(f"features -> {ML_DIR / 'jm_xgboost_features.csv'}")
    print(f"predictions -> {ML_DIR / 'jm_xgboost_predictions.csv'}")
    print(f"latest signal -> {ML_DIR / 'jm_xgboost_latest_signal.csv'}")
    print(f"metrics -> {ML_DIR / 'jm_xgboost_backtest_metrics.csv'}")
    print(f"report -> {report_path}")
    print(
        "summary -> "
        f"rows={metrics.rows}, directional_accuracy={metrics.directional_accuracy:.3f}, "
        f"strategy_total_return_pct={metrics.strategy_total_return_pct:.2f}, "
        f"max_drawdown_pct={metrics.max_drawdown_pct:.2f}"
    )


if __name__ == "__main__":
    main()
