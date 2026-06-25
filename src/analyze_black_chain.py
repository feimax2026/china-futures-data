from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "db" / "china_futures.duckdb"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PROCESSED_DIR = ANALYSIS_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SYMBOL_ORDER = ["I0", "JM0", "SM0"]
SYMBOL_LABELS = {
    "I0": "iron ore",
    "JM0": "coking coal",
    "SM0": "manganese silicon",
}
ROLLING_WINDOW = 60
IRF_PERIODS = 20
MAX_LAGS = 10


@dataclass(frozen=True)
class AnalysisResult:
    start_date: str
    end_date: str
    observation_count: int
    selected_lag: int
    feature_path: Path
    report_path: Path


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_futures_daily() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run src/build_duckdb.py first.")

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.sql(
            """
            SELECT
                date,
                symbol,
                close,
                settle,
                volume,
                hold
            FROM futures_daily
            WHERE symbol IN ('I0', 'JM0', 'SM0')
            ORDER BY date, symbol
            """
        ).df()


def pivot_field(df: pd.DataFrame, field: str) -> pd.DataFrame:
    pivoted = df.pivot(index="date", columns="symbol", values=field)
    pivoted.index = pd.to_datetime(pivoted.index)
    return pivoted.reindex(columns=SYMBOL_ORDER).sort_index()


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close = pivot_field(df, "close").dropna()
    volume = pivot_field(df, "volume").reindex(close.index)
    hold = pivot_field(df, "hold").reindex(close.index)

    log_close = np.log(close)
    returns = log_close.diff().mul(100).dropna()
    volatility = returns.rolling(20).std()
    volume_change = np.log(volume.replace(0, np.nan)).diff().mul(100)
    hold_change = np.log(hold.replace(0, np.nan)).diff().mul(100)

    features = pd.DataFrame(index=close.index)
    for symbol in SYMBOL_ORDER:
        features[f"close_{symbol}"] = close[symbol]
        features[f"ret_{symbol}"] = returns[symbol]
        features[f"vol20_{symbol}"] = volatility[symbol]
        features[f"volume_chg_{symbol}"] = volume_change[symbol]
        features[f"hold_chg_{symbol}"] = hold_change[symbol]

    features = features.replace([np.inf, -np.inf], np.nan)
    feature_path = PROCESSED_DIR / "black_chain_daily_features.csv"
    features.to_csv(feature_path, index_label="date")

    return close, returns, volatility, features


def run_adf_tests(close: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOL_ORDER:
        for series_name, series in [
            ("log_close", np.log(close[symbol])),
            ("return", returns[symbol]),
        ]:
            clean = series.dropna()
            stat, p_value, used_lag, nobs, *_ = adfuller(clean, autolag="AIC")
            rows.append(
                {
                    "symbol": symbol,
                    "series": series_name,
                    "adf_stat": stat,
                    "p_value": p_value,
                    "used_lag": used_lag,
                    "nobs": nobs,
                    "stationary_at_5pct": p_value < 0.05,
                }
            )

    output = pd.DataFrame(rows)
    output.to_csv(PROCESSED_DIR / "adf_tests.csv", index=False)
    return output


def fit_var(returns: pd.DataFrame):
    model_data = returns[SYMBOL_ORDER].dropna().copy()
    model_data.index = pd.RangeIndex(len(model_data))
    max_lags = min(MAX_LAGS, max(1, len(model_data) // 50))
    model = VAR(model_data)
    lag_selection = model.select_order(maxlags=max_lags)
    lag_order = lag_selection.selected_orders.get("aic")
    selected_lag = int(lag_order) if lag_order and lag_order > 0 else 1
    result = model.fit(selected_lag)

    pd.DataFrame([lag_selection.selected_orders]).to_csv(PROCESSED_DIR / "var_lag_order.csv", index=False)
    with (PROCESSED_DIR / "var_summary.txt").open("w", encoding="utf-8") as file:
        file.write(str(result.summary()))

    return model_data, selected_lag, result


def save_price_index_plot(close: pd.DataFrame) -> Path:
    indexed = close.div(close.iloc[0]).mul(100)
    path = FIGURES_DIR / "price_index.png"

    fig, ax = plt.subplots(figsize=(11, 5))
    for symbol in SYMBOL_ORDER:
        ax.plot(indexed.index, indexed[symbol], label=f"{symbol} {SYMBOL_LABELS[symbol]}", linewidth=1.5)
    ax.set_title("Black chain futures price index, first common date = 100")
    ax.set_ylabel("Index")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_correlation_plot(returns: pd.DataFrame) -> Path:
    corr = returns[SYMBOL_ORDER].corr()
    corr.to_csv(PROCESSED_DIR / "return_correlation.csv")
    path = FIGURES_DIR / "return_correlation.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(SYMBOL_ORDER)), SYMBOL_ORDER)
    ax.set_yticks(range(len(SYMBOL_ORDER)), SYMBOL_ORDER)
    for i in range(len(SYMBOL_ORDER)):
        for j in range(len(SYMBOL_ORDER)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Daily return correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_rolling_correlation_plot(returns: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "rolling_correlation_60d.png"

    fig, ax = plt.subplots(figsize=(11, 5))
    for left, right in combinations(SYMBOL_ORDER, 2):
        rolling = returns[left].rolling(ROLLING_WINDOW).corr(returns[right])
        ax.plot(rolling.index, rolling, label=f"{left}-{right}", linewidth=1.3)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_title(f"{ROLLING_WINDOW}-day rolling return correlation")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_irf_plot(var_result) -> tuple[Path, pd.DataFrame]:
    irf = var_result.irf(IRF_PERIODS)
    orth_irfs = irf.orth_irfs
    periods = np.arange(IRF_PERIODS + 1)
    path = FIGURES_DIR / "orthogonalized_irf.png"

    fig, axes = plt.subplots(len(SYMBOL_ORDER), len(SYMBOL_ORDER), figsize=(12, 10), sharex=True)
    for response_idx, response in enumerate(SYMBOL_ORDER):
        for shock_idx, shock in enumerate(SYMBOL_ORDER):
            ax = axes[response_idx, shock_idx]
            ax.plot(periods, orth_irfs[:, response_idx, shock_idx], color="#2f6f9f", linewidth=1.4)
            ax.axhline(0, color="black", linewidth=0.7, alpha=0.45)
            if response_idx == 0:
                ax.set_title(f"Shock: {shock}")
            if shock_idx == 0:
                ax.set_ylabel(f"Response: {response}")
            ax.grid(True, alpha=0.2)
    fig.suptitle("Orthogonalized impulse response functions", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    rows: list[dict[str, object]] = []
    for response_idx, response in enumerate(SYMBOL_ORDER):
        for shock_idx, shock in enumerate(SYMBOL_ORDER):
            series = orth_irfs[1:, response_idx, shock_idx]
            peak_idx = int(np.argmax(np.abs(series))) + 1
            rows.append(
                {
                    "response": response,
                    "shock": shock,
                    "day0_response": orth_irfs[0, response_idx, shock_idx],
                    "peak_abs_day_1_20": peak_idx,
                    "peak_response_1_20": orth_irfs[peak_idx, response_idx, shock_idx],
                    "cumulative_1_20d_response": orth_irfs[1:, response_idx, shock_idx].sum(),
                }
            )
    irf_summary = pd.DataFrame(rows)
    irf_summary.to_csv(PROCESSED_DIR / "irf_summary.csv", index=False)

    return path, irf_summary


def save_fevd_outputs(var_result) -> tuple[Path, pd.DataFrame]:
    fevd = var_result.fevd(IRF_PERIODS + 1)
    horizons = [1, 5, 10, 20]
    rows: list[dict[str, object]] = []

    for target_idx, target in enumerate(SYMBOL_ORDER):
        for horizon in horizons:
            row: dict[str, object] = {"target": target, "horizon_days": horizon}
            for shock_idx, shock in enumerate(SYMBOL_ORDER):
                row[f"shock_{shock}_pct"] = fevd.decomp[target_idx, horizon, shock_idx] * 100
            rows.append(row)

    fevd_table = pd.DataFrame(rows)
    fevd_table.to_csv(PROCESSED_DIR / "fevd_selected_horizons.csv", index=False)

    path = FIGURES_DIR / "fevd_20d.png"
    horizon = 20
    bottom = np.zeros(len(SYMBOL_ORDER))
    x = np.arange(len(SYMBOL_ORDER))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for shock_idx, shock in enumerate(SYMBOL_ORDER):
        values = np.array([fevd.decomp[target_idx, horizon, shock_idx] * 100 for target_idx in range(len(SYMBOL_ORDER))])
        ax.bar(x, values, bottom=bottom, label=f"Shock {shock}", color=colors[shock_idx])
        bottom += values
    ax.set_xticks(x, SYMBOL_ORDER)
    ax.set_ylabel("Share of forecast error variance (%)")
    ax.set_title("FEVD at 20 trading days")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path, fevd_table


def save_latest_dashboard(close: pd.DataFrame, returns: pd.DataFrame, volatility: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOL_ORDER:
        series = close[symbol].dropna()
        ret = returns[symbol].dropna()
        rows.append(
            {
                "symbol": symbol,
                "latest_date": series.index[-1].date().isoformat(),
                "latest_close": series.iloc[-1],
                "return_5d_pct": ret.tail(5).sum(),
                "return_20d_pct": ret.tail(20).sum(),
                "volatility_20d_pct": volatility[symbol].dropna().iloc[-1],
            }
        )
    dashboard = pd.DataFrame(rows)
    dashboard.to_csv(PROCESSED_DIR / "latest_dashboard.csv", index=False)
    return dashboard


def markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    formatted = df.copy()
    for column in formatted.select_dtypes(include=[float]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{float_digits}f}")
    return formatted.to_markdown(index=False)


def build_report(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    adf: pd.DataFrame,
    selected_lag: int,
    irf_summary: pd.DataFrame,
    fevd_table: pd.DataFrame,
    dashboard: pd.DataFrame,
) -> Path:
    report_path = REPORTS_DIR / "black_chain_var_report.md"
    start_date = close.index[0].date().isoformat()
    end_date = close.index[-1].date().isoformat()
    corr = returns[SYMBOL_ORDER].corr().round(3)

    cross_irf = irf_summary[irf_summary["response"] != irf_summary["shock"]].copy()
    cross_irf["abs_cumulative"] = cross_irf["cumulative_1_20d_response"].abs()
    cross_irf = cross_irf.sort_values("abs_cumulative", ascending=False).head(5)
    cross_irf = cross_irf[
        [
            "shock",
            "response",
            "day0_response",
            "peak_abs_day_1_20",
            "peak_response_1_20",
            "cumulative_1_20d_response",
        ]
    ]

    fevd_20 = fevd_table[fevd_table["horizon_days"] == 20].copy()
    adf_short = adf[["symbol", "series", "p_value", "stationary_at_5pct"]].copy()

    report = f"""# 黑色系 VAR-IRF-FEVD 第一版报告

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 研究问题

这一版先回答一个很具体的问题：在黑色系期货内部，铁矿石 (`I0`)、焦煤 (`JM0`) 和锰硅 (`SM0`) 的日收益冲击会如何相互传导？

模型使用的是对数收益率，单位是百分点，不直接使用价格水平。这样更接近平稳序列，也更适合用 VAR 来解释短期冲击响应。

## 2. 数据说明

- 样本区间：`{start_date}` 到 `{end_date}`
- 共同交易日数量：`{len(close):,}`
- 模型变量：`{", ".join(SYMBOL_ORDER)}`
- 正交化 IRF 和 FEVD 的 Cholesky 排序：`{", ".join(SYMBOL_ORDER)}`

这个排序是一个识别假设。当前报告先把它当作基准版本，后面需要用不同排序做稳健性检查。

## 3. 最新交易看板

{markdown_table(dashboard, 3)}

## 4. 平稳性检查

{markdown_table(adf_short, 4)}

通常来说，收益率序列应该比价格水平更适合 VAR。这里的 ADF 检验也支持先用收益率做第一版模型。

## 5. 收益率相关性

{corr.to_markdown()}

![日收益相关性](figures/return_correlation.png)

![60日滚动相关性](figures/rolling_correlation_60d.png)

## 6. VAR 模型设置

- 滞后阶选择准则：AIC
- 选出的滞后阶数：`{selected_lag}`
- IRF 和 FEVD 的观察窗口：`{IRF_PERIODS}` 个交易日

![价格指数](figures/price_index.png)

## 7. IRF：冲击如何传导

下图是正交化冲击响应函数。每一列是冲击来源，每一行是响应变量。

![Orthogonalized IRF](figures/orthogonalized_irf.png)

下面的表只看跨品种响应。`day0_response` 表示当日正交冲击的即时响应，`cumulative_1_20d_response` 表示后续 1-20 个交易日的累计响应。

{markdown_table(cross_irf, 4)}

## 8. FEVD：谁解释了谁的波动

FEVD 用来观察一个品种未来预测误差中，有多少比例可以归因于自身冲击，有多少比例来自其他品种冲击。

{markdown_table(fevd_table, 2)}

![20-day FEVD](figures/fevd_20d.png)

## 9. 第一版读法

这份报告还不是交易信号，它更像一个研究模板。现在比较值得注意的读法是：

1. 先看相关性，判断这些品种是否真的在同一个交易逻辑里。
2. 再看 IRF，观察冲击之后的方向和持续时间。
3. 再看 FEVD，判断某个品种的波动主要由自己解释，还是明显受其他品种影响。
4. 如果结果要用于交易研究，需要改变量排序、换样本区间、加新品种做稳健性检查。

## 10. 下一步扩展

下一批最值得加入的期货变量是 `J0` 焦炭、`RB0` 螺纹钢、`HC0` 热卷、`SF0` 硅铁。再往后，可以加入库存、基差、现货价格、钢厂利润、高炉开工率和宏观变量。
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> AnalysisResult:
    ensure_output_dirs()
    raw = read_futures_daily()
    close, returns, volatility, features = build_features(raw)
    adf = run_adf_tests(close, returns)
    _, selected_lag, var_result = fit_var(returns)

    save_price_index_plot(close)
    save_correlation_plot(returns)
    save_rolling_correlation_plot(returns)
    _, irf_summary = save_irf_plot(var_result)
    _, fevd_table = save_fevd_outputs(var_result)
    dashboard = save_latest_dashboard(close, returns, volatility)

    report_path = build_report(
        close=close,
        returns=returns,
        adf=adf,
        selected_lag=selected_lag,
        irf_summary=irf_summary,
        fevd_table=fevd_table,
        dashboard=dashboard,
    )

    result = AnalysisResult(
        start_date=close.index[0].date().isoformat(),
        end_date=close.index[-1].date().isoformat(),
        observation_count=len(close),
        selected_lag=selected_lag,
        feature_path=PROCESSED_DIR / "black_chain_daily_features.csv",
        report_path=report_path,
    )
    print(f"analysis sample: {result.start_date} -> {result.end_date}")
    print(f"observations: {result.observation_count}")
    print(f"selected VAR lag: {result.selected_lag}")
    print(f"features -> {result.feature_path}")
    print(f"report -> {result.report_path}")
    return result


if __name__ == "__main__":
    main()
