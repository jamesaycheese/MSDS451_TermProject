"""Performance of AMAN and its benchmarks over different periods.

The notebook reports one set of figures for the whole backtest. This script recomputes
the same statistics, with the same formulas, for several periods so the strategy can be
compared with its benchmarks on equal terms:

- full:   the whole backtest
- traded: from AMAN's first rebalance (it holds cash for the first year while it
          gathers price history, which lowers its full-period return)
- test:   2015 onward, the out-of-sample period for the XGBoost and LSTM models
- gfc:    October 2007 to December 2009, the financial crisis

Run from the repo folder:

    python period_stats.py

It prints the table and writes period_summary.csv.
"""
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def summarize_curve(series: pd.Series) -> dict:
    """Same formulas as summarize_curve() in the notebook."""
    dr = series.pct_change().dropna()
    return {
        "ann_return": (1.0 + dr.mean()) ** TRADING_DAYS_PER_YEAR - 1.0,
        "ann_vol": dr.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR),
        "sharpe": np.sqrt(TRADING_DAYS_PER_YEAR) * dr.mean() / dr.std(ddof=1),
        "max_drawdown": float((series / series.cummax() - 1.0).min()),
        "growth_of_1": float(series.iloc[-1] / series.iloc[0]),
    }


def load_curves() -> pd.DataFrame:
    """Net and gross growth curves for every portfolio, one column each."""
    net = pd.read_csv("all_value_curves_net.csv", index_col=0, parse_dates=True)
    aman = pd.read_csv("aman_values_nav.csv", index_col=0, parse_dates=True)
    bench = pd.read_csv("bench_values_nav.csv", index_col=0, parse_dates=True)
    gross = pd.DataFrame({
        "AMAN_gross": aman["AMAN_gross_nav"] / aman["AMAN_gross_nav"].iloc[0],
        "EW_gross": bench["EW_gross_nav"] / bench["EW_gross_nav"].iloc[0],
        "60_40_gross": bench["60_40_gross_nav"] / bench["60_40_gross_nav"].iloc[0],
    })
    return pd.concat([net, gross], axis=1)


def main() -> None:
    curves = load_curves()
    weights = pd.read_csv("aman_weights.csv", index_col=0, parse_dates=True)
    end = curves.index[-1]
    periods = {
        "full": (curves.index[0], end),
        "traded": (weights.index[0], end),
        "test": (pd.Timestamp("2015-01-01"), end),
        "gfc": (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-12-31")),
    }

    rows = []
    for name, (start, stop) in periods.items():
        window = curves.loc[start:stop]
        for portfolio in curves.columns:
            rows.append({
                "period": name,
                "start": window.index[0].date(),
                "end": window.index[-1].date(),
                "portfolio": portfolio,
                **summarize_curve(window[portfolio]),
            })

    table = pd.DataFrame(rows)
    table.to_csv("period_summary.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        for name in periods:
            print(f"\n== {name} ==")
            print(table[table["period"] == name].drop(columns="period").to_string(index=False))
    print("\nWrote period_summary.csv")


if __name__ == "__main__":
    main()
