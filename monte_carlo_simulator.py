"""Monte Carlo simulator built on this project's own return history.

Each simulated month is drawn at random, with replacement, from the real months in the
data (December 2005 to October 2025). For ETF mixes, all five ETFs are taken from the
same real month, so the way they moved together is preserved, and the mix is rebalanced
every month. Money moves in or out at the end of each year.

Examples (run from the repo folder):

    python monte_carlo_simulator.py
    python monte_carlo_simulator.py --portfolio aman_gross --years 20
    python monte_carlo_simulator.py --portfolio custom --weights 40 10 30 10 10 --cash 0
    python monte_carlo_simulator.py --portfolio ew --cash 12000        # add $12,000 a year

Portfolios:
    aman_net    AMAN strategy, after fund fees (default)
    aman_gross  AMAN strategy, before fees
    ens_net     AMAN with the ensemble overlay, after fees
    ew          equal weight of the five ETFs
    6040        20% each SPY, EFA and VNQ, 40% IEF
    custom      your own weights for SPY EFA IEF VNQ GLD (--weights, must total 100)

The simulator assumes the future resembles those 20 years and ignores taxes. AMAN is a
course project, not a real fund. This is a teaching tool, not financial advice.

The same simulator runs in the browser at
https://jamesquantai.com/projects/financial-backtesting.html
"""
import argparse

import numpy as np
import pandas as pd

ETFS = ["SPY", "EFA", "IEF", "VNQ", "GLD"]
PRESETS = {"ew": [20, 20, 20, 20, 20], "6040": [20, 20, 40, 20, 0]}
FIRST_MONTH, LAST_MONTH = "2005-12", "2025-10"   # whole months in which AMAN was trading


def monthly_returns() -> pd.DataFrame:
    """Monthly returns for the five ETFs and the strategy curves, one column each."""
    daily = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
    etf = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1

    net = pd.read_csv("all_value_curves_net.csv", index_col=0, parse_dates=True)
    nav = pd.read_csv("aman_values_nav.csv", index_col=0, parse_dates=True)
    curves = pd.DataFrame({
        "aman_net": net["AMAN_net"],
        "ens_net": net["AMAN_ENS_net"],
        "aman_gross": nav["AMAN_gross_nav"],
    })
    strat = curves.groupby(curves.index.to_period("M")).last().pct_change()
    return pd.concat([etf, strat], axis=1).loc[FIRST_MONTH:LAST_MONTH]


def portfolio_returns(months: pd.DataFrame, portfolio: str, weights=None) -> np.ndarray:
    if portfolio in ("aman_net", "aman_gross", "ens_net"):
        return months[portfolio].to_numpy()
    w = np.array(PRESETS.get(portfolio, weights), dtype=float)
    if abs(w.sum() - 100) > 0.05:
        raise ValueError(f"Weights add up to {w.sum():g}%. They must total 100%.")
    return months[ETFS].to_numpy() @ (w / 100)


def simulate(rets, start=500_000, years=30, cash=-20_000, inflation=0.025,
             inflation_adjust=True, real=True, sims=5000, seed=42):
    """Return (balances, failed).

    balances has shape (sims, years + 1): the balance at the end of each year.
    failed counts simulations where withdrawals ran the balance down to zero.
    cash is the yearly cash flow: negative to withdraw, positive to add.
    """
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(rets), size=(sims, years, 12))
    growth = (1 + rets[picks]).prod(axis=2)          # each simulated year's growth factor

    balances = np.empty((sims, years + 1))
    balances[:, 0] = start
    bal = np.full(sims, float(start))
    out = np.zeros(sims, dtype=bool)                 # ran out of money
    flow = float(cash)
    for y in range(years):
        bal = np.where(out, 0.0, bal * growth[:, y] + flow)
        if cash < 0:
            out |= bal <= 0
        bal = np.maximum(bal, 0.0)
        if inflation_adjust:
            flow *= 1 + inflation
        balances[:, y + 1] = bal / (1 + inflation) ** (y + 1) if real else bal
    return balances, int(out.sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--portfolio", default="aman_net", choices=["aman_net", "aman_gross", "ens_net", "ew", "6040", "custom"])
    p.add_argument("--weights", type=float, nargs=5, metavar=tuple(ETFS), help="weights in %% for a custom mix")
    p.add_argument("--start", type=float, default=500_000, help="starting balance")
    p.add_argument("--years", type=int, default=30)
    p.add_argument("--cash", type=float, default=-20_000, help="yearly cash flow: negative withdraws, positive adds")
    p.add_argument("--inflation", type=float, default=2.5, help="inflation in %% a year")
    p.add_argument("--no-inflation-adjust", action="store_true", help="keep the yearly cash flow fixed")
    p.add_argument("--nominal", action="store_true", help="report future dollars, not today's dollars")
    p.add_argument("--sims", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.portfolio == "custom" and a.weights is None:
        p.error("--portfolio custom needs --weights")

    months = monthly_returns()
    try:
        rets = portfolio_returns(months, a.portfolio, a.weights)
    except ValueError as err:
        p.error(str(err))
    hist_return = (1 + rets).prod() ** (12 / len(rets)) - 1
    hist_vol = rets.std(ddof=1) * np.sqrt(12)
    print(f"Portfolio: {a.portfolio}. In the data ({FIRST_MONTH} to {LAST_MONTH}, {len(rets)} months) "
          f"it grew {hist_return:.1%} a year with {hist_vol:.1%} volatility.")

    balances, failed = simulate(rets, a.start, a.years, a.cash, a.inflation / 100,
                                not a.no_inflation_adjust, not a.nominal, a.sims, a.seed)
    final = balances[:, -1]
    if a.cash < 0:
        print(f"{1 - failed / a.sims:.1%} of {a.sims:,} simulations still had money after {a.years} years.")
    else:
        print(f"{(final < a.start).mean():.1%} of {a.sims:,} simulations ended below the starting balance.")

    years = sorted(set(list(range(5, a.years, 5)) + [a.years]))
    table = pd.DataFrame(
        np.percentile(balances[:, years], [10, 25, 50, 75, 90], axis=0).T,
        index=[f"Year {y}" for y in years],
        columns=["10th", "25th", "Median", "75th", "90th"],
    )
    terms = "future dollars" if a.nominal else f"today's dollars at {a.inflation:g}% inflation"
    print(f"\nBalance by year, in {terms}:")
    print(table.to_string(float_format=lambda v: f"${v:,.0f}"))


if __name__ == "__main__":
    main()
