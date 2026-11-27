# Aurora Multi-Asset Navigator (AMAN)

Term project for MSDS 451 (Financial Engineering) at Northwestern: a rules-based multi-asset strategy with machine learning risk overlays, backtested as a hypothetical fund with fees.

Explore the results and run the simulator in the browser: https://jamesquantai.com/projects/financial-backtesting.html

## The strategy

- **Universe:** SPY (US stocks), EFA (international stocks), VNQ (real estate), IEF (Treasury bonds), GLD (gold).
- **Rules:** about once a month, hold only the assets with positive 12-month momentum (skipping the latest month), weight them by the inverse of their 60-day volatility, and cap each at 40%. If nothing qualifies, hold 60% IEF and 40% GLD.
- **Overlays:** an XGBoost classifier and an LSTM, trained on data before 2015 and tested from 2015, predict next-day direction and return for each ETF. When the combined signal for the stock ETFs is bearish, the overlay halves the stock and real estate weights and moves the money to bonds and gold. The ensemble acts only when both models agree.
- **Fees:** 2% management, 20% performance above the high-water mark, $100,000 startup cost on $1 million, and 5 basis points per 100% turnover. The benchmarks pay the same fees.

## Results, November 2004 to November 2025

After fees:

| Portfolio | Return a year | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| AMAN | 4.4% | 10.1% | 0.42 | -25.2% |
| AMAN + LSTM overlay | 4.1% | 9.6% | 0.41 | -23.5% |
| AMAN + XGBoost overlay | 4.4% | 9.9% | 0.43 | -25.2% |
| AMAN + ensemble overlay | 4.2% | 9.9% | 0.41 | -21.8% |
| Equal weight | 5.8% | 13.0% | 0.44 | -38.0% |
| 60/40 | 4.5% | 12.0% | 0.37 | -37.5% |

Before fees, AMAN returned 7.7% a year, equal weight 9.3% and 60/40 7.6%.

What the numbers say:

- The rules controlled risk: lower volatility and a much smaller worst fall than either benchmark.
- AMAN did not beat the benchmarks on return, and the gap widened after 2015.
- Fees took almost half of AMAN's return.
- The overlays added little. XGBoost's next-day accuracy on the test period was between 49.5% and 52.2% across the five ETFs.

Figures come from `master_summary.csv` and `xgb_diagnostics_summary.csv`.

## Files

| File | What it is |
|---|---|
| `MSDS451_Final_term_project.ipynb` | The full project: data, strategy, models, overlays, backtests, exports |
| `MSDS 451 - Term_Project_Final_JKWOK.pdf` | Final report |
| `master_summary.csv`, `summary*.csv`, `management_kpis_net.csv` | Performance tables |
| `all_value_curves_net.csv`, `aman_values_nav.csv`, `bench_values_nav.csv` | Daily value curves |
| `aman_weights.csv` | Portfolio weights at each rebalance |
| `overlay_signal_*.csv` | Overlay signals at each rebalance |
| `*_xgb_*`, `*_lstm_*` | Per-ETF model outputs, diagnostics and saved models |
| `prices.csv`, `returns.csv` | Daily ETF prices and returns |
| `period_stats.py` | Recomputes the performance table for four periods (full, since first trade, since 2015, 2007 to 2009) and writes `period_summary.csv` |
| `monte_carlo_simulator.py` | Monte Carlo simulator that resamples the project's real monthly returns |

## Running the scripts

Both scripts need `numpy` and `pandas` and read the CSV files in this folder.

```bash
python period_stats.py
python monte_carlo_simulator.py                         # AMAN after fees, $500,000, withdraw $20,000 a year
python monte_carlo_simulator.py --portfolio ew --years 20
python monte_carlo_simulator.py --portfolio custom --weights 40 10 30 10 10 --cash 0
python monte_carlo_simulator.py --help
```

## Note

AMAN is a course project, not a real fund. The backtest and the simulator describe the past and are not financial advice.
