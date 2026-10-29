# Research execution report — 2026-10-27

**Runs:** [strategy lab](https://github.com/jwcepapa-create/autotrade_new/actions/runs/40385921297), [independent QA](https://github.com/jwcepapa-create/autotrade_new/actions/runs/40385921217). Both succeeded.

**Data:** 7 Tiingo EOD tickers, 2016-01-04 to 2026-10-23, adjusted OHLCV, next-session open execution, 15bps transaction cost per side. Separate ticker sleeves; no combined portfolio or FX conversion.

## Critical audit
- Faithful original B engine and B-knobs default match **on all 7 tickers**, including returns/MDD/order count.
- Prior `exit_experiments.py` **does not match original B** on any of the seven. Previous B1-vs-B2 comparisons using that engine must be **withdrawn** as faithful baseline comparisons; discrepancy needs forensic investigation. New knob experiment uses the faithful B engine.
- B cooldown3 + 20% MA5/30% MA20 improves full-sample returns in QQQ/SPY/NVDA/AMD/AMZN/META but hurts TSLA; **in-sample selection, not proof**. In some cases MDD worsens. Need chronological OOS B evaluation before promotion.
- 2024-2026 period model returns below are *mark-to-market segments of continuously run strategies*, not fresh period-initialized independent accounts. No portfolio-level aggregate or Sharpe yet.

## 2024-10-23 holdout period (2024-01-01 onward): returns %
| Ticker | Buy & hold | D momentum | E ATR breakout | C daily prototype |
|---|---:|---:|---:|---:|
| QQQ | 77.48 | 36.55 | 20.80 | 38.38 |
| SPY | 66.05 | 41.19 | 12.18 | 13.08 |
| NVDA | 339.23 | 75.27 | 89.37 | 4.49 |
| TSLA | 49.84 | -3.85 | 14.47 | 25.69 |
| AMD | 272.92 | 37.82 | 109.74 | 94.72 |
| AMZN | 65.41 | 8.21 | -20.80 | -11.54 |
| META | 87.31 | 13.29 | 45.74 | 1.40 |

**Assessment:** No candidate consistently outperforms buy-and-hold on returns. Some reduce drawdowns; E has especially poor drawdown on TSLA and AMZN. D performs too many daily rebalances (hundreds to thousands of orders) and needs turnover/cost/rounding realism. C is a daily prototype, **not** full M/W/D/120min v0.2 reconstruction. C causal confirmation and append-only invariance tests pass but do not validate profitability.

## Remaining blockers (do not claim completed)
1. Forensic comparison of old `exit_experiments.py` against original B (share dust handling and state transitions), retire misleading B1/B2 reports.
2. Frozen train-only knob selection and walk-forward holdout for B; risk-adjusted aggregate portfolio with realistic capital, FX, cash, benchmark, order rounding.
3. Full C multi-timeframe completed-bar alignment, 120-minute historical entitlement and original labeled validation.
4. D monthly/weekly rebalance and volatility-sizing realism, and E cross-asset stress and structural C hybrid evaluation.
5. Independent code review, survivorship/universe and statistical multiple-testing controls.

**No live account connection, brokerage API or trading authorization.** PR remains draft.
