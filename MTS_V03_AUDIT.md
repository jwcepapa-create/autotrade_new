# Original v0.3 archive — independent audit

Source: user-supplied `Autotrade_new_final_model_v0_3.zip` (2026-10-27). Do not publish the original human-labeled CSV without explicit permission; repository is public.

## Strategy B — v0.3 daily breakout
- This branch contains an isolated faithful port in `mts_v03/breakout.py`.
- Entry: previous 60-session high breakout, volume above trailing 20-session average, close above MA20, MACD above signal.
- All signals formed at close; orders executed next session open.
- Partial exits: 30% of **remaining** shares under MA5, 50% under MA20, full exit under prior 60-session low. Repeated reductions can happen on consecutive days; this is original behavior, not an independently endorsed risk policy.
- Original v0.3 `strategy/backtest.py` reports total return and MDD for one ticker, not a multi-asset portfolio. Fees and slippage 0.1% + 0.05% per transaction by default.
- Tests cover prior-day extrema, append-only future invariance, execution timing and finite metrics.
- Not yet validated on Tiingo real price history; no performance claim.

## Strategy C — legacy v0.2 structural support/resistance
- Original `legacy/mts_trading_engine_v0_2/src/structure.py` uses centered pivots with right-side candles. `extract_observations` also scores subsequent reaction candles. Recomputing historical signals using the completed history can leak future data.
- Original `src/engine.py` produces a **current snapshot** with action score; it is not a historical trading execution engine.
- Correct conversion needs point-in-time timeframe candles, pivot confirmation delay, reaction confirmation delay, training-only zone calibration, and documented entry/exit execution rules.
- The archive includes 175 historical human S/R labels; labels are **not** performance evidence. Keep the originals private and audit provenance/split before using them.
- `data/labels/expert_actions.csv` in the archive contains only a header.
- No real-money execution, brokerage integration or claims of profitability.

## Acceptance gates
1. Run automated tests in GitHub Actions.
2. Compare port output against the original v0.3 implementation on identical synthetic and historical input.
3. Backtest with Tiingo-validated adjusted OHLCV and common trading costs/FX.
4. Independent reviewer checks same-bar execution, leakage, survivorship and selection bias.
5. Keep strategy C out of performance rankings until point-in-time reconstruction is complete.
