# Strategy lab — independent AI research roles

This is a software workflow of simulated roles, **not real employees, autonomous background agents or employment**. All experiments are research-only; **no brokerage access or orders without explicit final approval**.

## Role 1: Global strategy research analyst
Research publicly reproducible strategies; log primary papers, universe, frequency, cost assumptions, limitations, and independent replications. Do not copy reported returns as our results.

Initial primary sources:
- Moskowitz, Ooi, Pedersen (2012), *Time Series Momentum*, JFE, DOI 10.1016/j.jfineco.2011.11.003
- Moreira and Muir (2017), *Volatility-Managed Portfolios*, Journal of Finance, DOI 10.1111/jofi.12513

## Role 2: Strategy research lead
Prioritize falsifiable hypotheses, design controlled parameter sweeps, freeze baselines, require out-of-sample and stress tests. Independent QA/adversarial reviewer has veto on methodological flaws.

## B improvement knobs (one-factor-at-a-time)
1. MA5 exit fraction 10/20/30%, MA20 exit fraction 30/50/70%.
2. Partial exit cooldown 0/3/5 sessions; compare repeated, cooldown and edge-triggered.
3. Exit confirmation 1/2 closes, ATR tolerance 0/0.5.
4. Breakout lookback 40/60/100 days; volume 1.0/1.3/1.5 times trailing average.
5. Volatility-targeted position sizing and QQQ 200-day market regime filter.

**Priority 0:** resolve B1 original-vs-comparison baseline mismatch across runs (possibly dataset coverage or adjusted price changes); record CSV checksums, dates, code SHA and compare same-run outputs. Do not rank strategies until baseline parity is proven.

## C structural zones
- `mts_v03/causal_zones.py`: daily-only causal zone prototype, point-in-time confirmed pivot and reaction observations, with a retest signal; no complete legacy M/W/D/120m parity.
- Remaining: weekly/monthly completed-candle alignment, intraday data provenance, zone state/scoring reconstruction, 175-label calibration with train/test separation, order execution backtest.
- Never call C profitable or validated based on causal pivot unit tests alone.

## New candidates
D: 6-12 month time-series momentum, long/cash, monthly rebalance, volatility sizing.
E: Donchian 55-day breakout with ATR stop, volatility-scaled position and optional structural retest confirmation (hybrid B+C+D).

## Evaluation gates
- 2016-2021 training, 2022-2023 validation, 2024-present untouched holdout (illustrative frozen split; adjust only with documented data availability).
- Transaction costs, delistings/survivorship, next-open fills, cash drag, walk-forward, common capital/FX, turnover, Sharpe, CAGR, MDD, exposure and benchmark.
- Fail if leakage, poor baseline parity, no adequate trade count, or weak out-of-sample net returns. No guaranteed profits.
