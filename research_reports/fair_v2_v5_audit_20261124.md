# Fair V2/V4/V5 next-open research — 2026-11-24

## Decision
Research-only. No real orders, API credentials, portfolio management or live trading enabled.
**Use V4 recovery without 3x as the provisional V6 research baseline, not V5.** A higher-return V2 exists but its drawdown is materially worse. RSI/MACD recovery confirmation did not improve historical performance in the isolated ablation.

## Source and evaluation
- Real Tiingo adjusted Open and Close for QQQ, VOO, MGK, SOXX, SMH, QLD, USD, TQQQ, SOXL.
- Full price history starts 2016-01-04; mandatory 252-session warm-up; start all strategies at 10,000,000 KRW cash on 2017-01-03; evaluation ends at close 2026-11-23.
- Signal through close t-1, trade at adjusted OPEN t (without collecting that gap on newly entered holdings); existing holdings get the actual close-to-open gap. Natural weight drift between rebalances.
- Cost sensitivities: 15, 30, 60 bps *per traded notional*; optional assumed 0% or 3% fixed annualized cash return.
- Statistics below reflect 30bps per traded notional and 0% cash return.
- There is **no** untouched true out-of-sample performance; strategies and rules were designed after seeing historical market data.

## Same-period, same-open execution
| Strategy | CAGR | MDD | Sharpe (rf=0) | Final NAV, KRW | Rebalances |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 threshold rebalance | 25.55% | -47.20% | 0.862 | 94.81m | 138 |
| V2 daily target rebalance | 24.96% | -47.98% | 0.852 | 90.49m | 2486 |
| V4 recovery, no 3x | 22.82% | -29.47% | 1.009 | 76.30m | 99 |
| V4 strict, no 3x | 22.27% | -29.47% | 1.004 | 73.00m | 84 |
| V5 confirmed RSI/MACD | 21.80% | -29.47% | 0.983 | 70.24m | 122 |
| V5 volatility cap | 21.06% | -28.84% | 0.968 | 66.18m | 124 |
| V5 volatility cap **without RSI/MACD** | 22.12% | -28.84% | 0.999 | 72.12m | 102 |
| Four 1x ETFs buy and hold | 24.08% | -40.06% | 0.889 | 84.43m | 1 |
| Four 1x ETFs monthly rebalanced | 22.87% | -38.00% | 0.909 | 76.58m | 119 |
| QQQ buy and hold | 19.45% | -35.12% | 0.888 | 57.95m | 1 |

## Interpretation
1. V2 threshold adds only 1.46 percentage points CAGR versus 1x basket buy-and-hold while making worst drawdown ~7.14 pp deeper. It also rebalances frequently: 138 trades; lifetime total modeled trading fees ~5.22m KRW. Do not deploy this aggressively to recover investment losses.
2. V4 recovery/no3x improves Sharpe by ~0.12 vs the 1x basket and reduces MDD by ~10.6 pp but gives up ~1.26 pp CAGR. This was the best Sharpe among the compared strategic variants.
3. V5 volatility lowers MDD only 0.63 pp relative to V4 but reduces CAGR 1.76 pp and Sharpe from 1.009 to 0.968.
4. V5 volatility without oscillator confirmation improves CAGR from 21.06% to 22.12%, Sharpe 0.968 to 0.999, with identical rounded MDD. The RSI/MACD condition, as currently implemented, is not adding value in this sample. This does not prove oscillators can never be useful.
5. In the descriptive 2022 period the returns were: V2 threshold -38.15%; V4 -25.18%; V5 volatility -24.14%; 1x buy-and-hold -32.72%; QQQ -33.22%.
6. On 2020-02-19 to 2020-06-30: V2 threshold +3.35%; V4 -6.14%; V5 volatility -7.33%. These subperiods do not reinitialize strategy equity and should not be described as independent OOS tests.

## Cost and cash sensitivity
At 0% cash assumption, 15 bps / 60 bps:
- V2 threshold CAGR: 26.59% / 23.49%.
- V4 recovery CAGR: 23.54% / 21.40%.
- V5 volatility CAGR: 21.86% / 19.49%.
At 30 bps and a hypothetical fixed 3% cash yield, V4 CAGR 23.11%, V5 volatility 21.40%.

## Validation and known limits
- Separate open/close engine fixes the original inconsistent V2 previous-close and V5 next-close simulation definitions, and corrects portfolio-weight drift and benchmark rebalancing accounting.
- Synthetic unit tests check no capture of prior-close/open gap on first entry, capture of gaps by existing holdings, allocation drift, cash interest, cost direction, causal signals, and complete latest-period date.
- Each real ticker is validated against full XNYS expected sessions; no silent forward-filling or missing-session deletion.
- Data are adjusted OHLC, approximating dividend reinvestment, not exact execution. No live quotes, tick spreads, real order fills, taxes, FX risk, historical cash rate, or provider-adjustment hindsight correction. Slippage proxy may not match an individual account.
- There is still selection/hindsight bias; a walk-forward frozen-rule test and forward paper trading are mandatory before real-money deployment.
- Neither drawdown target of -25% nor statistically robust incremental alpha has been demonstrated.

## Next action
Hold V4 no-3x and V5 no-RSI/MACD as competing candidates. Freeze their rules, verify recent independent data and risk of tail drawdowns, and research a market-trend/volume confirmation only as a clearly isolated experimental component.
