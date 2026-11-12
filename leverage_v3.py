"""V3 trend-filtered leverage strategy research.

Signals use only information available at the prior daily close.
Trades use next-close execution proxy, and subsequent close-to-close returns.
This is intentionally more conservative in timing than V2 but is NOT next-open fill.
"""
import numpy as np
import pandas as pd
from leverage_drawdown import ALL, BASE, DOUBLE, TRIPLE, weights


def trend_targets(prices, cap_3x=.10, cap_2x=.20, weekly_filter=True):
    q = prices["QQQ"]
    dd = q / q.cummax() - 1
    ma200 = q.rolling(200, min_periods=200).mean()
    # W-FRI: completed weekly observations only; forward-fill from last Friday.
    weekly = q.resample("W-FRI").last()
    weekly_ma = weekly.rolling(40, min_periods=40).mean()
    weekly_ok = (weekly >= weekly_ma).reindex(prices.index, method="ffill").fillna(False)
    result = []
    for i in range(len(prices)):
        if pd.isna(ma200.iloc[i]):
            result.append({t: 0. for t in BASE + DOUBLE + TRIPLE})
            continue
        target = weights(float(dd.iloc[i]))
        if q.iloc[i] < ma200.iloc[i] or (weekly_filter and not bool(weekly_ok.iloc[i])):
            # No leverage in a downtrend; hold 50% 1x and 50% cash.
            target = {t: (.5 / len(BASE) if t in BASE else 0.) for t in BASE + DOUBLE + TRIPLE}
        else:
            old3 = sum(target[t] for t in TRIPLE)
            old2 = sum(target[t] for t in DOUBLE)
            scale3 = min(1., cap_3x / old3) if old3 else 1.
            scale2 = min(1., cap_2x / old2) if old2 else 1.
            for t in TRIPLE:
                target[t] *= scale3
            for t in DOUBLE:
                target[t] *= scale2
            # Unused leverage allocation remains cash, not silently reallocated.
        result.append(target)
    return result


def simulate_v3(prices, cap_3x=.10, cap_2x=.20, fee_bps=30.,
                weekly_filter=True, initial=10_000_000):
    if not set(ALL).issubset(prices.columns):
        raise ValueError("Missing real ETF price series")
    p = prices.loc[:, ALL].dropna()
    if len(p) < 252 or (p <= 0).any().any() or p.index.has_duplicates or not p.index.is_monotonic_increasing:
        raise ValueError("Invalid history")
    if not (0 <= cap_3x <= .30 and 0 <= cap_2x <= .30 and 0 <= fee_bps <= 100):
        raise ValueError("Invalid risk parameters")
    tickers = BASE + DOUBLE + TRIPLE
    targets = trend_targets(p, cap_3x, cap_2x, weekly_filter)
    nav = float(initial)
    current = {t: 0. for t in tickers}
    previous_target = None
    turnover = 0.
    trades = 0
    cost_total = 0.
    curve = [nav]
    # Decision after close i-1; execute at close i; gain exposure for i -> i+1.
    # Thus first date's return is cash, no same-bar signal execution.
    for i in range(1, len(p)):
        r = p.iloc[i] / p.iloc[i-1] - 1
        growth = 1 + sum(current[t] * float(r[t]) for t in tickers)
        if growth <= 0:
            raise ValueError("NAV depleted")
        nav *= growth
        current = {t: current[t] * (1 + float(r[t])) / growth for t in tickers}
        target = targets[i-1]
        if target != previous_target:
            volume = sum(abs(target[t] - current[t]) for t in tickers)
            cost = nav * volume * fee_bps / 10000
            nav -= cost
            turnover += volume
            cost_total += cost
            if volume > 1e-10:
                trades += 1
            current = target.copy()
        previous_target = target
        curve.append(nav)
    series = pd.Series(curve, index=p.index, name="nav_krw")
    returns = series.pct_change().dropna()
    years = (p.index[-1] - p.index[0]).days / 365.25
    return {
        "start": str(p.index[0].date()), "end": str(p.index[-1].date()),
        "total_return": float(nav / initial - 1),
        "cagr": float((nav / initial) ** (1 / years) - 1),
        "mdd": float((series / series.cummax() - 1).min()),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else None,
        "turnover": float(turnover), "rebalances": trades,
        "transaction_cost_krw": float(cost_total),
        "ending_krw": float(nav),
        "parameters": {"cap_3x": cap_3x, "cap_2x": cap_2x,
                       "weekly_filter": weekly_filter, "execution": "next close proxy"},
    }, series
