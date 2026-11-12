"""V4 hybrid research: V2 drawdown sleeves + V3 trend defence.

No future data: indicators use closes through signal day; execution is next close.
Uses genuine Tiingo ETF series supplied by leverage_drawdown.py. No live trading.
"""
import numpy as np
import pandas as pd
from leverage_drawdown import ALL, BASE, DOUBLE, TRIPLE, weights


def hybrid_targets(prices, variant="recovery", cap3=.10, cap2=.25):
    if variant not in ("recovery", "strict", "balanced"):
        raise ValueError("Unknown V4 variant")
    q = prices["QQQ"]
    ma200 = q.rolling(200, min_periods=200).mean()
    ma50 = q.rolling(50, min_periods=50).mean()
    dd = q / q.cummax() - 1
    weekly = q.resample("W-FRI").last()
    wma = weekly.rolling(40, min_periods=40).mean()
    weekly_up = (weekly >= wma).reindex(prices.index, method="ffill").fillna(False)
    out = []
    for i in range(len(prices)):
        target = {t: 0. for t in BASE + DOUBLE + TRIPLE}
        if pd.isna(ma200.iloc[i]) or pd.isna(ma50.iloc[i]):
            out.append(target)
            continue
        above200 = q.iloc[i] >= ma200.iloc[i]
        above50 = q.iloc[i] >= ma50.iloc[i]
        weekly_good = bool(weekly_up.iloc[i])
        # Recovery entry: permit a partial position above 50d MA, even while
        # still below 200d MA, avoiding the delayed V3 re-entry.
        if variant == "strict":
            mode = "risk_on" if above200 and weekly_good else "defensive"
        elif variant == "balanced":
            mode = "risk_on" if above200 and weekly_good else (
                "recovery" if above50 and above200 else "defensive")
        else:
            mode = "risk_on" if above200 and weekly_good else (
                "recovery" if above50 else "defensive")
        if mode == "defensive":
            for t in BASE:
                target[t] = .50 / len(BASE)
        elif mode == "recovery":
            for t in BASE:
                target[t] = .80 / len(BASE)
            # No leverage until both long-term filters recover.
        else:
            target = weights(float(dd.iloc[i]))
            s2 = sum(target[t] for t in DOUBLE)
            s3 = sum(target[t] for t in TRIPLE)
            for t in DOUBLE:
                target[t] *= min(1., cap2 / s2) if s2 else 1.
            for t in TRIPLE:
                target[t] *= min(1., cap3 / s3) if s3 else 1.
        out.append(target)
    return out


def simulate_v4(prices, variant="recovery", cap3=.10, cap2=.25,
                fee_bps=30., initial=10_000_000):
    if not set(ALL).issubset(prices.columns):
        raise ValueError("Missing real ETF prices")
    p = prices.loc[:, ALL].dropna()
    if len(p) < 252 or p.index.has_duplicates or not p.index.is_monotonic_increasing or (p <= 0).any().any():
        raise ValueError("Invalid aligned price history")
    if not (0 <= cap3 <= .30 and 0 <= cap2 <= .30 and 0 <= fee_bps <= 100):
        raise ValueError("Invalid limits")
    tickers = BASE + DOUBLE + TRIPLE
    targets = hybrid_targets(p, variant, cap3, cap2)
    holdings = {t: 0. for t in tickers}
    last_target = None
    nav = float(initial)
    curve = [nav]
    turnover = 0.
    fees = 0.
    trades = 0
    exposure = []
    for i in range(1, len(p)):
        # Existing holdings earn i-1 -> i returns. Signal i-1 is traded
        # at close i; newly bought exposure earns from i -> i+1.
        returns = p.iloc[i] / p.iloc[i-1] - 1
        growth = 1 + sum(holdings[t] * float(returns[t]) for t in tickers)
        if growth <= 0:
            raise ValueError("NAV depleted")
        nav *= growth
        holdings = {t: holdings[t] * (1 + float(returns[t])) / growth for t in tickers}
        target = targets[i-1]
        if target != last_target:
            volume = sum(abs(target[t] - holdings[t]) for t in tickers)
            cost = nav * volume * fee_bps / 10000
            if cost >= nav:
                raise ValueError("NAV depleted by fees")
            nav -= cost
            fees += cost
            turnover += volume
            trades += volume > 1e-10
            holdings = target.copy()
        last_target = target
        exposure.append(sum(holdings.values()))
        curve.append(nav)
    curve = pd.Series(curve, index=p.index, name="nav_krw")
    daily = curve.pct_change().dropna()
    years = (p.index[-1] - p.index[0]).days / 365.25
    return {
        "start": str(p.index[0].date()), "end": str(p.index[-1].date()),
        "total_return": float(nav / initial - 1),
        "cagr": float((nav / initial) ** (1 / years) - 1),
        "mdd": float((curve / curve.cummax() - 1).min()),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else None,
        "turnover": float(turnover), "rebalances": int(trades),
        "transaction_cost_krw": float(fees),
        "ending_krw": float(nav),
        "average_invested_fraction": float(np.mean(exposure)),
        "parameters": {"variant": variant, "cap3": cap3, "cap2": cap2,
                       "execution": "next close proxy", "fee_bps": fee_bps},
    }, curve
