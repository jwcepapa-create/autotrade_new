"""Holdings-based leverage ETF research, correcting V1 target-weight return accounting."""
import numpy as np
import pandas as pd
from leverage_drawdown import ALL, BASE, DOUBLE, TRIPLE, weights


def simulate_v2(prices, managed=False, fee_bps=30., initial=10_000_000,
                rebalance="daily"):
    """Signals at prior close; rebalance at that close, then hold through next close.

    Daily: maintain target weights daily (pay for drift corrections).
    threshold: trade only on changes in QQQ drawdown regime, otherwise drift.
    Uses close-to-close execution proxy, not realizable next-open fills.
    """
    if rebalance not in ("daily", "threshold"):
        raise ValueError("Invalid rebalance policy")
    if not 0 <= fee_bps <= 100:
        raise ValueError("Invalid cost")
    if not set(ALL).issubset(prices.columns):
        raise ValueError("Missing ETF series")
    prices = prices.loc[:, ALL].dropna()
    if len(prices) < 252 or (prices <= 0).any().any() or prices.index.has_duplicates:
        raise ValueError("Invalid genuine history")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be sorted")
    tickers = BASE + DOUBLE + TRIPLE
    qqq = prices.QQQ
    dd = qqq / qqq.cummax() - 1
    nav = float(initial)
    current = {t: 0. for t in tickers}
    prev_target = None
    turnover = 0.
    rebalances = 0
    total_cost_krw = 0.
    series = [nav]
    events = []
    for i in range(1, len(prices)):
        target = weights(float(dd.iloc[i - 1]), managed)
        do_trade = rebalance == "daily" or target != prev_target
        if do_trade:
            traded = sum(abs(target[t] - current[t]) for t in tickers)
            cost = nav * traded * fee_bps / 10000
            if cost >= nav:
                raise ValueError("Transaction costs depleted capital")
            nav -= cost
            total_cost_krw += cost
            turnover += traded
            if traded > 1e-10:
                rebalances += 1
                events.append({"date": str(prices.index[i-1].date()),
                               "qqq_drawdown": float(dd.iloc[i-1]),
                               "turnover": traded, "cost_krw": cost})
            current = target.copy()
        prev_target = target
        daily = prices.iloc[i] / prices.iloc[i-1] - 1
        growth = 1 + sum(current[t] * float(daily[t]) for t in tickers)
        if growth <= 0:
            raise ValueError("Portfolio NAV depleted")
        new_nav = nav * growth
        # Reconstruct post-return weights, including cash.
        current = {t: current[t] * (1 + float(daily[t])) / growth for t in tickers}
        nav = new_nav
        series.append(nav)
    curve = pd.Series(series, index=prices.index, name="nav_krw")
    returns = curve.pct_change().dropna()
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    result = {
        "start": str(prices.index[0].date()),
        "end": str(prices.index[-1].date()),
        "total_return": float(nav / initial - 1),
        "cagr": float((nav / initial) ** (1 / years) - 1),
        "mdd": float((curve / curve.cummax() - 1).min()),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252))
                  if returns.std() > 0 else None,
        "turnover": float(turnover), "rebalances": rebalances,
        "transaction_cost_krw": float(total_cost_krw),
        "ending_krw": float(nav),
        "rebalance_policy": rebalance,
    }
    return result, curve, pd.DataFrame(events)


def regime_metrics(curve):
    """Ex-post calendar regimes, descriptive only, not independently reinitialized."""
    result = {}
    windows = {
        "covid_2020": ("2020-02-19", "2020-06-30"),
        "bear_2022": ("2022-01-03", "2022-12-30"),
        "recent_2023_2026": ("2023-01-03", "2026-11-11"),
    }
    for name, (start, end) in windows.items():
        sub = curve.loc[start:end]
        if len(sub) < 2:
            continue
        result[name] = {
            "start": str(sub.index[0].date()), "end": str(sub.index[-1].date()),
            "return": float(sub.iloc[-1] / sub.iloc[0] - 1),
            "mdd_within_window": float((sub / sub.cummax() - 1).min()),
        }
    return result
