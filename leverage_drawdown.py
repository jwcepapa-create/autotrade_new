"""Research-only daily-close backtest of a drawdown-triggered leveraged ETF allocation.

Requires genuine provider-adjusted daily closes. Signals at close t are applied to
close-to-close returns t+1; no synthetic 2x/3x price series are generated.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tiingo_data import Client, DataError, adjusted

BASE = ("VOO", "MGK", "SOXX", "SMH")
DOUBLE = ("QLD", "USD")
TRIPLE = ("TQQQ", "SOXL")
ALL = ("QQQ",) + BASE + DOUBLE + TRIPLE


def weights(drawdown, managed=False):
    # Equal-weighted within each sleeve. Gross ETF exposure <= 100% of NAV.
    if drawdown <= -0.30:
        portions = (.50, .20, .30)
    elif drawdown <= -0.20:
        portions = (.65, .25, .10)
    elif drawdown <= -0.10:
        portions = (.80, .20, 0.)
    else:
        portions = (1., 0., 0.)
    if managed:
        # Conservative alternative: cap 3x sleeve at 10% and hold remainder in cash.
        portions = (portions[0], portions[1], min(portions[2], .10))
    result = {t: portions[0] / len(BASE) for t in BASE}
    result.update({t: portions[1] / len(DOUBLE) for t in DOUBLE})
    result.update({t: portions[2] / len(TRIPLE) for t in TRIPLE})
    return result


def simulate(prices, managed=False, fee_bps=30.0, initial=10_000_000):
    if not set(ALL).issubset(prices.columns):
        raise ValueError("Missing required real ETF price series")
    prices = prices.loc[:, ALL].dropna()
    if len(prices) < 252 or (prices <= 0).any().any():
        raise ValueError("Insufficient aligned genuine ETF history")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be sorted and unique")
    qqq = prices.QQQ
    drawdown = qqq / qqq.cummax() - 1
    nav = float(initial)
    curve = [nav]
    holdings = {t: 0. for t in BASE + DOUBLE + TRIPLE}
    turnover = 0.
    rebalances = 0
    # Day i target uses only yesterday's QQQ close. Today's close returns
    # are applied to yesterday's close holdings; transaction cost charged
    # on the reallocation at yesterday's close.
    for i in range(1, len(prices)):
        target = weights(float(drawdown.iloc[i - 1]), managed)
        delta = sum(abs(target[t] - holdings[t]) for t in target)
        if delta > 1e-12:
            nav *= 1 - delta * fee_bps / 10000
            turnover += delta
            rebalances += 1
        holdings = target
        returns = prices.iloc[i] / prices.iloc[i - 1] - 1
        nav *= 1 + sum(holdings[t] * float(returns[t]) for t in holdings)
        if nav <= 0:
            raise ValueError("Portfolio NAV depleted")
        curve.append(nav)
    series = pd.Series(curve, index=prices.index, name="nav_krw")
    daily = series.pct_change().dropna()
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    return {
        "start": str(prices.index[0].date()), "end": str(prices.index[-1].date()),
        "total_return": float(nav / initial - 1),
        "cagr": float((nav / initial) ** (1 / years) - 1),
        "mdd": float((series / series.cummax() - 1).min()),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else None,
        "turnover": float(turnover), "rebalances": rebalances,
        "ending_krw": float(nav),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", required=True)
    p.add_argument("--output", default="leverage_results")
    p.add_argument("--fee-bps", type=float, default=30.)
    args = p.parse_args()
    if args.fee_bps < 0 or args.fee_bps > 100:
        p.error("Fee+slippage must be 0..100 bps per unit of turnover")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    client = Client()
    frames = {}
    failures = {}
    for ticker in ALL:
        try:
            frames[ticker] = adjusted(client.daily(ticker, args.start, args.end))["Close"]
        except DataError as exc:
            failures[ticker] = str(exc)
    if failures:
        (output / "status.json").write_text(json.dumps(
            {"status": "unavailable", "failures": failures}, indent=2))
        raise SystemExit("Real price coverage unavailable for one or more ETFs; no results fabricated")
    prices = pd.concat(frames, axis=1).sort_index()
    # Intersection of genuine trading sessions; no forward filling or
    # retroactive leveraged-ETF reconstruction before fund inception.
    common = prices.dropna()
    if len(common) < 252:
        raise SystemExit("Insufficient common ETF history; no results fabricated")
    # Compare the original accounting to holdings-based execution accounting.
    from leverage_v2 import simulate_v2, regime_metrics
    v2 = {}
    v2_curves = {}
    v2_events = {}
    for name, managed, policy in [
        ('original_daily', False, 'daily'),
        ('managed_daily', True, 'daily'),
        ('original_threshold', False, 'threshold'),
        ('managed_threshold', True, 'threshold'),
    ]:
        v2[name], v2_curves[name], v2_events[name] = simulate_v2(
            common, managed=managed, fee_bps=args.fee_bps, rebalance=policy)
        v2[name]['regimes'] = regime_metrics(v2_curves[name])
        v2_curves[name].to_csv(output / f'{name}_equity.csv', index_label='Date')
        v2_events[name].to_csv(output / f'{name}_rebalance_events.csv', index=False)
    from leverage_v3 import simulate_v3
    v3 = {}
    for name, cap3, weekly in [
        ("trend_daily_weekly_cap10", .10, True),
        ("trend_daily_only_cap10", .10, False),
        ("trend_daily_weekly_cap0", 0., True),
    ]:
        v3[name], curve = simulate_v3(
            common, cap_3x=cap3, weekly_filter=weekly, fee_bps=args.fee_bps)
        v3[name]["regimes"] = regime_metrics(curve)
        curve.to_csv(output / f"{name}_equity.csv", index_label="Date")
    from leverage_v4 import simulate_v4
    v4 = {}
    for name, variant, cap3 in [
        ("hybrid_recovery_cap10", "recovery", .10),
        ("hybrid_strict_cap10", "strict", .10),
        ("hybrid_balanced_cap10", "balanced", .10),
        ("hybrid_recovery_no3x", "recovery", 0.),
    ]:
        v4[name], curve = simulate_v4(
            common, variant=variant, cap3=cap3, fee_bps=args.fee_bps)
        v4[name]["regimes"] = regime_metrics(curve)
        curve.to_csv(output / f"{name}_equity.csv", index_label="Date")
    result = {
        "status": "completed",
        "source": "Tiingo adjusted daily ETF closes",
        "tickers": list(ALL),
        "common_sessions": len(common),
        "start_requested": args.start,
        "end_requested": args.end,
        "assumptions": {
            "signals": "QQQ close drawdown vs running all-time high, applied next session",
            "trading": "Close-to-close return approximation, no live orders",
            "cost_bps_per_turnover": args.fee_bps,
            "limitations": [
                "Leverage ETFs reset daily and can suffer path-dependent decay.",
                "ETF adjusted-close total-return proxy, not exact fill simulation.",
                "No pre-inception synthetic leveraged returns; intersection shortens history.",
                "Historical constituents and survivorship/selection biases remain.",
                "No tax, FX fluctuations, borrowing, or intraday slippage model.",
            ],
        },
        "v4_strategies": v4,
        "v3_strategies": v3,
        "v2_strategies": v2,
        "strategies": {
            "drawdown_original": simulate(common, fee_bps=args.fee_bps),
            "drawdown_risk_managed": simulate(common, managed=True, fee_bps=args.fee_bps),
            "base_1x_only": simulate_base(common, fee_bps=args.fee_bps),
            "qqq_buy_hold": simulate_qqq(common, fee_bps=args.fee_bps),
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


def simulate_base(prices, fee_bps=30.):
    # Constant 25% allocation to four unleveraged ETFs.
    proxy = prices.copy()
    qqq = proxy.QQQ
    proxy.loc[:, "QQQ"] = qqq.iloc[0]  # no QQQ drawdown threshold crossings
    return simulate(proxy, fee_bps=fee_bps)


def simulate_qqq(prices, fee_bps=30.):
    q = prices.QQQ
    curve = (q / q.iloc[0]) * 10_000_000 * (1 - fee_bps / 10000)
    daily = curve.pct_change().dropna()
    years = (q.index[-1] - q.index[0]).days / 365.25
    return {
        "start": str(q.index[0].date()), "end": str(q.index[-1].date()),
        "total_return": float(curve.iloc[-1] / 10_000_000 - 1),
        "cagr": float((curve.iloc[-1] / 10_000_000) ** (1 / years) - 1),
        "mdd": float((curve / curve.cummax() - 1).min()),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else None,
        "turnover": 1., "rebalances": 1,
        "ending_krw": float(curve.iloc[-1]),
    }


if __name__ == "__main__":
    main()
