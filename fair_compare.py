"""Research-only, apples-to-apples V2/V4/V5 leverage study.

All strategies see the same historical bars and START with cash on the same
evaluation date (after a 252-session signal warm-up).
Signals are known only through close t-1 and executed at adjusted OPEN t.
Existing holdings experience close(t-1)->open(t) gap before rebalancing.
New holdings experience open(t)->close(t) price movement.
This model is NOT a live brokerage execution/volume/liquidity simulation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leverage_drawdown import ALL, BASE, DOUBLE, TRIPLE, weights
from leverage_v4 import hybrid_targets
from leverage_v5 import v5_targets
from tiingo_data import Client, DataError, adjusted


INITIAL = 10_000_000
WARMUP = 252
REGIMES = {
    "covid_2020": ("2020-02-19", "2020-06-30"),
    "bear_2022": ("2022-01-03", "2022-12-30"),
    "recent_2023_onward": ("2023-01-03", None),
}


def prepare_targets(close):
    """Only close-based, causal targets; all signal histories include warm-up."""
    qqq = close["QQQ"]
    dd = qqq / qqq.cummax() - 1
    v2 = [weights(float(d)) for d in dd]
    v4 = hybrid_targets(close, variant="recovery", cap3=0., cap2=.25)
    v4strict = hybrid_targets(close, variant="strict", cap3=0., cap2=.25)
    confirmed = v5_targets(close, "confirmed")
    volatile = v5_targets(close, "volatility")
    fast = v5_targets(close, "fast")
    ablation = v5_no_oscillator_targets(close)
    base = {t: .25 for t in BASE}
    return {
        "v2_threshold": (v2, "change"),
        "v2_daily": (v2, "daily"),
        "v4_recovery_no3x": (v4, "change"),
        "v4_strict_no3x": (v4strict, "change"),
        "v5_confirmed": (confirmed, "change"),
        "v5_volatility": (volatile, "change"),
        "v5_fast": (fast, "change"),
        "v5_no_rsi_macd": (ablation, "change"),
        "basket_1x_buyhold": ([base] * len(close), "once"),
        "basket_1x_monthly": ([base] * len(close), "monthly"),
        "qqq_buyhold": ([{"QQQ": 1.}] * len(close), "once"),
    }


def v5_no_oscillator_targets(close):
    """V5 volatility rule with only RSI/MACD recovery check removed.

    This controls for the value of the oscillator confirmation; unlike
    comparing V4 to V5, trend filters and volatility cap remain identical.
    """
    q = close["QQQ"]
    ma20 = q.rolling(20, min_periods=20).mean()
    ma50 = q.rolling(50, min_periods=50).mean()
    ma200 = q.rolling(200, min_periods=200).mean()
    completed_weeks = q.resample("W-FRI").last()
    weekly_up = (completed_weeks >= completed_weeks.rolling(40, min_periods=40).mean()).reindex(
        q.index, method="ffill").fillna(False)
    dd = q / q.cummax() - 1
    vol = q.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
    result = []
    for i in range(len(q)):
        target = {t: 0. for t in BASE + DOUBLE + TRIPLE}
        if i < 200 or pd.isna(ma200.iloc[i]):
            result.append(target)
            continue
        long_up = bool(q.iloc[i] >= ma200.iloc[i] and weekly_up.iloc[i])
        short_up = bool(q.iloc[i] >= ma20.iloc[i] and q.iloc[i] >= ma50.iloc[i])
        if long_up:
            target = weights(float(dd.iloc[i]))
            sleeve2 = sum(target[t] for t in DOUBLE)
            cap2 = .10 if vol.iloc[i] > .25 else .25
            for t in DOUBLE:
                target[t] *= min(1., cap2 / sleeve2) if sleeve2 else 1.
            for t in TRIPLE:
                target[t] = 0.
        else:
            base_fraction = .80 if short_up else .50
            for t in BASE:
                target[t] = base_fraction / len(BASE)
        result.append(target)
    return result


def validate_bars(opens, closes):
    if not opens.index.equals(closes.index) or list(opens.columns) != list(closes.columns):
        raise ValueError("Open and Close index/columns must match")
    if not set(ALL).issubset(opens.columns):
        raise ValueError("Incomplete ETF universe")
    if len(opens) <= WARMUP + 50:
        raise ValueError("Insufficient history for fair warm-up")
    if not opens.index.is_monotonic_increasing or opens.index.has_duplicates:
        raise ValueError("Unsorted or duplicate trading sessions")
    for p in (opens, closes):
        if p.isna().any().any() or not np.isfinite(p.to_numpy()).all() or (p <= 0).any().any():
            raise ValueError("Invalid OHLC data")


def run_strategy(opens, closes, targets, policy="change", cost_bps=30.,
                 cash_rate=0., initial=INITIAL, first_index=WARMUP):
    """Rebalance once at open i based solely on target[i-1].

    Cash is the unallocated fraction and optionally earns a specified
    annualized rate over calendar days between closes.
    """
    validate_bars(opens, closes)
    if policy not in ("daily", "change", "once", "monthly"):
        raise ValueError("Invalid policy")
    if not (0 <= cost_bps <= 100 and 0 <= cash_rate <= .20):
        raise ValueError("Invalid costs/cash rate")
    if len(targets) != len(opens) or first_index < WARMUP or first_index >= len(opens) - 2:
        raise ValueError("Invalid targets/evaluation start")
    assets = list(opens.columns)
    w = np.zeros(len(assets), dtype=float)
    prev_target = None
    last_trade_month = None
    nav = float(initial)
    nav_values = [nav]
    rebalance_dates = []
    all_cost = 0.
    all_turnover = 0.
    invested = []
    for i in range(first_index + 1, len(opens)):
        previous_close = closes.iloc[i - 1].to_numpy(dtype=float)
        open_px = opens.iloc[i].to_numpy(dtype=float)
        close_px = closes.iloc[i].to_numpy(dtype=float)
        gap_ratios = open_px / previous_close
        elapsed_days = (opens.index[i] - opens.index[i - 1]).days
        cash_growth = (1 + cash_rate) ** (elapsed_days / 365.25)
        open_growth = (1 - w.sum()) * cash_growth + float(w @ gap_ratios)
        if open_growth <= 0 or not np.isfinite(open_growth):
            raise ValueError("Capital depleted on overnight gap")
        nav *= open_growth
        w = w * gap_ratios / open_growth
        raw_target = targets[i - 1]
        target = np.array([float(raw_target.get(asset, 0.)) for asset in assets])
        if (target < -1e-12).any() or target.sum() > 1 + 1e-9:
            raise ValueError("Invalid target weights")
        month = opens.index[i].to_period("M")
        trade = (policy == "daily"
                 or policy == "change" and (prev_target is None or not np.array_equal(target, prev_target))
                 or policy == "once" and prev_target is None
                 or policy == "monthly" and month != last_trade_month)
        if trade:
            turnover = float(np.abs(target - w).sum())
            cost = nav * turnover * cost_bps / 10000
            if cost >= nav:
                raise ValueError("Capital depleted by costs")
            nav -= cost
            all_cost += cost
            all_turnover += turnover
            if turnover > 1e-12:
                rebalance_dates.append(str(opens.index[i].date()))
            w = target.copy()
            last_trade_month = month
        prev_target = target.copy()
        intraday = close_px / open_px
        growth = 1 - w.sum() + float(w @ intraday)
        if growth <= 0 or not np.isfinite(growth):
            raise ValueError("Capital depleted intraday")
        nav *= growth
        w = w * intraday / growth
        invested.append(float(w.sum()))
        nav_values.append(nav)
    curve = pd.Series(nav_values, index=closes.index[first_index:], name="nav_krw")
    daily_ret = curve.pct_change().dropna()
    ann_years = (curve.index[-1] - curve.index[0]).days / 365.25
    dd = curve / curve.cummax() - 1
    std = daily_ret.std()
    return {
        "cagr": float((nav / initial) ** (1 / ann_years) - 1),
        "total_return": float(nav / initial - 1),
        "mdd": float(dd.min()),
        "sharpe_zero_rf": float(daily_ret.mean() / std * np.sqrt(252)) if std > 0 else None,
        "volatility": float(std * np.sqrt(252)) if std > 0 else 0.,
        "ending_krw": float(nav), "fee_paid_krw": float(all_cost),
        "turnover": float(all_turnover), "rebalances": len(rebalance_dates),
        "average_exposure": float(np.mean(invested)),
        "start": str(curve.index[0].date()), "end": str(curve.index[-1].date()),
        "max_drawdown_krw_per_10m_at_peak": float((1+dd.min()) * 10_000_000),
        "regimes": regime_metrics(curve),
        "trade_dates_sample": rebalance_dates[:5],
    }, curve


def regime_metrics(curve):
    result = {}
    for name, (begin, end) in REGIMES.items():
        section = curve.loc[begin:end] if end else curve.loc[begin:]
        if len(section) < 2:
            continue
        result[name] = {
            "start": str(section.index[0].date()),
            "end": str(section.index[-1].date()),
            "total_return": float(section.iloc[-1] / section.iloc[0] - 1),
            "mdd": float((section / section.cummax() - 1).min())
        }
    return result


def collect_price_bars(client, start, end):
    opens, closes, issues = {}, {}, {}
    for ticker in ALL:
        try:
            bars = adjusted(client.daily(ticker, start, end))
            opens[ticker] = bars.Open
            closes[ticker] = bars.Close
        except DataError as e:
            issues[ticker] = str(e)
    if issues:
        raise DataError("Missing required original Tiingo prices for " + ",".join(sorted(issues)))
    opening = pd.concat(opens, axis=1).sort_index()
    closing = pd.concat(closes, axis=1).sort_index()
    valid = opening.notna().all(axis=1) & closing.notna().all(axis=1)
    opening, closing = opening.loc[valid], closing.loc[valid]
    validate_bars(opening, closing)
    return opening, closing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="fair_compare_results")
    args = parser.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    client = Client()
    opens, closes = collect_price_bars(client, args.start, args.end)
    targets = prepare_targets(closes)
    results = {}
    for cost in (15., 30., 60.):
        for cash_rate in (0., .03):
            key = f"cost{int(cost)}bps_cash{int(cash_rate*100)}pct"
            scenario = {}
            for name, (target, policy) in targets.items():
                stat, curve = run_strategy(opens, closes, target, policy, cost, cash_rate)
                scenario[name] = stat
                if cost == 30 and cash_rate == 0:
                    curve.to_csv(outdir / f"{name}_equity.csv", index_label="Date")
            results[key] = scenario
    report = {
        "status": "completed",
        "source": "Tiingo original adjusted Open and Close, no synthetic leveraged bars",
        "universe": list(ALL),
        "price_start": str(closes.index[0].date()),
        "evaluation_start": str(closes.index[WARMUP].date()),
        "evaluation_end": str(closes.index[-1].date()),
        "sessions": len(closes) - WARMUP,
        "methods": {
            "signals": "Closing signals from t-1 or earlier",
            "execution": "At adjusted Open(t), earn overnight gap on existing holdings, then Open-to-Close return after rebalance",
            "positions": "Holdings drift naturally; cash is unallocated part; no margin/shorting",
            "warmup": f"All start with cash and first trade after {WARMUP} common trading sessions",
            "fee": "All-in round-trip modeled as per-direction 15/30/60 bps of traded notional; no separate market impact",
            "cash_rates": "0% and 3% annualized fixed assumptions, not observed historical risk-free rates",
            "benchmarks": "QQQ buy-hold, 4x 1x ETF buy-hold, monthly rebalanced 1x ETF basket",
            "rsi_macd_ablation": "V5 volatility controls intact, only oscillator-confirmed recovery condition removed",
            "no_lookahead": "Signals use only closes to t-1; no intraday signal use",
        },
        "limitations": [
            "Not true out of sample: rule selection made using historical performance",
            "Adjusted Open/Close is dividend reinvestment proxy, not exact cash-flow execution",
            "No per-ticker bid/ask spread, liquidity, market-impact, fill probability or taxes",
            "Cash yield assumed flat, not contemporaneous money market actuals",
            "ETF survivorship/selection and provider historical data revisions possible",
            "V5 family includes no volume signal",
            "Historical subperiods are descriptive, not independent walk-forward tests",
        ],
        "scenarios": results,
    }
    (outdir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    out = results["cost30bps_cash0pct"]
    print(json.dumps({"status": "completed", "evaluation_start": report["evaluation_start"],
                      "evaluation_end": report["evaluation_end"],
                      "summary": {k: {"cagr": round(v["cagr"], 5),
                                      "mdd": round(v["mdd"], 5),
                                      "sharpe": round(v["sharpe_zero_rf"], 4)}
                                  for k, v in out.items()}}, indent=2))


if __name__ == "__main__":
    main()
