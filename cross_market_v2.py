"""Cross-market V2 validation: genuine 1x/2x/3x ETF adjusted closes only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tiingo_data import Client, DataError, adjusted

MARKETS = {
    "nasdaq100": ("QQQ", "QQQ", "QLD", "TQQQ"),
    "sp500": ("SPY", "SPY", "SSO", "UPRO"),
    "semiconductors": ("SOXX", "SOXX", "USD", "SOXL"),
    "technology": ("XLK", "XLK", "ROM", "TECL"),
    "smallcap": ("IWM", "IWM", "UWM", "TNA"),
    "financials": ("XLF", "XLF", "UYG", "FAS"),
}
SPLIT = "2022-01-03"


def allocation(drawdown):
    if drawdown <= -.30:
        return np.array([.50, .20, .30])
    if drawdown <= -.20:
        return np.array([.65, .25, .10])
    if drawdown <= -.10:
        return np.array([.80, .20, 0.])
    return np.array([1., 0., 0.])


def simulate(prices, mode="v2", fee_bps=30., initial=10_000_000):
    if mode not in ("v2", "buy_hold_1x", "static_leveraged"):
        raise ValueError("Invalid strategy")
    if len(prices) < 252 or prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Invalid common price history")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Unsorted or duplicate sessions")
    p = prices[["signal", "one", "two", "three"]]
    dd = p.signal / p.signal.cummax() - 1
    weights = np.zeros(3)
    prev = None
    nav = float(initial)
    fees = turnover = 0.
    trades = 0
    values = [nav]
    for i in range(1, len(p)):
        if mode == "v2":
            target = allocation(float(dd.iloc[i - 1]))
        elif mode == "buy_hold_1x":
            target = np.array([1., 0., 0.])
        else:
            target = np.array([.8, .15, .05])
        if prev is None or not np.array_equal(target, prev):
            volume = float(np.abs(target - weights).sum())
            cost = nav * volume * fee_bps / 10000.
            nav -= cost
            fees += cost
            turnover += volume
            trades += volume > 1e-12
            weights = target.copy()
        prev = target.copy()
        returns = p[["one", "two", "three"]].iloc[i].to_numpy() / p[["one", "two", "three"]].iloc[i - 1].to_numpy() - 1
        growth = 1 + float(weights @ returns)
        if growth <= 0:
            raise ValueError("Portfolio depleted")
        nav *= growth
        weights = weights * (1 + returns) / growth
        values.append(nav)
    curve = pd.Series(values, index=p.index)
    daily = curve.pct_change().dropna()
    years = (p.index[-1] - p.index[0]).days / 365.25
    return {"cagr": float((nav / initial) ** (1 / years) - 1),
            "mdd": float((curve / curve.cummax() - 1).min()),
            "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() else None,
            "ending_krw": float(nav), "fees_krw": float(fees),
            "turnover": float(turnover), "rebalances": int(trades)}, curve


def segment(curve, start, end):
    x = curve.loc[start:end]
    if len(x) < 2:
        return None
    years = (x.index[-1] - x.index[0]).days / 365.25
    return {"start": str(x.index[0].date()), "end": str(x.index[-1].date()),
            "return": float(x.iloc[-1] / x.iloc[0] - 1),
            "cagr": float((x.iloc[-1] / x.iloc[0]) ** (1 / years) - 1),
            "mdd": float((x / x.cummax() - 1).min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", required=True)
    ap.add_argument("--output", default="cross_market_results")
    ap.add_argument("--fee-bps", type=float, default=30.)
    args = ap.parse_args()
    if not 0 <= args.fee_bps <= 100:
        ap.error("fee-bps must be 0..100")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    client = Client()
    tickers = sorted(set(t for group in MARKETS.values() for t in group))
    data, failures = {}, {}
    for t in tickers:
        try:
            data[t] = adjusted(client.daily(t, args.start, args.end))["Close"]
        except DataError as exc:
            failures[t] = str(exc)
    results = {}
    for name, (signal, one, two, three) in MARKETS.items():
        if any(t not in data for t in (signal, one, two, three)):
            results[name] = {"status": "unavailable", "reason": "Required real ETF data missing",
                             "missing": [t for t in (signal, one, two, three) if t not in data]}
            continue
        p = pd.concat({"signal": data[signal], "one": data[one], "two": data[two], "three": data[three]}, axis=1).dropna()
        if len(p) < 252:
            results[name] = {"status": "unavailable", "reason": "Insufficient overlapping actual ETF history"}
            continue
        market = {"status": "completed", "tickers": {"signal": signal, "one": one, "two": two, "three": three},
                  "start": str(p.index[0].date()), "end": str(p.index[-1].date()),
                  "sessions": len(p), "strategies": {}}
        for mode in ("v2", "buy_hold_1x", "static_leveraged"):
            metrics, curve = simulate(p, mode, args.fee_bps)
            metrics["segments"] = {"training_2016_2021": segment(curve, "2016-01-01", "2021-12-31"),
                                   "holdout_2022_2026": segment(curve, SPLIT, args.end),
                                   "covid_2020": segment(curve, "2020-02-19", "2020-06-30"),
                                   "bear_2022": segment(curve, "2022-01-03", "2022-12-30")}
            market["strategies"][mode] = metrics
            curve.to_csv(out / f"{name}_{mode}_equity.csv", index_label="Date")
        market["v2_vs_buyhold_cagr_pp"] = 100 * (market["strategies"]["v2"]["cagr"] - market["strategies"]["buy_hold_1x"]["cagr"])
        results[name] = market
    report = {"status": "completed" if any(x["status"] == "completed" for x in results.values()) else "unavailable",
              "source": "Tiingo genuine adjusted ETF daily closes", "start_requested": args.start,
              "end_requested": args.end, "cost_bps_per_turnover": args.fee_bps,
              "method": "V2 threshold allocation; signal from own market benchmark; rebalance on threshold change; prior-close execution proxy",
              "cautions": ["Per-market histories differ; compare aligned dates within each market.",
                           "Training/holdout split is descriptive: V2 rules were developed with hindsight and this is not a truly untouched out-of-sample test.",
                           "Static leveraged benchmark is 80% 1x, 15% 2x, 5% 3x, drifting after initial allocation.",
                           "No FX, taxes, true next-open fills or financing/slippage beyond flat turnover fee."],
              "provider_failures": failures, "markets": results}
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"status": report["status"], "markets": {k: v["status"] for k, v in results.items()}}, indent=2))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
