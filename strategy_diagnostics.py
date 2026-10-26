"""Diagnostic funnel for the unchanged swing entry logic. Run after Tiingo validation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from backtest import indicators, swing
from tiingo_data import TICKERS, adjusted


def diagnose(ticker, raw):
    d = indicators(adjusted(raw)).dropna().copy()
    # Exact eligibility window used by swing(): first eligible row is index 1.
    d = d.iloc[1:max(1, len(d) - 12)].copy()
    if d.empty:
        return {'ticker': ticker, 'sessions': 0, 'trades': 0}
    filters = [
        ('price_gt_5', d.Close > 5),
        ('dollar_volume_gt_20m', d.dv20 > 20e6),
        ('return_21d_gt_30pct', d.ret21 > .30),
        ('return_63d_positive', d.ret63 > 0),
        ('return_126d_positive', d.ret126 > 0),
        ('ema21_gt_ema50', d.ema21 > d.ema50),
        ('low_near_ema21', d.Low <= d.ema21 * 1.02),
        ('close_above_ema21', d.Close > d.ema21),
        ('bullish_candle', d.Close > d.Open),
    ]
    # Previous candle comes from the full indicator frame, not the sliced subset.
    full = indicators(adjusted(raw)).dropna()
    prev_red = full.Close.shift(1).lt(full.Open.shift(1)).reindex(d.index).fillna(False)
    filters.append(('previous_candle_bearish', prev_red))
    mask = pd.Series(True, index=d.index)
    stages = {'eligible_sessions': len(d)}
    for name, condition in filters:
        mask &= condition.fillna(False)
        stages[name] = int(mask.sum())
    # Next-open risk rule: calculate on candidates, without treating overlap as executed trades.
    candidates = d.index[mask]
    accepted = 0
    for idx in candidates:
        i = full.index.get_loc(idx)
        if i + 1 >= len(full):
            continue
        row = full.iloc[i]
        entry = float(full.iloc[i + 1].Open)
        stop = min(float(row.Low), float(row.ema21) * .985)
        if entry > stop and (entry - stop) / entry <= .08:
            accepted += 1
    return {'ticker': ticker, 'sessions': len(d), 'funnel': stages,
            'risk_rule_pass_candidates': accepted,
            'executed_trades': len(swing(ticker, adjusted(raw)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data')
    ap.add_argument('--output', default='results/strategy_diagnostics.json')
    args = ap.parse_args()
    rows = []
    for ticker in TICKERS:
        path = Path(args.cache) / f'{ticker}.csv'
        if not path.is_file():
            raise SystemExit(f'Missing validated CSV: {ticker}')
        raw = pd.read_csv(path, index_col='Date', parse_dates=True)
        rows.append(diagnose(ticker, raw))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'note': 'Sequential condition funnel; not a performance backtest.', 'tickers': rows}, indent=2))
    print(out.read_text())


if __name__ == '__main__':
    main()
