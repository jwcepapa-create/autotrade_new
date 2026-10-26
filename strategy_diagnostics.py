"""Per-ticker diagnostics for the unchanged daily swing strategy."""
import argparse
import json
from pathlib import Path

import pandas as pd

from backtest import indicators, swing
from tiingo_data import TICKERS, adjusted, validate_values

FILTER_NAMES = (
    'price_gt_5', 'dollar_volume_gt_20m', 'return_21d_gt_30pct',
    'return_63d_positive', 'return_126d_positive', 'ema21_gt_ema50',
    'low_near_ema21', 'close_above_ema21', 'bullish_candle',
    'previous_candle_bearish',
)


def diagnose(ticker, raw, max_hold=10):
    """Sequential funnel counts candidate bars, not distinct executed trades."""
    d = indicators(adjusted(raw)).dropna().copy()
    eligible = d.iloc[1:len(d) - max_hold - 2]
    counts = {'eligible_sessions': len(eligible), **{k: 0 for k in FILTER_NAMES}}
    accepted = 0
    if len(eligible):
        prev = d.Close.shift(1) < d.Open.shift(1)
        conditions = (
            eligible.Close > 5,
            eligible.dv20 > 20e6,
            eligible.ret21 > .30,
            eligible.ret63 > 0,
            eligible.ret126 > 0,
            eligible.ema21 > eligible.ema50,
            eligible.Low <= eligible.ema21 * 1.02,
            eligible.Close > eligible.ema21,
            eligible.Close > eligible.Open,
            prev.reindex(eligible.index).fillna(False),
        )
        mask = pd.Series(True, index=eligible.index)
        for name, condition in zip(FILTER_NAMES, conditions):
            mask &= condition.fillna(False)
            counts[name] = int(mask.sum())
        for idx in eligible.index[mask]:
            i = d.index.get_loc(idx)
            row = d.iloc[i]
            entry = float(d.iloc[i + 1].Open)
            stop = min(float(row.Low), float(row.ema21) * .985)
            if entry > stop and (entry - stop) / entry <= .08:
                accepted += 1
    trades = swing(ticker, adjusted(raw), max_hold=max_hold)
    return {'ticker': ticker, 'sessions': len(eligible), 'funnel': counts,
            'risk_rule_pass_candidates': accepted, 'executed_trades': len(trades)}


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
        validate_values(raw)
        rows.append(diagnose(ticker, raw))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'note': 'Sequential filter candidate counts; risk-pass candidates can overlap existing positions.',
        'tickers': rows,
    }, indent=2))
    print(out.read_text())


if __name__ == '__main__':
    main()
