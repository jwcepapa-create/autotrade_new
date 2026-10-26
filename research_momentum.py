"""Research-only sensitivity study. Keeps the production swing strategy unchanged.

The same price data, fills, stops, targets and holding rules are used; only the
21-day momentum threshold and/or EMA pullback proximity change per experiment.
This is hypothesis generation, not a walk-forward or out-of-sample evaluation.
"""
import argparse
import csv
from pathlib import Path

from backtest import indicators
from tiingo_data import TICKERS, adjusted, validate_values
import pandas as pd


def trades_for(ticker, prices, momentum=.30, pullback=.02, max_hold=10):
    d = indicators(prices).dropna().copy()
    trades = []
    i = 1
    while i < len(d) - max_hold - 2:
        r = d.iloc[i]
        p = d.iloc[i - 1]
        trend = (r.Close > 5 and r.dv20 > 20e6 and
                 r.ret21 > momentum and r.ret63 > 0 and r.ret126 > 0 and
                 r.ema21 > r.ema50)
        signal = (trend and r.Low <= r.ema21 * (1 + pullback) and
                  r.Close > r.ema21 and r.Close > r.Open and p.Close < p.Open)
        if not signal:
            i += 1
            continue
        entry_idx = i + 1
        entry = float(d.iloc[entry_idx].Open)
        stop = min(float(r.Low), float(r.ema21) * .985)
        if entry <= stop or (entry - stop) / entry > .08:
            i += 1
            continue
        risk = entry - stop
        target = entry + 3 * risk
        exit_idx = entry_idx + max_hold - 1
        exit_price = float(d.iloc[exit_idx].Close)
        for j in range(entry_idx, min(len(d), entry_idx + max_hold)):
            bar = d.iloc[j]
            if bar.Open <= stop or bar.Open >= target:
                exit_idx, exit_price = j, float(bar.Open)
                break
            if bar.Low <= stop:
                exit_idx, exit_price = j, stop
                break
            if bar.High >= target:
                exit_idx, exit_price = j, target
                break
        trades.append({'ticker': ticker, 'entry_date': str(d.index[entry_idx].date()),
                       'exit_date': str(d.index[exit_idx].date()),
                       'entry': entry, 'stop': stop, 'exit': exit_price,
                       'R': (exit_price - entry) / risk})
        i = exit_idx + 1
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data')
    ap.add_argument('--output', default='results/momentum_variants.csv')
    args = ap.parse_args()
    experiments = [
        ('baseline', .30, .02),
        ('momentum_20pct', .20, .02),
        ('momentum_10pct', .10, .02),
        ('pullback_4pct', .30, .04),
        ('pullback_6pct', .30, .06),
    ]
    rows = []
    for ticker in TICKERS:
        raw = pd.read_csv(Path(args.cache) / f'{ticker}.csv', index_col='Date', parse_dates=True)
        validate_values(raw)
        prices = adjusted(raw)
        for label, momentum, pullback in experiments:
            trades = trades_for(ticker, prices, momentum, pullback)
            rows.append({'ticker': ticker, 'variant': label, 'momentum': momentum,
                         'pullback': pullback, 'trades': len(trades),
                         'sum_R': round(sum(t['R'] for t in trades), 6)})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(path)
    print('Research only: trade count and sum_R are NOT portfolio returns. Reuse portfolio.py for CAGR/MDD.')


if __name__ == '__main__':
    main()
