"""Runs only after all seven EOD datasets pass validation. No brokerage integration."""
import argparse
from dataclasses import asdict
import html
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest import swing
from portfolio import Costs, account, metrics
from tiingo_data import TICKERS, DataError, adjusted, validate_coverage, clear_performance

LIMITATIONS = [
    'Research only; selected current tickers create selection/survivorship bias.',
    'Adjusted OHLC includes splits and dividends; returns are an adjusted-price total-return proxy, not a cash-dividend ledger.',
    'Adjusted Volume is used to preserve original strategy inputs; historical Close>5 and dollar-volume filters depend on vendor adjustment vintage.',
    'Seven fixed initial sleeves, no rebalancing; each risks 0.5% of current sleeve cash per trade; unused cash earns zero.',
    'Fractional adjusted units, cash cap and no leverage. QQQ benchmark receives the full same initial capital.',
    'KRW/USD held constant at a configurable assumption; excludes FX moves, FX conversion costs and capital-gains tax.',
    'Commission 25bps/side and slippage 5bps/side by default; sell levy 0.3bps is a constant estimate, not historical exchange fee rates.',
    'Original 30%/21-day momentum filter, pullback, 3R target, 10-session hold and final-window exclusion are retained.',
    'Swing requires 126-session indicator warmup after start; benchmark begins on first session.',
    'Daily bars cannot reveal intraday path: opening gap executes first, otherwise both-hit bars assume stop first.',
    'All signals use completed bars, entry is next open; sizing uses available sleeve cash, never future exits.',
    'No 2016-present five-minute dataset has been validated: intraday rules preserved but intraday backtest skipped.',
    'Sharpe uses daily equity returns, 252 sessions/year and zero risk-free rate; CAGR uses calendar years.',
]


def write_outputs(output, summary, curves, fills, records):
    output.mkdir(parents=True, exist_ok=True)
    for name, curve in curves.items():
        curve.to_csv(output / f'{name}_equity.csv', index_label='Date')
        pd.DataFrame(fills[name], columns=['ticker', 'entry_date', 'exit_date', 'entry', 'stop', 'exit',
            'R', 'exit_reason', 'units', 'entry_fill', 'exit_fill', 'entry_debit_usd', 'exit_credit_usd',
            'net_pnl_usd']).to_csv(output / f'{name}_trades.csv', index=False)
    pd.DataFrame(records, columns=['ticker','entry_date','exit_date','entry','stop','exit','R','exit_reason']).to_csv(output / 'swing_signals.csv', index=False)
    pd.DataFrame(summary['metrics']).T.to_csv(output / 'metrics.csv', index_label='strategy')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    table = pd.DataFrame(summary['metrics']).T.to_html(border=0, na_rep='N/A')
    rows = ''.join('<li>' + html.escape(x) + '</li>' for x in LIMITATIONS)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Tiingo research backtest</title>'
            '<style>body{font:16px system-ui;margin:2rem;max-width:1200px}td,th{padding:.5rem;text-align:right}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>'
            '<h1>Tiingo research backtest</h1><p>Validated daily data; intraday NOT evaluated. '
            'Returns and drawdowns below are fractions (0.10 = 10%).</p>' + table +
            '<h2>Configuration</h2><pre>' + html.escape(json.dumps(summary['config'], indent=2)) +
            '</pre><h2>Assumptions and limits</h2><ul>' + rows + '</ul></html>')
    (output / 'report.html').write_text(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data')
    ap.add_argument('--output', default='results')
    ap.add_argument('--initial-krw', type=float, default=10_000_000)
    ap.add_argument('--krw-per-usd', type=float, default=1350.)
    ap.add_argument('--fee-bps', type=float, default=25.)
    ap.add_argument('--slippage-bps', type=float, default=5.)
    ap.add_argument('--sell-levy-bps', type=float, default=.3)
    args = ap.parse_args()
    output = Path(args.output)
    clear_performance(output)
    try:
        validation = json.loads((output / 'data_validation.json').read_text())
        if validation['status'] != 'validated' or set(validation['tickers']) != set(TICKERS):
            raise DataError('All seven tickers must pass API/data validation before backtesting.')
        start, end = validation['start'], validation['end']
        if start != '2016-01-01':
            raise DataError('This research run requires a 2016-01-01 start.')
        frames = {}
        for ticker in TICKERS:
            path = Path(args.cache) / f'{ticker}.csv'
            if hashlib.sha256(path.read_bytes()).hexdigest() != validation['tickers'][ticker]['sha256']:
                raise DataError('CSV changed since API validation; rerun data verification.')
            raw = pd.read_csv(path, index_col='Date', parse_dates=True)
            validate_coverage(raw, start, end)
            frames[ticker] = adjusted(raw.loc[start:end])
        costs = Costs(args.fee_bps, args.slippage_bps, args.sell_levy_bps)
        config = {'initial_krw': args.initial_krw, 'krw_per_usd_fixed_assumption': args.krw_per_usd,
                  'start': start, 'end': end, 'risk_pct_per_sleeve': .005, **asdict(costs)}
        records = [t for ticker, d in frames.items() for t in swing(ticker, d)]
        equity, trades = account(frames, records, args.initial_krw, args.krw_per_usd, costs=costs)
        benchmark, bh_trades = account({'QQQ': frames['QQQ']}, [], args.initial_krw,
                                        args.krw_per_usd, costs=costs, buy_hold=True)
        summary = {'status': 'completed', 'config': config,
            'metrics': {'swing': metrics(equity, trades, args.initial_krw, start),
                        'qqq_buy_hold': metrics(benchmark, bh_trades, args.initial_krw, start)},
            'intraday': {'status': 'skipped_unverified_data', 'metrics': None},
            'limitations': LIMITATIONS}
        write_outputs(output, summary, {'swing': equity, 'qqq_buy_hold': benchmark},
                      {'swing': trades, 'qqq_buy_hold': bh_trades}, records)
        print(json.dumps(summary, indent=2, allow_nan=False))
    except (DataError, ValueError, KeyError, OSError) as exc:
        # Inputs are trusted local files; no token or HTTP responses are read here.
        print(f'Backtest not completed: {type(exc).__name__}. Check data_validation.json and local data.')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
