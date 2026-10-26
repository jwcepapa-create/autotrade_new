"""Run isolated v0.3 strategy B on validated Tiingo EOD CSVs (no orders)."""
import argparse
import json
from pathlib import Path
import pandas as pd
from tiingo_data import TICKERS, adjusted, validate_values
from mts_v03.breakout import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='data')
    p.add_argument('--output', default='results/mts_v03_breakout.json')
    args = p.parse_args()
    results = []
    for ticker in TICKERS:
        path = Path(args.cache) / f'{ticker}.csv'
        raw = pd.read_csv(path, index_col='Date', parse_dates=True)
        validate_values(raw)
        result = run(adjusted(raw))
        results.append({
            'ticker': ticker,
            'total_return_pct': round(100*result['total_return'], 5),
            'max_drawdown_pct': round(100*result['max_drawdown'], 5),
            'order_count': len(result['trades']),
            'buy_count': sum(t[1] == 'BUY' for t in result['trades']),
            'sell_count': sum(t[1] == 'SELL' for t in result['trades']),
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'strategy': 'B: v0.3 daily breakout, independent per-ticker sleeves',
        'initial_capital_per_ticker_krw': 10000000,
        'transaction_cost_each_side': 0.0015,
        'note': 'Independent sleeves, not a combined portfolio. No broker orders. No FX conversion.',
        'results': results,
    }, indent=2))
    print(out.read_text())


if __name__ == '__main__':
    main()
