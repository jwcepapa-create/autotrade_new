"""Compare original repeat exits against edge-triggered exits on the same bars."""
import argparse
import json
from pathlib import Path
import pandas as pd
from tiingo_data import TICKERS, adjusted, validate_values
from mts_v03.exit_experiments import simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data')
    ap.add_argument('--output', default='results/mts_exit_comparison.json')
    args = ap.parse_args()
    result = []
    for ticker in TICKERS:
        raw = pd.read_csv(Path(args.cache)/f'{ticker}.csv', index_col='Date', parse_dates=True)
        validate_values(raw)
        for policy in ('repeat', 'edge'):
            s = simulate(adjusted(raw), policy)
            result.append({'ticker':ticker, 'policy':policy,
                           'total_return_pct':round(s['total_return']*100,5),
                           'max_drawdown_pct':round(s['max_drawdown']*100,5),
                           'buy_orders':sum(t['side']=='BUY' for t in s['orders']),
                           'sell_orders':sum(t['side']=='SELL' for t in s['orders']),
                           'exposure_pct':round(s['exposure']*100,3),
                           'gross_turnover_over_initial':round(s['turnover_over_initial'],5)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'description':'Research-only, separate per-ticker sleeves; not a portfolio',
                                  'cost_each_side':0.0015,
                                  'policy_edge':'Partial sell only when exit level changes; full support exit always active',
                                  'results':result},indent=2))
    print(output.read_text())


if __name__ == '__main__':
    main()
