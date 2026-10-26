"""Independent diagnostic checks using deterministic synthetic daily OHLCV."""
import numpy as np
import pandas as pd

from backtest import swing
from strategy_diagnostics import FILTER_NAMES, diagnose


def fixture(n=260, daily_return=0.004):
    dates = pd.bdate_range('2020-01-01', periods=n)
    close = 20 * (1 + daily_return) ** np.arange(n)
    return pd.DataFrame({
        'adjOpen': close * 0.995,
        'adjHigh': close * 1.02,
        'adjLow': close * 0.98,
        'adjClose': close,
        'adjVolume': np.full(n, 5_000_000.0),
    }, index=dates)


def test_empty_eligible_schema():
    result = diagnose('SYN', fixture(130))
    assert result['sessions'] == 0
    assert result['executed_trades'] == 0
    assert result['risk_rule_pass_candidates'] == 0
    assert set(FILTER_NAMES).issubset(result['funnel'])


def test_no_momentum_means_no_trades():
    raw = fixture(daily_return=0.001)
    result = diagnose('SYN', raw)
    assert result['funnel']['return_21d_gt_30pct'] == 0
    assert result['executed_trades'] == 0
    assert result['risk_rule_pass_candidates'] == 0


def test_funnel_monotonic_and_executions_match():
    raw = fixture(daily_return=0.016)
    result = diagnose('SYN', raw)
    counts = [result['funnel']['eligible_sessions']] + [
        result['funnel'][key] for key in FILTER_NAMES
    ]
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    adjusted = raw.rename(columns={
        'adjOpen': 'Open', 'adjHigh': 'High', 'adjLow': 'Low',
        'adjClose': 'Close', 'adjVolume': 'Volume'
    })
    assert result['executed_trades'] == len(swing('SYN', adjusted))
    assert result['risk_rule_pass_candidates'] >= result['executed_trades']
