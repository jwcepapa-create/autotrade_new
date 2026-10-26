"""Independent tests for user-provided v0.3 model port."""
import numpy as np
import pandas as pd
from mts_v03.breakout import features, signals, run


def bars(n=150):
    close = np.arange(n, dtype=float) + 100
    return pd.DataFrame({"Open": close, "High": close+1,
                         "Low": close-1, "Close": close,
                         "Volume": np.full(n, 100.)},
                        index=pd.bdate_range("2020-01-01", periods=n))


def test_prior_resistance_never_includes_today():
    df = bars()
    f = features(df)
    assert f.resistance_prior.iloc[60] == df.High.iloc[:60].max()
    assert f.support_prior.iloc[60] == df.Low.iloc[:60].min()


def test_appending_future_candles_does_not_change_past_signals():
    df = bars()
    a = signals(df.iloc[:100])
    b = signals(df).iloc[:100]
    for col in ("entry", "exit_5", "exit_20", "exit_support"):
        assert a[col].equals(b[col])


def test_v03_no_future_backtest_and_finite_metrics():
    result = run(bars())
    assert np.isfinite(result["total_return"])
    assert np.isfinite(result["max_drawdown"])
    assert all(np.isfinite(t[2]) and t[2] >= 0 for t in result["trades"])


def test_next_open_execution_not_same_day():
    df = bars()
    s = signals(df)
    r = run(df)
    signal_dates = set(s.index[s.entry])
    for date, side, qty, price in r["trades"]:
        if side == "BUY":
            d = pd.Timestamp(date)
            assert d in df.index
            assert df.index[df.index.get_loc(d)-1] in signal_dates
