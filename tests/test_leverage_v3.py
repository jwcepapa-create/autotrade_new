import pandas as pd
import pytest
from leverage_drawdown import ALL
from leverage_v3 import trend_targets, simulate_v3


def sample(n=300):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({t: [100 + .1 * i for i in range(n)] for t in ALL}, index=idx)


def test_leverage_caps_and_no_lookahead():
    p = sample()
    targets = trend_targets(p, cap_3x=.10, cap_2x=.20)
    assert all(sum(w[t] for t in ("TQQQ", "SOXL")) <= .10000001 for w in targets)
    assert all(sum(w[t] for t in ("QLD", "USD")) <= .20000001 for w in targets)
    assert all(sum(w.values()) <= 1.00000001 for w in targets)
    changed = p.copy()
    changed.iloc[-1, changed.columns.get_loc("QQQ")] *= .4
    assert targets[:-1] == trend_targets(changed, .10, .20)[:-1]


def test_no_trade_same_bar_and_zero_returns():
    p = pd.DataFrame({t: [100.] * 300 for t in ALL},
                     index=pd.bdate_range("2020-01-01", periods=300))
    r, curve = simulate_v3(p, fee_bps=0)
    assert r["total_return"] == pytest.approx(0)
    assert len(curve) == 300


def test_invalid_limits():
    with pytest.raises(ValueError):
        simulate_v3(sample(), cap_3x=.5)
