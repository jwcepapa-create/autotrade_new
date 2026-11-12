import pandas as pd
import pytest
from leverage_drawdown import ALL
from leverage_v4 import hybrid_targets, simulate_v4


def sample():
    ix = pd.bdate_range("2020-01-01", periods=310)
    p = pd.DataFrame({t: [100. + i * .1 for i in range(len(ix))] for t in ALL}, index=ix)
    return p


def test_caps_and_budget():
    p = sample()
    for variant in ("strict", "recovery", "balanced"):
        for t in hybrid_targets(p, variant):
            assert sum(t.values()) <= 1.0000001
            assert sum(t[x] for x in ("TQQQ", "SOXL")) <= .1000001
            assert sum(t[x] for x in ("QLD", "USD")) <= .2500001


def test_no_future_leakage():
    p = sample()
    before = hybrid_targets(p)
    altered = p.copy()
    altered.iloc[-1, altered.columns.get_loc("QQQ")] *= .5
    assert before[:-1] == hybrid_targets(altered)[:-1]


def test_constant_prices_and_no_orders():
    p = pd.DataFrame({t: [100.] * 310 for t in ALL},
                     index=pd.bdate_range("2020-01-01", periods=310))
    result, curve = simulate_v4(p, fee_bps=0)
    assert result["total_return"] == pytest.approx(0)
    assert len(curve) == 310


def test_cost_nonnegative_and_limits():
    p = sample()
    free, _ = simulate_v4(p, fee_bps=0)
    costly, _ = simulate_v4(p, fee_bps=30)
    assert costly["ending_krw"] <= free["ending_krw"]
    with pytest.raises(ValueError):
        simulate_v4(p, cap3=.5)
