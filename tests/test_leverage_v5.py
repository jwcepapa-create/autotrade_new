import numpy as np
import pandas as pd
import pytest
from leverage_drawdown import ALL, BASE, DOUBLE, TRIPLE
from leverage_v5 import v5_targets, simulate_v5


def prices(n=340):
    idx = pd.bdate_range("2023-01-02", periods=n)
    base = np.linspace(100, 150, n)
    return pd.DataFrame({t: base * (1 + .001 * k) for k, t in enumerate(ALL)}, index=idx)


@pytest.mark.parametrize("variant", ["fast", "confirmed", "volatility"])
def test_budget_and_no_3x(variant):
    for t in v5_targets(prices(), variant):
        assert 0 <= sum(t.values()) <= 1 + 1e-10
        assert sum(t[k] for k in TRIPLE) == 0
        assert sum(t[k] for k in DOUBLE) <= .25 + 1e-10


def test_causality_no_future_prices():
    p = prices()
    a = v5_targets(p.iloc[:280], "fast")
    changed = p.copy()
    changed.iloc[280:] *= 10
    b = v5_targets(changed, "fast")
    assert a == b[:280]


def test_fee_impact_and_flat_prices():
    p = prices()
    a, _ = simulate_v5(p, "confirmed", fee_bps=0)
    b, _ = simulate_v5(p, "confirmed", fee_bps=30)
    assert a["ending_krw"] >= b["ending_krw"]
    flat = p.copy()
    flat.loc[:, :] = 100.
    result, _ = simulate_v5(flat, "fast", fee_bps=0)
    assert abs(result["ending_krw"] - 10_000_000) < 1e-5


def test_invalid_variant():
    with pytest.raises(ValueError):
        v5_targets(prices(), "unknown")
