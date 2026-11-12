import numpy as np
import pandas as pd
from cross_market_v2 import allocation, simulate, segment, MARKETS


def sample(n=300):
    idx = pd.bdate_range("2023-01-02", periods=n)
    x = np.linspace(100, 140, n)
    return pd.DataFrame({"signal": x, "one": x, "two": x * 1.1, "three": x * 1.2}, index=idx)


def test_six_markets():
    assert len(MARKETS) == 6


def test_thresholds():
    assert np.allclose(allocation(-.31), [.5, .2, .3])
    assert np.allclose(allocation(-.21), [.65, .25, .1])
    assert np.allclose(allocation(-.11), [.8, .2, 0])
    assert np.allclose(allocation(-.01), [1, 0, 0])


def test_causality():
    p = sample()
    a, curve = simulate(p.iloc[:260])
    changed = p.copy()
    changed.iloc[260:] *= 10
    b, curve2 = simulate(changed.iloc[:260])
    assert a == b


def test_fee_and_flat():
    p = sample()
    a, _ = simulate(p, fee_bps=0)
    b, _ = simulate(p, fee_bps=30)
    assert a["ending_krw"] >= b["ending_krw"]
    p.loc[:, :] = 100.
    m, _ = simulate(p, fee_bps=0)
    assert abs(m["ending_krw"] - 10_000_000) < 1e-5


def test_segments():
    _, curve = simulate(sample())
    assert segment(curve, "2023-01-02", "2023-04-01") is not None
