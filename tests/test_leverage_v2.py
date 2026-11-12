import pandas as pd
import pytest
from leverage_drawdown import ALL, simulate
from leverage_v2 import simulate_v2


def prices():
    dates = pd.bdate_range("2020-01-01", periods=260)
    d = {t: [100.] * len(dates) for t in ALL}
    d["VOO"] = [100 + i * .5 for i in range(len(dates))]
    d["MGK"] = [100 + i * .1 for i in range(len(dates))]
    return pd.DataFrame(d, index=dates)


def test_daily_rebalancing_costs_more_than_v1_when_weights_drift():
    p = prices()
    v1 = simulate(p, fee_bps=30)
    v2, curve, events = simulate_v2(p, fee_bps=30)
    assert v2["turnover"] > v1["turnover"]
    assert v2["rebalances"] > v1["rebalances"]
    assert v2["transaction_cost_krw"] > 0
    assert len(curve) == len(p)


def test_threshold_policy_trades_less_than_daily():
    p = prices()
    daily, _, _ = simulate_v2(p, fee_bps=30, rebalance="daily")
    threshold, _, _ = simulate_v2(p, fee_bps=30, rebalance="threshold")
    assert daily["turnover"] > threshold["turnover"]
    assert daily["rebalances"] > threshold["rebalances"]


def test_constant_prices_no_drift():
    p = pd.DataFrame({t: [100.] * 260 for t in ALL},
                     index=pd.bdate_range("2020-01-01", periods=260))
    v2, _, _ = simulate_v2(p, fee_bps=0)
    assert v2["total_return"] == pytest.approx(0)
    assert v2["rebalances"] == 1
