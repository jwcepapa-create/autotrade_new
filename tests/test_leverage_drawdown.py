import pandas as pd
import pytest
from leverage_drawdown import ALL, simulate, weights


def test_thresholds_and_allocation():
    assert sum(weights(0).values()) == pytest.approx(1)
    assert weights(-.099)["QLD"] == 0
    assert weights(-.10)["QLD"] > 0
    assert weights(-.20)["TQQQ"] > 0
    assert weights(-.30)["TQQQ"] > weights(-.20)["TQQQ"]
    assert sum(weights(-.30, managed=True).values()) < 1


def test_no_future_data_and_no_fabricated_results():
    dates = pd.bdate_range("2020-01-01", periods=260)
    prices = pd.DataFrame({t: [100.] * 260 for t in ALL}, index=dates)
    result = simulate(prices, fee_bps=0)
    assert result["total_return"] == pytest.approx(0)
    assert result["mdd"] == pytest.approx(0)
    with pytest.raises(ValueError):
        simulate(prices.drop(columns=["SOXL"]))


def test_no_trade_on_same_day_signal():
    dates = pd.bdate_range("2020-01-01", periods=260)
    prices = pd.DataFrame({t: [100.] * 260 for t in ALL}, index=dates)
    prices.loc[dates[100], "QQQ"] = 70
    # The -30% close signal cannot influence the same day's returns.
    for t in ("TQQQ", "SOXL"):
        prices.loc[dates[100], t] = 200
    r = simulate(prices, fee_bps=0)
    assert r["rebalances"] >= 2
