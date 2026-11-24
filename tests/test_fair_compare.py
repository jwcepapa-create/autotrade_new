"""Invariants for the fair next-open portfolio engine (synthetic only)."""
import numpy as np
import pandas as pd
import pytest
from fair_compare import (
    WARMUP, prepare_targets, regime_metrics, run_strategy,
    v5_no_oscillator_targets,
)
from leverage_drawdown import ALL, BASE, TRIPLE


def bars(n=350):
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = pd.DataFrame(100., index=dates, columns=ALL)
    opened = close.copy()
    return opened, close


def test_first_trade_does_not_earn_prior_close_to_open_gap():
    opened, close = bars()
    i = WARMUP + 1
    opened.loc[opened.index[i], "QQQ"] = 150.
    close.loc[close.index[i]:, "QQQ"] = 150.
    # Signal on day i-1 cannot capture the 100->150 overnight gap.
    targets = [{"QQQ": 1.}] * len(close)
    metrics, curve = run_strategy(opened, close, targets, "once", cost_bps=0.)
    assert np.isclose(curve.iloc[1], 10_000_000.)
    assert metrics["rebalances"] == 1


def test_existing_holdings_do_earn_overnight_gap():
    opened, close = bars()
    i = WARMUP + 1
    close.loc[close.index[i]:, "QQQ"] = 110.
    opened.loc[opened.index[i], "QQQ"] = 100.
    opened.loc[opened.index[i + 1]:, "QQQ"] = 220.
    close.loc[close.index[i + 1]:, "QQQ"] = 220.
    metrics, curve = run_strategy(
        opened, close, [{"QQQ": 1.}] * len(close), "once", cost_bps=0.)
    assert np.isclose(curve.iloc[1], 11_000_000.)
    assert np.isclose(curve.iloc[2], 22_000_000.)
    assert metrics["rebalances"] == 1


def test_holdings_drift_without_forced_daily_rebalance():
    opened, close = bars()
    i = WARMUP + 1
    t = BASE[0]
    opened.loc[opened.index[i + 1]:, t] = 150.
    close.loc[close.index[i + 1]:, t] = 150.
    weights = [{x: .25 for x in BASE}] * len(close)
    held, _ = run_strategy(opened, close, weights, "once", cost_bps=0.)
    daily, _ = run_strategy(opened, close, weights, "daily", cost_bps=0.)
    assert np.isclose(held["ending_krw"], 11_250_000.)
    assert np.isclose(daily["ending_krw"], 11_250_000.)
    assert held["rebalances"] == 1
    assert daily["rebalances"] > 1  # drift causes corrective trades


def test_change_policy_rebalances_only_on_target_change():
    opened, close = bars()
    targets = [{BASE[0]: 1.}] * len(close)
    targets[WARMUP + 12:] = [{BASE[1]: 1.}] * (len(close) - WARMUP - 12)
    m, _ = run_strategy(opened, close, targets, "change", cost_bps=30.)
    assert m["rebalances"] == 2


def test_cash_is_not_a_free_return_and_interest_is_explicit():
    opened, close = bars()
    zeros = [{}] * len(close)
    no_interest, _ = run_strategy(opened, close, zeros, "once", cost_bps=0.)
    with_interest, _ = run_strategy(opened, close, zeros, "once", cost_bps=0., cash_rate=.03)
    assert np.isclose(no_interest["ending_krw"], 10_000_000)
    assert with_interest["ending_krw"] > 10_000_000


def test_fees_monotonically_reduce_value():
    opened, close = bars()
    targets = [{BASE[0]: .5, BASE[1]: .5}] * len(close)
    no_cost, _ = run_strategy(opened, close, targets, "once", cost_bps=0.)
    fee_cost, _ = run_strategy(opened, close, targets, "once", cost_bps=60.)
    assert fee_cost["ending_krw"] < no_cost["ending_krw"]


def test_no_3x_and_no_future_prices_in_v5():
    opened, close = bars()
    for t in ALL:
        close[t] = np.linspace(100., 150., len(close))
    ref = prepare_targets(close)
    for key in ("v5_volatility", "v5_confirmed", "v5_no_rsi_macd"):
        targets = ref[key][0]
        assert all(sum(row.get(x, 0.) for x in TRIPLE) == 0 for row in targets)
        assert all(0 <= sum(row.values()) <= 1. + 1e-9 for row in targets)
    changed = close.copy()
    changed.iloc[300:] *= 1.5
    alt = prepare_targets(changed)
    for k in ("v2_threshold", "v4_recovery_no3x",
              "v5_volatility", "v5_no_rsi_macd"):
        assert ref[k][0][:290] == alt[k][0][:290]


def test_invalid_overallocation_and_dates_fail():
    opened, close = bars()
    t = [{BASE[0]: .8, BASE[1]: .6}] * len(close)
    with pytest.raises(ValueError, match="Invalid target"):
        run_strategy(opened, close, t)
    with pytest.raises(ValueError):
        run_strategy(opened.iloc[:-1], close, [{}] * len(close))


def test_recent_regime_includes_actual_final_date():
    dates = pd.bdate_range("2023-01-03", "2026-11-20")
    curve = pd.Series(np.linspace(100, 125, len(dates)), index=dates)
    result = regime_metrics(curve)
    assert result["recent_2023_onward"]["end"] == "2026-11-20"
