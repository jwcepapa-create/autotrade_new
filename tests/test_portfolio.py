import numpy as np
import pandas as pd
import pytest
from portfolio import Costs, account, metrics


def prices():
    return pd.DataFrame({'Open': [100., 100., 110.], 'Close': [100., 110., 120.]},
                        index=pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06']))


def trade(exit_price=120.):
    return {'ticker': 'QQQ', 'entry_date': '2020-01-02', 'exit_date': '2020-01-06',
            'entry': 100., 'stop': 90., 'exit': exit_price, 'R': 2., 'exit_reason': 'target'}


def test_buy_hold_exact_costs_and_cash_cap():
    costs = Costs(25, 5, .3)
    equity, trades = account({'QQQ': prices()}, [], 10_000_000, 1000, costs=costs, buy_hold=True)
    units = 10_000 / (100 * 1.0005 * 1.0025)
    expected = units * 120 * .9995 * (1 - 25.3 / 10000) * 1000
    assert equity.iloc[-1] == pytest.approx(expected)
    assert trades[0]['units'] == pytest.approx(units)
    assert trades[0]['entry_debit_usd'] <= 10_000 + 1e-8


def test_future_exit_never_changes_initial_size_or_prior_nav():
    a, fills_a = account({'QQQ': prices()}, [trade()], 10000, 1, costs=Costs(0,0,0))
    b, fills_b = account({'QQQ': prices()}, [trade(50)], 10000, 1, costs=Costs(0,0,0))
    assert fills_a[0]['units'] == fills_b[0]['units'] == 5
    pd.testing.assert_series_equal(a.iloc[:-1], b.iloc[:-1])
    assert a.iloc[-1] == 10100
    assert b.iloc[-1] == 9750


def test_no_trades_and_initial_drawdown_baseline():
    equity, fills = account({'QQQ': prices()}, [], 10000, 1)
    m = metrics(equity, fills, 10000, '2020-01-01')
    assert m['trades'] == 0 and m['win_rate'] is None and m['profit_factor'] is None
    assert m['total_return'] == m['mdd'] == 0
    down = pd.Series([9000.,9500.], index=pd.to_datetime(['2020-01-02','2020-01-03']))
    assert metrics(down, [], 10000, '2020-01-01')['mdd'] == pytest.approx(-.1)


def test_profit_factor_net_of_costs():
    m = metrics(pd.Series([11000.], index=pd.to_datetime(['2021-01-01'])),
                [{'net_pnl_usd': 200}, {'net_pnl_usd': -100}, {'net_pnl_usd': 0}],
                10000, '2020-01-01')
    assert m['profit_factor'] == 2
    assert m['win_rate'] == pytest.approx(1/3)
    assert m['cagr'] == pytest.approx(1.1 ** (365.25 / 366) - 1)


def test_fixed_sleeves_do_not_reuse_another_tickers_cash():
    t = trade()
    equity, fills = account({'QQQ': prices(), 'SPY': prices()}, [t], 10000, 1, costs=Costs(0,0,0))
    assert fills[0]['units'] == 2.5
    assert equity.iloc[-1] == 10050


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'),10000])
def test_invalid_costs(value):
    with pytest.raises(ValueError):
        Costs(fee_bps=value)
