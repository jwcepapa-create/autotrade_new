import numpy as np
import pandas as pd
import pytest
import backtest


def fixture(monkeypatch):
    dates = pd.bdate_range('2020-01-01', periods=30)
    d = pd.DataFrame({'Open':100.,'Close':99.,'High':102.,'Low':98.,'Volume':1e6,'ema21':100.}, index=dates)
    d.loc[dates[1], 'Close'] = 101.
    d.loc[dates[2], ['Open','Close','High','Low']] = [100.,102.,105.,99.]
    d.loc[dates[3], ['Open','Close','High','Low']] = [108.,101.,110.,95.]
    monkeypatch.setattr(backtest, 'indicators', lambda x: x.copy())
    monkeypatch.setattr(backtest, 'screen', lambda x: pd.Series(True,index=x.index))
    return d


def test_completed_signal_next_open_gap_target_before_later_low(monkeypatch):
    d = fixture(monkeypatch)
    trades = backtest.swing('QQQ', d)
    t = trades[0]
    assert t['entry_date'] == str(d.index[2].date())
    assert t['entry'] == 100. and t['stop'] == 98.
    assert t['exit'] == 108. and t['exit_reason'] == 'gap_target'


def test_both_hit_without_opening_gap_stops_first(monkeypatch):
    d = fixture(monkeypatch)
    d.iloc[2, d.columns.get_indexer(['Open','Close','High','Low'])] = [100.,101.,107.,97.]
    t = backtest.swing('QQQ', d)[0]
    assert t['exit'] == 98. and t['exit_reason'] == 'stop'


def test_first_bar_cannot_wrap_to_last_bar(monkeypatch):
    d = fixture(monkeypatch)
    d.iloc[0, d.columns.get_loc('Close')] = 101.
    trades = backtest.swing('QQQ', d)
    assert all(t['entry_date'] != str(d.index[1].date()) for t in trades)


def test_indicators_are_causal():
    rng = np.random.default_rng(4)
    close = 100 + rng.normal(size=220).cumsum()
    d = pd.DataFrame({'Open':close,'Close':close,'High':close+2,'Low':close-2,'Volume':1e6},
                     index=pd.bdate_range('2020-01-01', periods=220))
    full = backtest.indicators(d)
    pd.testing.assert_frame_equal(full.iloc[:180], backtest.indicators(d.iloc[:180]))
    changed = d.copy(); changed.iloc[180:, changed.columns.get_loc('Close')] *= 2
    pd.testing.assert_frame_equal(full.iloc[:180], backtest.indicators(changed).iloc[:180])


def test_intraday_ignores_same_day_daily_signal(monkeypatch):
    dates = pd.to_datetime(['2020-01-01','2020-01-02'])
    daily = pd.DataFrame({'eligible':[False,True]}, index=dates)
    bars = pd.DataFrame({'Open':100.,'Close':101.,'High':102.,'Low':99.,'Volume':1e6},
                        index=pd.date_range('2020-01-02 14:30',periods=78,freq='5min',tz='UTC'))
    monkeypatch.setattr(backtest, 'indicators', lambda x:x)
    monkeypatch.setattr(backtest, 'screen', lambda x:x.eligible)
    assert backtest.intraday('QQQ',bars,daily) == []
