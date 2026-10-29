import numpy as np
import pandas as pd
import pytest
from research_defense_overlay import target_exposure,portfolio

def bars(n=300):
    dates=pd.bdate_range("2019-01-01",periods=n)
    v=100+np.arange(n)*.1
    return pd.DataFrame({"Open":v,"High":v+1,"Low":v-1,"Close":v,"Volume":1000.},index=dates)

def test_signal_append_only():
    x=bars(300)
    a=target_exposure(x.Close)
    b=target_exposure(x.Close.iloc[:250])
    pd.testing.assert_series_equal(a.iloc[:250],b)

def test_true_hold_has_only_initial_buys():
    x=bars()
    _,execution=portfolio({"QQQ":x,"SPY":x},mode="hold")
    assert execution["orders"]==2

def test_no_negative_core():
    with pytest.raises(ValueError):
        target_exposure(bars().Close,core=-.1)

def test_flat_market_overlay_matches_hold():
    x=bars()
    a,_=portfolio({"QQQ":x,"SPY":x},mode="hold")
    b,_=portfolio({"QQQ":x,"SPY":x},mode="overlay",core=.8)
    pd.testing.assert_series_equal(a,b)
