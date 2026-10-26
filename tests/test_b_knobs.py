import numpy as np
import pandas as pd
import pytest
from mts_v03.breakout import run
from mts_v03.b_knobs import simulate_exit

def bars(n=160):
    c=100+np.arange(n)*.3+6*np.sin(np.arange(n)*.22)
    return pd.DataFrame({"Open":c,"High":c+1,"Low":c-1,"Close":c,"Volume":1000+np.arange(n)*2},
                        index=pd.bdate_range("2020-01-01",periods=n))

def test_default_matches_original():
    d=bars()
    a=run(d)
    b=simulate_exit(d)
    assert b["total_return"]==pytest.approx(a["total_return"],abs=1e-10)
    assert b["max_drawdown"]==pytest.approx(a["max_drawdown"],abs=1e-10)
    assert b["orders"]==len(a["trades"])

def test_cooldown_is_valid():
    d=bars()
    a=simulate_exit(d,cooldown=3)
    assert np.isfinite(a["total_return"])
    assert 0<=a["exposure"]<=1

def test_invalid_knob():
    with pytest.raises(ValueError):
        simulate_exit(bars(),ma5_fraction=1.5)
