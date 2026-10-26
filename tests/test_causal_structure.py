"""Adversarial point-in-time tests: no future observations or retroactive scores."""
import numpy as np
import pandas as pd
import pytest
from mts_v03.causal_structure import available_pivots, confirmed_pivots


def bars(n=100):
    idx = pd.bdate_range("2020-01-01", periods=n)
    v = 100 + 10*np.sin(np.arange(n)*.35)
    return pd.DataFrame({"Open":v, "High":v+1, "Low":v-1,
                         "Close":v, "Volume":1000}, index=idx)


def test_future_append_does_not_change_confirmed_history():
    df = bars()
    for n in (40, 55, 75, 90):
        prior = available_pivots(df, df.index[n-1])
        after = [p for p in confirmed_pivots(df) if p.available_date <= str(df.index[n-1])]
        assert prior == after


def test_reaction_confirmation_cannot_be_backdated():
    df = bars()
    pivots = confirmed_pivots(df)
    assert pivots
    for p in pivots:
        i = df.index.get_loc(pd.Timestamp(p.pivot_date))
        j = df.index.get_loc(pd.Timestamp(p.available_date))
        assert j - i >= 9
        assert p.available_date > p.pivot_date


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        confirmed_pivots(bars(), right=4, reaction_bars=3)
