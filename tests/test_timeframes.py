import numpy as np
import pandas as pd
from mts_v03.timeframes import completed_timeframe,intraday_status

def sample():
    idx=pd.bdate_range("2020-01-01",periods=200)
    v=np.arange(len(idx),dtype=float)+100
    return pd.DataFrame({"Open":v,"High":v+1,"Low":v-1,"Close":v,"Volume":1000.},index=idx)

def test_no_future_append():
    x=sample();asof=x.index[140]
    for freq in ("W","M"):
        pd.testing.assert_frame_equal(completed_timeframe(x,freq,asof),
                                      completed_timeframe(x.iloc[:141],freq,asof))

def test_weekly_no_partial_bar():
    x=sample()
    z=completed_timeframe(x,"W",pd.Timestamp("2020-03-11"))
    assert z.index[-1]==pd.Timestamp("2020-03-06")

def test_intraday_not_faked():
    assert intraday_status()["status"]=="blocked"
