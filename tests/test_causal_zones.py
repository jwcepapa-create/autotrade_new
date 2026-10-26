import numpy as np
import pandas as pd
from mts_v03.causal_zones import zones_asof,retest_signal

def bars(n=120):
    idx=pd.bdate_range("2020-01-01",periods=n)
    v=100+10*np.sin(np.arange(n)*.35)
    return pd.DataFrame({"Open":v,"High":v+1,"Low":v-1,"Close":v,"Volume":1000.},index=idx)

def test_append_only_invariance():
    df=bars()
    for n in (45,65,90,110):
        assert zones_asof(df,df.index[n-1])==zones_asof(df.iloc[:n],df.index[n-1])
        assert retest_signal(df,df.index[n-1])==retest_signal(df.iloc[:n],df.index[n-1])

def test_no_unconfirmed_recent_pivots():
    df=bars()
    z=zones_asof(df,df.index[-1])
    assert all(p.available_at<=str(df.index[-1]) for p in z)
