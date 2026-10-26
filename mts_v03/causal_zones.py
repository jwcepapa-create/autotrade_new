"""Daily-only causal support/resistance prototype; NOT the complete v0.2 multi-timeframe engine."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Zone:
    center: float
    low: float
    high: float
    score: float
    touches: int
    available_at: str

def zones_asof(df, asof, left=3, right=3, reaction_bars=9, tolerance=.0075):
    """Only completed daily bars through asof; pivot and reaction confirmation delayed."""
    if min(left,right) < 1 or reaction_bars < right:
        raise ValueError("Invalid confirmation windows")
    x=df.loc[:asof]
    if not x.index.is_monotonic_increasing or not x.index.is_unique:
        raise ValueError("Expected unique chronological dates")
    if len(x) < 30:
        return []
    h,l,c,v=(x[k].to_numpy(float) for k in ("High","Low","Close","Volume"))
    tr=np.maximum(h-l,np.maximum(abs(h-np.r_[np.nan,c[:-1]]),abs(l-np.r_[np.nan,c[:-1]])))
    tr[0]=h[0]-l[0]
    atr=pd.Series(tr).rolling(14).mean().to_numpy()
    vol20=pd.Series(v).rolling(20).mean().to_numpy()
    delay=max(right,reaction_bars)
    obs=[]
    for i in range(max(left,19),len(x)-delay):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        for side,series in (("high",h),("low",l)):
            w=series[i-left:i+right+1]
            if not np.isfinite(w).all():
                continue
            if not (series[i]>=w.max() if side=="high" else series[i]<=w.min()):
                continue
            reaction=((series[i]-l[i+1:i+reaction_bars+1].min())/atr[i] if side=="high"
                      else (h[i+1:i+reaction_bars+1].max()-series[i])/atr[i])
            rv=v[i]/vol20[i] if vol20[i]>0 else 1.
            strength=20+min(rv,8)*4+min(max(reaction,0),8)*2.5
            obs.append((float(series[i]),float(strength),str(x.index[i+delay])))
    if not obs:
        return []
    price=float(c[-1]); tol=price*tolerance
    groups=[]
    for o in sorted(obs):
        if not groups:
            groups.append([o])
        else:
            g=groups[-1]
            center=np.average([t[0] for t in g],weights=[t[1] for t in g])
            (g if abs(o[0]-center)<=tol else groups.append([o]) or groups[-1]).append(o) if abs(o[0]-center)<=tol else None
            if abs(o[0]-center)>tol:
                continue
    zones=[]
    for g in groups:
        weights=[o[1] for o in g]
        center=float(np.average([o[0] for o in g],weights=weights))
        zones.append(Zone(center,min(o[0] for o in g)-tol*.3,max(o[0] for o in g)+tol*.3,
                          min(100,float(np.mean(weights)*.65+len(g)*2.2)),len(g),
                          max(o[2] for o in g)))
    return zones

def retest_signal(df, asof, min_score=25):
    """Yesterday breaks an existing resistance; today retests; next-open entry only."""
    x=df.loc[:asof]
    if len(x)<40:
        return False
    history=x.iloc[:-2]
    zones=zones_asof(history,history.index[-1])
    prior,today=x.iloc[-2],x.iloc[-1]
    for z in zones:
        if z.center<=float(history.Close.iloc[-1]) or z.score<min_score:
            continue
        if history.Close.iloc[-1]<=z.high and prior.Close>z.high and today.Low<=z.high and today.Close>z.high:
            return True
    return False
