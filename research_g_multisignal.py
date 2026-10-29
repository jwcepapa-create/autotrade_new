"""G research: causal support/resistance zones + volume + RSI divergence + MACD + stochastic + weekly trend.
Research only. Parameters are predeclared hypotheses, not optimized on holdout.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tiingo_data import TICKERS, adjusted, validate_values
from research_defense_overlay import metrics, STRESS
from research_strategy_lab import COST

def indicators(x):
    c=x.Close.astype(float); d=c.diff()
    up=d.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    down=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=up/down.replace(0,np.nan)
    rsi=(100-100/(1+rs)).where(down.ne(0),100)
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    hist=macd-macd.ewm(span=9,adjust=False).mean()
    low=x.Low.rolling(14).min(); high=x.High.rolling(14).max()
    k=(100*(c-low)/(high-low).replace(0,np.nan)).rolling(3).mean()
    stoch=k.rolling(3).mean()
    # Completed prior Friday bar only, available from next trading session.
    w=c.resample("W-FRI").last()
    weekly=(w>w.rolling(10,min_periods=10).mean()).shift(1)
    weekly=weekly.reindex(c.index,method="ffill").fillna(False).astype(bool)
    return rsi,hist,stoch,weekly

def zones(x,lookback=120,tolerance=.012,min_separation=5):
    """Select repeatedly visited horizontal pivot zones, using ONLY previously confirmed pivots.
    Avoid raw extrema; count separated touches and reward reactions.
    """
    h=x.High.to_numpy(float);l=x.Low.to_numpy(float);c=x.Close.to_numpy(float)
    n=len(x);support=np.full(n,np.nan);resistance=np.full(n,np.nan)
    confirmed=[]
    for i in range(n):
        # Pivot at i-3 is confirmed only after three subsequent completed bars.
        p=i-3
        if p>=3:
            if l[p]<=min(l[p-3:p+4]):confirmed.append((p,float(l[p]),"S"))
            if h[p]>=max(h[p-3:p+4]):confirmed.append((p,float(h[p]),"R"))
        if i<lookback:continue
        for side,out in (("S",support),("R",resistance)):
            pts=[(j,v) for j,v,t in confirmed if t==side and i-lookback<=j<=i-3]
            candidates=[]
            for _,v in pts:
                cluster=[(j,z) for j,z in pts if abs(z/v-1)<=tolerance]
                separated=[];last=-999
                for j,z in cluster:
                    if j-last>=min_separation:separated.append((j,z));last=j
                if len(separated)>=2:
                    level=float(np.median([z for _,z in separated]))
                    # Prefer closest meaningful zone on relevant side of last close.
                    if (side=="S" and level<=c[i]) or (side=="R" and level>=c[i]):
                        candidates.append((len(separated),-abs(level/c[i]-1),level))
            if candidates:
                out[i]=max(candidates)[2]
    # At close i, zone is built from confirmed pivots. Use on next open.
    return pd.Series(support,index=x.index),pd.Series(resistance,index=x.index)

def divergence(x,rsi,left=3,right=3,lookback=75):
    """Bull/bear divergences from paired confirmed price pivots; signal only at confirmation."""
    lo=x.Low.to_numpy(float);hi=x.High.to_numpy(float);rv=rsi.to_numpy(float)
    bull=np.zeros(len(x),bool);bear=np.zeros(len(x),bool); lows=[];highs=[]
    for i in range(left+right,len(x)):
        p=i-right
        if np.isfinite(rv[p]):
            if lo[p]<=min(lo[p-left:p+right+1]):
                prev=next(((j,v,r) for j,v,r in reversed(lows) if p-j<=lookback),None)
                if prev is not None and lo[p]<prev[1] and rv[p]>prev[2]:bull[i]=True
                lows.append((p,lo[p],rv[p]))
            if hi[p]>=max(hi[p-left:p+right+1]):
                prev=next(((j,v,r) for j,v,r in reversed(highs) if p-j<=lookback),None)
                if prev is not None and hi[p]>prev[1] and rv[p]<prev[2]:bear[i]=True
                highs.append((p,hi[p],rv[p]))
    return pd.Series(bull,index=x.index).rolling(5,min_periods=1).max().astype(bool),pd.Series(bear,index=x.index).rolling(5,min_periods=1).max().astype(bool)

def signal(x,variant):
    rsi,hist,stoch,weekly=indicators(x)
    sup,res=zones(x)
    bull,bear=divergence(x,rsi)
    r_up=(rsi.diff()>0)&(rsi.shift(1).diff()<=0)
    r_dn=(rsi.diff()<0)&(rsi.shift(1).diff()>=0)
    m_up=(hist.diff()>0)&(hist.shift(1).diff()<=0)
    m_dn=(hist.diff()<0)&(hist.shift(1).diff()>=0)
    s_up=(stoch.diff()>0)&(stoch.shift(1).diff()<=0)
    s_dn=(stoch.diff()<0)&(stoch.shift(1).diff()>=0)
    # Within 3 sessions of a reversal, count at most one vote per indicator.
    votes_buy=sum(z.rolling(3,min_periods=1).max().fillna(0).astype(int) for z in (r_up,m_up,s_up))
    votes_sell=sum(z.rolling(3,min_periods=1).max().fillna(0).astype(int) for z in (r_dn,m_dn,s_dn))
    buy=votes_buy>=2;sell=votes_sell>=2
    if variant>=2:
        buy &= ((rsi<=35)|(rsi.shift(1)<=30))
        sell &= ((rsi>=75)|(rsi.shift(1)>=80))
    if variant>=3:
        buy &= bull
        sell &= bear
    if variant>=4:
        buy &= weekly & (hist.diff()>0)
        sell &= (~weekly) & (hist.diff()<0)
    if variant>=5:
        # Prior confirmed zone, with proximity or decisive volume-backed breakout.
        avgvol=x.Volume.rolling(20,min_periods=20).mean().shift(1)
        ratio=x.Volume/avgvol.replace(0,np.nan)
        near_support=(abs(x.Close/sup-1)<=.02)
        near_resistance=(abs(x.Close/res-1)<=.02)
        breakout=(x.Close>res*1.005)&(x.Close.shift(1)<=res.shift(1)*1.005)&(ratio>=1.5)
        breakdown=(x.Close<sup*.995)&(ratio>=1.5)
        buy &= (near_support | breakout) & (ratio>=1.0)
        sell |= (breakdown & (~weekly))
        sell &= near_resistance | breakdown | bear
    return buy.fillna(False),sell.fillna(False)

def simulate(x,variant,capital=10_000_000):
    buy,sell=signal(x,variant)
    cash=float(capital);shares=0.;curve=[];orders=0
    # Decision on completed close i-1, execute at open i.
    for i in range(1,len(x)):
        px=float(x.Open.iloc[i])
        if not np.isfinite(px) or px<=0:raise ValueError("invalid open")
        if shares>0 and bool(sell.iloc[i-1]):
            cash+=shares*px*(1-COST);shares=0.;orders+=1
        elif shares==0 and bool(buy.iloc[i-1]):
            shares=cash/(px*(1+COST));cash=0.;orders+=1
        curve.append((x.index[i],cash+shares*float(x.Close.iloc[i])))
    return pd.Series(dict(curve)),orders

def main():
    out={"research_only":True,"method":"per-ticker long/cash, completed bars, next-open, 15bps each side; not common-capital portfolio","results":{}}
    for t in TICKERS:
        raw=pd.read_csv(Path("data")/f"{t}.csv",index_col="Date",parse_dates=True)
        validate_values(raw);x=adjusted(raw)
        out["results"][t]={}
        for v in range(1,6):
            curve,orders=simulate(x,v)
            out["results"][t][f"G{v}"]={"full":metrics(curve),"orders":orders,"stress":{k:metrics(curve.loc[a:b]) for k,(a,b) in STRESS.items()}}
    p=Path("results/g_multisignal.json");p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out,indent=2));print(p.read_text())
if __name__=="__main__":main()
