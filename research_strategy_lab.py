"""Frozen, reproducible research evaluation; no broker integration or orders."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tiingo_data import TICKERS, adjusted, validate_values
from mts_v03.breakout import run
from mts_v03.b_knobs import simulate_exit
from mts_v03.exit_experiments import simulate

COST=.0015
SPLITS={"train":("2016-01-01","2021-12-31"),
        "validation":("2022-01-01","2023-12-31"),
        "holdout":("2024-01-01","2099-12-31")}

def evaluate_curve(curve, initial=10_000_000):
    if len(curve)<2: return {"return_pct":0.,"mdd_pct":0.,"cagr_pct":0.}
    curve=pd.Series(curve).astype(float)
    years=max((curve.index[-1]-curve.index[0]).days/365.25,1/365.25)
    return {"return_pct":round((curve.iloc[-1]/initial-1)*100,4),
            "mdd_pct":round((curve/curve.cummax()-1).min()*100,4),
            "cagr_pct":round(((curve.iloc[-1]/initial)**(1/years)-1)*100,4)}

def signals_momentum(x, lookback=252, vol_target=.15, vol_window=63):
    c=x.Close
    past=c.pct_change(lookback)
    realized=c.pct_change().rolling(vol_window).std()*np.sqrt(252)
    # Daily decisions known only at close, filled at following open.
    return (past>0).astype(float)*(vol_target/realized.replace(0,np.nan)).clip(upper=1).fillna(0)

def signals_hybrid(x, lookback=55, atr_stop=3.):
    # Causal daily Donchian breakout + ATR trailing stop; structural C is separately audited.
    c=x.Close; high=x.High.rolling(lookback).max().shift(1)
    low=x.Low.rolling(lookback).min().shift(1)
    prevclose=c.shift(1)
    tr=pd.concat([x.High-x.Low,(x.High-prevclose).abs(),(x.Low-prevclose).abs()],axis=1).max(axis=1)
    atr=tr.rolling(20).mean()
    state=0.; peak=None; out=[]
    for i in range(len(x)):
        if i<lookback or not np.isfinite(atr.iloc[i]):
            out.append(0.); continue
        if state==0 and c.iloc[i]>high.iloc[i]:
            state=1.;peak=float(c.iloc[i])
        elif state:
            peak=max(peak,float(c.iloc[i]))
            if c.iloc[i]<max(float(low.iloc[i]),peak-atr_stop*float(atr.iloc[i])):
                state=0.;peak=None
        out.append(state)
    return pd.Series(out,index=x.index)

def signals_c_daily(x, left=3, right=3, confirm=9):
    """C daily causal prototype: only pivot highs confirmed by right+reaction bars."""
    h=x.High.to_numpy(float); l=x.Low.to_numpy(float); cl=x.Close.to_numpy(float)
    n=len(x); available=np.full(n,np.nan)
    delay=max(right,confirm)
    for i in range(left,n-delay):
        if h[i]>=np.max(h[i-left:i+right+1]):
            available[i+delay]=h[i]
    resistance=pd.Series(available,index=x.index).ffill().shift(1)
    breakout=(x.Close>resistance)&(x.Close.shift(1)<=resistance.shift(1))
    # Enter on retest after a confirmed breakout; only completed bars used.
    level=resistance.where(breakout).ffill()
    retest=(x.Low<=level)&(x.Close>level)&(breakout.shift(1).fillna(False))
    exit_signal=x.Close<x.Close.rolling(20).mean()
    state=0.; out=[]
    for enter,leave in zip(retest.fillna(False),exit_signal.fillna(False)):
        if state and leave:state=0.
        elif not state and enter:state=1.
        out.append(state)
    return pd.Series(out,index=x.index)

def execute_target(x, targets, initial=10_000_000):
    """Next-open rebalance to target exposure; account for costs and overnight gaps."""
    cash=float(initial); shares=0.; curve=[]; orders=0; turnover=0.
    for i in range(1,len(x)):
        op=float(x.Open.iloc[i]); cl=float(x.Close.iloc[i])
        target=float(targets.iloc[i-1])
        if not np.isfinite(op) or op<=0: raise ValueError("Invalid open")
        equity_open=cash+shares*op
        desired=equity_open*target/op
        delta=desired-shares
        if abs(delta)>1e-8:
            # Pay fee+slippage on gross, including buy. Prevent negative cash.
            if delta>0:
                affordable=max(cash,0)/(op*(1+COST))
                delta=min(delta,affordable)
            cash-=delta*op+abs(delta)*op*COST
            shares+=delta; orders+=1; turnover+=abs(delta)*op
        curve.append((x.index[i],cash+shares*cl))
    return pd.Series(dict(curve)).sort_index(),orders,turnover/initial

def evaluate_ticker(ticker,raw):
    x=adjusted(raw)
    baseline=run(x)
    replicate=simulate_exit(x)
    edge=simulate(x,"repeat")
    parity=all(np.isclose(baseline[k],replicate[{"total_return":"total_return","max_drawdown":"max_drawdown"}[k]],atol=1e-10) for k in ("total_return","max_drawdown"))
    parity=parity and len(baseline["trades"])==replicate["orders"]
    edge_parity=bool(np.isclose(baseline["total_return"],edge["total_return"],atol=1e-10) and len(baseline["trades"])==len(edge["orders"]))
    if not parity: raise AssertionError(f"Baseline parity failure: {ticker}")
    trials=[]
    for f5,f20,cool in [(0.3,0.5,0),(0.1,0.3,0),(0.2,0.5,0),(0.3,0.5,3),(0.3,0.5,5),(0.2,0.3,3)]:
        r=simulate_exit(x,ma5_fraction=f5,ma20_fraction=f20,cooldown=cool)
        trials.append({"f5":f5,"f20":f20,"cooldown":cool,
                       "return_pct":round(r["total_return"]*100,4),
                       "mdd_pct":round(r["max_drawdown"]*100,4),
                       "orders":r["orders"],"turnover":round(r["turnover"],3)})
    models={}
    for name,targets in [("D_momentum",signals_momentum(x)),("E_breakout_ATR",signals_hybrid(x)),("C_daily_prototype",signals_c_daily(x))]:
        curve,orders,turnover=execute_target(x,targets)
        models[name]={"full":evaluate_curve(curve),"orders":orders,"turnover":round(turnover,4),
                      "periods":{}}
        for label,(start,end) in SPLITS.items():
            segment=curve.loc[start:end]
            if len(segment)>1:
                models[name]["periods"][label]=evaluate_curve(segment,initial=float(segment.iloc[0]))
    return {"ticker":ticker,"data":{"start":str(raw.index[0].date()),
             "end":str(raw.index[-1].date()),"bars":len(raw),
             "sha256":hashlib.sha256(raw.to_csv().encode()).hexdigest()},
            "baseline_parity":parity,"legacy_exit_experiment_parity":edge_parity,"B_knobs":trials,"models":models}

def main():
    results=[]
    for ticker in TICKERS:
        raw=pd.read_csv(Path("data")/f"{ticker}.csv",index_col="Date",parse_dates=True)
        validate_values(raw)
        results.append(evaluate_ticker(ticker,raw))
    out=Path("results/strategy_lab.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"research_only":True,"no_orders":True,
        "cost_each_side":COST,"note":"No FX conversion; independent ticker sleeves; not a combined portfolio; split return values are marked to market, not independently reinitialized trades.",
        "results":results},indent=2))
    print(out.read_text())

if __name__=="__main__":main()
