"""Second-generation causal G research: moving-average pullbacks and retests.

Research only: no broker calls, no live orders. All decisions use completed bars;
fills occur at the following session open. Predeclared rules, no fitting.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_g_multisignal import indicators, signal
from research_g_improvements import run
from research_defense_overlay import metrics, STRESS
from tiingo_data import TICKERS, adjusted, validate_values

CAPITAL=10_000_000

def candidate_signals(x):
    c=x.Close.astype(float)
    ma20=c.rolling(20,min_periods=20).mean()
    ma50=c.rolling(50,min_periods=50).mean()
    ma200=c.rolling(200,min_periods=200).mean()
    atr=pd.concat([x.High-x.Low,(x.High-c.shift(1)).abs(),(x.Low-c.shift(1)).abs()],axis=1).max(axis=1).rolling(14).mean()
    rsi,hist,stoch,weekly=indicators(x)
    up50=ma50>ma50.shift(10)
    up200=ma200>ma200.shift(20)
    improving=(rsi.diff()>0)&(hist.diff()>0)
    recovering=(c>c.shift(1)) & improving
    # Only earlier completed closes define a pullback; don't require RSI<30.
    recent_pullback=(c.shift(1)<ma20.shift(1)).rolling(10,min_periods=1).max().fillna(0).astype(bool)
    touch50=(x.Low<=ma50+atr*.5)&(c>=ma50-atr*.5)
    buy27=up50 & (ma50>ma200) & touch50 & recovering & (rsi<65)
    sell27=((c<ma50-atr)&(~up50)) | ((c<ma200)&(~up200))
    yield "G2_7_PULLBACK",buy27,sell27

    reclaim50=(c>ma50)&(c.shift(1)<=ma50.shift(1))
    reclaim20=(c>ma20)&(c.shift(1)<=ma20.shift(1))
    buy28=(up50 & (reclaim50 | (reclaim20 & recent_pullback)) &
           (hist.diff()>0) & (rsi>35) & (rsi<75))
    sell28=((c<ma50-atr*.5)&(~up50)) | ((c<ma200)&(~up200))
    yield "G2_8_RECOVERY",buy28,sell28

    # Prior 60-bar high and prior 20-bar volume mean; all levels causal.
    resistance=x.High.rolling(60,min_periods=60).max().shift(1)
    volref=x.Volume.rolling(20,min_periods=20).mean().shift(1)
    breakout=(c>resistance*1.005)&(c.shift(1)<=resistance.shift(1)*1.005)&(x.Volume>=volref*1.5)
    # Track most recent breakout and its contemporaneous resistance.
    level=resistance.where(breakout).ffill()
    event_index=pd.Series(np.where(breakout,np.arange(len(x)),np.nan),index=x.index).ffill()
    age=pd.Series(np.arange(len(x)),index=x.index)-event_index
    # Revisit within 10 sessions, close back above the former ceiling.
    retest=age.between(1,10)&(x.Low<=level+atr*.5)&(c>level)&(c>c.shift(1))
    buy53=retest & up50 & (ma50>ma200)
    sell53=((c<ma50-atr)&(~up50)) | ((c<level-atr)&age.between(1,20))
    yield "G5_3_RETEST",buy53,sell53

def benchmark(x):
    # All-in on first executable open, same capital and cost as strategy run().
    b=pd.Series(False,index=x.index)
    s=b.copy()
    if len(x):b.iloc[0]=True
    return run(x,b,s,CAPITAL)

def evaluate(x,buy,sell):
    curve,orders=run(x,buy,sell,CAPITAL)
    return {"full":metrics(curve),"orders":orders,
            "stress":{k:metrics(curve.loc[a:z]) for k,(a,z) in STRESS.items()},
            "segments":{k:metrics(curve.loc[a:z]) for k,(a,z) in {
                "train":("2016-01-01","2021-12-31"),
                "validation":("2022-01-01","2023-12-31"),
                "holdout":("2024-01-01","2099-12-31")}.items()}}

def main():
    out={"research_only":True,"capital_per_ticker":CAPITAL,
         "method":"Independent per-ticker sleeves; adjusted Tiingo OHLCV; completed close/next open; 15bps each side; no FX; no shared-capital comparison with F70",
         "warning":"Stress/segment metrics are within-window mark-to-market, not restarted trading accounts; empty periods yield null; historical event windows are not holdout.",
         "results":{}}
    for ticker in TICKERS:
        raw=pd.read_csv(Path("data")/f"{ticker}.csv",index_col="Date",parse_dates=True)
        validate_values(raw)
        x=adjusted(raw)
        baseline=signal(x,2)
        baseline_bh=benchmark(x)
        out["results"][ticker]={
            "BUY_HOLD":evaluate(x,pd.Series([True]+[False]*(len(x)-1),index=x.index),pd.Series(False,index=x.index)),
            "G2_BASE":evaluate(x,*baseline)}
        for name,buy,sell in candidate_signals(x):
            out["results"][ticker][name]=evaluate(x,buy.fillna(False),sell.fillna(False))
    dest=Path("results/g_second_generation.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(dest.read_text())

if __name__=="__main__":
    main()
