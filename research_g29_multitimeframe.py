"""G2.9 multi-timeframe trend research; no broker connection.
Weekly/monthly signals use only completed prior bars, aligned to daily sessions.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tiingo_data import TICKERS, adjusted, validate_values
from research_g_multisignal import signal, zones, indicators
from research_g_improvements import run
from research_defense_overlay import metrics, STRESS
from research_strategy_lab import COST

CAPITAL=10_000_000

def completed_frame(close, rule, fast, slow):
    # Resampled bar becomes usable only after the NEXT completed higher-timeframe bar
    # boundary; conservative shift(1) avoids lookahead on incomplete week/month.
    bars=close.resample(rule).last()
    ma_fast=bars.rolling(fast,min_periods=fast).mean()
    ma_slow=bars.rolling(slow,min_periods=slow).mean()
    macd=bars.ewm(span=12,adjust=False).mean()-bars.ewm(span=26,adjust=False).mean()
    hist=macd-macd.ewm(span=9,adjust=False).mean()
    state=pd.DataFrame({
        "up":(bars>ma_fast)&(ma_fast>ma_slow)&(ma_fast>ma_fast.shift(2)),
        "macd_up":(hist>hist.shift(1)),
        "weak":(bars<ma_fast)&(hist<hist.shift(1)),
    }).shift(1)
    return state.reindex(close.index,method="ffill").fillna(False).astype(bool)

def signals(x):
    c=x.Close.astype(float)
    week=completed_frame(c,"W-FRI",10,40)
    month=completed_frame(c,"ME",10,20)
    rsi,hist,stoch,_=indicators(x)
    ma20=c.rolling(20,min_periods=20).mean()
    ma50=c.rolling(50,min_periods=50).mean()
    trend=week.up & (month.up | (month.macd_up & week.macd_up))
    rising=(hist.diff()>0)&(rsi.diff()>0)
    pullback=(x.Low<=ma20*1.015)&(c>ma20)&(c>ma50)&rising
    # Resistance zones are confirmed from multiple prior pivot touches.
    _,res=zones(x)
    prior=res.shift(1)
    volbase=x.Volume.rolling(20,min_periods=20).mean().shift(1)
    breakout=(c>prior*1.005)&(c.shift(1)<=prior.shift(1)*1.005)&(x.Volume>=1.5*volbase)
    # Strong weekly/monthly trends should not be sold merely on daily RSI > 80.
    exit_trend=(week.weak & ~month.up) | (month.weak & week.weak)
    exit_price=(c<ma50*.97)&(hist.diff()<0)&(~week.up)
    sell=(exit_trend | exit_price).fillna(False)
    yield "G2_9_A_PULLBACK", (trend & pullback).fillna(False),sell
    yield "G2_9_B_BREAKOUT", (trend & breakout & week.macd_up).fillna(False),sell
    yield "G2_9_C_COMBINED", (trend & (pullback | breakout)).fillna(False),sell

def evaluate(x,buy,sell):
    curve,orders=run(x,buy,sell,CAPITAL)
    return {"full":metrics(curve),"orders":orders,
        "stress":{k:metrics(curve.loc[a:b]) for k,(a,b) in STRESS.items()},
        "segments":{k:metrics(curve.loc[a:b]) for k,(a,b) in {
            "train_2016_2021":("2016-01-01","2021-12-31"),
            "validation_2022_2023":("2022-01-01","2023-12-31"),
            "holdout_2024_2026":("2024-01-01","2026-12-31")}.items()}}

def main():
    out={"research_only":True,"capital_per_ticker":CAPITAL,
        "cost_per_side":COST,
        "method":"Independent per-ticker 10m KRW, adjusted Tiingo daily OHLCV, completed higher-timeframe bars shifted one period, next-open fills; no FX, not F70 shared capital",
        "results":{}}
    for ticker in TICKERS:
        raw=pd.read_csv(Path("data")/f"{ticker}.csv",index_col="Date",parse_dates=True)
        validate_values(raw);x=adjusted(raw)
        baseline=signal(x,2)
        buyhold=pd.Series(False,index=x.index);buyhold.iloc[0]=True
        empty=pd.Series(False,index=x.index)
        out["results"][ticker]={"BUY_HOLD":evaluate(x,buyhold,empty),
                                  "G2_BASE":evaluate(x,*baseline)}
        for name,buy,sell in signals(x):
            out["results"][ticker][name]=evaluate(x,buy,sell)
    path=Path("results/g29_multitimeframe.json");path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(path.read_text())

if __name__=="__main__":
    main()
