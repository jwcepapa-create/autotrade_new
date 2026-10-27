"""Experimental G2 overlays; research only. Uses completed daily bars and next-session open."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_g_multisignal import signal, indicators, simulate
from research_defense_overlay import metrics, STRESS
from research_strategy_lab import COST
from tiingo_data import TICKERS, adjusted, validate_values

def variants(x):
    base_buy, base_sell = signal(x, 2)
    close = x.Close.astype(float)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    slope50 = ma50 > ma50.shift(10)
    # G2-MA: oversold entries only in an established intermediate uptrend.
    yield "G2_MA50", base_buy & slope50 & (close >= ma50 * .98), base_sell | ((close < ma50) & (~slope50))
    # G2-RECLAIM: tolerate 20-day breaks, buy after a confirmed 50-day reclaim.
    reclaim = (close > ma50) & (close.shift(1) <= ma50.shift(1)) & slope50
    recent = base_buy.rolling(10, min_periods=1).max().astype(bool)
    yield "G2_RECLAIM", base_buy & slope50 | (recent & reclaim), base_sell | ((close < ma50 * .97) & (~slope50))
    # G2-TREND: suppress oversold entries below a falling 200-day average.
    trend = (close > ma200) & (ma200 > ma200.shift(20))
    yield "G2_TREND200", base_buy & trend, base_sell | ((close < ma50) & (~slope50))
    # G5-BREAKOUT: price breaks prior 60-session resistance on high relative volume.
    # Shifted extrema are known before current close; current signal executes next open.
    resistance = x.High.rolling(60, min_periods=60).max().shift(1)
    volume_mean = x.Volume.rolling(20, min_periods=20).mean().shift(1)
    breakout = (close > resistance * 1.005) & (x.Volume > 1.5 * volume_mean)
    hold_trend = (ma50 > ma200) & slope50
    yield "G5_BREAKOUT", breakout & hold_trend, (close < ma50) & (~slope50)
    # G5-RETEST: a breakout occurred in past 10 sessions and price retests the old ceiling.
    level = resistance.where(breakout).ffill()
    age = pd.Series(np.where(breakout, np.arange(len(x)), np.nan), index=x.index).ffill()
    age = pd.Series(np.arange(len(x)), index=x.index) - age
    retest = age.between(1, 10) & (x.Low <= level * 1.015) & (close >= level) & hold_trend
    yield "G5_RETEST", retest, (close < ma50) & (~slope50)

def run(x, buy, sell, capital=10_000_000):
    cash=float(capital); shares=0.; orders=0; curve=[]
    buy=buy.fillna(False); sell=sell.fillna(False)
    for i in range(1,len(x)):
        op=float(x.Open.iloc[i]); cl=float(x.Close.iloc[i])
        if not np.isfinite(op) or op<=0 or not np.isfinite(cl) or cl<=0:
            raise ValueError("invalid execution price")
        if shares and bool(sell.iloc[i-1]):
            cash += shares*op*(1-COST);shares=0.;orders+=1
        elif not shares and bool(buy.iloc[i-1]):
            shares=cash/(op*(1+COST));cash=0.;orders+=1
        curve.append((x.index[i],cash+shares*cl))
    return pd.Series(dict(curve)), orders

def main():
    out={"research_only":True,"comparison":"per-ticker 10M KRW independent capital, not F70 common-capital portfolio","execution":"completed close signal; next open fill; 15bps each side","results":{}}
    for ticker in TICKERS:
        raw=pd.read_csv(Path("data")/f"{ticker}.csv",index_col="Date",parse_dates=True)
        validate_values(raw); x=adjusted(raw)
        out["results"][ticker]={}
        strategies=[("G2_BASE", *signal(x,2)), *list(variants(x))]
        for name,buy,sell in strategies:
            curve,orders=run(x,buy,sell)
            out["results"][ticker][name]={"full":metrics(curve),"orders":orders,"stress":{k:metrics(curve.loc[a:b]) for k,(a,b) in STRESS.items()}}
    p=Path("results/g_improvements.json");p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(p.read_text())
if __name__=="__main__":main()
