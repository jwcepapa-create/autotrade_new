"""G3.0 research: use higher-timeframe trend for holding and re-entry, not entry veto.
Research only; never connects to a broker. All signals use completed daily bars.
"""
import json
from pathlib import Path
import pandas as pd
from tiingo_data import TICKERS, adjusted, validate_values
from research_g_multisignal import signal, indicators, zones
from research_g29_multitimeframe import completed_frame, evaluate, CAPITAL
from research_strategy_lab import COST

def variants(x):
    close=x.Close.astype(float)
    week=completed_frame(close,"W-FRI",10,40)
    month=completed_frame(close,"ME",10,20)
    rsi,hist,_,_=indicators(x)
    base_buy,base_sell=signal(x,2)
    ma20=close.rolling(20,min_periods=20).mean()
    ma50=close.rolling(50,min_periods=50).mean()
    # Regime permits holding through daily overbought reversals.
    strong=week.up & (month.up | month.macd_up)
    broken=(week.weak & month.weak) | ((close<ma50*.97)&(~week.up)&(hist.diff()<0))
    # Weekly pullback recovery and volume-confirmed resistance breakout.
    prior_high=x.High.rolling(60,min_periods=60).max().shift(1)
    vol_mean=x.Volume.rolling(20,min_periods=20).mean().shift(1)
    breakout=(close>prior_high*1.005)&(close.shift(1)<=prior_high.shift(1)*1.005)&(x.Volume>=1.5*vol_mean)
    reclaim=(close>ma20)&(close.shift(1)<=ma20.shift(1))&(hist.diff()>0)
    # A: original G2 entries; delay overbought exits until higher-timeframe weakness.
    yield "G3_0_A_TREND_HOLD",base_buy,(base_sell & ~strong)|broken
    # B: original entries plus non-overbought re-entry during established trends.
    yield "G3_0_B_REENTRY",base_buy|(strong & (reclaim|breakout)),(base_sell & ~strong)|broken
    # C: retain a risk-off exit on unusually weak daily momentum even in strong regime.
    sharp_break=(close<ma50*.95)&(hist<0)&(hist.diff()<0)
    yield "G3_0_C_BALANCED",base_buy|(strong & (reclaim|breakout)),(base_sell & ~strong)|broken|sharp_break

def main():
    out={"research_only":True,"capital_per_ticker":CAPITAL,"cost_per_side":COST,
         "method":"G2 base entry preserved; weekly/monthly completed-bar signals for holding and re-entry; next daily open fills; independent 10m KRW per ticker, not F70 shared capital",
         "results":{}}
    for ticker in TICKERS:
        raw=pd.read_csv(Path("data")/f"{ticker}.csv",index_col="Date",parse_dates=True)
        validate_values(raw);x=adjusted(raw)
        bh=pd.Series(False,index=x.index);bh.iloc[0]=True
        empty=pd.Series(False,index=x.index)
        out["results"][ticker]={
            "BUY_HOLD":evaluate(x,bh,empty),
            "G2_BASE":evaluate(x,*signal(x,2)),
        }
        for name,buy,sell in variants(x):
            out["results"][ticker][name]=evaluate(x,buy.fillna(False),sell.fillna(False))
    p=Path("results/g30_trend_hold_reentry.json");p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(p.read_text())

if __name__=="__main__":main()
