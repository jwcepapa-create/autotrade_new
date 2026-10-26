"""Long-hold plus defensive risk overlay, tested with next-open execution.

Research only; no orders or brokerage integration. Parameters are hypotheses,
not optimized on the holdout.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tiingo_data import TICKERS, adjusted, validate_values
from research_strategy_lab import COST

STRESS={
    "covid_crash":("2020-02-19","2020-04-30"),
    "inflation_bear":("2022-01-03","2022-12-30"),
    "recent_2024_2026":("2024-01-02","2026-10-23"),
}

def target_exposure(index_close, core=.8, fast=50, slow=200, vol_window=20,
                    vol_trigger=.35, confirmation=2, reentry=5):
    """Use only completed close data; next trading open is execution time."""
    if not 0<=core<=1:raise ValueError("core must be 0..1")
    c=index_close.astype(float)
    ma_fast=c.rolling(fast,min_periods=fast).mean()
    ma_slow=c.rolling(slow,min_periods=slow).mean()
    vol=c.pct_change().rolling(vol_window).std()*np.sqrt(252)
    risk=((c<ma_fast)&(vol>vol_trigger)) | ((c<ma_slow)&(ma_fast<ma_slow))
    # Require consecutive risk signals to reduce whipsaw, likewise reentry.
    risk_confirm=risk.rolling(confirmation,min_periods=confirmation).sum()==confirmation
    recover=((c>ma_fast)&(c>ma_slow))
    recover_confirm=recover.rolling(reentry,min_periods=reentry).sum()==reentry
    exposure=[]; defensive=False
    for danger,clear in zip(risk_confirm.fillna(False),recover_confirm.fillna(False)):
        if danger:defensive=True
        elif clear:defensive=False
        exposure.append(core if defensive else 1.)
    return pd.Series(exposure,index=c.index)

def portfolio(frames, core=.8, mode="overlay", capital=10_000_000):
    tickers=sorted(frames)
    dates=frames[tickers[0]].index
    for t in tickers[1:]:dates=dates.intersection(frames[t].index)
    if len(dates)<260:raise ValueError("Insufficient common history")
    data={t:frames[t].loc[dates] for t in tickers}
    # QQQ is a market-wide proxy; do not use individual-stock hindsight.
    market=data["QQQ"].Close
    target=target_exposure(market,core=core) if mode=="overlay" else pd.Series(1.,index=dates)
    cash=float(capital);shares={t:0. for t in tickers};curve=[];orders=0;turnover=0.
    for i in range(1,len(dates)):
        opens={t:float(data[t].Open.iloc[i]) for t in tickers}
        closes={t:float(data[t].Close.iloc[i]) for t in tickers}
        if i==1:
            # Initial equal-dollar allocation. Subsequent buy/hold never rebalances.
            budget=cash/len(tickers)
            for t in tickers:
                q=budget/(opens[t]*(1+COST))
                cash-=q*opens[t]*(1+COST);shares[t]+=q;orders+=1;turnover+=q*opens[t]
        if mode=="overlay":
            # Preserve equal shares in the baseline. Adjust only total exposure
            # by proportional liquidation/re-entry; do not force equal-weight daily.
            equity=cash+sum(shares[t]*opens[t] for t in tickers)
            invested=sum(shares[t]*opens[t] for t in tickers)
            desired=equity*float(target.iloc[i-1])
            if invested>desired+1e-7:
                ratio=desired/invested
                for t in tickers:
                    qty=shares[t]*(1-ratio)
                    if qty>1e-9:
                        cash+=qty*opens[t]*(1-COST);shares[t]-=qty
                        turnover+=qty*opens[t];orders+=1
            elif invested<desired-1e-7 and cash>0:
                # Add to existing exposures proportional to market value;
                # when all sold, restore original equal-weight allocations.
                amount=min(cash/(1+COST),desired-invested)
                weights={t:(shares[t]*opens[t]/invested if invested>1e-7 else 1/len(tickers)) for t in tickers}
                for t in tickers:
                    qty=amount*weights[t]/opens[t]
                    if qty>1e-9:
                        cash-=qty*opens[t]*(1+COST);shares[t]+=qty
                        turnover+=qty*opens[t];orders+=1
        curve.append((dates[i],cash+sum(shares[t]*closes[t] for t in tickers)))
    series=pd.Series(dict(curve))
    return series,{"orders":orders,"turnover_initial":round(turnover/capital,3)}

def metrics(series):
    if len(series)<2:return None
    years=(series.index[-1]-series.index[0]).days/365.25
    return {"return_pct":round((series.iloc[-1]/series.iloc[0]-1)*100,3),
            "cagr_pct":round(((series.iloc[-1]/series.iloc[0])**(1/years)-1)*100,3),
            "mdd_pct":round((series/series.cummax()-1).min()*100,3)}

def run(frames):
    results={}
    for name,core in (("true_buy_hold",1.),("F_core90",.9),("F_core80",.8),("F_core70",.7)):
        curve,execution=portfolio(frames,core=core,mode="hold" if core==1 else "overlay")
        results[name]={"full":metrics(curve),"execution":execution,
                       "stress":{k:metrics(curve.loc[start:end]) for k,(start,end) in STRESS.items()}}
    return results

def main():
    frames={}
    for t in TICKERS:
        raw=pd.read_csv(Path("data")/f"{t}.csv",index_col="Date",parse_dates=True)
        validate_values(raw);frames[t]=adjusted(raw)
    result=run(frames)
    out=Path("results/defense_overlay.json");out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"research_only":True,"cost_each_side":COST,
        "method":"QQQ market regime, next-open, common-capital 7 sleeves, fractional shares, no FX",
        "warning":"Stress windows chosen as known events; not independent holdout. Compare drawdowns AND recovery opportunity cost.",
        "results":result},indent=2))
    print(out.read_text())

if __name__=="__main__":main()
