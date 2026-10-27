"""Quarterly continuous-account scorecards and annual fresh-capital experiments.

Research only. Completed-bar signals, next-session-open fills, no broker connection.
Quarterly results are slices of uninterrupted full-history equity curves.
Annual experiments reset cash/positions at each year, but calculate indicators
on the full prior history (warm-up) without allowing pre-period trades.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tiingo_data import TICKERS, adjusted, validate_values
from research_g_multisignal import signal
from research_g_second_generation import candidate_signals
from research_g_improvements import run
from research_strategy_lab import COST

INITIAL=10_000_000

def strategies(x):
    no=pd.Series(False,index=x.index)
    hold=no.copy()
    hold.iloc[0]=True
    yield "BUY_HOLD",hold,no
    yield "G2_BASE",*signal(x,2)
    yield from candidate_signals(x)

def score(series):
    if len(series)<2 or not np.isfinite(series.to_numpy(float)).all():
        return None
    first=float(series.iloc[0]);last=float(series.iloc[-1])
    if first<=0:return None
    return {"return_pct":round((last/first-1)*100,3),
            "mdd_pct":round((series/series.cummax()-1).min()*100,3),
            "start_equity":round(first,2),"end_equity":round(last,2),
            "bars":len(series)}

def quarterly(x, curves):
    periods=x.index[1:].to_period("Q").unique()
    rows=[]
    for p in periods:
        label=str(p)
        row={"quarter":label,"strategies":{}}
        for name,curve in curves.items():
            segment=curve.loc[curve.index.to_period("Q")==p]
            row["strategies"][name]=score(segment)
        bh=row["strategies"]["BUY_HOLD"]
        if bh is not None:
            for name,record in row["strategies"].items():
                if record is not None:
                    record["excess_vs_bh_pct_points"]=round(record["return_pct"]-bh["return_pct"],3)
        rows.append(row)
    return rows

def annual_reset(x, buy, sell, year):
    # The first trading session of the year is the initial purchase opportunity.
    # Indicator state is calculated using history preceding the year.
    indices=np.flatnonzero(x.index.year==year)
    if len(indices)<2:return None
    cash=float(INITIAL);shares=0.;orders=0;curve=[]
    first=int(indices[0])
    for i in indices:
        op=float(x.Open.iloc[i]);cl=float(x.Close.iloc[i])
        if not np.isfinite(op) or op<=0 or not np.isfinite(cl) or cl<=0:
            raise ValueError("invalid execution price")
        # For the first session, use only the last completed bar from the prior year.
        b=bool(buy.iloc[i-1]) if i>0 else False
        s=bool(sell.iloc[i-1]) if i>0 else False
        if shares>0 and s:
            cash+=shares*op*(1-COST);shares=0.;orders+=1
        elif shares==0 and b:
            shares=cash/(op*(1+COST));cash=0.;orders+=1
        curve.append((x.index[i],cash+shares*cl))
    series=pd.Series(dict(curve))
    # Initial investment is INITIAL, not first close (which includes day-one return).
    result=score(pd.concat([pd.Series([INITIAL],index=[series.index[0]-pd.Timedelta(nanoseconds=1)]),series]))
    result["orders"]=orders
    result["initial_capital"]=INITIAL
    return result

def annual_bh(x,year):
    indices=np.flatnonzero(x.index.year==year)
    if len(indices)<2:return None
    first=int(indices[0]); op=float(x.Open.iloc[first])
    shares=INITIAL/(op*(1+COST))
    curve=pd.Series({x.index[i]:shares*float(x.Close.iloc[i]) for i in indices})
    result=score(pd.concat([pd.Series([INITIAL],index=[curve.index[0]-pd.Timedelta(nanoseconds=1)]),curve]))
    result["orders"]=1
    result["initial_capital"]=INITIAL
    return result

def main():
    out={"research_only":True,"capital_per_ticker":INITIAL,
         "cost_per_side":COST,
         "quarterly_method":"Continuous full-history trading, no forced quarterly liquidation; per-quarter marked equity first-close to last-close; first-session returns are not allocated to the quarter.",
         "annual_method":"Independent 10M KRW fresh capital each calendar year; full prior history for indicators; no prior-year positions; first open of year executable using previous close; B&H buys first open.",
         "warning":"Quarterly values omit each quarter's first open-to-close move and gap from preceding quarter. Do not compound these quarter returns to reconstruct full-period CAGR. Annual MDD includes initial capital. No FX; per-ticker independent sleeves, not F70 common-capital.",
         "results":{}}
    for ticker in TICKERS:
        raw=pd.read_csv(Path("data")/f"{ticker}.csv",index_col="Date",parse_dates=True)
        validate_values(raw);x=adjusted(raw)
        sig=dict((name,(buy.fillna(False),sell.fillna(False))) for name,buy,sell in strategies(x))
        curves={name:run(x,buy,sell,INITIAL)[0] for name,(buy,sell) in sig.items()}
        quarters=quarterly(x,curves)
        years=sorted(set(x.index.year))
        annual={}
        for year in years:
            if year<2016:continue
            annual[str(year)]={}
            for name,(buy,sell) in sig.items():
                annual[str(year)][name]=annual_bh(x,year) if name=="BUY_HOLD" else annual_reset(x,buy,sell,year)
            base=annual[str(year)]["BUY_HOLD"]
            if base:
                for name,item in annual[str(year)].items():
                    if item is not None:
                        item["excess_vs_bh_pct_points"]=round(item["return_pct"]-base["return_pct"],3)
        summary={}
        for name in sig:
            valid=[q["strategies"][name] for q in quarters if q["strategies"][name] is not None]
            valid_years=[a[name] for a in annual.values() if a[name] is not None]
            summary[name]={"quarters_evaluated":len(valid),
                "quarterly_outperformance_count":sum(v["excess_vs_bh_pct_points"]>0 for v in valid),
                "quarterly_outperformance_rate_pct":round(100*sum(v["excess_vs_bh_pct_points"]>0 for v in valid)/len(valid),2) if valid else None,
                "years_evaluated":len(valid_years),
                "annual_outperformance_count":sum(v["excess_vs_bh_pct_points"]>0 for v in valid_years)}
        out["results"][ticker]={"quarterly_continuous":quarters,"annual_independent":annual,"summary":summary}
    dest=Path("results/g_periodic_comparison.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(dest.read_text())

if __name__=="__main__":
    main()
