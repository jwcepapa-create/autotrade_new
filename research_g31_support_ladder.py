"""G3.1 support-ladder research: exit, hold, and partial-risk variants.
Research only. Confirmed pivots and completed bars; next-open executions.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tiingo_data import TICKERS, adjusted, validate_values
from research_g_multisignal import zones, indicators
from research_g29_multitimeframe import completed_frame
from research_defense_overlay import metrics, STRESS
from research_strategy_lab import COST

CAPITAL=10_000_000
LEVEL_WEIGHTS=(1/7,2/7,4/7)

def signals(x):
    c=x.Close.astype(float)
    support,resistance=zones(x)
    # Last observed confirmed zones, never forward-looking future pivots.
    support=support.ffill().shift(1)
    resistance=resistance.ffill().shift(1)
    week=completed_frame(c,"W-FRI",10,40)
    month=completed_frame(c,"ME",10,20)
    _,hist,_,_=indicators(x)
    vol=x.Volume/x.Volume.rolling(20,min_periods=20).mean().shift(1)
    return support,resistance,week,month,hist,vol

def simulate(x,variant):
    support,resistance,week,month,hist,vol=signals(x)
    cash=float(CAPITAL);shares=0.;orders=0
    # Three tiers are defined relative to the FIRST entry support, not
    # recomputed from future price action or reset on each bar.
    levels=None;filled=[False]*3;entry_spend=0.; realized=0.
    curve=[];events=[]
    for i in range(1,len(x)):
        prev=i-1
        op=float(x.Open.iloc[i]);cl=float(x.Close.iloc[i])
        if not np.isfinite(op) or op<=0 or not np.isfinite(cl) or cl<=0:
            raise ValueError("invalid prices")
        close=float(x.Close.iloc[prev])
        sup=float(support.iloc[prev]);res=float(resistance.iloc[prev])
        # Only information at the preceding close is used to place orders.
        if levels is None and np.isfinite(sup) and close<=sup*1.025 and close>=sup*.975:
            levels=(sup,sup*.91,sup*.78)
            filled=[False]*3
            entry_spend=0.
        if levels is not None:
            last=levels[2]
            broken=close<last*.985
            trend_weak=bool(week.weak.iloc[prev] and month.weak.iloc[prev])
            # D: full exit; E: hold without adding; F: reduce 30%.
            if broken and shares>0:
                if variant=="G3_1_D_EXIT":
                    qty=shares
                elif variant=="G3_1_F_PARTIAL" and not any(e["type"]=="risk_reduce" for e in events if e["cycle"]==str(levels)):
                    qty=shares*.30
                else:qty=0.
                if qty>0:
                    cash+=qty*op*(1-COST);shares-=qty;orders+=1
                    events.append({"type":"risk_reduce" if variant.endswith("PARTIAL") else "stop","cycle":str(levels),"date":str(x.index[i].date()),"qty":qty})
                    if variant=="G3_1_D_EXIT":
                        levels=None
                        continue
            # Never average down after last support fails; F also freezes adding on weak higher timeframes.
            can_add=not broken and not (variant=="G3_1_F_PARTIAL" and trend_weak)
            if can_add:
                for j,level in enumerate(levels):
                    if not filled[j] and close<=level*1.015 and close>=level*.95:
                        # Fixed 1:2:4 original capital budget, not compounding profits.
                        budget=min(CAPITAL*LEVEL_WEIGHTS[j],cash)
                        if budget>1:
                            qty=budget/(op*(1+COST))
                            shares+=qty;cash-=budget;entry_spend+=budget;orders+=1;filled[j]=True
                            events.append({"type":"buy","cycle":str(levels),"date":str(x.index[i].date()),"tier":j+1})
                        break
            # Confirmed resistance profit-taking; cannot sell below average cost as "profit".
            avg_cost=(entry_spend/shares) if shares>0 else float("inf")
            if shares>0 and np.isfinite(res) and close>=res*.995 and op>avg_cost:
                # First resistance: 30%; subsequent confirmed resistance: 50% of remainder.
                sells=sum(e["type"]=="profit" and e["cycle"]==str(levels) for e in events)
                qty=shares*(.30 if sells==0 else .50)
                cash+=qty*op*(1-COST);shares-=qty;orders+=1
                events.append({"type":"profit","cycle":str(levels),"date":str(x.index[i].date())})
            if shares<1e-8 and all(filled):
                levels=None
        curve.append((x.index[i],cash+shares*cl))
    series=pd.Series(dict(curve))
    return series,orders,events

def report(x,variant):
    curve,orders,events=simulate(x,variant)
    return {"full":metrics(curve),"orders":orders,
        "events_count":{k:sum(e["type"]==k for e in events) for k in ("buy","profit","risk_reduce","stop")},
        "stress":{k:metrics(curve.loc[a:b]) for k,(a,b) in STRESS.items()},
        "holdout_2024_2026":metrics(curve.loc["2024-01-01":"2026-12-31"])}

def main():
    out={"research_only":True,"capital_per_ticker":CAPITAL,"cost_per_side":COST,
         "warning":"Experimental rule-based support ladder; not a validated quality-stock filter. No leverage or broker orders. Levels 0.91 and 0.78 are research hypotheses.",
         "results":{}}
    for t in TICKERS:
        raw=pd.read_csv(Path("data")/f"{t}.csv",index_col="Date",parse_dates=True)
        validate_values(raw);x=adjusted(raw)
        out["results"][t]={name:report(x,name) for name in ("G3_1_D_EXIT","G3_1_E_HOLD","G3_1_F_PARTIAL")}
    p=Path("results/g31_support_ladder.json");p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(p.read_text())
if __name__=="__main__":main()
