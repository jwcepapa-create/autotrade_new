"""Research-only equal-weight independent sleeve portfolio from daily next-open target fills."""
import numpy as np
import pandas as pd
from research_strategy_lab import COST,signals_momentum,signals_hybrid,signals_c_daily

def portfolio(frames,model="D",capital=10_000_000,rebalance="D"):
    if model not in ("D","E","C","hold"):raise ValueError("Unknown model")
    tickers=sorted(frames)
    dates=frames[tickers[0]].index
    for ticker in tickers[1:]:
        dates=dates.intersection(frames[ticker].index)
    if len(dates)<260:raise ValueError("Insufficient common data")
    data={t:frames[t].loc[dates] for t in tickers}
    sig={}
    for t,x in data.items():
        sig[t]={"D":lambda:signals_momentum(x),"E":lambda:signals_hybrid(x),
                "C":lambda:signals_c_daily(x),"hold":lambda:pd.Series(1.,index=dates)}[model]()
    cash=float(capital); shares={t:0. for t in tickers}; records=[]; gross_turnover=0.
    n=len(tickers)
    for i in range(1,len(dates)):
        today=dates[i]; yesterday=dates[i-1]
        opens={t:float(data[t].Open.iloc[i]) for t in tickers}
        closes={t:float(data[t].Close.iloc[i]) for t in tickers}
        equity_open=cash+sum(shares[t]*opens[t] for t in tickers)
        if rebalance=="D" or (rebalance=="M" and today.month!=yesterday.month):
            targets={t:float(sig[t].iloc[i-1])/n for t in tickers}
            desired={t:equity_open*targets[t]/opens[t] for t in tickers}
            # Sells first to finance buys. Pro-rata scale purchases to cash after costs.
            for t in tickers:
                delta=min(0.,desired[t]-shares[t])
                if delta:
                    cash-=delta*opens[t]+abs(delta)*opens[t]*COST
                    shares[t]+=delta;gross_turnover+=abs(delta)*opens[t]
            buys={t:max(0.,desired[t]-shares[t]) for t in tickers}
            needed=sum(q*opens[t]*(1+COST) for t,q in buys.items())
            scale=min(1.,max(cash,0)/needed) if needed else 0.
            for t,q in buys.items():
                delta=q*scale
                cash-=delta*opens[t]*(1+COST)
                shares[t]+=delta;gross_turnover+=delta*opens[t]
        equity=cash+sum(shares[t]*closes[t] for t in tickers)
        records.append((today,equity))
    curve=pd.Series(dict(records))
    years=(curve.index[-1]-curve.index[0]).days/365.25
    return {"return_pct":round((curve.iloc[-1]/capital-1)*100,3),
            "cagr_pct":round(((curve.iloc[-1]/capital)**(1/years)-1)*100,3),
            "mdd_pct":round((curve/curve.cummax()-1).min()*100,3),
            "turnover_initial":round(gross_turnover/capital,3),
            "curve":curve}

def main():
    import json
    from pathlib import Path
    from tiingo_data import TICKERS,adjusted,validate_values
    frames={}
    for ticker in TICKERS:
        raw=pd.read_csv(Path("data")/f"{ticker}.csv",index_col="Date",parse_dates=True)
        validate_values(raw);frames[ticker]=adjusted(raw)
    results=[]
    for model in ("hold","D","E","C"):
        for frequency in (("D","M") if model=="D" else ("D",)):
            r=portfolio(frames,model,rebalance=frequency)
            r.pop("curve")
            results.append({"model":model,"rebalance":frequency,**r})
    out=Path("results/portfolio_lab.json")
    out.write_text(json.dumps({"currency":"nominal KRW sizing with USD price returns; no FX conversion",
        "assumption":"Equal capital weights, common daily sessions, adjusted data, 15bps each-side cost, fractional shares; research only",
        "results":results},indent=2))
    print(out.read_text())

if __name__=="__main__":main()
