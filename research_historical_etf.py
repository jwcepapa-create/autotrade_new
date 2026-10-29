"""Separate pre-2016 ETF crisis research; no retrospective inclusion of unlisted stocks."""
import json
from pathlib import Path
import pandas as pd
from tiingo_data import Client,DataError,load_daily,adjusted
from research_defense_overlay import portfolio,metrics,STRESS

START="1999-03-10"
END="2010-12-31"
TICKERS=("QQQ","SPY")

def main():
    out=Path("results/etf_historical_stress.json")
    out.parent.mkdir(exist_ok=True)
    report={"status":"unavailable","requested":f"{START}..{END}",
            "tickers":list(TICKERS),"source":"Tiingo EOD",
            "note":"1999+ ETF-only separate study; no survivorship-contaminated 7-stock portfolio"}
    try:
        client=Client()
        frames={}
        for ticker in TICKERS:
            raw,info=load_daily(client,ticker,START,END,"data/historical_etf")
            frames[ticker]=adjusted(raw)
        results={}
        for name,core in (("buy_hold",1.),("F90",.9),("F80",.8),("F70",.7)):
            curve,execution=portfolio(frames,core=core,mode="hold" if core==1 else "overlay")
            results[name]={"full":metrics(curve),"execution":execution,
                           "stress":{key:metrics(curve.loc[start:end])
                                     for key,(start,end) in STRESS.items()
                                     if key in ("dotcom_crash","global_financial_crisis")}}
        report.update(status="validated",results=results)
    except (DataError,ValueError,KeyError,IndexError) as exc:
        report["reason"]=str(exc)
    out.write_text(json.dumps(report,indent=2))
    print(out.read_text())
    # Unavailable data is explicitly reported; do not fabricate crisis metrics.

if __name__=="__main__":main()
