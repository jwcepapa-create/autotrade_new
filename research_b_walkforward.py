"""Chronological B validation: choose on train only, then report untouched future slices."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from tiingo_data import TICKERS,adjusted,validate_values
from mts_v03.b_knobs import simulate_exit

TRAIN_END="2021-12-31"
VAL_END="2023-12-31"
CONFIGS=[(.3,.5,0),(.1,.3,0),(.2,.5,0),(.3,.5,3),(.3,.5,5),(.2,.3,3)]

def select_on_training(x):
    train=x.loc[:TRAIN_END]
    if len(train)<250: raise ValueError("Insufficient training data")
    trials=[]
    for f5,f20,cooldown in CONFIGS:
        r=simulate_exit(train,ma5_fraction=f5,ma20_fraction=f20,cooldown=cooldown)
        # Penalize drawdown; fixed selection rule before looking at future segments.
        score=r["total_return"]+min(0.,r["max_drawdown"])
        trials.append({"params":[f5,f20,cooldown],"train_return":r["total_return"],
                       "train_mdd":r["max_drawdown"],"score":score})
    return max(trials,key=lambda t:t["score"]),trials

def evaluate_future(x,params,start,end):
    # Warm-up with past bars; trade only within selected period using fresh capital.
    # This evaluates a reset account, not a continuing historical portfolio.
    warm=x.loc[:end].tail(len(x.loc[start:end])+260)
    period=warm.loc[start:end]
    if len(period)<30:return None
    # Start with no position at beginning of period, warmup signals from prior history.
    from mts_v03.breakout import signals
    from mts_v03.b_knobs import simulate_exit
    # Exact reset implementation: trim warm-up to prior 260 bars and set start date via explicit engine.
    return simulate_exit(warm,ma5_fraction=params[0],ma20_fraction=params[1],cooldown=params[2],
                         start_trading=start)

def main():
    rows=[]
    for ticker in TICKERS:
        path=Path("data")/f"{ticker}.csv"
        raw=pd.read_csv(path,index_col="Date",parse_dates=True)
        validate_values(raw)
        x=adjusted(raw)
        chosen,trials=select_on_training(x)
        params=chosen["params"]
        rows.append({"ticker":ticker,"chosen_on_train":chosen,
                     "validation":evaluate_future(x,params,"2022-01-01",VAL_END),
                     "holdout":evaluate_future(x,params,"2024-01-01",str(x.index[-1].date())),
                     "data_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    out=Path("results/b_walkforward.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"train_end":TRAIN_END,"validation_end":VAL_END,
                               "selection":"max(train return + train MDD)","results":rows},indent=2))
    print(out.read_text())

if __name__=="__main__":main()
