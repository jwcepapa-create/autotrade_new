"""Completed-bar only higher-timeframe causal filter. Intraday requires licensed bars."""
import pandas as pd

def completed_timeframe(df,frequency,asof):
    """Only include periods whose right boundary is strictly before asof."""
    x=df.loc[df.index < pd.Timestamp(asof)]
    if x.empty:return x
    rule={"W":"W-FRI","M":"ME"}[frequency]
    grouped=x.resample(rule,label="right",closed="right").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    return grouped.loc[grouped.index < pd.Timestamp(asof)]

def confirmed_trend(df,asof,period="W",length=10):
    bars=completed_timeframe(df,period,asof)
    if len(bars)<length:return None
    return bool(bars.Close.iloc[-1]>bars.Close.iloc[-length:].mean())

def intraday_status():
    return {"status":"blocked","reason":"No verified complete 120-minute historical dataset; no synthetic intraday reconstruction"}
