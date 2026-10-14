"""Research-only momentum pullback backtest. Uses Yahoo Finance via yfinance.
Survivorship bias: supplied tickers are current constituents; do not treat as institutional-grade validation.
"""
import argparse
import numpy as np
import pandas as pd
import yfinance as yf


def indicators(df):
    d=df.copy()
    for n in (9,21,50): d[f'ema{n}']=d['Close'].ewm(span=n,adjust=False).mean()
    d['dollar_vol']=d['Close']*d['Volume']
    d['dv20']=d['dollar_vol'].rolling(20).mean()
    d['ret21']=d['Close'].pct_change(21)
    d['ret63']=d['Close'].pct_change(63)
    d['ret126']=d['Close'].pct_change(126)
    d['atr']=(pd.concat([(d.High-d.Low),(d.High-d.Close.shift()).abs(),(d.Low-d.Close.shift()).abs()],axis=1).max(axis=1)).rolling(14).mean()
    return d

def screen(d):
    return (d.Close>5)&(d.dv20>20e6)&(d.ret21>.30)&(d.ret63>0)&(d.ret126>0)&(d.ema21>d.ema50)

def swing(ticker, d, max_hold=10, risk_pct=.005):
    """Signals on close, fills next open; stop/target checked from following bars.
    Same-bar stop and target: pessimistically assume stop first. One position per ticker.
    """
    d=indicators(d).dropna().copy(); mask=screen(d)
    records=[]; i=0
    while i<len(d)-max_hold-2:
        row=d.iloc[i]
        # Previous trend + pullback near 21EMA + bullish reversal on close
        if not(mask.iloc[i] and row.Low<=row.ema21*1.02 and row.Close>row.ema21 and row.Close>row.Open and d.iloc[i-1].Close<d.iloc[i-1].Open):
            i+=1; continue
        entry_idx=i+1; entry=float(d.iloc[entry_idx].Open)
        stop=min(float(row.Low),float(row.ema21)*.985)
        if entry<=stop or (entry-stop)/entry>.08: i+=1; continue
        r=entry-stop; target=entry+3*r
        exit_price=None; why='time'; exit_idx=entry_idx+max_hold-1
        for j in range(entry_idx,min(len(d),entry_idx+max_hold)):
            b=d.iloc[j]
            if b.Open<=stop: exit_price=float(b.Open);why='gap_stop';exit_idx=j;break
            if b.Low<=stop: exit_price=stop;why='stop';exit_idx=j;break
            if b.Open>=target: exit_price=float(b.Open);why='gap_target';exit_idx=j;break
            if b.High>=target: exit_price=target;why='target';exit_idx=j;break
        if exit_price is None: exit_price=float(d.iloc[exit_idx].Close)
        records.append({'ticker':ticker,'entry_date':str(d.index[entry_idx].date()),'exit_date':str(d.index[exit_idx].date()),'entry':entry,'stop':stop,'exit':exit_price,'R':(exit_price-entry)/r,'exit_reason':why})
        i=exit_idx+1
    return records

def intraday(ticker, bars, daily):
    """Intraday approximation with 5-min bars. Signal from previous completed daily bar.
    Entries only after 10:00 NY, close at 15:55 NY; excludes overnight holding.
    AVWAP proxy: intraday session VWAP, NOT anchored VWAP from a prior swing high.
    """
    bars=bars.copy()
    if bars.empty: return []
    if bars.index.tz is None: bars.index=bars.index.tz_localize('UTC')
    bars=bars.tz_convert('America/New_York')
    daily=indicators(daily)
    eligible=screen(daily).fillna(False)
    records=[]
    for date,g in bars.groupby(bars.index.date):
        prev=daily.index[daily.index.date<date]
        if len(prev)==0: continue
        day=prev[-1]
        if not bool(eligible.loc[day]): continue
        g=g.between_time('09:30','15:55').copy()
        if len(g)<15: continue
        typical=(g.High+g.Low+g.Close)/3
        g['vwap']=(typical*g.Volume).cumsum()/g.Volume.cumsum().replace(0,np.nan)
        g['ema9']=g.Close.ewm(span=9,adjust=False).mean()
        g['ema21']=g.Close.ewm(span=21,adjust=False).mean()
        entry_i=None
        for k in range(7,len(g)-2):
            b=g.iloc[k];p=g.iloc[k-1]
            if g.index[k].hour<10:continue
            if p.Low<=p.vwap*1.005 and b.Close>p.High and b.Close>b.vwap and b.ema9>b.ema21:
                entry_i=k+1;break
        if entry_i is None:continue
        entry=float(g.iloc[entry_i].Open);stop=min(float(g.iloc[entry_i-1].Low),float(g.iloc[entry_i-2].Low))
        if stop>=entry or (entry-stop)/entry>.04:continue
        r=entry-stop;target=entry+3*r;why='eod';exit_i=len(g)-1;exit_price=float(g.iloc[-1].Close)
        for j in range(entry_i,len(g)):
            b=g.iloc[j]
            if b.Open<=stop:exit_price=float(b.Open);why='gap_stop';exit_i=j;break
            if b.Low<=stop:exit_price=stop;why='stop';exit_i=j;break
            if b.Open>=target:exit_price=float(b.Open);why='gap_target';exit_i=j;break
            if b.High>=target:exit_price=target;why='target';exit_i=j;break
        records.append({'ticker':ticker,'entry_date':str(g.index[entry_i]),'exit_date':str(g.index[exit_i]),'entry':entry,'stop':stop,'exit':exit_price,'R':(exit_price-entry)/r,'exit_reason':why})
    return records

def fetch(ticker,period,interval):
    x=yf.download(ticker,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    return x.dropna()

def report(trades, fee_bps=10):
    if not trades:return {'trades':0,'note':'No trades; no performance claim possible'}
    d=pd.DataFrame(trades)
    # Conservative cost approximation as R based on entry risk distance
    d['net_R']=d.R-2*fee_bps/10000*d.entry/(d.entry-d.stop)
    return {'trades':len(d),'win_rate':round(float((d.net_R>0).mean()),4),'mean_net_R':round(float(d.net_R.mean()),4),'profit_factor':round(float(d.loc[d.net_R>0,'net_R'].sum()/abs(d.loc[d.net_R<0,'net_R'].sum())),3) if (d.net_R<0).any() else None}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tickers',default='NVDA,AMD,PLTR,IONQ,TSLA,AMZN,META,APP,COIN,HOOD');ap.add_argument('--period',default='5y');ap.add_argument('--intraday-period',default='60d');ap.add_argument('--output',default='results');args=ap.parse_args()
    import os,json
    os.makedirs(args.output,exist_ok=True)
    sw=[];day=[]; errors={}
    for ticker in args.tickers.split(','):
        ticker=ticker.strip().upper()
        try:
            daily=fetch(ticker,args.period,'1d')
            if len(daily)<160:continue
            sw+=swing(ticker,daily)
            minute=fetch(ticker,args.intraday_period,'5m')
            if not minute.empty:day+=intraday(ticker,minute,daily)
        except Exception as e:errors[ticker]=str(e)
    pd.DataFrame(sw).to_csv(os.path.join(args.output,'swing_trades.csv'),index=False)
    pd.DataFrame(day).to_csv(os.path.join(args.output,'intraday_trades.csv'),index=False)
    summary={'swing':report(sw),'intraday':report(day),'errors':errors,'limitations':['Current tickers: survivorship and selection bias','Intraday uses session VWAP, not swing-anchored AVWAP','Free intraday history is limited','Independent trades: no portfolio CAGR or MDD','Fee estimate, no realistic spread/order-book simulation']}
    with open(os.path.join(args.output,'summary.json'),'w') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
