"""B exit policy sweep with common signals, costs and next-open fills; research only."""
import pandas as pd
from mts_v03.breakout import signals

def simulate_exit(df, ma5_fraction=.3, ma20_fraction=.5, cooldown=0,
                  initial=10_000_000, fee=.001, slippage=.0005, start_trading=None):
    if not 0 <= ma5_fraction <= 1 or not 0 <= ma20_fraction <= 1 or cooldown < 0:
        raise ValueError("Invalid exit parameters")
    x=signals(df).dropna(subset=["ma20","resistance_prior","support_prior"])
    cash=float(initial); shares=0.; last_partial=-100000
    trades=[]; equity=[]; exposure=0; turnover=0.
    for i in range(1,len(x)):
        prev,today=x.iloc[i-1],x.iloc[i]
        if start_trading is not None and x.index[i] < pd.Timestamp(start_trading):
            continue
        price=float(today.Open)
        if price<=0:
            raise ValueError("Invalid open")
        if shares>0:
            if bool(prev.exit_support):
                fraction=1.
            elif i-last_partial>cooldown:
                fraction=ma20_fraction if bool(prev.exit_20) else ma5_fraction if bool(prev.exit_5) else 0.
            else:
                fraction=0.
            if fraction>0:
                qty=shares*fraction
                gross=qty*price
                cash+=gross*(1-fee-slippage)
                turnover+=gross
                shares-=qty
                trades.append((str(x.index[i]),"SELL",qty,price))
                if fraction<1:
                    last_partial=i
        if shares==0 and bool(prev.entry):
            qty=cash/(price*(1+fee+slippage))
            turnover+=qty*price
            shares,cash=qty,0.
            last_partial=-100000
            trades.append((str(x.index[i]),"BUY",qty,price))
        exposure+=int(shares>0)
        equity.append(cash+shares*float(today.Close))
    if not equity:
        return dict(total_return=0.,max_drawdown=0.,orders=0,exposure=0.,turnover=0.)
    curve=pd.Series(equity)
    return dict(total_return=float(curve.iloc[-1]/initial-1),
                max_drawdown=float((curve/curve.cummax()-1).min()),
                orders=len(trades),exposure=exposure/len(curve),
                turnover=turnover/initial)
