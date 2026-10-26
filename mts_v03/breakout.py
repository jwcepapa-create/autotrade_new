"""Faithful research port of v0.3 daily breakout model; no live orders."""
import pandas as pd


def features(df):
    x = df.copy()
    c = x["Close"].astype(float)
    x["ma5"] = c.rolling(5).mean()
    x["ma20"] = c.rolling(20).mean()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    x["macd"] = e12 - e26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False).mean()
    x["resistance_prior"] = x.High.rolling(60, min_periods=60).max().shift(1)
    x["support_prior"] = x.Low.rolling(60, min_periods=60).min().shift(1)
    x["volume_avg20"] = x.Volume.rolling(20).mean().shift(1)
    return x


def signals(df):
    x = features(df)
    x["breakout"] = (x.Close > x.resistance_prior) & (x.Volume > x.volume_avg20)
    x["trend_ok"] = (x.Close > x.ma20) & (x.macd > x.macd_signal)
    x["entry"] = x.breakout & x.trend_ok
    x["exit_5"] = x.Close < x.ma5
    x["exit_20"] = x.Close < x.ma20
    x["exit_support"] = x.Close < x.support_prior
    return x


def run(df, initial=10_000_000, fee=.001, slippage=.0005):
    """Reproduce original v0.3 behavior, including repeated partial exits."""
    x = signals(df).dropna(subset=["ma20", "resistance_prior", "support_prior"])
    cash = float(initial)
    shares = 0.0
    history, trades = [], []
    for i in range(1, len(x)):
        prev, today = x.iloc[i-1], x.iloc[i]
        date = str(x.index[i])
        price = float(today.Open)
        if price <= 0:
            continue
        fraction = 0.
        if shares > 0:
            if bool(prev.exit_support):
                fraction = 1.
            elif bool(prev.exit_20):
                fraction = .5
            elif bool(prev.exit_5):
                fraction = .3
            if fraction:
                qty = shares * fraction
                cash += qty * price * (1 - fee - slippage)
                shares -= qty
                trades.append((date, "SELL", qty, price))
        if shares == 0 and bool(prev.entry):
            qty = cash / (price * (1 + fee + slippage))
            shares, cash = qty, 0.
            trades.append((date, "BUY", qty, price))
        history.append((date, cash + shares * float(today.Close)))
    curve = pd.DataFrame(history, columns=["date", "equity"])
    if curve.empty:
        total_return = max_dd = 0.
    else:
        total_return = float(curve.equity.iloc[-1] / initial - 1)
        max_dd = float((curve.equity / curve.equity.cummax() - 1).min())
    return {"total_return": total_return, "max_drawdown": max_dd,
            "trades": trades, "equity": curve}
