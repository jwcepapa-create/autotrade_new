"""Isolated exit-policy comparison; original strategy B remains unchanged."""
import pandas as pd
from mts_v03.breakout import signals


def simulate(df, policy="repeat", initial=10_000_000, fee=.001, slippage=.0005):
    if policy not in ("repeat", "edge"):
        raise ValueError("Unknown exit policy")
    x = signals(df).dropna(subset=["ma20", "resistance_prior", "support_prior"])
    cash, shares = float(initial), 0.
    previous_exit_level = None
    rows, orders = [], []
    exposure_days = 0
    turnover = 0.
    for i in range(1, len(x)):
        prev, today = x.iloc[i-1], x.iloc[i]
        date = str(x.index[i])
        price = float(today.Open)
        if price <= 0:
            raise ValueError("Nonpositive execution price")
        level = ("support" if bool(prev.exit_support) else
                 "ma20" if bool(prev.exit_20) else
                 "ma5" if bool(prev.exit_5) else None)
        active_level = level
        if shares > 1e-10:
            fraction = {"support": 1., "ma20": .5, "ma5": .3, None: 0.}[level]
            if policy == "edge" and level == previous_exit_level and level != "support":
                fraction = 0.
            if fraction:
                qty = shares * fraction
                gross = qty * price
                cash += gross * (1 - fee - slippage)
                turnover += gross
                shares -= qty
                if shares < 1e-10:
                    shares = 0.
                orders.append({"date":date, "side":"SELL", "level":level, "quantity":qty, "gross":gross})
        if shares == 0. and bool(prev.entry):
            qty = cash / (price * (1 + fee + slippage))
            gross = qty * price
            turnover += gross
            shares, cash = qty, 0.
            orders.append({"date":date, "side":"BUY", "level":"entry", "quantity":qty, "gross":gross})
            active_level = None
        previous_exit_level = active_level if shares > 0 else None
        exposure_days += int(shares > 0)
        rows.append({"date":date, "equity":cash + shares * float(today.Close),
                     "cash":cash, "shares":shares})
    curve = pd.DataFrame(rows)
    if curve.empty:
        return {"total_return":0., "max_drawdown":0., "orders":orders,
                "exposure":0., "turnover_over_initial":0., "equity":curve}
    return {"total_return":float(curve.equity.iloc[-1]/initial - 1),
            "max_drawdown":float((curve.equity/curve.equity.cummax()-1).min()),
            "orders":orders, "exposure":exposure_days/len(curve),
            "turnover_over_initial":turnover/initial, "equity":curve}
