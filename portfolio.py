"""Chronological, cash-limited accounting of strategy records, without changing signals."""
from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    fee_bps: float = 25.0  # illustrative Korean retail broker US equity commission per side
    slippage_bps: float = 5.0
    sell_levy_bps: float = 0.3  # conservative constant approximation, not historical SEC/FINRA rates

    def __post_init__(self):
        if (self.fee_bps + self.sell_levy_bps >= 10000 or
                any(not math.isfinite(v) or v < 0 or v >= 10000 for v in asdict(self).values())):
            raise ValueError('Costs must be finite and between 0 and 10000 bps.')

    def buy(self, price):
        return price * (1 + self.slippage_bps / 10000)

    def sell(self, price):
        return price * (1 - self.slippage_bps / 10000)

    def entry_cost(self, price):
        return self.buy(price) * (1 + self.fee_bps / 10000)

    def exit_proceeds(self, price):
        return self.sell(price) * (1 - (self.fee_bps + self.sell_levy_bps) / 10000)


def metrics(equity, trades, initial_krw, start):
    """Calendar-day CAGR; MDD includes initial capital; daily Sharpe with RF=0."""
    baseline = pd.Series([initial_krw], index=[pd.Timestamp(start) - pd.Timedelta(nanoseconds=1)])
    nav = pd.concat([baseline, equity]).sort_index()
    returns = nav.pct_change().dropna()
    years = (equity.index[-1] - pd.Timestamp(start)).total_seconds() / (365.25 * 86400)
    ending = float(equity.iloc[-1])
    pnl = pd.Series([t['net_pnl_usd'] for t in trades], dtype=float)
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {'initial_krw': initial_krw, 'ending_krw': ending,
            'total_return': ending / initial_krw - 1,
            'cagr': (ending / initial_krw) ** (1 / years) - 1 if years > 0 else None,
            'mdd': float((nav / nav.cummax() - 1).min()),
            'sharpe_ratio': float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
                if len(returns) > 1 and returns.std(ddof=1) > 0 else None,
            'win_rate': float((pnl > 0).mean()) if len(pnl) else None,
            'profit_factor': float(gains / losses) if losses > 0 else None,
            'profit_factor_note': 'Undefined without losing trades' if losses == 0 else None,
            'trades': len(trades)}


def account(frames, records, initial_krw=10_000_000, krw_per_usd=1350.,
            risk_pct=.005, costs=None, buy_hold=False):
    costs = costs or Costs()
    if (not all(math.isfinite(v) for v in (initial_krw, krw_per_usd, risk_pct))
            or initial_krw <= 0 or krw_per_usd <= 0 or not 0 < risk_pct <= 1):
        raise ValueError('Invalid capital, FX or risk fraction.')
    if not frames:
        raise ValueError('No price frames.')
    dates = next(iter(frames.values())).index
    if any(not d.index.equals(dates) for d in frames.values()):
        raise ValueError('All tickers must have the same validated trading sessions.')
    sleeves = []
    fills = []
    allocation = initial_krw / krw_per_usd / len(frames)
    for ticker, daily in frames.items():
        cash, qty, position = allocation, 0., None
        schedule = {pd.Timestamp(t['entry_date']): t for t in records if t['ticker'] == ticker}
        if buy_hold:
            schedule = {dates[0]: {'ticker': ticker, 'entry_date': str(dates[0].date()),
                'exit_date': str(dates[-1].date()), 'entry': float(daily.Open.iloc[0]),
                'exit': float(daily.Close.iloc[-1]), 'exit_reason': 'end_of_test'}}
        nav = []
        for date, bar in daily.iterrows():
            if date in schedule:
                if position is not None:
                    raise ValueError('Overlapping positions in one ticker.')
                t = schedule[date]
                if pd.Timestamp(t['exit_date']) < date:
                    raise ValueError('Exit precedes entry.')
                # Size from cash and entry/stop only. Future exit is never used for sizing.
                affordable = cash / costs.entry_cost(t['entry'])
                risk_size = cash * risk_pct / (costs.buy(t['entry']) - t['stop']) if not buy_hold else affordable
                qty = max(0., min(affordable, risk_size))
                if qty > 0:
                    debit = qty * costs.entry_cost(t['entry'])
                    cash -= debit
                    position = {**t, 'units': qty, 'entry_debit_usd': debit,
                                'entry_fill': costs.buy(t['entry'])}
            if position is not None and pd.Timestamp(position['exit_date']) == date:
                credit = qty * costs.exit_proceeds(position['exit'])
                cash += credit
                fills.append({**position, 'exit_fill': costs.sell(position['exit']),
                              'exit_credit_usd': credit,
                              'net_pnl_usd': credit - position['entry_debit_usd']})
                qty, position = 0., None
            nav.append(cash + qty * float(bar.Close))
            if cash < -1e-7:
                raise ValueError('Cash borrowing is prohibited.')
        if position is not None:
            raise ValueError('Unclosed trade; no performance claim possible.')
        sleeves.append(pd.Series(nav, index=dates, name=ticker))
    curve = pd.concat(sleeves, axis=1).sum(axis=1) * krw_per_usd
    curve.name = 'equity_krw'
    return curve, fills
