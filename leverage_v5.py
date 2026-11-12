"""V5 research-only momentum recovery variants on genuine adjusted daily closes.

RSI/MACD signals use only QQQ closes through signal date; execution is next close.
No volume breakout is claimed because the aligned input contains closes only.
"""
import numpy as np
import pandas as pd
from leverage_drawdown import BASE, DOUBLE, TRIPLE, weights
from leverage_v4 import simulate_v4


def v5_targets(prices, variant="confirmed"):
    if variant not in ("fast", "confirmed", "volatility"):
        raise ValueError("Unknown V5 variant")
    q = prices["QQQ"]
    ma20 = q.rolling(20, min_periods=20).mean()
    ma50 = q.rolling(50, min_periods=50).mean()
    ma200 = q.rolling(200, min_periods=200).mean()
    weekly = q.resample("W-FRI").last()
    weekly_up = (weekly >= weekly.rolling(40, min_periods=40).mean()).reindex(
        q.index, method="ffill").fillna(False)
    change = q.diff()
    up = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = up / down.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(100)
    ema12 = q.ewm(span=12, adjust=False, min_periods=26).mean()
    ema26 = q.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()
    dd = q / q.cummax() - 1
    vol = q.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
    out = []
    for i in range(len(q)):
        target = {t: 0.0 for t in BASE + DOUBLE + TRIPLE}
        if i < 200 or pd.isna(ma200.iloc[i]) or pd.isna(hist.iloc[i]):
            out.append(target)
            continue
        long_up = q.iloc[i] >= ma200.iloc[i] and bool(weekly_up.iloc[i])
        rsi_turn = i >= 2 and rsi.iloc[i] >= 40 and rsi.iloc[i] > rsi.iloc[i - 1]
        macd_turn = i >= 2 and hist.iloc[i] > hist.iloc[i - 1] and hist.iloc[i - 1] > hist.iloc[i - 2]
        short_up = q.iloc[i] >= ma20.iloc[i] and q.iloc[i] >= ma50.iloc[i]
        if variant == "fast":
            recovery = q.iloc[i] >= ma20.iloc[i] and (rsi_turn or macd_turn)
        else:
            recovery = short_up and rsi_turn and macd_turn
        if long_up:
            target = weights(float(dd.iloc[i]))
            s2 = sum(target[t] for t in DOUBLE)
            cap2 = .25
            if variant == "volatility" and vol.iloc[i] > .25:
                cap2 = .10
            for t in DOUBLE:
                target[t] *= min(1., cap2 / s2) if s2 else 1.
            for t in TRIPLE:
                target[t] = 0.
        elif recovery:
            for t in BASE:
                target[t] = .80 / len(BASE)
        else:
            for t in BASE:
                target[t] = .50 / len(BASE)
        out.append(target)
    return out


def simulate_v5(prices, variant="confirmed", fee_bps=30., initial=10_000_000):
    aligned = prices.dropna()
    targets = v5_targets(aligned, variant=variant)
    metrics, curve = simulate_v4(
        aligned, variant="strict", cap3=0., cap2=.25, fee_bps=fee_bps,
        initial=initial, target_override=targets)
    metrics["parameters"] = {
        "variant": variant, "fee_bps": fee_bps, "execution": "next close proxy",
        "volume_signal": False, "lookahead": False}
    return metrics, curve
