"""Point-in-time support/resistance pivot observations.

A pivot at bar i is *not* available until i+right closes. A reaction
score is available only after its full confirmation window. Each record
has separate pivot_date and available_date to prevent backdating.
This is a causal reconstruction, not an exact replacement of legacy scoring.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfirmedPivot:
    pivot_date: str
    available_date: str
    side: str
    price: float
    reaction_atr: float


def confirmed_pivots(df, left=3, right=3, reaction_bars=9):
    if left < 1 or right < 1 or reaction_bars < right:
        raise ValueError("Require left/right >=1 and reaction_bars >= right")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Data must be chronologically sorted and unique")
    result = []
    h, l, c = [df[k].to_numpy(dtype=float) for k in ("High", "Low", "Close")]
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.r_[np.nan,c[:-1]]),
                                      np.abs(l-np.r_[np.nan,c[:-1]])))
    tr[0] = h[0]-l[0] if len(df) else np.nan
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    # Reaction score is frozen only when all required subsequent bars exist.
    confirmation = max(right, reaction_bars)
    for i in range(left, len(df)-confirmation):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        for side, values in (("high", h), ("low", l)):
            window = values[i-left:i+right+1]
            if not np.isfinite(window).all():
                continue
            is_pivot = values[i] >= window.max() if side == "high" else values[i] <= window.min()
            if not is_pivot:
                continue
            future = slice(i+1, i+reaction_bars+1)
            reaction = ((values[i] - np.min(l[future])) / atr[i] if side == "high"
                        else (np.max(h[future]) - values[i]) / atr[i])
            result.append(ConfirmedPivot(str(df.index[i]), str(df.index[i+confirmation]),
                                         side, float(values[i]), float(max(0, reaction))))
    return result


def available_pivots(df, asof, **kwargs):
    """Only records confirmed by asof; never evaluate later candles."""
    history = df.loc[:asof]
    return confirmed_pivots(history, **kwargs)
