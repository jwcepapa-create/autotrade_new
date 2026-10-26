import numpy as np
import pandas as pd
import pytest
from mts_v03.breakout import run
from mts_v03.exit_experiments import simulate


def bars(n=160):
    close = 100 + np.arange(n)*.2 + 6*np.sin(np.arange(n)*.2)
    return pd.DataFrame({"Open":close, "High":close+1, "Low":close-1,
                         "Close":close, "Volume":1000+np.arange(n)*2},
                        index=pd.bdate_range("2020-01-01",periods=n))


def test_repeat_matches_original():
    d = bars()
    baseline = run(d)
    variant = simulate(d,"repeat")
    assert variant["total_return"] == pytest.approx(baseline["total_return"],abs=1e-10)
    assert variant["max_drawdown"] == pytest.approx(baseline["max_drawdown"],abs=1e-10)
    assert len(variant["orders"]) == len(baseline["trades"])


def test_edge_outputs_valid_and_is_research_only():
    result = simulate(bars(),"edge")
    assert np.isfinite(result["total_return"])
    assert np.isfinite(result["max_drawdown"])
    assert 0 <= result["exposure"] <= 1
    assert result["turnover_over_initial"] >= 0


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        simulate(bars(),"unsupported")
