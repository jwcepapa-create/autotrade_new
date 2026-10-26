# MTS Trading Engine — support/resistance strategy (research branch)

This branch is reserved for the user's **previously developed** MTS Trading Engine v0.1/v0.2, not for a new, invented S/R system.

## Historical strategy recovered from earlier discussions

- Structural support/resistance **zones**, not single exact price lines.
- Zone confidence dimensions: timeframe, touches, volume, reaction strength, role reversal, breakout, recency.
- Entry sequence: identify resistance zone -> confirmed breakout with volume -> retest -> prior resistance acts as support -> enter.
- Earlier human-labeled calibration examples: 계룡건설 18,900–20,800 KRW, 삼영 5,400–5,550 KRW, 풍산 around 50,000 KRW.
- Earlier project artifacts reportedly named `mts_trading_engine_v0_1.zip`, `mts_trading_engine_v0_2.zip`; GitHub-style layout `sr-engine/`, `data/labels.csv`, `src/detector.py`, `calibrate.py`, `config.yaml`. Historical conversation refers to 175 human S/R labels.

## Integrity / adversarial-review gate

The original ZIP, original `labels.csv`, and exact threshold/exit parameters are **not currently available in this branch**. Do **not** invent the labels or silently claim to reproduce the original model.

Before backtesting:
1. Recover original source/labels, preserve their checksums and document provenance.
2. Audit pivot/zone detection for look-ahead bias (a pivot needing right-side candles must become available only after those candles close).
3. Fit zone-scoring thresholds only on training history; no future price, volume or reactions can contribute to historical zones.
4. Compare against momentum baseline and QQQ buy-and-hold on common dates, tickers, costs, FX, exposure and capital.
5. Test zone-score sensitivity and ablation; walk-forward and untouched out-of-sample holdout.
6. Require independent QA and adversarial review before merge.

**Status: strategy specification recovered; implementation and performance unverified.**
