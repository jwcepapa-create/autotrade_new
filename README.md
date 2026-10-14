# Momentum backtest (research prototype)

GitHub Actions downloads Yahoo Finance daily (5y) and 5-minute (60d, subject to provider limits) prices for 10 example US stocks and runs separate swing and intraday tests.

## Run
GitHub repository > Actions > Momentum backtest > Run workflow. After it finishes, open the run and download the `momentum-backtest-results` artifact (CSV and JSON). It is also scheduled weekdays at 23:30 UTC; scheduled runs may be delayed or disabled by GitHub.

## Limitations
This is a prototype, not a validated all-US-stocks test. The ticker list is a current hand-picked sample with selection/survivorship bias. Intraday uses session VWAP rather than swing-anchored VWAP; 3R full exits rather than 3R/5R split. No portfolio-level CAGR/MDD or realistic execution model. Successful execution is not proof of profitability.
