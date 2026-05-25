# Phase 6 Backtest Results Note

- Period: 2025-03-01 to 2026-03-30 (395 days)
- Evaluation frequency: 10 minutes
- Transaction costs: ignored
- Hold-period weight drift: enabled
- Primary comparison: minimum variance versus equal-weight on the same 50-asset universe.
- BTC HODL is retained only as a market reference, not the main benchmark for this strategy design.

## Performance

| Strategy | Policy | Cycle | Total Return | Ann. Return | Ann. Vol | Sharpe | MDD | IR vs BTC | Turnover Mean | Turnover Sum | Realized Risk |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc_hodl | buy_and_hold |  | -18.48% | -17.21% | 41.24% | -0.2517 | -47.71% | n/a | n/a | 0.0000 | n/a |
| minimum_variance | enter_once_then_drift | 7 | -7.48% | -6.93% | 33.14% | -0.0510 | -28.54% | 0.2486 | 1.0000 | 13.0000 | 11.15% |
| equal_weight | enter_once_then_drift | 7 | -23.02% | -21.47% | 39.16% | -0.4172 | -36.94% | -0.1670 | 1.0000 | 13.0000 | 14.00% |
| minimum_variance | daily_rebalance_to_target | 7 | -7.47% | -6.93% | 32.77% | -0.0548 | -28.44% | 0.2469 | 0.1503 | 14.5251 | 10.99% |
| equal_weight | daily_rebalance_to_target | 7 | -22.12% | -20.63% | 38.93% | -0.3946 | -36.16% | -0.1402 | 0.1525 | 14.7271 | 13.94% |
| minimum_variance | enter_once_then_drift | 14 | 29.51% | 26.99% | 40.56% | 0.7918 | -29.06% | 1.2839 | 1.0000 | 13.0000 | 21.56% |
| equal_weight | enter_once_then_drift | 14 | 11.27% | 10.37% | 51.74% | 0.4543 | -35.68% | 0.8761 | 1.0000 | 13.0000 | 27.57% |
| minimum_variance | daily_rebalance_to_target | 14 | 31.10% | 28.43% | 40.64% | 0.8187 | -29.59% | 1.3232 | 0.0840 | 16.2045 | 21.16% |
| equal_weight | daily_rebalance_to_target | 14 | 8.54% | 7.87% | 51.76% | 0.4098 | -35.20% | 0.8202 | 0.0874 | 16.8166 | 27.40% |

## DM Test

| Comparison | Policy | Cycle | Loss | Mean Loss Diff | DM Stat | p-value |
|---|---|---:|---|---:|---:|---:|
| minimum_variance_vs_equal_weight | enter_once_then_drift | 7 | squared_daily_return | -0.000118998 | -1.765 | 0.07763 |
| minimum_variance_vs_equal_weight | daily_rebalance_to_target | 7 | squared_daily_return | -0.000120778 | -1.885 | 0.05938 |
| minimum_variance_vs_equal_weight | enter_once_then_drift | 14 | squared_daily_return | -0.000281699 | -2.823 | 0.00476 |
| minimum_variance_vs_equal_weight | daily_rebalance_to_target | 14 | squared_daily_return | -0.000280379 | -2.897 | 0.003773 |

## Interpretation

- The defensible benchmark framing is minimum variance versus equal-weight, because both use the same selected 50-asset universe and rebalance cycle.
- BTC HODL is useful context, but it answers a different question: whether the active multi-asset strategy beats passive BTC exposure.
- Minimum variance portfolios should be judged first on realized risk, drawdown, and loss reduction versus equal-weight.
- Concentration remains a key diagnostic; uncapped runs reached above 90% top weight, so capped variants should be compared before final model-portfolio framing.
