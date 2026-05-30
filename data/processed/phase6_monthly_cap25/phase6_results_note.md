# Phase 6 Backtest Results Note

- Period: 2025-03-01 to 2026-03-30 (395 days)
- Evaluation frequency: 10 minutes
- Transaction costs: ignored
- Performance table: active monthly hold windows only.
- Each monthly product starts from the same initial AUM at the first available date of the month.
- Primary comparison: minimum variance versus equal-weight on the same 50-asset universe.
- BTC HODL is retained only in daily/monthly reference outputs, not in the main performance table.

## Hold-Window Performance

| Strategy | Mode | Cycle | Months | Invested Days | Total Return | Mean Monthly Return | Ann. Vol | Sharpe | MDD | Realized Risk |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| equal_weight | Managed Mode | 7 | 13 | 91 | -22.12% | -1.67% | 81.39% | -0.8193 | -36.16% | 60.53% |
| minimum_variance | Managed Mode | 7 | 13 | 91 | -7.47% | -0.40% | 68.57% | -0.1137 | -28.44% | 47.70% |
| equal_weight | Simple Mode | 7 | 13 | 91 | -23.02% | -1.75% | 81.86% | -0.8662 | -36.94% | 60.78% |
| minimum_variance | Simple Mode | 7 | 13 | 91 | -7.48% | -0.41% | 69.35% | -0.1059 | -28.54% | 48.39% |
| equal_weight | Managed Mode | 14 | 13 | 182 | 8.54% | 1.26% | 76.35% | 0.6029 | -35.20% | 59.47% |
| minimum_variance | Managed Mode | 14 | 13 | 182 | 31.10% | 2.62% | 59.89% | 1.2056 | -29.59% | 45.93% |
| equal_weight | Simple Mode | 14 | 13 | 182 | 11.27% | 1.45% | 76.31% | 0.6685 | -35.68% | 59.84% |
| minimum_variance | Simple Mode | 14 | 13 | 182 | 29.51% | 2.52% | 59.78% | 1.1659 | -29.06% | 46.79% |

## DM Test

| Comparison | Policy | Cycle | Loss | Mean Loss Diff | DM Stat | p-value |
|---|---|---:|---|---:|---:|---:|
| minimum_variance_vs_equal_weight | daily_rebalance_to_target | 7 | squared_daily_return_active_hold_windows | -0.000524256 | -1.967 | 0.04918 |
| minimum_variance_vs_equal_weight | daily_rebalance_to_target | 14 | squared_daily_return_active_hold_windows | -0.000608515 | -3.05 | 0.002286 |
| minimum_variance_vs_equal_weight | enter_once_then_drift | 7 | squared_daily_return_active_hold_windows | -0.000516531 | -1.826 | 0.06779 |
| minimum_variance_vs_equal_weight | enter_once_then_drift | 14 | squared_daily_return_active_hold_windows | -0.000611379 | -2.966 | 0.003015 |

## Interpretation

- The defensible benchmark framing is minimum variance versus equal-weight, because both use the same selected 50-asset universe and rebalance cycle.
- Rows are computed only on 7-day / 14-day active hold windows, matching the monthly short-product framing.
- Minimum variance portfolios should be judged first on realized risk, drawdown, and loss reduction versus equal-weight.
- Concentration remains a key diagnostic; uncapped runs reached above 90% top weight, so capped variants should be compared before final model-portfolio framing.
