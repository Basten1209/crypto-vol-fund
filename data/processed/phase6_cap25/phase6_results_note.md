# Phase 6 Backtest Results Note

- Period: 2025-03-01 to 2026-03-30 (395 days)
- Evaluation frequency: 10 minutes
- Transaction costs: ignored
- Hold-period weight drift: enabled
- Primary comparison: minimum variance versus equal-weight on the same 50-asset universe.
- BTC HODL is retained only as a market reference, not the main benchmark for this strategy design.

## Performance

| Strategy | Cycle | Total Return | Ann. Return | Ann. Vol | Sharpe | MDD | IR vs BTC | Turnover | Realized Risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc_hodl |  | -18.48% | -17.21% | 41.24% | -0.2517 | -47.71% | n/a | n/a | n/a |
| minimum_variance | 7 | -36.39% | -34.16% | 47.53% | -0.6409 | -54.61% | -0.6091 | 0.3858 | 40.57% |
| equal_weight | 7 | -57.67% | -54.82% | 64.35% | -0.9077 | -65.49% | -1.1287 | 0.0564 | 55.07% |
| minimum_variance | 14 | -36.22% | -34.00% | 49.63% | -0.5895 | -53.65% | -0.5321 | 0.4846 | 41.75% |
| equal_weight | 14 | -58.69% | -55.83% | 64.44% | -0.9408 | -65.58% | -1.1769 | 0.0804 | 55.11% |

## DM Test

| Comparison | Cycle | Loss | Mean Loss Diff | DM Stat | p-value |
|---|---:|---|---:|---:|---:|
| minimum_variance_vs_equal_weight | 7 | squared_daily_return | -0.000516232 | -5.521 | 3.37e-08 |
| minimum_variance_vs_equal_weight | 14 | squared_daily_return | -0.000463617 | -4.359 | 1.308e-05 |

## Interpretation

- The defensible benchmark framing is minimum variance versus equal-weight, because both use the same selected 50-asset universe and rebalance cycle.
- BTC HODL is useful context, but it answers a different question: whether the active multi-asset strategy beats passive BTC exposure.
- Minimum variance portfolios should be judged first on realized risk, drawdown, and loss reduction versus equal-weight.
- Concentration remains a key diagnostic; uncapped runs reached above 90% top weight, so capped variants should be compared before final model-portfolio framing.
