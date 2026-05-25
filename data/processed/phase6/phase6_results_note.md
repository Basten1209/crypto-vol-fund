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
| minimum_variance | 7 | -46.52% | -43.92% | 42.17% | -1.1566 | -56.61% | -1.6495 | 0.2961 | 35.14% |
| equal_weight | 7 | -57.67% | -54.82% | 64.35% | -0.9077 | -65.49% | -1.1287 | 0.0564 | 55.07% |
| minimum_variance | 14 | -48.20% | -45.55% | 43.37% | -1.1813 | -57.16% | -1.7171 | 0.3679 | 35.87% |
| equal_weight | 14 | -58.69% | -55.83% | 64.44% | -0.9408 | -65.58% | -1.1769 | 0.0804 | 55.11% |

## DM Test

| Comparison | Cycle | Loss | Mean Loss Diff | DM Stat | p-value |
|---|---:|---|---:|---:|---:|
| minimum_variance_vs_equal_weight | 7 | squared_daily_return | -0.000646431 | -5.418 | 6.029e-08 |
| minimum_variance_vs_equal_weight | 14 | squared_daily_return | -0.000621437 | -5.126 | 2.959e-07 |

## Interpretation

- The defensible benchmark framing is minimum variance versus equal-weight, because both use the same selected 50-asset universe and rebalance cycle.
- BTC HODL is useful context, but it answers a different question: whether the active multi-asset strategy beats passive BTC exposure.
- Minimum variance portfolios should be judged first on realized risk, drawdown, and loss reduction versus equal-weight.
- Concentration remains a key diagnostic; uncapped runs reached above 90% top weight, so capped variants should be compared before final model-portfolio framing.
