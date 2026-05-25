# Phase 5 Results Note

## Optimization Output

| Cycle | Rebalances | Active Mean | Active Min-Max | Top Weight Mean | Top Weight Max | Turnover Mean |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 57 | 9.67 | 4-19 | 0.6258 | 0.9206 | 0.3059 |
| 14 | 29 | 9.59 | 4-17 | 0.6322 | 0.9206 | 0.3892 |

## Interpretation

- Phase 5 is a weight-generation step, not a completed investment product validation.
- The strategy is better framed as a monthly virtual asset model portfolio after Phase 6 backtesting.
- Concentration risk is material: top weights can exceed 90% in the unconstrained long-only setup.
- This behavior is consistent with minimum variance optimization, but it creates a clear Phase 6 cap-review item.

## Phase 6 Implications

- Complete Phase 6 backtest against equal-weight on the same 50-asset universe.
- Evaluate drawdown, turnover, active asset count, and top-weight time series.
- Test single-asset caps such as 20% or 30% if concentration materially worsens realized risk.
- Use research wording such as monthly virtual asset model portfolio rather than public fund wording.
