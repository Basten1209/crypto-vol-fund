# Phase 5 Results Note

## Optimization Output

- Rebalance frequency: monthly

| Cycle | Rebalances | Active Mean | Active Min-Max | Min Positive | Top Weight Mean | Top Weight Max | Turnover Mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 13 | 14.00 | 8-20 | 0.0028 | 0.2500 | 0.2500 | 0.6535 |
| 14 | 13 | 14.00 | 8-20 | 0.0028 | 0.2500 | 0.2500 | 0.6535 |

## Interpretation

- Phase 5 is a weight-generation step, not a completed investment product validation.
- The strategy is better framed as a monthly virtual asset model portfolio after Phase 6 backtesting.
- Positive weights at or below 0.10% are pruned to zero before artifacts are written.
- A single-asset cap of 25% is active; top weights are mechanically bounded at that level.
- This capped variant should be compared against the uncapped strategy and equal-weight in Phase 6.

## Phase 6 Implications

- Complete Phase 6 backtest against equal-weight on the same 50-asset universe.
- Compare capped versus uncapped minimum variance on return, volatility, drawdown, and turnover.
- Use BTC HODL as market context rather than the primary benchmark.
- Use research wording such as monthly virtual asset model portfolio rather than public fund wording.
