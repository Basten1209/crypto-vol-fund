# Rebalance Policy Comparison Note

## Modes

- `enter_once_then_drift`: enter once at the first day of the monthly hold window, then allow weights to drift until the hold window ends.
- `daily_rebalance_to_target`: enter at the first day of the monthly hold window, then restore the original target weights at the start of each hold-window day.
- Both modes go to cash outside the first 7-day or 14-day monthly hold window.
- Transaction costs are still ignored, so daily rebalancing results are optimistic until costs/slippage are modeled.

## Performance Summary

| Strategy | Policy | Cycle | Total Return | Ann. Vol | Sharpe | MDD | Turnover Sum | Realized Risk |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minimum_variance | enter_once_then_drift | 7 | -7.47% | 33.14% | -0.0508 | -28.54% | 13.0000 | 11.15% |
| equal_weight | enter_once_then_drift | 7 | -23.02% | 39.16% | -0.4172 | -36.94% | 13.0000 | 14.00% |
| minimum_variance | daily_rebalance_to_target | 7 | -7.47% | 32.77% | -0.0546 | -28.44% | 14.5252 | 10.99% |
| equal_weight | daily_rebalance_to_target | 7 | -22.12% | 38.93% | -0.3946 | -36.16% | 14.7271 | 13.94% |
| minimum_variance | enter_once_then_drift | 14 | 29.51% | 40.56% | 0.7917 | -29.06% | 13.0000 | 21.56% |
| equal_weight | enter_once_then_drift | 14 | 11.27% | 51.74% | 0.4543 | -35.68% | 13.0000 | 27.57% |
| minimum_variance | daily_rebalance_to_target | 14 | 31.09% | 40.64% | 0.8186 | -29.59% | 16.2046 | 21.16% |
| equal_weight | daily_rebalance_to_target | 14 | 8.54% | 51.76% | 0.4098 | -35.20% | 16.8166 | 27.40% |

## Readout

- For the 7-day minimum variance portfolio, daily rebalancing barely changes total return but slightly lowers volatility and MDD.
- For the 14-day minimum variance portfolio, daily rebalancing improves total return from 29.51% to 31.09%, with similar volatility and slightly worse MDD.
- Daily rebalancing requires many more order actions: 91 actions for 7-day windows and 182 actions for 14-day windows, versus 13 monthly entries in simple mode.
- The dashboard should therefore expose both modes: Simple Mode for easy product storytelling, Managed Mode for operational order guidance.

## Dashboard Implication

- Simple Mode order flow: monthly entry order, hold, then cash after window end.
- Managed Mode order flow: monthly entry order, daily rebalance orders inside the hold window, then cash after window end.
- Add a transaction-cost toggle before making Managed Mode the headline result.
