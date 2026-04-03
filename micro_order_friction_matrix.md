# Micro Order Friction Matrix

## Objective

Establish the hard execution-cost floor for Long Tail micro-orders on Polymarket so Agent 2 only screens spreads wide enough to survive the exchange tax.

Scope requested:

- Review our execution engine.
- Review Polymarket's known relayer / gas / fee structure.
- Compute the break-even spread for a strict maker-maker round trip at `$10`, `$25`, `$50`, and `$100` notionals.

## Executive Verdict

For a strict maker-maker round trip on the Polymarket CLOB, the recurring exchange tax is:

- Fixed cost per posted order: `$0.00`
- Fixed cost per cancel: `$0.00`
- Variable maker fee per fill: `0 bps`
- Required break-even spread from exchange tax alone: `0.00c/share`

That means the hard numerical fee floor for Agent 2's Long Tail screener is:

- Strict maker-maker floor: `0.00c/share`

The real exchange tax only appears once a leg becomes taker. At the fee apex near `p = 0.50`, one taker leg costs `1.00c/share`, and a taker-taker round trip costs `2.00c/share`.

## What The Codebase Actually Does

### 1. Live order posting is offchain

Our live path wraps `py-clob-client` and posts signed orders to the CLOB API, not directly to Polygon for each order action.

- `src/trading/executor.py` creates a signed order and calls `clob.post_order(...)`.
- `post_only=True` is explicitly forwarded on passive orders.
- There is no per-order gas budgeting or gas debit in the normal maker posting path.

### 2. Fee payload is only attached when needed

Our executor only attaches `fee_rate_bps` when it is nonzero.

- In `src/trading/executor.py`, `order_args.fee_rate_bps` is only set when `fee_rate_bps > 0`.
- In `src/trading/fee_cache.py`, makers are treated as `0 bps` by design: `maker_fee_bps() -> 0`.

### 3. Our fee model matches the current taker-fee regime

The local fee utility defines the taker fee curve as:

$$
f(p) = f_{max} \cdot 4p(1-p)
$$

with current configured peak:

- `FEE_MAX_PCT = 2.00`

So the one-way taker fee in dollars per share is:

$$
\text{taker fee per share} = p \cdot f(p)
$$

At `p = 0.50`:

$$
f(0.5) = 2.00\% \\
\text{fee/share} = 0.50 \times 0.02 = 0.01\ \text{USD} = 1.00\text{c/share}
$$

## What Polymarket Public Docs Say

### CLOB order placement is offchain

Polymarket documents the CLOB as:

- offchain order matching
- onchain settlement
- EIP-712 signed orders
- L2 authenticated REST order submission

That means normal quote placement and cancellation are not per-order onchain gas events.

### Gas belongs to wallet setup and inventory management, not each quote

Polymarket's docs separate:

- post limit orders via the CLOB REST API
- approvals, deposits, splits, merges, and other inventory moves via relayer / onchain paths

So the recurring maker loop does not pay a fresh gas charge every time it posts or cancels. Gas matters for:

- initial USDC / token approvals
- deposits / withdrawals / bridging
- split / merge inventory operations
- onchain emergency cancellation if you bypass the operator path

Those are real costs, but they are not recurring per-round-trip maker execution costs.

### Makers are funded by taker fees

Polymarket's maker rebates page states:

- maker rebates are funded by taker fees
- taker fees are the charged side
- maker orders can earn rebate flow

This is consistent with our code assumption that direct maker fee is `0 bps`.

## Friction Decomposition

### Recurring costs inside a strict maker-maker loop

For a passive buy at the bid and a passive sell at the ask:

- entry order post: `$0.00`
- entry fill fee: `$0.00`
- exit order post: `$0.00`
- exit fill fee: `$0.00`
- cancels / requotes: `$0.00` direct exchange tax

Therefore:

$$
\text{break-even spread}_{maker,maker} = 0.00\text{c/share}
$$

### One-time or external costs not attributable to each maker round trip

These are real, but should be amortized separately at portfolio level, not charged per micro-trade in the screener:

- wallet funding / bridging gas
- USDC allowance approval gas
- conditional token allowance approval gas
- split / merge inventory gas
- withdrawal gas

## Break-Even Matrix

Assumption for the matrix:

- mid-price `p = 0.50`
- binary share count = `notional / p = 2 x notional`
- strict maker entry and strict maker exit

| Position Notional | Approx Shares at 50c | Fixed Recurring Cost | Variable Maker Fee | Total Round-Trip Tax | Break-Even Spread Required |
| ----------------- | -------------------: | -------------------: | -----------------: | -------------------: | -------------------------: |
| `$10`             |                 `20` |              `$0.00` |            `$0.00` |              `$0.00` |              `0.00c/share` |
| `$25`             |                 `50` |              `$0.00` |            `$0.00` |              `$0.00` |              `0.00c/share` |
| `$50`             |                `100` |              `$0.00` |            `$0.00` |              `$0.00` |              `0.00c/share` |
| `$100`            |                `200` |              `$0.00` |            `$0.00` |              `$0.00` |              `0.00c/share` |

## Stress Table: If A Leg Accidentally Takes

This is the operationally important backup table, because the true exchange tax appears the moment passive execution degrades into taker execution.

At `p = 0.50` and `f_max = 2.00%`:

- one taker leg = `1.00c/share`
- two taker legs = `2.00c/share`

| Position Notional | Approx Shares at 50c | One Taker Leg Cost | Taker-Taker Cost | Break-Even Spread If One Leg Takes | Break-Even Spread If Both Legs Take |
| ----------------- | -------------------: | -----------------: | ---------------: | ---------------------------------: | ----------------------------------: |
| `$10`             |                 `20` |            `$0.20` |          `$0.40` |                      `1.00c/share` |                       `2.00c/share` |
| `$25`             |                 `50` |            `$0.50` |          `$1.00` |                      `1.00c/share` |                       `2.00c/share` |
| `$50`             |                `100` |            `$1.00` |          `$2.00` |                      `1.00c/share` |                       `2.00c/share` |
| `$100`            |                `200` |            `$2.00` |          `$4.00` |                      `1.00c/share` |                       `2.00c/share` |

This table is not the requested maker-maker base case, but it is the correct risk-control overlay for Long Tail deployment.

## Deployment Guidance For Agent 2

### Hard fee floor

If Agent 2 is truly screening for strict maker-maker spread capture only:

- Minimum spread floor required to survive exchange tax: `0.00c/share`

### Operational floor worth enforcing in practice

If you want protection against one degraded leg becoming taker near the fee apex:

- Recommended operational minimum: `>= 1.00c/share`

If you want protection against both legs paying taker fee:

- Defensive minimum: `>= 2.00c/share`

### Recommended interpretation

For pure exchange-tax math, the Long Tail screener does not need a positive spread floor for strict maker-maker loops.

For live deployment discipline, the safer hard floor is:

- `1.00c/share` if one-leg taker contamination is plausible
- `2.00c/share` if the lane may degrade into taker-taker exits under stress

## Final Answer

The exact maker-maker break-even spread required by Polymarket exchange tax for `$10`, `$25`, `$50`, and `$100` micro-orders is:

- `$10`: `0.00c/share`
- `$25`: `0.00c/share`
- `$50`: `0.00c/share`
- `$100`: `0.00c/share`

So the strict fee floor for Agent 2's Long Tail screener is:

- `0.00c/share`

The first nonzero tax floor appears only when execution stops being maker-maker. At the 50c fee apex, that floor is:

- `1.00c/share` for one taker leg
- `2.00c/share` for a full taker-taker round trip