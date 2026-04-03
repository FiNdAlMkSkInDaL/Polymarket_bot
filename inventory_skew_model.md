# Inventory Skew Model

## Purpose

`InventorySkewMaker` is meant for long-tail markets where the spread can look attractive but the real risk is being stuck with inventory after one side fills. The model below prices that risk directly by shifting the quote center as a function of current inventory.

## Inputs

- `current_inventory_usd`: signed inventory notional. Positive means long inventory that must be sold down. Negative means short inventory that must be bought back.
- `max_inventory_usd`: hard inventory cap for the market or region.
- `base_spread`: normal full quoted spread when inventory is flat.

The standalone quote helper also takes `mid_price` and optional `best_bid` / `best_ask` so the skew can be converted into actual bid and ask prices.

## Core Math

Let

$$
x = \mathrm{clip}\left(\frac{I}{I_{\max}}, -1, 1\right)
$$

where $I$ is current inventory and $I_{\max}$ is the maximum allowed inventory.

Define cubic risk pressure:

$$
u = |x|^3
$$

This is the key shape choice.

- Near flat inventory, $u$ stays tiny, so quotes remain close to symmetric.
- As inventory approaches the cap, $u$ accelerates sharply.

The skewed quote center is shifted by:

$$
\Delta c = -\operatorname{sign}(x) \cdot S \cdot \left(u + 0.25u^2\right)
$$

where $S$ is the base spread.

Interpretation:

- If inventory is long, $x > 0$, so $\Delta c < 0$: both quotes move down, making the ask more aggressive and the bid less attractive.
- If inventory is short, $x < 0$, so $\Delta c > 0$: both quotes move up, making the bid more aggressive and the ask less attractive.

The half-spread is widened slightly under stress:

$$
h = \frac{S}{2} \cdot \left(1 + 0.25u\right)
$$

Quoted prices become:

$$
\text{bid} = m - h + \Delta c
$$

$$
\text{ask} = m + h + \Delta c
$$

where $m$ is the mid-price.

## Risk Behavior

### Flat inventory

If $I = 0$, then $x = 0$, $u = 0$, and $\Delta c = 0$.

The model quotes symmetrically:

$$
\text{bid} = m - \frac{S}{2}, \quad \text{ask} = m + \frac{S}{2}
$$

### Moderate inventory

At half the cap, $|x| = 0.5$ and:

$$
u = 0.5^3 = 0.125
$$

The shift is still modest. The maker is nudged toward flattening, but not yet desperate.

### Extreme inventory

Near the cap, $u$ approaches 1 and the center shift approaches about $1.25 \times$ the base spread.

That is intentional. At full inventory stress the model is no longer trying to optimize maker edge first. It is trying to survive. The skew becomes strong enough that the exit-side quote can move through the touch and behave like a taker if a real BBO is provided.

## Aggressive Flattening Rule

The module also exposes an `aggressive_exit` flag.

- If `urgency >= 0.85` and inventory is long, the ask is allowed to cross down to `best_bid`.
- If `urgency >= 0.85` and inventory is short, the bid is allowed to cross up to `best_ask`.

This gives the strategy a smooth progression:

1. Flat: symmetric maker.
2. Mildly imbalanced: skewed maker.
3. Severely imbalanced: urgent exit, even if it means paying spread.

## Implementation Notes

- Module: `src/models/inventory_skew.py`
- Main pure math function: `compute_inventory_skew(...)`
- Quote conversion helper: `compute_inventory_skew_quotes(...)`
- Output includes `inventory_ratio`, `urgency`, `center_shift`, `adjusted_half_spread`, final quote prices, and whether the model recommends an aggressive exit.

## Recommended Injection Pattern

For Agent 2's long-tail strategy work:

- Use `compute_inventory_skew(...)` when you only need the curve state for ranking or telemetry.
- Use `compute_inventory_skew_quotes(...)` inside the actual maker quoting loop once mid-price and BBO are known.
- Keep `max_inventory_usd` local to each long-tail name or region basket, because skew quality depends on the cap being meaningful relative to actual fill sizes.