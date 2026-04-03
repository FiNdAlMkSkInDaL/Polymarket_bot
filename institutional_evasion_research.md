# Institutional Evasion Research

## Executive Summary

Our recent maker-style failures all point to the same structural conclusion: we are losing where institutional market makers are strongest and fastest.

- `OBI Evader` failed because wide spreads do not stay wide long enough for a non-colocated REST participant to claim queue priority.
- `Vacuum Maker` failed because the presumed post-sweep vacuum is refilled almost immediately by hybrid-AMM or institutional replenishment logic.
- `Whale Wall Jumper` failed because deep displayed liquidity is not trustworthy; the observed out-of-sample wall universe was overwhelmingly spoof-like, with near-mid walls pulled at effectively total frequency.
- `ExhaustionFader` added a second warning: even when the idea is conceptually sound, the trigger can collapse to zero executed opportunity if it depends on the exact microstructure regime that institutions already dominate.
- The five-day long-tail universe scan also matters: there was no robust archive evidence of a broad, stable ecosystem of wide-spread low-volume books waiting for passive harvesting.

The implication is blunt. We should stop searching for edge in queue position, displayed depth, or short-lived spread vacuums. Those are institutional domains. Our edge, if it exists, must live in slower structural dislocations that are too sparse, too research-heavy, too operationally awkward, or too balance-sheet inefficient for larger market makers to prioritize.

This memo proposes two such pivots:

1. `Hybrid Venue Dislocation Arb`: exploit lag or convexity mismatch between executable CLOB prices and executable AMM prices, using only immediately executable quotes rather than trusted resting depth.
2. `Resolution-Clock Hazard Short`: exploit the market's systematic overpricing of lingering YES optionality when time passes without confirming evidence and the fair resolution probability decays toward NO.

Both frameworks are explicitly designed to avoid dependence on latency races, queue rank, and spoofable book depth.

## Hard Constraints

These are not annoyances. They define the feasible strategy space.

- We do not have colocation.
- We enter through a REST API boundary, so our quote placement latency is structurally worse than institutional or in-venue participants.
- We cannot trust displayed L2 depth as a first-order signal because deep walls are often pulled before impact.
- We operate with roughly `$5,000` max capital, so strategies must be capital-efficient and robust at small notional.
- The dynamic fee curve is nontrivial around the middle of the probability range:

$$
\text{fee}(p) = f_{\max} \cdot 4p(1-p)
$$

This means naive mid-range taker trading gets taxed hardest exactly where generic mean-reversion or directional intraday ideas usually want to play.

## Autopsy Synthesis

### What failed structurally

The common failure mode is not "bad parameterization". It is that our prior designs were competing in the wrong game.

`OBI Evader` and `Vacuum Maker` assumed that a visible spread dislocation would remain open long enough for a slower participant to monetize it. Out-of-sample, that assumption failed. The book was replenished before we could acquire meaningful queue position, so the only fills that survived were the toxic ones.

`Whale Wall Jumper` assumed that large displayed resting interest signaled real intent. Out-of-sample, that assumption also failed. The pulled-wall rate was overwhelming, and the only useful slice was a highly filtered far-from-mid subset. The lesson is that raw depth is not information; it is adversarial theater unless confirmed by other structure.

`ExhaustionFader` showed a second-order issue: even if a hypothesis is directionally plausible, if it is keyed to the wrong microstructure window, the opportunity set can vanish entirely.

### What this means for edge discovery

We should seek edge in quantities institutions are less motivated to optimize:

- slower-moving cross-venue or cross-mechanism inconsistencies,
- event-specific hazard misestimation,
- state transitions driven by information processing rather than by queue speed,
- sparse, awkward, non-scalable setups that are unattractive to large market makers.

## Framework 1: Hybrid Venue Dislocation Arbitrage

## Core Thesis

Polymarket's hybrid structure likely creates moments when the executable AMM price and the executable CLOB price disagree by more than transaction costs, slippage, and a latency safety buffer. This is not a queue-priority problem. It is a pricing-consistency problem.

Institutional market makers dominate displayed quote replenishment, but they do not necessarily eliminate every hybrid dislocation because doing so requires capital, inventory, router complexity, and cross-mechanism state management. Sparse, low-notional inconsistencies can persist longer than pure CLOB microsecond opportunities, especially when one side is warehouse-driven and the other is order-flow-driven.

The key design principle is that we do not trust resting depth. We only compare executable prices.

## Tradeable Object

At time $t$, define:

- $P^{\text{clob}}_{\text{buy}}(q)$: the executable average price to buy size $q$ on the CLOB,
- $P^{\text{clob}}_{\text{sell}}(q)$: the executable average price to sell size $q$ on the CLOB,
- $P^{\text{amm}}_{\text{buy}}(q)$: the executable average price to buy size $q$ from the AMM,
- $P^{\text{amm}}_{\text{sell}}(q)$: the executable average price to sell size $q$ to the AMM.

These must include the full path cost, not just top-of-book marks.

Then define two directional dislocations:

$$
\Delta_{1}(q) = P^{\text{amm}}_{\text{sell}}(q) - P^{\text{clob}}_{\text{buy}}(q)
$$

$$
\Delta_{2}(q) = P^{\text{clob}}_{\text{sell}}(q) - P^{\text{amm}}_{\text{buy}}(q)
$$

If either is positive by enough margin after costs, there is a cross-mechanism arbitrage or quasi-arbitrage.

## Mathematical Entry Logic

For size $q$, enter only if:

$$
\Delta(q) > C_{\text{fees}}(q) + C_{\text{slippage}}(q) + C_{\text{gas}}(q) + C_{\text{latency}} + z\sigma_{\Delta}
$$

Where:

- $C_{\text{fees}}(q)$ includes Polymarket's dynamic taker-fee curve on the CLOB leg and any AMM fee component.
- $C_{\text{slippage}}(q)$ is the expected impact for both legs at size $q$.
- $C_{\text{gas}}(q)$ is the chain-routing cost if relevant.
- $C_{\text{latency}}$ is a fixed safety haircut for route/approval delays.
- $\sigma_{\Delta}$ is recent realized noise of the venue spread process.
- $z$ is a conservative multiple, for example `2` to `3`.

Because we only have `$5,000` max capital, size should be solved from the worst-case unwind condition rather than from theoretical edge alone:

$$
q^* = \min\left(q_{\text{capital}}, q_{\text{inventory}}, q_{\text{venue-depth-safe}}\right)
$$

The trade is not "buy what looks cheap". It is "buy on venue A and flatten on venue B only when the executable spread clears a large conservative hurdle".

## Where the Edge Likely Lives

The most plausible regimes are not the most crowded ones.

- Wings of the probability distribution, where the dynamic fee curve is lower and smaller imbalances can remain economically meaningful.
- Thin but not dead markets, where a warehouse or AMM state can lag order-flow repricing.
- Post-news repricing windows measured in seconds to minutes rather than milliseconds, especially when the CLOB and the AMM respond through different participant mixes.
- Complement/parity pockets where YES, NO, and AMM routing imply inconsistent synthetic prices.

There is also a more general convexity angle. The AMM pricing function is continuous and inventory-sensitive. The CLOB is discrete and participant-sensitive. Those are different microeconomic machines. When they disagree, the edge is structural rather than queue-based.

## Why Institutions May Not Fully Eliminate It

This is the critical institutional-evasion point.

- The edge is sparse and research-heavy, not continuous like passive quoting.
- The notional is small; many institutions will not deploy engineering and balance-sheet complexity to capture low-dollar, fragmented dislocations.
- Cross-mechanism routing creates operational drag: approvals, routing state, inventory fragmentation, and hedging complexity.
- The opportunity often requires carrying weird inventory or resolving awkward partial fills, which is unattractive relative to clean maker flow.
- Larger firms optimize scalable quote capture first. They do not necessarily chase every cross-mechanism convexity mismatch in thin event markets.

## Failure Modes

This framework will fail if:

- the AMM quote is itself stale, inaccessible, or costly to route against in real size,
- the observed dislocation is just a measurement artifact caused by partial path depth,
- operational frictions dominate the theoretical spread,
- the trade requires both legs to complete inside a latency window that is still too short.

So the strategy must be explicitly conservative and sparse.

## Why It Sidesteps Our Current Weaknesses

It does not require us to win the quote race. It requires us to detect when two pricing mechanisms disagree by enough to survive slow execution. It uses executable path prices rather than spoofable displayed depth. And it can be run as a taker-first or taker-plus-immediate-hedge system instead of a passive queue game.

## Framework 2: Resolution-Clock Hazard Short

## Core Thesis

Prediction markets systematically overvalue residual YES optionality in many event classes when time passes without confirming evidence. That is especially true in markets where retail demand is long optionality and emotionally sticky, while informed participants are selective and inventory-aware.

The intuition is simple: as the deadline approaches, the probability that the event still occurs is not constant if nothing has happened yet. In many markets, "nothing happened" is itself information. The fair price should decay toward NO, but the market often updates too slowly because participants anchor on narrative possibility instead of hazard-adjusted probability.

This is not a microsecond edge. It is a calendar-and-information-processing edge.

## Mathematical Model

Let the YES event occur before deadline $T$. Define a conditional hazard process $\lambda(t \mid \mathcal{I}_t)$ based on currently observed information $\mathcal{I}_t$.

Then the fair remaining YES probability at time $t$ is:

$$
P_{\text{fair}}(t) = 1 - \exp\left(-\int_t^T \lambda(s \mid \mathcal{I}_t) \, ds\right)
$$

The key is that $\mathcal{I}_t$ includes not only positive news, but also the absence of expected news.

For many event classes, a practical decomposition is:

$$
\lambda(t \mid \mathcal{I}_t) = \lambda_0(t) \cdot M_{\text{news}}(t) \cdot M_{\text{schedule}}(t) \cdot M_{\text{state}}(t)
$$

Where:

- $\lambda_0(t)$ is the baseline event hazard,
- $M_{\text{news}}(t)$ boosts or suppresses hazard based on external evidence,
- $M_{\text{schedule}}(t)$ penalizes the event after key checkpoints pass,
- $M_{\text{state}}(t)$ accounts for market-specific state such as score, vote count, weather path, filing status, or countdown windows.

The signal is then the market residual:

$$
R(t) = P_{\text{mkt}}(t) - P_{\text{fair}}(t)
$$

Enter a NO-biased trade only when:

$$
R(t) > C_{\text{fees}} + C_{\text{liquidity}} + C_{\text{model-risk}} + k\sigma_R
$$

This is a model-divergence trade, not a chart pattern.

## Practical Entry Design

The cleanest initial implementation is not a universal hazard model. It is a family of event templates.

Examples:

- Deadline-driven political or regulatory events: if a filing, nomination, court action, or executive step has not occurred by successive milestone windows, YES probability should decay faster than retail intuition.
- Sports or weather event derivatives: if state updates arrive and the triggering path narrows, lingering YES optionality can remain overpriced.
- Crypto and macro event markets: scheduled catalysts often produce a front-loaded hazard profile; after the catalyst window passes quietly, YES should decay sharply.

An initial template-driven rule can use milestone penalties rather than continuous hazard integration.

Example discrete approximation:

$$
P_{\text{fair}}(t) = P_0 \cdot e^{-\beta \tau} \cdot \prod_{j=1}^{m(t)} (1 - \delta_j)
$$

Where:

- $P_0$ is a baseline prior,
- $\tau$ is time elapsed since the last confirmatory update,
- $\beta$ is a stale-news decay coefficient,
- $\delta_j$ are milestone penalties once key windows pass without confirmation.

That gives a directly tradeable rule:

1. map the market to an event template,
2. compute the current hazard-adjusted fair YES price,
3. short YES or buy NO only when the market trades materially above that fair value,
4. size modestly because the holding period is hours to days, not milliseconds.

## Why Institutions May Not Fully Eliminate It

This edge is unattractive to many market makers for reasons that play to our strengths.

- It is research-intensive and event-specific rather than flow-generic.
- It can require holding risk for hours or days, which is balance-sheet inefficient for firms optimized for intraday spread capture.
- It depends on external information modeling and domain templates, not just better market making.
- The opportunity is sparse and heterogeneous, which makes it hard to scale with a standard quoting stack.
- With only `$5,000` capital, we do not need this to scale institutionally. We only need it to be positively asymmetric on a modest set of high-confidence markets.

In short, large market makers are better than us at microstructure extraction. They are not automatically better at small-cap, event-specific hazard modeling when the payoff is slow and operationally awkward.

## Why It Sidesteps Our Current Weaknesses

This framework does not depend on seeing a wall, winning queue, or reacting faster than replenishment logic. It relies on the market being wrong about the effect of time and non-events on resolution probability. That is a cognitive and modeling inefficiency, not a latency inefficiency.

## Why Toxic Exhaustion Is Not the First Replacement Pivot

The idea remains intellectually plausible: let the market makers fight toxic takers, then enter only after both the aggressive flow and the replenishment response are exhausted. But right now it is the third-priority direction, not the first.

Why:

- It still lives partly inside the microstructure battlefield that just killed our prior maker ideas.
- It requires very careful separation of true exhaustion from merely delayed continuation.
- The evidence so far includes zero realized opportunity under our first fade implementation.

That does not kill the concept forever. It means we should treat it as a secondary research track after we have tested slower structural edges that do not require us to infer intent from adversarial L2 behavior.

## Recommended Strategic Pivot

If we are choosing where to spend the next research cycle, the priority order should be:

1. `Resolution-Clock Hazard Short`
2. `Hybrid Venue Dislocation Arb`

That ordering is deliberate.

`Resolution-Clock Hazard Short` is the cleanest institutional-evasion thesis because it attacks a slower, cognition-heavy inefficiency that is naturally underexploited by firms optimized for continuous spread capture.

`Hybrid Venue Dislocation Arb` is the second-best pivot because it still depends on execution quality, but it is competing on pricing consistency rather than on queue speed, and it uses executable path quotes rather than spoofable displayed depth.

## Concrete Next Research Questions

For `Resolution-Clock Hazard Short`:

- Which event classes exhibit the strongest non-event decay toward NO?
- How should markets be templated so that milestone failures can be encoded systematically?
- What external data source is reliable enough to timestamp confirming versus absent evidence?
- How large is the average residual $R(t)$ after fees and spread in practice?

For `Hybrid Venue Dislocation Arb`:

- Can we obtain stable executable AMM quote snapshots at the same timestamps as the CLOB stream?
- How persistent are cross-mechanism dislocations once all path costs are included?
- Are the best opportunities concentrated in wings, in thin books, or around specific event classes?
- Does complement parity between YES, NO, and AMM pricing create a more robust signal than direct pairwise venue spreads?

## Final Conclusion

The evidence from the failed OOS strategies is strong enough to stop pretending that better spread-gating or cleaner spoof filters alone will produce a durable edge. Institutional market makers own the short-horizon quote-replenishment game. We should not keep attacking them on their preferred terrain.

Our next edge should come from structural slowness, not mechanical speed.

The two best candidates are:

- `Hybrid Venue Dislocation Arb`: a fee-aware executable-price arbitrage across the hybrid AMM and CLOB layers.
- `Resolution-Clock Hazard Short`: a hazard-model-based short-YES or long-NO framework that monetizes the market's slow recognition that time passing without confirmation is bearish information.

Those are the pivots most likely to survive our disadvantages in latency, capital, and queue access, because they target inefficiencies that institutional market makers are less structurally motivated to erase.