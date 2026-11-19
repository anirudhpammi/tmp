# Counterfactual Fill Labels from L3 (MBO) Data

**Question the model answers:** *If I had been at position X in the queue at price p at time t, would I have been filled within Δ?*

**Idea:** Under time priority, a bid with priority `s` at price `p` would have been **first filled** at the earliest of:

- **(a)** the first fill at `p` of any order with priority **≥ s** (anyone behind that spot, or the order itself, getting filled means everything ahead of it is gone), or
- **(b)** the first fill at a price **below p** (trade-through: level `p` was emptied).

Real cancellations ahead are already reflected in the timing of real fills, so they never need to be tracked. No per-order replay is needed.

All rules below are written for **bids**. For asks, flip the price comparisons (fills **above** `p` are trade-throughs).

---

## Pipeline

### 1. Rebuild the book from the MBO feed
For every event keep: order ID, side, price, quantity, and the exchange's **priority field** (`MDOrderPriority` in CME MDP 3.0). Use this field rather than timestamps so that modifications that reset priority are handled correctly.

### 2. Extract a fill log
For every resting order that gets filled, record:
`time, side, price, order_id, priority, qty`

### 3. Build the fill frontier per (side, price)
For each price level, take its fills in time order and store two arrays:

- `times`
- `F` = running maximum of filled priority

`F` never decreases, so it can be binary-searched.

### 4. Build the trade-through index per side
A segment tree over bid-side fill prices in time order, storing the **minimum** price per range (maximum for asks). To find "first fill after `t` with price < `p`", descend the tree: O(log n).

### 5. Generate queries `(t, p, s)`
- **t:** sample on events, not on every message.
- **p:** price levels of interest (e.g., best bid down to k ticks below it).
- **s:** priorities of real orders resting at `p` at time `t`, plus synthetic positions. To place a position X lots deep, cumulate volume through the queue at `t` and set `s` between the two neighbouring orders.
- **Features (as of t only):** volume ahead (X), size at level, distance from best, spread, imbalance, recent flow, volatility.

### 6. Label each query
```
T_a    = first time with F_p >= s          (binary search; must be > t)
T_b    = first fill below p after t        (segment tree)
T_fill = min(T_a, T_b)                     (∞ if neither before session end → censored)
label  = T_fill <= t + Δ
```
Alternatively keep `(T_fill − t, censored)` as a survival target, which covers every Δ at once.

### 7. Verify against brute force
On a random subsample, run the full per-order replay (below) and confirm both methods give identical labels.

### 8. Train
Either a classifier for P(fill within Δ), or a survival model (discrete-time hazard or gradient-boosted survival).

### 9. Validate
Split **by date**. Check calibration: bucket predicted fill probabilities and compare to realized fill rates.

### 10. Query
Input: book state at `t`, price `p`, position `X`. Output: P(fill within Δ), or a full fill-time curve.

**Cost:** O(n) precompute, O(log n) per label.

**Full fills (instead of first fill):** replace the running max with cumulative filled volume at `p` from orders with priority ≥ `s`, plus fills below `p`; find when it reaches your size `q`.

---

## Reference: brute-force replay (for step 7)

```python
ahead = {oid: qty for oid, qty in level_queue(t, side, p)
         if priority(oid) < priority(o)}
filled = 0
for ev in events(t, t + dt):          # o's own cancel is ignored
    if ev.type in ("cancel", "modify_lose_priority") and ev.oid in ahead:
        del ahead[ev.oid]
    elif ev.type == "modify_reduce" and ev.oid in ahead:
        ahead[ev.oid] = ev.new_qty
    elif ev.type == "fill" and ev.side == side:
        if ev.oid in ahead:
            ahead[ev.oid] -= ev.qty
            if ahead[ev.oid] <= 0:
                del ahead[ev.oid]
        elif ev.price <= p and ev.oid != o.oid:   # bid side
            filled += min(ev.qty, q - filled)     # volume that would have hit o
    if filled >= q:
        break
```

---

## Toy example

### Setup at t = 0
Six orders rest at a bid of **100**, one lot each:

| Order | Priority s |
|---|---|
| A | 1 |
| B | 2 |
| C | 3 |
| D | 4 |
| E | 5 |
| F | 6 |

### What actually happens
| Time | Event |
|---|---|
| 2 | C cancels |
| 3 | A filled |
| 4 | B filled |
| 5 | D cancels |
| 6 | E filled |

No fills below 100.

### Step 3: frontier at 100
Fill log at 100: A at t=3, B at t=4, E at t=6.

| time | 3 | 4 | 6 |
|---|---|---|---|
| F (running max priority) | 1 | 2 | 5 |

### Step 4: trade-through index
No fills below 100 → `T_b = ∞` for every query.

### Step 6: labels for queries at t = 0
| Order (s) | First F ≥ s | T_fill | Filled within Δ=5? | Filled within Δ=10? |
|---|---|---|---|---|
| A (1) | t=3 | 3 | yes | yes |
| B (2) | t=4 | 4 | yes | yes |
| C (3) | t=6 (F=5) | 6 | no | yes |
| D (4) | t=6 (F=5) | 6 | no | yes |
| E (5) | t=6 (F=5) | 6 | no | yes |
| F (6) | never | ∞ | no (censored) | no (censored) |

### Check: C was cancelled in reality
Had C stayed: by t=6, A and B are filled, so nothing is ahead of C. E's fill at t=6 would have hit C instead. The frontier gives t=6 without tracking any cancellations. Same reasoning for D (A, B, C all gone by t=6).

### Synthetic positions
| Position | s | First F ≥ s | T_fill |
|---|---|---|---|
| Between D and E | 4.5 | t=6 | 6 |
| Back of queue | 7 | never | ∞ |

### Extension: a trade-through
Add: F cancels at t=6.5 (level 100 now empty), then an order at **99** is filled at t=7.

- `T_b = 7` for any query at 100.
- Back-of-queue (s=7), and F if it had stayed, get `T_fill = 7`: that seller would have hit you at 100 first.
