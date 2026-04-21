# baseline_agent.py
# Rule-based inventory agent — no ML required.
#
# STRATEGY (Plain English):
#   Instead of waiting until stock is already low to order,
#   this agent thinks AHEAD:
#
#   "If the current stock + pending order is not enough to survive
#    the entire lead time at worst-case demand, place an order NOW."
#
# WHY THIS FIXES THE HARD TASK:
#   Task 3 has a 5-day lead time and occasional demand spikes (up to 22 units/day).
#   The old agent waited too long — by the time it noticed stock was low,
#   it had already placed an order and had to wait 5 days with zero stock.
#
#   This agent uses PREDICTIVE restocking:
#     expected_stock_at_delivery = current_stock + pending_order - (worst_case_demand * lead_time)
#   If that number drops below a safety buffer → order immediately.
#
# This logic works for all 3 tasks because:
#   - Task 1: low worst-case demand, rarely triggers
#   - Task 2: moderate demand variance, orders proportionally
#   - Task 3: high worst-case demand, orders proactively before stock runs out

from env.models import RestockAction


def simple_agent(obs) -> RestockAction:
    """
    Predictive threshold agent.

    Works by estimating how much stock will remain
    when the next order actually arrives (after lead_time days).

    If that projected stock is below a safety buffer → order now.
    """

    lead_time = obs.lead_time

    # Worst-case planning demand per day (accounts for spikes in Task 3)
    # Normal days: ~7 units. Spike days: up to 22 units.
    # Use a conservative estimate of 14 to be safe without massively overstocking.
    WORST_CASE_DAILY = 14
    SAFETY_BUFFER = 10      # extra units we want as cushion after delivery arrives

    # Target stock level we want when the next delivery arrives
    target_stock = WORST_CASE_DAILY * lead_time + SAFETY_BUFFER
    # Task 3: 14 * 5 + 10 = 80 (but max order is 50, so we stay aggressive)

    # Projected stock at delivery time:
    # What we have now + pending order - what we'll sell during lead time (worst case)
    projected_stock = (
        obs.current_stock
        + obs.pending_order
        - WORST_CASE_DAILY * lead_time
    )

    # Only place an order if there's no pending order AND we're below target
    if obs.pending_order == 0 and projected_stock < SAFETY_BUFFER:
        # Order enough to reach target (from current stock alone, ignoring pending)
        order_qty = target_stock - obs.current_stock
        order_qty = max(0, min(50, order_qty))   # clamp to valid range [0, 50]
        return RestockAction(restock_quantity=order_qty)

    # Secondary safety: if no pending order and stock is critically low, order anyway
    if obs.pending_order == 0 and obs.current_stock < WORST_CASE_DAILY * 2:
        order_qty = min(50, target_stock - obs.current_stock)
        return RestockAction(restock_quantity=max(0, order_qty))

    # Otherwise, do nothing this step
    return RestockAction(restock_quantity=0)