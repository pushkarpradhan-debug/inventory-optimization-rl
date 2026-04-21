# server/environment.py
# The CORE LOGIC of the environment.
# This is where step(), reset(), and state() are implemented.
#
# PLAIN ENGLISH EXPLANATION:
# - Imagine a small shop that sells items every day.
# - The agent decides each day: "Should I order more stock? If yes, how much?"
# - If the shop runs out of stock (stockout), it loses sales -> bad reward
# - If the shop has too much stock (overstock), it wastes money -> small penalty
# - The goal is to keep stock at a healthy level, not too low, not too high.

import random
from openenv.core.env_server import Environment
from .models import RestockAction, RestockObservation, RestockState


# ─── TASK CONFIGURATIONS ────────────────────────────────────────────────────
# Three tasks with increasing difficulty:
#   Task 1 (Easy)   - Stable demand, short lead time, plenty of starting stock
#   Task 2 (Medium) - Variable demand, moderate lead time, less starting stock
#   Task 3 (Hard)   - Unpredictable demand spikes, long lead time, tight stock

TASK_CONFIGS = {
    1: {
        "name": "Easy - Stable Demand",
        "description": "Demand is fixed at 5 units/day. Lead time is 1 day. Start with 20 units.",
        "starting_stock": 20,
        "demand_type": "fixed",       # demand never changes
        "demand_base": 5,
        "demand_variance": 0,
        "lead_time": 1,
        "max_days": 20,
        "difficulty": "easy",
    },
    2: {
        "name": "Medium - Variable Demand",
        "description": "Demand varies 3-8 units/day. Lead time is 3 days. Start with 15 units.",
        "starting_stock": 15,
        "demand_type": "variable",    # demand changes each day
        "demand_base": 5,
        "demand_variance": 3,         # can be base +/- variance
        "lead_time": 3,
        "max_days": 30,
        "difficulty": "medium",
    },
    3: {
        "name": "Hard - Demand Spikes",
        "description": "Demand spikes randomly (5-20 units/day). Lead time is 5 days. Start with 10 units.",
        "starting_stock": 10,
        "demand_type": "spiky",       # occasional big demand spikes
        "demand_base": 7,
        "demand_variance": 5,
        "lead_time": 5,
        "max_days": 40,
        "difficulty": "hard",
    },
}


class InventoryRestockEnvironment(Environment):
    """
    Inventory Restock Decision Environment.

    The agent plays the role of a shop manager.
    Each day (step), the agent sees the current stock and decides
    how many units to reorder.

    REWARD LOGIC (per step):
      +1.0  for every unit of demand that was satisfied (sold)
      -2.0  for every unit of unmet demand (stockout penalty)
      -0.1  for every unit of excess stock above a safety level (overstock penalty)
      -0.5  flat penalty if the agent orders nothing 3 days in a row while stock < 5
    """

    def __init__(self):
        self._state = RestockState()
        self._task_config = TASK_CONFIGS[1]
        self._pending_delivery_day = -1   # which day the pending order arrives
        self._days_without_order = 0      # track consecutive no-order days
        self._rng = random.Random(0)

    # ─── RESET ──────────────────────────────────────────────────────────────
    def reset(self, task_id: int = 1, seed: int = None) -> RestockObservation:
        """
        Start a fresh episode for the given task.
        task_id: 1 = Easy, 2 = Medium, 3 = Hard
        """
        task_id = max(1, min(3, task_id))  # clamp to valid range
        cfg = TASK_CONFIGS[task_id]
        self._rng = random.Random(seed if seed is not None else task_id)

        self._task_config = cfg
        self._pending_delivery_day = -1
        self._days_without_order = 0

        initial_demand = self._generate_demand(cfg, day=1)

        self._state = RestockState(
            current_stock=cfg["starting_stock"],
            daily_demand=initial_demand,
            lead_time=cfg["lead_time"],
            day=1,
            max_days=cfg["max_days"],
            pending_order=0,
            total_reward=0.0,
            task_id=task_id,
        )

        return RestockObservation(
            current_stock=self._state.current_stock,
            daily_demand=self._state.daily_demand,
            lead_time=self._state.lead_time,
            day=self._state.day,
            pending_order=self._state.pending_order,
            done=False,
            reward=0.0,
            message=f"Episode started. Task: {cfg['name']}. You have {cfg['starting_stock']} units.",
        )

    # ─── STEP ───────────────────────────────────────────────────────────────
    def step(self, action: RestockAction) -> RestockObservation:
        """
        Advance the simulation by one day.

        1. Receive any pending delivery (if lead_time days have passed)
        2. Sell units to meet demand
        3. Calculate reward for this day
        4. Process the agent's restock order
        5. Generate tomorrow's demand
        6. Advance day counter
        """
        s = self._state
        cfg = self._task_config
        order_qty = action.restock_quantity
        messages = []

        # --- 1. RECEIVE PENDING DELIVERY ---
        if s.pending_order > 0 and s.day >= self._pending_delivery_day:
            s.current_stock += s.pending_order
            messages.append(f"Delivery of {s.pending_order} units arrived!")
            s.pending_order = 0
            self._pending_delivery_day = -1

        # --- 2. MEET DEMAND ---
        demand = s.daily_demand
        units_sold = min(s.current_stock, demand)
        unmet_demand = demand - units_sold
        s.current_stock -= units_sold

        if unmet_demand > 0:
            messages.append(f"STOCKOUT: {unmet_demand} units of demand unmet!")
        else:
            messages.append(f"Sold {units_sold} units. Stock left: {s.current_stock}.")

        # --- 3. CALCULATE REWARD ---
        reward = self._calculate_reward(units_sold, unmet_demand, s.current_stock, order_qty)
        s.total_reward += reward

        # --- 4. PROCESS RESTOCK ORDER ---
        if order_qty > 0:
            if s.pending_order > 0:
                messages.append(f"Order ignored: previous order of {s.pending_order} still pending.")
            else:
                s.pending_order = order_qty
                self._pending_delivery_day = s.day + cfg["lead_time"]
                messages.append(f"Ordered {order_qty} units. Arrives in {cfg['lead_time']} days.")
            self._days_without_order = 0
        else:
            self._days_without_order += 1

        # --- 5. ADVANCE DAY ---
        s.day += 1
        done = s.day > s.max_days

        # --- 6. GENERATE TOMORROW'S DEMAND ---
        if not done:
            s.daily_demand = self._generate_demand(cfg, s.day)

        return RestockObservation(
            current_stock=s.current_stock,
            daily_demand=s.daily_demand,
            lead_time=cfg["lead_time"],
            day=s.day,
            pending_order=s.pending_order,
            done=done,
            reward=round(reward, 3),
            message=" | ".join(messages),
        )

    # ─── STATE ──────────────────────────────────────────────────────────────
    @property
    def state(self) -> RestockState:
        """Return the full internal state (useful for debugging)."""
        return self._state

    # ─── HELPERS ────────────────────────────────────────────────────────────
    def _generate_demand(self, cfg: dict, day: int) -> int:
        """Generate how many units customers want today, based on task type."""
        demand_type = cfg["demand_type"]
        base = cfg["demand_base"]
        variance = cfg["demand_variance"]

        if demand_type == "fixed":
            return base

        elif demand_type == "variable":
            # Random demand within a range
            return max(1, base + self._rng.randint(-variance, variance))

        elif demand_type == "spiky":
            # Most days: normal demand. Every ~5 days: big spike.
            if self._rng.random() < 0.20:  # 20% chance of spike
                return base + self._rng.randint(variance, variance * 3)
            else:
                return max(1, base + self._rng.randint(-variance, variance))

        return base

    def _calculate_reward(
        self,
        units_sold: int,
        unmet_demand: int,
        remaining_stock: int,
        order_qty: int,
    ) -> float:
        """
        Reward function — designed to be simple and logical.

        Good behaviours rewarded:
          - Selling units to customers

        Bad behaviours penalized:
          - Running out of stock (stockout)
          - Sitting on huge excess inventory (overstock)
          - Refusing to order when stock is critically low
        """
        reward = 0.0

        # Reward for every unit sold
        reward += units_sold * 1.0

        # Heavy penalty for every unit of unmet demand
        reward -= unmet_demand * 2.0

        # Mild penalty for holding too much excess stock (above 15 units)
        safety_level = 15
        if remaining_stock > safety_level:
            overstock = remaining_stock - safety_level
            reward -= overstock * 0.1

        # Penalty for ignoring restock when stock is dangerously low
        if remaining_stock < 5 and order_qty == 0 and self._days_without_order >= 3:
            reward -= 0.5

        return reward
