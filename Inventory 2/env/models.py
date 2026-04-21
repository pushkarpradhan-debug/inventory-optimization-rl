# models.py
# Defines what the agent SEES (Observation), what it DOES (Action),
# and the current internal state of the environment (State).

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class RestockAction(Action):
    """
    What the agent decides to do each step.

    restock_quantity: How many units to order.
      - 0 means "don't order anything"
      - 1-50 is a valid range for a small shop
    """
    restock_quantity: int = Field(
        ...,
        ge=0,
        le=50,
        description="Number of units to restock (0 = do nothing, max 50)"
    )


class RestockObservation(Observation):
    """
    What the agent sees after each step.

    Inherits: done (bool), reward (float) from Observation base class.
    """
    current_stock: int = Field(..., description="Units currently in the shop")
    daily_demand: int = Field(..., description="How many units customers want today")
    lead_time: int = Field(..., description="Days until a restock order arrives")
    day: int = Field(..., description="Current simulation day (1 to max_days)")
    pending_order: int = Field(..., description="Units already on order (not yet arrived)")
    message: str = Field(default="", description="Human-readable description of what happened")


class RestockState(State):
    """
    Full internal state of the environment (used by state() endpoint).
    """
    current_stock: int = 0
    daily_demand: int = 0
    lead_time: int = 0
    day: int = 0
    max_days: int = 30
    pending_order: int = 0
    total_reward: float = 0.0
    task_id: int = 1
