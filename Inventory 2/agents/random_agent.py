# agents/random_agent.py
# Naive baseline — orders a random quantity every day.
# Purpose: establishes the performance FLOOR.
# Any reasonable agent should score higher than this.

import random


class RandomAgent:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def act(self, obs) -> int:
        """Order a random quantity between 0 and 50."""
        return self.rng.randint(0, 50)

    def reset(self):
        pass  # nothing to reset between episodes
