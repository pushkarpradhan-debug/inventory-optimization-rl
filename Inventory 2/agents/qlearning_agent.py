# agents/qlearning_agent.py
# Q-Learning agent — learns the optimal restocking policy through experience.
#
# How it works in plain English:
#   The agent keeps a table of (situation → best action).
#   At first the table is empty and it tries random actions.
#   After each action it updates the table: "that action in that situation
#   gave me X reward, so I'll adjust my preference accordingly."
#   Over thousands of episodes, the table fills up with learned preferences.
#
# The Bellman update (one line of math):
#   Q(state, action) += α × (reward + γ × best_future_Q - current_Q)
#   Where α=learning rate, γ=discount (how much future rewards matter)

import json
import numpy as np


class QLearningAgent:

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.4,
                 epsilon_decay=0.997, epsilon_min=0.01):
        self.alpha         = alpha           # learning rate
        self.gamma         = gamma           # discount factor
        self.epsilon       = epsilon         # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q_table       = {}              # state → Q-values for each action
        self.training      = True

    # ── TURN OBSERVATION INTO A DISCRETE STATE ───────────────────────────────
    def _state(self, obs) -> tuple:
        """
        We can't store a Q-value for every possible stock number (0, 1, 2...).
        Instead we group similar situations into buckets.
        stock=23 and stock=24 behave the same → put in same bucket.
        """
        s = obs.current_stock
        if   s == 0:    sb = 0   # empty — critical
        elif s <= 5:    sb = 1   # danger zone
        elif s <= 10:   sb = 2   # low
        elif s <= 20:   sb = 3   # healthy
        elif s <= 35:   sb = 4   # high
        else:           sb = 5   # overstock

        d = obs.daily_demand
        if   d <= 4:    db = 0   # low demand
        elif d <= 7:    db = 1   # normal demand
        elif d <= 12:   db = 2   # high demand
        else:           db = 3   # spike

        lt = {1: 0, 3: 1, 5: 2}.get(obs.lead_time, 1)

        p = obs.pending_order
        if   p == 0:    pb = 0   # no pending order
        elif p <= 25:   pb = 1   # moderate order incoming
        else:           pb = 2   # large order incoming

        return (sb, db, lt, pb)

    # ── DECIDE WHAT TO DO ────────────────────────────────────────────────────
    def act(self, obs) -> int:
        state = self._state(obs)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(51)

        # During training: explore randomly with probability epsilon
        if self.training and np.random.random() < self.epsilon:
            return np.random.randint(0, 51)

        # Otherwise: pick the action with the highest Q-value
        return int(np.argmax(self.q_table[state]))

    # ── LEARN FROM WHAT JUST HAPPENED ───────────────────────────────────────
    def update(self, obs, action, reward, next_obs, done):
        s  = self._state(obs)
        s2 = self._state(next_obs)

        if s  not in self.q_table: self.q_table[s]  = np.zeros(51)
        if s2 not in self.q_table: self.q_table[s2] = np.zeros(51)

        current_q = self.q_table[s][action]
        target    = reward if done else reward + self.gamma * np.max(self.q_table[s2])
        self.q_table[s][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        pass  # Q-table persists across episodes — that's how it learns

    # ── SAVE AND LOAD ────────────────────────────────────────────────────────
    def save(self, path):
        data = {str(k): v.tolist() for k, v in self.q_table.items()}
        with open(path, "w") as f:
            json.dump({"q_table": data, "epsilon": self.epsilon}, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.q_table = {eval(k): np.array(v) for k, v in data["q_table"].items()}
        self.epsilon = 0.0  # no exploration when loaded for evaluation
        self.training = False
