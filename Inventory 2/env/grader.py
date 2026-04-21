# grader.py
# Scores the agent's performance on each task from 0.01 to 0.99
#
# HOW GRADING WORKS:
# Each task has a known "worst possible score" and a "target (good) score".
# We normalize the agent's actual score into the 0.01–0.99 range.
#
# Score 0.01 = agent did terribly (stockouts every day, or did nothing)
# Score 0.99 = agent perfectly managed inventory across the episode
#
# The grader is DETERMINISTIC for Task 1 (fixed demand),
# and bounded for Tasks 2 and 3 (random demand).
 # grader.py
from typing import Dict
from dataclasses import dataclass


# ── DATA STRUCTURE ───────────────────────────────

@dataclass
class GradeResult:
    task_id: int
    grade: float


# ── SCORE NORMALIZATION ─────────────────────────

SCORE_BOUNDS = {
    1: (-80.0, 85.0),
    2: (-150.0, 120.0),
    3: (-250.0, 200.0),
}


def normalize_score(task_id: int, total_reward: float) -> float:
    min_s, target_s = SCORE_BOUNDS[task_id]

    score = (total_reward - min_s) / (target_s - min_s)

    # STRICT range (HF requirement: not 0 or 1)
    score = max(0.011, min(0.989, score))

    return float(round(score, 4))


# ── CORE TASK RUNNER ────────────────────────────

def run_single_task(task_id: int) -> float:
    from env.environment import InventoryRestockEnvironment
    from agent.baseline_agent import simple_agent

    env = InventoryRestockEnvironment()

    obs = env.reset(task_id=task_id, seed=task_id)

    while not obs.done:
        action = simple_agent(obs)
        obs = env.step(action)

    total_reward = env.state.total_reward

    return normalize_score(task_id, total_reward)


# ── MAIN ENTRY (HF EXPECTS THIS) ────────────────

def run_grader() -> Dict[str, float]:
    """
    HF validator entry point
    Must return exactly 3 tasks
    """

    results = {}

    for task_id in [1, 2, 3]:
        score = run_single_task(task_id)
        results[f"task_{task_id}"] = score

    return results


# ── OPTIONAL FALLBACK (SOME VALIDATORS USE THIS) ─

def evaluate() -> Dict[str, float]:
    return run_grader()