# train.py
# ─────────────────────────────────────────────────────────────────────────────
# One script that does everything:
#
#   1. Trains Q-Learning agent on all 3 tasks  (~45 seconds)
#   2. Evaluates all 3 agents × 3 tasks × 5 seeds
#   3. Prints a clean results table to the terminal
#   4. Saves results/results.json
#   5. Saves results/comparison.png  — bar chart (your headline result)
#   6. Saves results/learning.png    — learning curves (shows agent improving)
#
# Run: python train.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file

from env.environment import InventoryRestockEnvironment
from env.models import RestockAction
from env.grader import normalize_score
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleBasedAgent
from agents.qlearning_agent import QLearningAgent

os.makedirs("results", exist_ok=True)

TASK_NAMES  = {1: "Easy", 2: "Medium", 3: "Hard"}
NUM_SEEDS   = 5
TRAIN_EPS   = {1: 1000, 2: 2000, 3: 3000}


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_qlearning():
    """
    Train Q-Learning agent separately on each task.
    Each task gets its own Q-table because the optimal policy is different
    for 1-day vs 5-day lead times.
    Returns: dict of {task_id: (trained_agent, episode_rewards)}
    """
    print("\n" + "="*55)
    print("STEP 1 — Training Q-Learning Agent")
    print("="*55)

    trained = {}
    env = InventoryRestockEnvironment()

    for task_id in [1, 2, 3]:
        episodes = TRAIN_EPS[task_id]
        agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.4,
                               epsilon_decay=0.997, epsilon_min=0.01)
        rewards = []

        print(f"\n  Task {task_id} ({TASK_NAMES[task_id]}) — {episodes} episodes...")

        for ep in range(1, episodes + 1):
            obs = env.reset(task_id=task_id)   # random seed during training
            total = 0.0

            while not obs.done:
                action   = agent.act(obs)
                prev_obs = obs
                obs      = env.step(RestockAction(restock_quantity=action))
                agent.update(prev_obs, action, obs.reward, obs, obs.done)
                total   += obs.reward

            agent.decay_epsilon()
            rewards.append(round(total, 2))

            if ep % (episodes // 4) == 0:
                recent = sum(rewards[-100:]) / min(100, len(rewards))
                print(f"    ep {ep:4d}/{episodes} | "
                      f"avg reward (last 100): {recent:7.1f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"states learned: {len(agent.q_table)}")

        # Save trained model
        agent.training = False
        agent.epsilon  = 0.0
        model_path = f"results/qlearning_task{task_id}.json"
        agent.save(model_path)
        print(f"    Saved model → {model_path}")

        trained[task_id] = (agent, rewards)

    return trained


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env, agent, task_id, seed):
    """Run one full episode and return grade + step rewards."""
    agent.reset()
    obs = env.reset(task_id=task_id, seed=seed)
    step_rewards = []
    stockout_days = 0

    while not obs.done:
        action = agent.act(obs)
        obs    = env.step(RestockAction(restock_quantity=action))
        step_rewards.append(obs.reward)
        if "STOCKOUT" in obs.message:
            stockout_days += 1

    grade = normalize_score(task_id, env.state.total_reward)
    return grade, step_rewards, stockout_days


def evaluate_all(trained_agents):
    """
    Run all 3 agents on all 3 tasks × NUM_SEEDS seeds.
    Returns structured results dict.
    """
    print("\n" + "="*55)
    print(f"STEP 2 — Evaluating All Agents ({NUM_SEEDS} seeds each)")
    print("="*55)

    env = InventoryRestockEnvironment()

    agents = {
        "Random Agent":    RandomAgent(),
        "Rule-Based Agent": RuleBasedAgent(),
    }
    # Q-Learning: separate trained model per task
    ql_agents = {tid: agent for tid, (agent, _) in trained_agents.items()}

    results = {}

    for agent_name, agent in agents.items():
        results[agent_name] = {}
        for task_id in [1, 2, 3]:
            grades = []
            for seed in range(NUM_SEEDS):
                grade, _, _ = run_episode(env, agent, task_id, seed)
                grades.append(grade)
            avg = round(sum(grades) / len(grades), 4)
            results[agent_name][task_id] = avg
            print(f"  {agent_name:20s} | Task {task_id} ({TASK_NAMES[task_id]:6s}) | {avg:.4f}")

    results["Q-Learning Agent"] = {}
    for task_id in [1, 2, 3]:
        agent = ql_agents[task_id]
        grades = []
        for seed in range(NUM_SEEDS):
            grade, _, _ = run_episode(env, agent, task_id, seed)
            grades.append(grade)
        avg = round(sum(grades) / len(grades), 4)
        results["Q-Learning Agent"][task_id] = avg
        print(f"  {'Q-Learning Agent':20s} | Task {task_id} ({TASK_NAMES[task_id]:6s}) | {avg:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results):
    print("\n" + "="*55)
    print("RESULTS SUMMARY")
    print("="*55)
    print(f"{'Agent':22s} | {'Easy':>6} | {'Medium':>6} | {'Hard':>6} | {'Avg':>6}")
    print("-"*55)

    for name, scores in results.items():
        t1  = scores.get(1, 0)
        t2  = scores.get(2, 0)
        t3  = scores.get(3, 0)
        avg = round((t1 + t2 + t3) / 3, 4)
        print(f"{name:22s} | {t1:>6.4f} | {t2:>6.4f} | {t3:>6.4f} | {avg:>6.4f}")

    print("-"*55)
    print("Pass threshold = 0.5 | Score range = 0.0 to 1.0")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_chart(results):
    """
    Bar chart: one group of 3 bars per task, one bar per agent.
    This is the HEADLINE result — put it in your README.
    """
    agents   = list(results.keys())
    colors   = ["#e74c3c", "#f39c12", "#27ae60"]   # red, amber, green
    x        = np.arange(3)   # 3 tasks
    width    = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (agent, color) in enumerate(zip(agents, colors)):
        scores = [results[agent].get(t, 0) for t in [1, 2, 3]]
        bars   = ax.bar(x + i * width, scores, width,
                        label=agent, color=color, alpha=0.85, edgecolor="white")
        for bar, score in zip(bars, scores):
            if score > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{score:.2f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(2.87, 0.51, "Pass threshold", fontsize=9, color="gray")

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Task 1 — Easy", "Task 2 — Medium", "Task 3 — Hard"], fontsize=11)
    ax.set_ylabel("Average Grade (0–1)", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Agent Performance Comparison\nInventory Restock Environment",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved → results/comparison.png")


def save_learning_chart(trained_agents):
    """
    3-panel chart showing Q-Learning reward improving over training episodes.
    This proves the agent is actually learning, not just running a fixed policy.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, task_id in enumerate([1, 2, 3]):
        _, rewards = trained_agents[task_id]
        n      = len(rewards)
        window = 50

        # Rolling average to smooth the noisy curve
        smoothed = [
            sum(rewards[max(0, i - window): i + 1]) / len(rewards[max(0, i - window): i + 1])
            for i in range(n)
        ]

        ax = axes[idx]
        ax.plot(range(n), rewards,   color="#27ae60", alpha=0.15, linewidth=0.5)
        ax.plot(range(n), smoothed,  color="#27ae60", linewidth=2.0,
                label=f"Rolling avg ({window} ep)")

        ax.set_title(f"Task {task_id} — {TASK_NAMES[task_id]}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Episode", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Total Episode Reward", fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("Q-Learning Training Curves — Reward Increases as Agent Learns",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/learning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → results/learning.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\nInventory Restock — Training & Evaluation Pipeline")
    print("This will take about 45–60 seconds.\n")

    # Step 1: Train
    trained_agents = train_qlearning()

    # Step 2: Evaluate
    results = evaluate_all(trained_agents)

    # Step 3: Print table
    print_results_table(results)

    # Step 4: Save charts
    print("\nSTEP 3 — Saving Charts")
    save_comparison_chart(results)
    save_learning_chart(trained_agents)

    # Step 5: Save results JSON (for reference)
    with open("results/results.json", "w") as f:
        serializable = {
            name: {str(tid): score for tid, score in scores.items()}
            for name, scores in results.items()
        }
        json.dump(serializable, f, indent=2)
    print("  Saved → results/results.json")

    print("\n" + "="*55)
    print("DONE. Open results/ folder to see:")
    print("  comparison.png  — bar chart (put this in your README)")
    print("  learning.png    — learning curves")
    print("  results.json    — raw numbers")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
