from env.environment import InventoryRestockEnvironment
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleBasedAgent
from agents.qlearning_agent import QLearningAgent
from env.grader import normalize_score

env = InventoryRestockEnvironment()

agents = {
    "Random": RandomAgent(),
    "Rule": RuleBasedAgent(),
    "Q-Learning": QLearningAgent()
}

for name, agent in agents.items():
    print(f"\n{name}")
    for task_id in [1, 2, 3]:
        obs = env.reset(task_id=task_id, seed=task_id)

        while not obs.done:
            action = agent.act(obs)
            obs = env.step(action)

        score = normalize_score(task_id, env.state.total_reward)
        print(f"Task {task_id}: {score}")