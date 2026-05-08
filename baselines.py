import numpy as np
from backend.rl.task_env import TaskSchedulerEnv
from backend.rl.dqn_agent import DQNAgent


def deadline_only(tasks, task_status):
    """Always pick the pending task with the nearest deadline."""
    candidates = [
        (t["deadline_days"], i)
        for i, t in enumerate(tasks)
        if task_status[i] == "pending"
    ]
    if not candidates:
        return len(tasks)  # take a break if nothing pending
    return min(candidates)[1]


def priority_only(tasks, task_status):
    """Always pick the highest priority pending task."""
    candidates = [
        (-t["priority"], i)
        for i, t in enumerate(tasks)
        if task_status[i] == "pending"
    ]
    if not candidates:
        return len(tasks)
    return min(candidates)[1]


def run_baseline(strategy_fn, name, episodes=500):
    env = TaskSchedulerEnv(seed=42)
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = strategy_fn(env.tasks, env.task_status)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    avg = np.mean(rewards)
    print(f"{name:25s}  avg reward = {avg:.3f}")
    return rewards


def run_rl_agent(episodes=500):
    env = TaskSchedulerEnv(seed=42)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load("saved_model/dqn_model.pth")
    agent.epsilon = 0.0  # no random exploration when evaluating

    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    avg = np.mean(rewards)
    print(f"{'RL Agent (DQN)':25s}  avg reward = {avg:.3f}")
    return rewards


if __name__ == "__main__":
    print("Running 500 evaluation episodes per strategy...\n")
    dl = run_baseline(deadline_only, "Deadline-only")
    pr = run_baseline(priority_only, "Priority-only")
    rl = run_rl_agent()

    print()
    if np.mean(rl) > np.mean(dl) and np.mean(rl) > np.mean(pr):
        print("RL agent outperforms both baselines.")
    else:
        print("RL agent does not beat both baselines yet.")
        print("Consider running train.py with more episodes.")
