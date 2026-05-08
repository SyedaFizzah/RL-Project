import numpy as np
from backend.rl.task_env import TaskSchedulerEnv
from backend.rl.dqn_agent import DQNAgent


# ══════════════════════════════════════════════════════════════════════════════
# Heuristic baselines
# ══════════════════════════════════════════════════════════════════════════════

def deadline_only(tasks, task_status):
    """Always pick the pending task with the nearest deadline."""
    candidates = [
        (t["deadline_days"], i)
        for i, t in enumerate(tasks)
        if task_status[i] == "pending"
    ]
    if not candidates:
        return len(tasks)
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


# ══════════════════════════════════════════════════════════════════════════════
# Runners
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(strategy_fn, name, seeds, episodes_per_seed=200):
    """
    Evaluate a heuristic strategy across multiple seeds.
    Returns a flat list of all episode rewards.
    """
    all_rewards = []

    for seed in seeds:
        env = TaskSchedulerEnv(seed=seed, training_mode=False)
        for _ in range(episodes_per_seed):
            state, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = strategy_fn(env.tasks, env.task_status)
                state, reward, done, _, _ = env.step(action)
                ep_reward += reward
            all_rewards.append(ep_reward)

    _print_stats(name, all_rewards, seeds, episodes_per_seed)
    return all_rewards


def run_rl_agent(seeds, episodes_per_seed=200):
    """
    Evaluate the trained DQN agent across multiple seeds.
    Returns a flat list of all episode rewards.
    """
    all_rewards = []

    for seed in seeds:
        env   = TaskSchedulerEnv(seed=seed, training_mode=False)
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
        agent.load("saved_model/dqn_model.pth")
        agent.epsilon = 0.0   # pure exploitation during evaluation

        for _ in range(episodes_per_seed):
            state, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = agent.act(state)
                state, reward, done, _, _ = env.step(action)
                ep_reward += reward
            all_rewards.append(ep_reward)

    _print_stats("RL Agent (DQN)", all_rewards, seeds, episodes_per_seed)
    return all_rewards


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def _print_stats(name, rewards, seeds, episodes_per_seed):
    """Print a full breakdown: overall stats + per-seed means."""
    arr = np.array(rewards)
    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")
    print(f"  Seeds evaluated : {seeds}")
    print(f"  Episodes / seed : {episodes_per_seed}  "
          f"(total: {len(arr)})")
    print(f"  Mean   : {arr.mean():+.3f}")
    print(f"  Std    : {arr.std():.3f}")
    print(f"  Min    : {arr.min():+.3f}")
    print(f"  Max    : {arr.max():+.3f}")
    print(f"  Median : {np.median(arr):+.3f}")

    # Per-seed breakdown so you can spot if one seed is an outlier
    print(f"\n  Per-seed means:")
    for i, seed in enumerate(seeds):
        chunk = rewards[i * episodes_per_seed : (i + 1) * episodes_per_seed]
        print(f"    seed={seed:>4d}  →  mean={np.mean(chunk):+.3f}  "
              f"std={np.std(chunk):.3f}")


def _print_verdict(results: dict):
    """Compare all strategies and print a clear winner."""
    print(f"\n{'═' * 55}")
    print("  FINAL VERDICT")
    print(f"{'═' * 55}")

    # Rank by mean reward
    ranked = sorted(results.items(), key=lambda x: np.mean(x[1]), reverse=True)

    for rank, (name, rewards) in enumerate(ranked, 1):
        medal = ["1st", "2nd", "3rd"][min(rank - 1, 2)]
        print(f"  {medal}  {name:25s}  mean={np.mean(rewards):+.3f}")

    winner_name, winner_rewards = ranked[0]
    rl_mean = np.mean(results["RL Agent (DQN)"])
    best_baseline_mean = max(
        np.mean(v) for k, v in results.items() if k != "RL Agent (DQN)"
    )

    print(f"\n  Winner: {winner_name}")

    if rl_mean > best_baseline_mean:
        gap = rl_mean - best_baseline_mean
        print(f"  ✓ RL agent beats best baseline by {gap:+.3f}")
    else:
        gap = best_baseline_mean - rl_mean
        print(f"  ✗ RL agent trails best baseline by {gap:.3f}")
        print("  → Retrain with more episodes or tune shaping weights.")

    print(f"{'═' * 55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SEEDS             = [42, 0, 123]   # three independent environments
    EPISODES_PER_SEED = 200            # 200 × 3 = 600 total per strategy

    print(f"\nEvaluating across seeds {SEEDS}, "
          f"{EPISODES_PER_SEED} episodes each...\n")

    results = {
        "Deadline-only" : run_baseline(deadline_only, "Deadline-only",
                                       SEEDS, EPISODES_PER_SEED),
        "Priority-only" : run_baseline(priority_only, "Priority-only",
                                       SEEDS, EPISODES_PER_SEED),
        "RL Agent (DQN)": run_rl_agent(SEEDS, EPISODES_PER_SEED),
    }

    _print_verdict(results)