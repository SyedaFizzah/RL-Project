import numpy as np
import json
import os
from backend.rl.task_env import TaskSchedulerEnv
from backend.rl.dqn_agent import DQNAgent


def train():
    env = TaskSchedulerEnv(seed=42)
    state_size = env.observation_space.shape[0]   # 36
    action_size = env.action_space.n               # 9

    agent = DQNAgent(state_size, action_size)

    EPISODES = 3000
    TRAIN_EVERY = 4       # do a gradient update every 4 steps
    TARGET_EVERY = 100    # sync target network every 100 steps

    training_log = []
    total_steps = 0

    print("=" * 50)
    print("DQN Training Started")
    print(f"Episodes: {EPISODES}")
    print(f"State size: {state_size}  |  Action size: {action_size}")
    print("=" * 50)

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps % TRAIN_EVERY == 0:
                agent.train_step()

            if total_steps % TARGET_EVERY == 0:
                agent.update_target()

        agent.decay_epsilon()

        training_log.append({
            "episode": episode,
            "reward": round(episode_reward, 3),
            "epsilon": round(agent.epsilon, 4)
        })

        if episode % 100 == 0:
            print(f"Episode {episode:5d} / {EPISODES}"
                  f"  |  Reward: {episode_reward:7.2f}"
                  f"  |  Epsilon: {agent.epsilon:.3f}")

    print("=" * 50)
    print("Training complete.")

    # Save trained model
    os.makedirs("saved_model", exist_ok=True)
    agent.save("saved_model/dqn_model.pth")

    # Save training log for charts later
    with open("training_log.json", "w") as f:
        json.dump(training_log, f)
    print("Training log saved to training_log.json")


if __name__ == "__main__":
    train()
