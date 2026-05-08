from backend.rl.task_env import TaskSchedulerEnv

print("Creating environment...")
env = TaskSchedulerEnv(seed=42)
obs, _ = env.reset()

print(f"Observation shape: {obs.shape}")
print(f"Expected shape:    (36,)")
print(f"Action space size: {env.action_space.n}")
print(f"Expected size:     9")
print()

total_reward = 0.0
steps = 0

print("Running one episode with random actions...")
while True:
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    steps += 1
    if "completed_task_id" in info:
        print(f"  Slot {steps}: completed task {info['completed_task_id']} — reward {reward:.2f}")
    if done:
        break

print()
print(f"Episode finished after {steps} steps")
print(f"Total reward: {total_reward:.3f}")
print()
print("If you see numbers above and no errors, the environment is working correctly.")
