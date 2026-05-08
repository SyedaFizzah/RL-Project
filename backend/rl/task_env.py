import gymnasium as gym
import numpy as np
from gymnasium import spaces

CATEGORIES = ["deep_work", "communication", "admin", "creative", "learning"]


class TaskSchedulerEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_tasks=8, seed=42, training_mode=True):
        super().__init__()

        self.num_tasks = num_tasks
        self.slots_per_day = 19
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(num_tasks + 1)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_tasks * 4 + 4,),
            dtype=np.float32
        )

        self.training_mode = training_mode

        self.tasks = []
        self.task_status = []
        self.current_slot = 0
        self.last_category = None

        # reward normalization
        self._reward_history = []
        self._NORM_WINDOW = 500
        self._NORM_CLIP = 5.0

    # ---------------- TASK GENERATION ----------------
    def _generate_task(self, idx):
        return {
            "id": idx,
            "category": self.rng.choice(CATEGORIES),
            "priority": int(self.rng.integers(1, 4)),
            "effort": int(self.rng.integers(1, 5)),
            "deadline_days": int(self.rng.integers(1, 8)),
            "slots_worked": 0,
        }

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_slot = 0
        self.last_category = None

        self.tasks = [self._generate_task(i) for i in range(self.num_tasks)]
        self.task_status = ["pending"] * self.num_tasks

        return self._get_obs(), {}

    # ---------------- ENERGY ----------------
    def _get_energy(self, slot):
        hour = 9.0 + (slot * 25.0 / 60.0)
        energy = 3.0 + np.sin(hour / 24 * 2 * np.pi)
        return float(np.clip(energy, 1.0, 5.0))

    # ---------------- OBS ----------------
    def _get_obs(self):
        obs = []

        for i, task in enumerate(self.tasks):
            if self.task_status[i] == "pending":
                obs += [
                    task["priority"] / 3.0,
                    task["effort"] / 4.0,
                    task["deadline_days"] / 7.0,
                    task["slots_worked"] / max(task["effort"], 1),
                ]
            else:
                obs += [0.0, 0.0, 0.0, 0.0]

        obs += [
            self.current_slot / self.slots_per_day,
            self._get_energy(self.current_slot) / 5.0,
            0.0,
            sum(s == "pending" for s in self.task_status) / self.num_tasks,
        ]

        return np.array(obs, dtype=np.float32)

    # ---------------- REWARD SHAPING ----------------
    def _shaped_reward(self, action: int, base_reward: float) -> float:

        task = self.tasks[action]
        status = self.task_status[action]
        priority = task.get("priority", 1)
        deadline = max(task["deadline_days"], 1)

        shaping = 0.0

        if status == "done":
            slack = deadline - (self.current_slot / self.slots_per_day * 7)

            if slack >= 0:
                shaping += (slack / deadline) * 15.0 * priority
            else:
                shaping -= (abs(slack) / deadline) * 25.0 * priority

        if status == "pending":
            days_left = deadline - (self.current_slot / self.slots_per_day * 7)
            if days_left <= 2:
                shaping -= 0.5 * priority

        return base_reward + shaping

    # ---------------- NORMALIZATION ----------------
    def _normalize_reward(self, raw):

        if not self.training_mode:
            return float(np.clip(raw, -self._NORM_CLIP, self._NORM_CLIP))

        self._reward_history.append(raw)

        if len(self._reward_history) > self._NORM_WINDOW:
            self._reward_history.pop(0)

        mean = np.mean(self._reward_history)
        std = np.std(self._reward_history) + 1e-8

        norm = (raw - mean) / std
        return float(np.clip(norm, -self._NORM_CLIP, self._NORM_CLIP))

    # ---------------- STEP ----------------
    def step(self, action):

        reward = 0.0
        info = {}

        # BREAK ACTION
        if action == self.num_tasks:
            energy = self._get_energy(self.current_slot)
            reward = 0.2 if energy < 3 else -0.2

        # INVALID ACTION
        elif self.task_status[action] != "pending":
            reward = -0.7

        # TASK ACTION
        else:
            task = self.tasks[action]

            if self.last_category and self.last_category != task["category"]:
                reward -= 0.3

            self.last_category = task["category"]
            task["slots_worked"] += 1

            energy = self._get_energy(self.current_slot)

            # completion
            if task["slots_worked"] >= task["effort"]:
                self.task_status[action] = "done"

                base_reward = task["priority"] * 2
                reward = base_reward + (0.3 if energy > 4 else 0.0)

                info["completed_task_id"] = task["id"]

            else:
                reward += 0.1

            # apply shaping
            reward = self._shaped_reward(action, reward)

        # time step
        self.current_slot += 1

        done = (
            self.current_slot >= self.slots_per_day or
            all(s != "pending" for s in self.task_status)
        )

        # normalize
        reward = self._normalize_reward(reward)

        return self._get_obs(), reward, done, False, info