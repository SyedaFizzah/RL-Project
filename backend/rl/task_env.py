import gymnasium as gym
import numpy as np
from gymnasium import spaces

CATEGORIES = ["deep_work", "communication", "admin", "creative", "learning"]


class TaskSchedulerEnv(gym.Env):
    """
    Custom RL environment simulating a single 8-hour work day.
    One episode = 19 Pomodoro slots of 25 minutes each.
    At each slot the agent picks a task (0-7) or takes a break (8).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_tasks=8, seed=42):
        super().__init__()
        self.num_tasks = num_tasks
        self.slots_per_day = 19
        self.rng = np.random.default_rng(seed)

        # Action space: pick task 0-7 or take a break (action 8)
        self.action_space = spaces.Discrete(num_tasks + 1)

        # Observation space: 8 tasks x 4 features each + 4 global features = 36
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_tasks * 4 + 4,),
            dtype=np.float32
        )

        # These will be set in reset()
        self.tasks = []
        self.task_status = []
        self.current_slot = 0
        self.last_category = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_slot = 0
        self.last_category = None
        self.tasks = [self._generate_task(i) for i in range(self.num_tasks)]
        self.task_status = ["pending"] * self.num_tasks
        return self._get_obs(), {}

    def _generate_task(self, idx):
        category = self.rng.choice(CATEGORIES)
        priority = int(self.rng.integers(1, 4))       # 1=low, 2=medium, 3=high
        effort = int(self.rng.integers(1, 5))          # slots needed to complete (1-4)
        deadline_days = int(self.rng.integers(1, 8))   # days until deadline (1-7)
        return {
            "id": idx,
            "category": category,
            "priority": priority,
            "effort": effort,
            "deadline_days": deadline_days,
            "value": priority * effort,
            "slots_worked": 0,
        }

    def _get_energy(self, slot):
        """
        Returns a float from 1.0 to 5.0 simulating realistic daily energy.
        High in the morning, dip after lunch, partial recovery in afternoon.
        """
        hour = 9.0 + (slot * 25.0 / 60.0)
        base = (
            2.0 * np.exp(-0.5 * ((hour - 10.0) / 1.5) ** 2) +
            1.5 * np.exp(-0.5 * ((hour - 16.0) / 1.5) ** 2) -
            1.0 * np.exp(-0.5 * ((hour - 13.0) / 1.0) ** 2)
        )
        energy = np.clip(base * 1.5 + 3.0, 1.0, 5.0)
        noise = self.rng.normal(0, 0.3)
        return float(np.clip(energy + noise, 1.0, 5.0))

    def _get_productivity_score(self):
        """
        Returns a float in [-1, 1] simulating how productive the last 2 hours were.
        In production this comes from the Chrome extension.
        In simulation it follows the energy level with some noise.
        """
        energy = self._get_energy(self.current_slot)
        score = (energy - 3.0) / 2.0 + self.rng.normal(0, 0.1)
        return float(np.clip(score, -1.0, 1.0))

    def _get_obs(self):
        """
        Builds and returns the 36-dimensional state vector.
        First 32 values: 8 tasks x 4 features each (zeros for completed tasks).
        Last 4 values: time of day, energy, productivity score, pending task count.
        """
        task_features = []
        for i, task in enumerate(self.tasks):
            if self.task_status[i] == "pending":
                task_features.extend([
                    task["priority"] / 3.0,
                    task["effort"] / 4.0,
                    task["deadline_days"] / 7.0,
                    task["slots_worked"] / max(task["effort"], 1),
                ])
            else:
                # Completed tasks contribute zeros — signals to agent this slot is unavailable
                task_features.extend([0.0, 0.0, 0.0, 0.0])

        pending_count = sum(1 for s in self.task_status if s == "pending")
        global_features = [
            self.current_slot / self.slots_per_day,
            self._get_energy(self.current_slot) / 5.0,
            self._get_productivity_score(),
            pending_count / self.num_tasks,
        ]

        return np.array(task_features + global_features, dtype=np.float32)

    def step(self, action):
        """
        Execute one action (one 25-minute Pomodoro slot).
        Returns: observation, reward, done, truncated, info
        """
        reward = 0.0
        info = {}

        if action == self.num_tasks:
            # Break action — only good when energy is low
            energy = self._get_energy(self.current_slot)
            reward = 0.1 if energy < 3.0 else -0.1

        elif self.task_status[action] != "pending":
            # Agent tried to pick a task that is already done — penalise
            reward = -0.5

        else:
            task = self.tasks[action]

            # Context switch penalty — penalise jumping between different categories
            if self.last_category and self.last_category != task["category"]:
                reward -= 0.3
            self.last_category = task["category"]

            # Work on this task for one slot
            task["slots_worked"] += 1

            # Check if the task is now fully complete
            if task["slots_worked"] >= task["effort"]:
                self.task_status[action] = "done"

                # Core reward formula from the project brief
                days_overdue = max(0, 1 - task["deadline_days"])
                base_reward = task["value"] * (1.0 / max(1, days_overdue))

                # Bonus for completing a task without switching away mid-effort
                focus_bonus = 0.5

                # Bonus for assigning hard tasks during high-energy slots
                energy = self._get_energy(self.current_slot)
                energy_bonus = 0.3 if (task["effort"] >= 3 and energy >= 4.0) else 0.0

                reward += base_reward + focus_bonus + energy_bonus
                info["completed_task_id"] = task["id"]
                info["task_value"] = task["value"]

        # Move time forward by one slot
        self.current_slot += 1

        # Deadlines get closer as the day progresses
        if self.current_slot % 10 == 0:
            for t in self.tasks:
                if self.task_status[t["id"]] == "pending":
                    t["deadline_days"] = max(0, t["deadline_days"] - 0.5)

        # Episode ends when the work day is over OR all tasks are complete
        done = (
            self.current_slot >= self.slots_per_day or
            all(s != "pending" for s in self.task_status)
        )

        return self._get_obs(), reward, done, False, info
