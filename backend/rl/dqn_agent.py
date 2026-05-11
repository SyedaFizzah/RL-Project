import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


# ══════════════════════════════════════════════════════════════════════════════
# Prioritized Experience Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """
    Binary tree where each leaf stores a transition's priority.
    Parent nodes store the SUM of their children's priorities.

    This lets us sample proportionally to priority in O(log n)
    instead of O(n) for a plain list.

              42          ← root = total priority sum
            /    \\
          29      13
         /  \\   /  \\
        13  16  3   10    ← leaf priorities (one per transition)

    To sample: draw a random number r in [0, 42], walk the tree
    left if r <= left_child, else subtract left and go right.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity          # max number of transitions (leaves)
        self.tree     = np.zeros(2 * capacity - 1)   # full binary tree array
        self.data     = np.zeros(capacity, dtype=object)  # actual transitions
        self.write    = 0                 # next write position (circular)
        self.size     = 0                 # how many transitions stored so far

    # ── internal helpers ──────────────────────────────────────────────

    def _propagate(self, idx: int, delta: float):
        """Walk up the tree, adding delta to every ancestor."""
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, value: float) -> int:
        """Walk down from idx, returning the leaf whose range contains value."""
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):   # we've reached a leaf
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    # ── public API ────────────────────────────────────────────────────

    @property
    def total(self) -> float:
        """Sum of all priorities (stored at root)."""
        return self.tree[0]

    def add(self, priority: float, transition):
        """Insert a new transition with the given priority."""
        leaf_idx = self.write + self.capacity - 1   # map write → leaf position

        self.data[self.write] = transition
        self.update(leaf_idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.size  = min(self.size + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float):
        """Change the priority of an existing leaf and fix the tree."""
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, value: float):
        """
        Sample one transition whose priority range contains `value`.
        Returns (leaf_idx, priority, transition).
        """
        leaf_idx   = self._retrieve(0, value)
        data_idx   = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Wraps SumTree to provide add() / sample() / update_priorities().

    Key ideas
    ---------
    alpha  : how much priority matters  (0 = uniform, 1 = fully prioritized)
    beta   : importance-sampling correction (anneals from beta_start → 1.0)
    eps    : small constant so zero-error transitions still get sampled
    """

    def __init__(
        self,
        capacity:    int   = 50_000,
        alpha:       float = 0.6,    # priority exponent
        beta_start:  float = 0.4,    # IS weight at the start of training
        beta_frames: int   = 100_000 # how many steps to anneal beta to 1.0
    ):
        self.tree        = SumTree(capacity)
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.frame       = 1          # counts total add() calls for beta annealing
        self.eps         = 1e-6       # prevents zero-priority transitions

        # New transitions get max priority so they are sampled at least once
        self._max_priority = 1.0

    # ── beta annealing ────────────────────────────────────────────────

    @property
    def beta(self) -> float:
        """
        Linearly anneal beta from beta_start → 1.0 over beta_frames steps.
        Higher beta = stronger correction for the sampling bias PER introduces.
        """
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    # ── public API ────────────────────────────────────────────────────

    def add(self, transition):
        """Store a new transition with maximum current priority."""
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, transition)
        self.frame += 1

    def sample(self, batch_size: int):
        """
        Draw batch_size transitions by priority-proportional sampling.

        Returns
        -------
        transitions : list of (s, a, r, s', done) tuples
        weights     : torch.FloatTensor of importance-sampling weights
        leaf_indices: list of ints — needed to call update_priorities() later
        """
        transitions  = []
        leaf_indices = []
        priorities   = []

        # Divide [0, total_priority] into equal segments,
        # sample one value uniformly from each segment
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            low  = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)

            leaf_idx, priority, transition = self.tree.get(value)
            transitions.append(transition)
            leaf_indices.append(leaf_idx)
            priorities.append(max(priority, self.eps))

        # ── importance-sampling weights ───────────────────────────────
        # PER over-samples high-priority transitions, which biases the
        # gradient. IS weights correct for this bias.
        # w_i = ( 1/N * 1/P(i) ) ^ beta,  normalised by max weight.
        total    = self.tree.total
        n        = self.tree.size
        probs    = np.array(priorities) / total
        weights  = (n * probs) ** (-self.beta)
        weights /= weights.max()                   # normalize so max weight = 1

        return transitions, torch.FloatTensor(weights), leaf_indices

    def update_priorities(self, leaf_indices: list, td_errors: np.ndarray):
        """
        After a train step, update each sampled transition's priority
        to |TD error| + eps so better-learned transitions get sampled less.
        """
        for idx, td_error in zip(leaf_indices, td_errors):
            priority = (abs(td_error) + self.eps) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size


# ══════════════════════════════════════════════════════════════════════════════
# Network
# ══════════════════════════════════════════════════════════════════════════════

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size

        # ── PER buffer (replaces plain deque) ─────────────────────────
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity    = 50_000,
            alpha       = 0.6,    # priority strength
            beta_start  = 0.4,    # IS correction strength (anneals to 1.0)
            beta_frames = 100_000 # anneal over ~100k environment steps
        )

        # Hyperparameters (unchanged from previous version)
        self.gamma         = 0.99
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.997 
        self.batch_size    = 64
        self.learning_rate = 0.0005

        self.main_net   = DQNNetwork(state_size, action_size)
        self.target_net = DQNNetwork(state_size, action_size)
        self.update_target()

        self.optimizer = torch.optim.Adam(
            self.main_net.parameters(),
            lr=self.learning_rate
        )
        self.loss_fn = nn.HuberLoss(reduction="none")  # ← "none" so we can
                                                        #   weight each sample
                                                        #   by its IS weight

    def update_target(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.main_net(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store transition — PER assigns max priority automatically."""
        self.replay_buffer.add(
            (state, action, reward, next_state, done)
        )

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # ── 1. Sample with priorities ──────────────────────────────────
        transitions, is_weights, leaf_indices = \
            self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*transitions)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        # ── 2. Double DQN target ───────────────────────────────────────
        with torch.no_grad():
            next_actions = self.main_net(next_states).argmax(dim=1)
            next_q       = self.target_net(next_states)
            max_next_q   = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets      = rewards + self.gamma * max_next_q * (1 - dones)

        q_values = self.main_net(states) \
                       .gather(1, actions.unsqueeze(1)) \
                       .squeeze(1)

        # ── 3. IS-weighted loss ────────────────────────────────────────
        # elementwise HuberLoss, then multiply each sample's loss by its
        # importance-sampling weight before averaging
        td_errors    = (targets - q_values).detach().cpu().numpy()
        element_loss = self.loss_fn(q_values, targets)           # shape [batch]
        loss         = (is_weights * element_loss).mean()        # weighted mean

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── 4. Update priorities in the tree ──────────────────────────
        self.replay_buffer.update_priorities(leaf_indices, td_errors)

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.main_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.main_net.load_state_dict(
            torch.load(path, weights_only=True)
        )
        self.main_net.eval()