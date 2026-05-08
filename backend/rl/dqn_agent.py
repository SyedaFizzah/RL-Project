import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


class DQNNetwork(nn.Module):
    """
    The neural network brain of the agent.
    Input:  36-dimensional state vector
    Output: 9 Q-values (one per action)
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Replay buffer stores past experiences to learn from later
        self.replay_buffer = deque(maxlen=10000)

        # Hyperparameters
        self.gamma = 0.95           # how much future rewards matter
        self.epsilon = 1.0          # starts fully random (100% exploration)
        self.epsilon_min = 0.01     # never goes below 1% random
        self.epsilon_decay = 0.995  # reduces exploration each episode
        self.batch_size = 32
        self.learning_rate = 0.001

        # Two identical networks: main (trained every step) and
        # target (frozen copy, updated every 100 steps for stability)
        self.main_net = DQNNetwork(state_size, action_size)
        self.target_net = DQNNetwork(state_size, action_size)
        self.update_target()

        self.optimizer = torch.optim.Adam(
            self.main_net.parameters(),
            lr=self.learning_rate
        )
        self.loss_fn = nn.MSELoss()

    def update_target(self):
        """Copy main network weights into the target network."""
        self.target_net.load_state_dict(self.main_net.state_dict())

    def act(self, state):
        """
        Choose an action.
        With probability epsilon: pick randomly (explore).
        Otherwise: pick the action with the highest Q-value (exploit).
        """
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.main_net(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store one experience tuple in the replay buffer."""
        self.replay_buffer.append(
            (state, action, reward, next_state, done)
        )

    def train_step(self):
        """
        Sample a random mini-batch from the replay buffer and
        do one gradient update using the Bellman equation.
        Returns the loss value (0 if not enough data yet).
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors for PyTorch
        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        # Q-value for the specific action that was actually taken
        q_values = self.main_net(states)\
                       .gather(1, actions.unsqueeze(1))\
                       .squeeze(1)

        # Bellman target: use frozen target network, no gradient needed
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Compute loss and update the main network
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        """Call once per episode to slowly reduce random exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save the trained model weights to a file."""
        torch.save(self.main_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load previously trained model weights from a file."""
        self.main_net.load_state_dict(
            torch.load(path, weights_only=True)
        )
        self.main_net.eval()
