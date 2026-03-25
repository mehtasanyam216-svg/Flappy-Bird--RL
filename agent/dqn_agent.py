"""
dqn_agent.py — Deep Q-Network agent for the Flappy Bird RL environment.

Components:
    DQNNetwork   — Small feed-forward Q-value network.
    ReplayBuffer — Fixed-size experience replay memory.
    DQNAgent     — Epsilon-greedy policy, training loop, target-net sync.
"""
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

STATE_SIZE = 4          
ACTION_SIZE = 2         

SCREEN_HEIGHT = 512
SCREEN_WIDTH = 400
MAX_VELOCITY = 10.0

class DQNNetwork(nn.Module):
    """
    Simple fully-connected Q-network.
        Input:  normalised state vector (4 floats)
        Output: Q-value for each action (2 floats)
    """

    def __init__(self, state_size: int = STATE_SIZE,
                 action_size: int = ACTION_SIZE,
                 hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    """
    Fixed-size circular buffer that stores experience tuples
    and samples uniform random mini-batches for training.
    """

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN agent with:
        • epsilon-greedy exploration (linear decay)
        • experience replay
        • target network (periodically synced)
    """

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50_000,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_freq: int = 500,
        device: str | None = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_step = (epsilon_start - epsilon_end) / epsilon_decay_steps

        self.policy_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(capacity=buffer_capacity)

        self.steps_done = 0

    @staticmethod
    def normalize_state(state: list[float]) -> list[float]:
        """
        Scale raw state values to roughly [-1, 1] range.
        This helps the network learn much faster.
        """
        bird_y, velocity, horiz_dist, vert_dist = state
        return [
            bird_y / SCREEN_HEIGHT,        
            velocity / MAX_VELOCITY,       
            horiz_dist / SCREEN_WIDTH,     
            vert_dist / SCREEN_HEIGHT,     
        ]

    def select_action(self, state: list[float], training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        During evaluation, set training=False to always pick the greedy action.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        norm_state = self.normalize_state(state)
        state_tensor = torch.tensor([norm_state], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        """Store a normalised transition in replay memory."""
        norm_state = self.normalize_state(state)
        norm_next = self.normalize_state(next_state)
        self.memory.push(norm_state, action, reward, norm_next, done)

    def train_step(self) -> float | None:
        """
        Sample a mini-batch from replay memory and do one gradient step.
        Returns the loss value, or None if the buffer is too small.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_step)

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str = "checkpoints/dqn_flappy.pth"):
        """Save the policy network weights and training state."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
        }, path)
        print(f"💾 Model saved to {path}")

    def load(self, path: str = "checkpoints/dqn_flappy.pth"):
        """Load weights and training state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]
        print(f"📂 Model loaded from {path}  (epsilon={self.epsilon:.4f}, steps={self.steps_done})")
