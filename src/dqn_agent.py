# src/dqn_agent.py

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------
# Replay Buffer
# -------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, max_actions: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_actions = max_actions

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.next_num_actions = np.zeros((capacity,), dtype=np.int64)

        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, next_num_actions):
        idx = self.ptr

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.next_num_actions[idx] = next_num_actions

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        assert self.size >= batch_size, "Not enough samples in buffer."

        indices = np.random.choice(self.size, size=batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.next_num_actions[indices],
        )

    def __len__(self):
        return self.size


# -------------------------
# Dueling Q-Network
# -------------------------

class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, max_actions: int, hidden_dim: int = 512):
        super().__init__()
        
        # Shared feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Stream 1: Value Function V(s) - How good is the state?
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Stream 2: Advantage Function A(s, a) - How good is the action?
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_actions)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine using the Dueling formula: Q = V + (A - mean(A))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals


# -------------------------
# DQN Agent
# -------------------------

class DQNAgent:

    def __init__(
        self,
        obs_dim: int,
        max_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        device: Optional[str] = None,
    ):
        self.obs_dim = obs_dim
        self.max_actions = max_actions
        self.gamma = gamma

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize Dueling Networks
        self.q_net = DuelingQNetwork(obs_dim, max_actions).to(self.device)
        self.target_net = DuelingQNetwork(obs_dim, max_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim, max_actions)

        self.train_steps = 0

    # -------------------------
    # Action selection
    # -------------------------

    def act(self, obs: np.ndarray, num_actions: int, epsilon: float) -> int:
        if num_actions == 0:
            return -1

        if np.random.rand() < epsilon:
            return np.random.randint(num_actions)

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(obs_t)
            
            # Mask valid actions (environment sorts them, so we take top N)
            valid_q = q_values[:, :num_actions]
            action = int(torch.argmax(valid_q).item())
            
        return action

    # -------------------------
    # Store transition
    # -------------------------

    def remember(self, state, action, reward, next_state, done, next_num_actions):
        self.buffer.add(state, action, reward, next_state, done, next_num_actions)

    # -------------------------
    # Train step
    # -------------------------

    def train_step(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_num_actions,
        ) = self.buffer.sample(batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        next_num_actions = torch.from_numpy(next_num_actions).long().to(self.device)

        # 1. Compute Q(s, a)
        q_values = self.q_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Compute Target Q(s', a')
        with torch.no_grad():
            next_q_all = self.target_net(next_states)

            # Mask invalid actions in next state
            mask = torch.arange(self.max_actions, device=self.device).unsqueeze(0)
            mask = mask.expand(next_q_all.size(0), -1)
            valid_mask = mask < next_num_actions.unsqueeze(1)

            # Fill invalid actions with negative infinity so max() ignores them
            very_neg = -1e9
            next_q_masked = torch.where(valid_mask, next_q_all, torch.tensor(very_neg, device=self.device))
            
            # Get max Q-value for next state
            next_q_max = next_q_masked.max(dim=1).values
            
            # Safety: if a state has 0 actions, value is 0 (handled by done, but good to be safe)
            next_q_max[next_num_actions == 0] = 0.0

            target = rewards + self.gamma * (1 - dones) * next_q_max

        # 3. Compute Loss & Update
        loss = nn.functional.mse_loss(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1

        return float(loss.item())

    # -------------------------
    # Target network sync
    # -------------------------

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())