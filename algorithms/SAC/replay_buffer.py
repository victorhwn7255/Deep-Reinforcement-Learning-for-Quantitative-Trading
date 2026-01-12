import numpy as np
import torch


class ReplayBuffer:
    """Uniform Experience Replay Buffer (for off-policy RL like SAC v2).

    Stores transitions (state, action, reward, next_state, done) in pre-allocated numpy arrays.
    Sampling is uniform with replacement.

    Conventions:
    - state: shape (state_dim,)
    - action: shape (action_dim,)
    - reward: scalar stored as shape (1,)
    - done: stored as shape (1,) float {0.0, 1.0}
    """

    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu"):
        self.max_size = int(max_size)
        if self.max_size <= 0:
            raise ValueError(f"max_size must be positive (got {self.max_size})")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive (got {self.state_dim})")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive (got {self.action_dim})")

        self.device = torch.device(device)

        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    def __len__(self):
        return self.size

    def is_ready(self, batch_size):
        return self.size >= int(batch_size)

    def add(self, state, action, reward, next_state, done):
        """Add one transition with shape validation."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if state.shape[0] != self.state_dim:
            raise ValueError(f"State dim mismatch: got {state.shape[0]}, expected {self.state_dim}")
        if next_state.shape[0] != self.state_dim:
            raise ValueError(f"Next state dim mismatch: got {next_state.shape[0]}, expected {self.state_dim}")
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Action dim mismatch: got {action.shape[0]}, expected {self.action_dim}")

        r = np.float32(reward)
        d = np.float32(bool(done))  # store 0.0/1.0

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr, 0] = r
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr, 0] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Sample a batch uniformly (with replacement)."""
        if self.size == 0:
            raise ValueError("Cannot sample from empty replay buffer")

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive (got {batch_size})")

        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "states": torch.as_tensor(self.states[idx], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=self.device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=self.device),
            "next_states": torch.as_tensor(self.next_states[idx], dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=self.device),
        }
