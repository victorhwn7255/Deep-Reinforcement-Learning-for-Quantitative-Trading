import numpy as np
import torch


class ReplayBuffer:
    """Uniform Experience Replay Buffer (for off-policy RL like SAC).

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

        self.device = torch.device(device)

        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    def __len__(self):
        return self.size

    def is_ready(self, batch_size):
        return self.size >= int(batch_size)

    def add(self, state, action, reward, next_state, done):
        """Add one transition.

        Safety:
        - Flattens state/action/next_state to 1D.
        - Forces dtype float32.
        - Converts reward/done to float32.

        Raises:
            ValueError on shape mismatch.
        """
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if state.shape[0] != self.states.shape[1]:
            raise ValueError(f"State dim mismatch: got {state.shape[0]}, expected {self.states.shape[1]}")
        if next_state.shape[0] != self.next_states.shape[1]:
            raise ValueError(f"Next state dim mismatch: got {next_state.shape[0]}, expected {self.next_states.shape[1]}")
        if action.shape[0] != self.actions.shape[1]:
            raise ValueError(f"Action dim mismatch: got {action.shape[0]}, expected {self.actions.shape[1]}")

        r = np.float32(reward)
        d = np.float32(bool(done))

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Sample a batch uniformly (with replacement).

        - Only raises if the buffer is empty.
        - Allows batch_size > size by sampling with replacement.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty replay buffer")

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive (got {batch_size})")

        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "states": torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            "next_states": torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device),
        }