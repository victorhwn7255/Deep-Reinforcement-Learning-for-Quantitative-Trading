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
    - done: scalar {0.0, 1.0} stored as shape (1,)
    """

    def __init__(self, state_dim, action_dim, max_size=1_000_000, device="cpu"):
        if int(state_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError(f"state_dim and action_dim must be positive (got {state_dim}, {action_dim})")
        if int(max_size) <= 0:
            raise ValueError(f"max_size must be positive (got {max_size})")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_size = int(max_size)
        self.device = device

        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    @staticmethod
    def _to_1d_float32(x):
        """Convert array-like / tensor-like input to a 1D np.float32 array."""
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def add(self, state, action, reward, next_state, done):
        """Add a single transition."""
        state = self._to_1d_float32(state)
        next_state = self._to_1d_float32(next_state)
        action = self._to_1d_float32(action)

        if state.size != self.state_dim:
            raise ValueError(f"State dim mismatch: expected {self.state_dim}, got {state.size}")
        if next_state.size != self.state_dim:
            raise ValueError(f"Next state dim mismatch: expected {self.state_dim}, got {next_state.size}")
        if action.size != self.action_dim:
            raise ValueError(f"Action dim mismatch: expected {self.action_dim}, got {action.size}")

        # Prevent NaNs from silently poisoning training
        if (not np.all(np.isfinite(state))) or (not np.all(np.isfinite(next_state))) or (not np.all(np.isfinite(action))):
            raise ValueError("Non-finite values detected in transition (state/action/next_state)")

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

    def is_ready(self, batch_size):
        return self.size >= int(batch_size)

    def clear(self):
        self.ptr = 0
        self.size = 0

    def get_stats(self):
        return {
            "size": int(self.size),
            "capacity": int(self.max_size),
            "utilization": float(self.size) / float(self.max_size),
            "ptr": int(self.ptr),
            "is_full": bool(self.size >= self.max_size),
        }

    def __len__(self):
        return int(self.size)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay (PER) buffer.

    Sampling probability:
        P(i) = p_i / sum_j p_j
    Priority definition:
        p_i = (|td_error| + eps) ** alpha

    Importance sampling weights (normalized):
        w_i = (N * P(i)) ** (-beta)
        w_i <- w_i / max(w_i)

    Notes:
    - This implementation stores priorities in "priority-space" (already alpha-applied).
      Therefore, when adding a new transition, set its priority to current max_priority directly
      (do NOT raise max_priority to alpha again).
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=1_000_000,
        device="cpu",
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100_000,
        epsilon=1e-6,
    ):
        super().__init__(state_dim, action_dim, max_size, device)

        if float(alpha) < 0:
            raise ValueError(f"alpha must be >= 0 (got {alpha})")
        if not (0.0 <= float(beta_start) <= 1.0):
            raise ValueError(f"beta_start must be in [0,1] (got {beta_start})")

        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = max(1, int(beta_frames))
        self.epsilon = float(epsilon)

        self.frame = 1

        self.priorities = np.zeros(self.max_size, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """Add transition and initialize its priority to current max_priority."""
        current_ptr = self.ptr
        super().add(state, action, reward, next_state, done)

        # The transition was written at current_ptr.
        self.priorities[current_ptr] = np.float32(self.max_priority)

    def _beta(self):
        # Anneal beta to 1.0 over beta_frames sampling calls.
        beta = self.beta_start + (1.0 - self.beta_start) * (float(self.frame) / float(self.beta_frames))
        return float(min(1.0, beta))

    def sample(self, batch_size):
        """Sample a batch according to priorities, returning IS weights and indices."""
        if self.size == 0:
            raise ValueError("Cannot sample from empty prioritized replay buffer")

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive (got {batch_size})")

        priorities = self.priorities[: self.size]
        psum = float(priorities.sum())
        if (psum <= 0.0) or (not np.isfinite(psum)):
            probs = np.ones(self.size, dtype=np.float32) / float(self.size)
        else:
            probs = priorities / psum

        replace = self.size < batch_size
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=replace)

        beta = self._beta()
        self.frame += 1

        # Importance sampling weights (numerically safe)
        p_i = np.maximum(probs[indices], 1e-12)
        weights = (self.size * p_i) ** (-beta)
        w_max = float(weights.max())
        if w_max > 0.0:
            weights = weights / w_max

        return {
            "states": torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            "next_states": torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device),
            "weights": torch.as_tensor(weights, dtype=torch.float32, device=self.device).reshape(-1, 1),
            "indices": indices,
        }

    def update_priorities(self, indices, td_errors):
        """Update priorities for sampled indices.

        Args:
            indices: array-like of buffer indices
            td_errors: array-like or torch tensor of TD errors corresponding to indices
        """
        if torch.is_tensor(td_errors):
            td_errors = td_errors.detach().cpu().numpy()
        td_errors = np.asarray(td_errors, dtype=np.float32).reshape(-1)

        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(indices).reshape(-1).astype(np.int64)

        if indices.size != td_errors.size:
            raise ValueError(f"Length mismatch: indices ({indices.size}) vs td_errors ({td_errors.size})")

        # Bounds guard (defensive)
        if (indices < 0).any() or (indices >= self.size).any():
            raise ValueError("Some indices are out of valid range for current buffer size")

        new_p = (np.abs(td_errors) + self.epsilon) ** self.alpha
        new_p = np.clip(new_p, self.epsilon, 1e6).astype(np.float32)

        self.priorities[indices] = new_p
        mp = float(new_p.max())
        if mp > self.max_priority:
            self.max_priority = mp

    def clear(self):
        super().clear()
        self.priorities.fill(0.0)
        self.max_priority = 1.0
        self.frame = 1

    def get_stats(self):
        base = super().get_stats()
        if self.size > 0:
            p = self.priorities[: self.size]
            psum = float(p.sum())
            base.update(
                {
                    "alpha": float(self.alpha),
                    "beta": float(self._beta()),
                    "epsilon": float(self.epsilon),
                    "max_priority": float(self.max_priority),
                    "mean_priority": float(p.mean()),
                    "min_priority": float(p.min()),
                    "priority_std": float(p.std()),
                    "priority_sum": float(psum),
                }
            )
        else:
            base.update(
                {
                    "alpha": float(self.alpha),
                    "beta": float(self._beta()),
                    "epsilon": float(self.epsilon),
                    "max_priority": float(self.max_priority),
                    "mean_priority": 0.0,
                    "min_priority": 0.0,
                    "priority_std": 0.0,
                    "priority_sum": 0.0,
                }
            )
        return base
