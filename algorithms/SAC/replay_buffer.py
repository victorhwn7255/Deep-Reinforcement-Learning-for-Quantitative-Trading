import numpy as np
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer for SAC
    
    Stores transitions (state, action, reward, next_state, done) for off-policy learning.
    Uses numpy arrays for efficient storage and random sampling.
    """
    
    def __init__(self, state_dim, action_dim, max_size=1000000, device='cpu'):
        """
        Initialize replay buffer
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_size: Maximum number of transitions to store
            device: Device to return tensors on
        """
        self.max_size = max_size
        self.device = device
        self.ptr = 0  # Pointer to current position
        self.size = 0  # Current number of stored transitions
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a single transition to the buffer
        
        Args:
            state: Current state (numpy array or flattened)
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Flatten state if needed
        if isinstance(state, np.ndarray):
            state = state.flatten()
        if isinstance(next_state, np.ndarray):
            next_state = next_state.flatten()
        if isinstance(action, np.ndarray):
            action = action.flatten()
            
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # Update pointer (circular buffer)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions uniformly at random
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary of tensors: states, actions, rewards, next_states, dones
        """
        # Random indices
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Convert to tensors and move to device
        batch = {
            'states': torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            'rewards': torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            'next_states': torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            'dones': torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        }
        
        return batch
    
    def __len__(self):
        """Return current size of buffer"""
        return self.size
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples for a batch"""
        return self.size >= batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer (optional enhancement)
    
    Samples transitions based on TD error priority, with importance sampling weights.
    """
    
    def __init__(self, state_dim, action_dim, max_size=1000000, device='cpu',
                 alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Initialize prioritized replay buffer
        
        Args:
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling exponent
            beta_frames: Number of frames over which to anneal beta to 1
        """
        super().__init__(state_dim, action_dim, max_size, device)
        
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Priority tree
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        """Add transition with max priority"""
        super().add(state, action, reward, next_state, done)
        # New transitions get max priority
        self.priorities[self.ptr - 1] = self.max_priority ** self.alpha
        
    def sample(self, batch_size):
        """Sample batch based on priorities"""
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = {
            'states': torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            'rewards': torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            'next_states': torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            'dones': torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device),
            'weights': torch.as_tensor(weights, dtype=torch.float32, device=self.device),
            'indices': indices
        }
        
        return batch
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
