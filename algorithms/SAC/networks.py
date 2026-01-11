import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

class SoftQNetwork(nn.Module):
    """
    Soft Q-Network (Critic) for SAC

    Estimates Q(s, a) - the expected return starting from state s,
    taking action a, and following the policy thereafter.

    SAC uses two Q-networks to mitigate overestimation bias.
    """

    def __init__(self, n_input, n_action, n_hidden=256):
        """
        Initialize Q-network

        Args:
            n_input: State dimension
            n_action: Action dimension
            n_hidden: Number of hidden units
        """
        super().__init__()

        # Q-network takes state-action pair as input
        self.network = nn.Sequential(
            nn.Linear(n_input + n_action, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state, action):
        """
        Forward pass

        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)

        Returns:
            Q-value tensor (batch_size, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class PolicyNetwork(nn.Module):
    """Policy Network (Actor) for SAC with a Dirichlet distribution.

    Produces concentration parameters `alpha` for a Dirichlet distribution over portfolio
    weights (assets + cash). Dirichlet samples lie on the simplex (non-negative and sum to 1).

    Stability / integration notes:
    - We clamp actions away from exact 0 before computing log_prob to avoid -inf / NaNs.
    - We renormalize after clamping to preserve the simplex constraint.
    - MPS backend: Dirichlet rsample/log_prob may be unsupported/unstable for training gradients.
      This implementation raises if gradients are enabled on MPS to avoid silent failures.
      Use CPU/CUDA for training with Dirichlet, or switch to a different policy parameterization.
    """

    def __init__(
        self,
        n_input: int,
        n_action: int,
        n_hidden: int = 256,
        alpha_min: float = 0.6,
        alpha_max: float = 100.0,
        action_eps: float = 1e-8,
    ):
        super().__init__()

        self.n_action = int(n_action)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.action_eps = float(action_eps)

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        # Output Dirichlet concentration parameters (alpha)
        self.alpha_head = nn.Linear(n_hidden, n_action)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

        # Smaller gain for stability; bias=1 encourages near-uniform early allocations.
        nn.init.xavier_uniform_(self.alpha_head.weight, gain=0.1)
        nn.init.constant_(self.alpha_head.bias, 1.0)

    @staticmethod
    def _device_of(x: torch.Tensor):
        return x.device

    @staticmethod
    def _ensure_device(x: torch.Tensor, device):
        """Move x to device if provided and different."""
        if device is None:
            return x
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if x.device != dev:
            return x.to(dev)
        return x

    @staticmethod
    def _safe_simplex(action: torch.Tensor, eps: float) -> torch.Tensor:
        """Clamp away from exact zero and renormalize to sum to 1."""
        action = torch.clamp(action, min=eps)
        action = action / action.sum(dim=-1, keepdim=True)
        return action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Dirichlet concentration parameters alpha.

        Returns:
            alpha: (batch_size, n_action), strictly positive.
        """
        features = self.feature_net(state)

        # softplus ensures positivity; add alpha_min to avoid tiny concentrations.
        alpha = F.softplus(self.alpha_head(features)) + self.alpha_min

        # Prevent extremely large concentrations which can harm exploration / stability.
        if self.alpha_max is not None:
            alpha = torch.clamp(alpha, max=self.alpha_max)

        return alpha

    def sample(self, state: torch.Tensor, device=None):
        """Sample an action using rsample (reparameterized) and compute its log-prob.

        Returns:
            action: (batch_size, n_action)
            log_prob: (batch_size,)
            alpha: (batch_size, n_action)
        """
        # Keep state (and therefore alpha) on the intended device.
        state = self._ensure_device(state, device)
        out_device = state.device

        alpha = self.forward(state)

        # MPS safety: prevent silent actor-gradient failure.
        if out_device.type == "mps" and torch.is_grad_enabled():
            raise RuntimeError(
                "Dirichlet rsample/log_prob may be unsupported on MPS with gradients. "
                "Use device='cpu' or CUDA for training, or change the policy parameterization."
            )

        if out_device.type == "mps":
            # Evaluation-only fallback: do distribution math on CPU.
            alpha_cpu = alpha.detach().cpu()
            dist = Dirichlet(alpha_cpu)
            action_cpu = dist.rsample()
            action_cpu = self._safe_simplex(action_cpu, self.action_eps)
            log_prob_cpu = dist.log_prob(action_cpu)
            return action_cpu.to(out_device), log_prob_cpu.to(out_device), alpha

        # Normal path: keep everything on-device.
        dist = Dirichlet(alpha)
        action = dist.rsample()
        action = self._safe_simplex(action, self.action_eps)
        log_prob = dist.log_prob(action)

        return action, log_prob, alpha

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, device=None):
        """Evaluate log_prob and entropy of a provided action under the current policy.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, n_action)

        Returns:
            log_prob: (batch_size,)
            entropy: (batch_size,)
        """
        state = self._ensure_device(state, device)
        action = self._ensure_device(action, state.device)
        out_device = state.device

        if not torch.isfinite(action).all():
            raise ValueError("Non-finite values detected in action during PolicyNetwork.evaluate()")

        alpha = self.forward(state)

        # MPS safety: prevent silent actor-gradient failure.
        if out_device.type == "mps" and torch.is_grad_enabled():
            raise RuntimeError(
                "Dirichlet log_prob/entropy may be unsupported on MPS with gradients. "
                "Use device='cpu' or CUDA for training, or change the policy parameterization."
            )

        if out_device.type == "mps":
            # Evaluation-only fallback on CPU.
            alpha_cpu = alpha.detach().cpu()
            dist = Dirichlet(alpha_cpu)
            action_cpu = action.detach().cpu()
            action_cpu = self._safe_simplex(action_cpu, self.action_eps)
            log_prob_cpu = dist.log_prob(action_cpu)
            entropy_cpu = dist.entropy()
            return log_prob_cpu.to(out_device), entropy_cpu.to(out_device)

        dist = Dirichlet(alpha)
        action = self._safe_simplex(action, self.action_eps)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic action (mean of Dirichlet): alpha / sum(alpha)."""
        alpha = self.forward(state)
        return alpha / alpha.sum(dim=-1, keepdim=True)


class ValueNetwork(nn.Module):
    """
    Value Network for SAC (optional)

    Estimates V(s) - the expected return starting from state s.
    While the original SAC paper uses a separate value network,
    later versions compute V from Q and the policy.

    Including it here for stability, as mentioned in the paper.
    """

    def __init__(self, n_input, n_hidden=256):
        """
        Initialize value network

        Args:
            n_input: State dimension
            n_hidden: Number of hidden units
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state):
        """
        Forward pass

        Args:
            state: State tensor

        Returns:
            V-value tensor
        """
        return self.network(state)


# For backward compatibility with the naming convention
class TwinQNetwork(nn.Module):
    """
    Twin Q-Networks for SAC

    Wraps two Q-networks and provides convenient access to both.
    """

    def __init__(self, n_input, n_action, n_hidden=256):
        super().__init__()
        self.q1 = SoftQNetwork(n_input, n_action, n_hidden)
        self.q2 = SoftQNetwork(n_input, n_action, n_hidden)

    def forward(self, state, action):
        """
        Get Q-values from both networks

        Returns:
            q1_value, q2_value
        """
        return self.q1(state, action), self.q2(state, action)

    def q1_forward(self, state, action):
        """Get Q-value from first network only"""
        return self.q1(state, action)

    def q2_forward(self, state, action):
        """Get Q-value from second network only"""
        return self.q2(state, action)
