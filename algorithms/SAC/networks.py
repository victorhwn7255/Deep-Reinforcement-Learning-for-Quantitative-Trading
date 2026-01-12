from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


# =============================================================================
# Helpers
# =============================================================================

def _get_activation(name: str) -> Callable[[], nn.Module]:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "elu":
        return nn.ELU
    if name == "gelu":
        return nn.GELU
    if name == "leaky_relu":
        return lambda: nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unknown activation: {name}")


def _xavier_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_size: int,
    num_layers: int,
    activation: str = "relu",
    layer_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Generic MLP builder.

    num_layers counts total linear layers including the output layer.
    - num_layers=1: Linear(in_dim -> out_dim)
    - num_layers>=2: (Linear+Act+...)* + Linear(out_dim)
    """
    in_dim = int(in_dim)
    out_dim = int(out_dim)
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)

    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    Act = _get_activation(activation)
    layers: list[nn.Module] = []

    if num_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    # Hidden layers
    dims = [in_dim] + [hidden_size] * (num_layers - 1)  # last hidden feeds to output separately
    for i in range(num_layers - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(Act())
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(p=float(dropout)))

    # Output layer
    layers.append(nn.Linear(hidden_size, out_dim))
    return nn.Sequential(*layers)


# =============================================================================
# Critic Networks (SAC v2 uses twin Q networks + target copies)
# =============================================================================

class SoftQNetwork(nn.Module):
    """Q(s,a) critic. Output shape: [B, 1]."""

    def __init__(self, state_dim: int, action_dim: int, net_cfg):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.q_net = build_mlp(
            in_dim=self.state_dim + self.action_dim,
            out_dim=1,
            hidden_size=int(net_cfg.hidden_size),
            num_layers=max(1, int(net_cfg.num_layers)),
            activation=str(net_cfg.activation),
            layer_norm=bool(net_cfg.layer_norm),
            dropout=float(net_cfg.dropout),
        )
        self.apply(_xavier_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Ensure [B, D]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        x = torch.cat([state, action], dim=-1)
        q = self.q_net(x)
        # Always [B, 1]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        return q


# =============================================================================
# Value Network (compatibility shim; SAC v2 does NOT need this)
# =============================================================================

class ValueNetwork(nn.Module):
    """DEPRECATED for SAC v2.

    Kept temporarily so older imports don't break while we migrate step-by-step.
    You will remove usage from agent.py in the next migration step.
    """

    def __init__(self, state_dim: int, net_cfg):
        super().__init__()
        self.state_dim = int(state_dim)

        self.v_net = build_mlp(
            in_dim=self.state_dim,
            out_dim=1,
            hidden_size=int(net_cfg.hidden_size),
            num_layers=max(1, int(net_cfg.num_layers)),
            activation=str(net_cfg.activation),
            layer_norm=bool(net_cfg.layer_norm),
            dropout=float(net_cfg.dropout),
        )
        self.apply(_xavier_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        v = self.v_net(state)
        if v.dim() == 1:
            v = v.unsqueeze(1)
        return v


# =============================================================================
# Policy Network (Dirichlet over simplex)
# =============================================================================

class PolicyNetwork(nn.Module):
    """Dirichlet policy π(a|s) over the simplex (long-only weights).

    Outputs Dirichlet concentration parameters alpha > 0.
    """

    def __init__(self, state_dim: int, action_dim: int, net_cfg):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.action_eps = float(net_cfg.action_eps)
        self.alpha_min = float(net_cfg.alpha_min)
        self.alpha_max = float(net_cfg.alpha_max)

        self.feature_net = build_mlp(
            in_dim=self.state_dim,
            out_dim=int(net_cfg.hidden_size),
            hidden_size=int(net_cfg.hidden_size),
            num_layers=max(1, int(net_cfg.num_layers)),
            activation=str(net_cfg.activation),
            layer_norm=bool(net_cfg.layer_norm),
            dropout=float(net_cfg.dropout),
        )
        self.alpha_head = nn.Linear(int(net_cfg.hidden_size), self.action_dim)

        self.apply(_xavier_init)

    @staticmethod
    def _safe_simplex(action: torch.Tensor, eps: float) -> torch.Tensor:
        """Clamp to (eps, 1) and renormalize to sum=1."""
        a = torch.clamp(action, min=eps)
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-12)
        return a

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return alpha parameters, shape [B, action_dim]."""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        h = self.feature_net(state)
        raw = self.alpha_head(h)

        # Softplus to ensure positivity + floor, then clamp for stability
        alpha = F.softplus(raw) + self.alpha_min
        alpha = torch.clamp(alpha, min=self.alpha_min, max=self.alpha_max)
        return alpha

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action with reparameterization; returns:
        - action: [B, action_dim]
        - log_prob: [B, 1]
        """
        alpha = self.forward(state)

        # Dirichlet gradients can be problematic on some backends; compute log_prob on CPU for MPS safety
        dist = Dirichlet(alpha)
        action = dist.rsample()  # [B, action_dim], differentiable
        action = self._safe_simplex(action, self.action_eps)

        log_prob = dist.log_prob(action)  # [B]
        log_prob = log_prob.unsqueeze(1)  # [B,1]
        return action, log_prob

    def log_prob_and_entropy(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log_prob and entropy for given (state, action).
        Returns:
        - log_prob: [B, 1]
        - entropy:  [B, 1]
        """
        alpha = self.forward(state)
        dist = Dirichlet(alpha)
        action = self._safe_simplex(action, self.action_eps)

        log_prob = dist.log_prob(action).unsqueeze(1)
        entropy = dist.entropy().unsqueeze(1)
        return log_prob, entropy

    def get_deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """A deterministic proxy action: mean of Dirichlet = alpha / alpha0."""
        alpha = self.forward(state)
        return alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-12)
