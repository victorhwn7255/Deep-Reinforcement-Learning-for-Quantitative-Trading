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
    """Generic MLP builder. num_layers counts linear layers including output."""
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
    dims = [in_dim] + [hidden_size] * (num_layers - 1)
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
# Critic Network
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
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        q = self.q_net(x)
        if q.dim() == 1:
            q = q.unsqueeze(1)
        return q


# =============================================================================
# Policy Network (Dirichlet)
# =============================================================================

class PolicyNetwork(nn.Module):
    """Dirichlet policy π(a|s) over simplex (long-only weights incl cash).

    Key production fix:
      - DO NOT clamp/renormalize sampled actions inside sample() before log_prob.
        Dirichlet samples are already on the simplex.
      - Provide entropy(state) for Dirichlet-native temperature tuning.
    """

    def __init__(self, state_dim: int, action_dim: int, net_cfg):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return concentration alpha > 0, shape [B, action_dim]."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.feature_net(state)
        raw = self.alpha_head(h)
        alpha = F.softplus(raw) + self.alpha_min
        alpha = torch.clamp(alpha, min=self.alpha_min, max=self.alpha_max)
        return alpha

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample + log_prob.
        Returns:
          action: [B, action_dim]
          log_prob: [B, 1]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        out_device = state.device
        alpha = self.forward(state)

        # MPS safety: Dirichlet rsample/log_prob gradients are often problematic
        if out_device.type == "mps" and torch.is_grad_enabled():
            raise RuntimeError(
                "Dirichlet rsample/log_prob gradients may be unsupported on MPS. Use CPU or CUDA."
            )

        # For eval-only on MPS, compute on CPU and move back
        if out_device.type == "mps":
            alpha_cpu = alpha.detach().cpu()
            dist = Dirichlet(alpha_cpu)
            action_cpu = dist.rsample()
            logp_cpu = dist.log_prob(action_cpu)
            return action_cpu.to(out_device), logp_cpu.unsqueeze(1).to(out_device)

        dist = Dirichlet(alpha)
        action = dist.rsample()                # already on simplex
        logp = dist.log_prob(action)           # exact log_prob of sampled action
        return action, logp.unsqueeze(1)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Return policy entropy H(Dir(alpha(s))) as [B,1]."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        alpha = self.forward(state)

        out_device = state.device
        if out_device.type == "mps":
            alpha_cpu = alpha.detach().cpu()
            dist = Dirichlet(alpha_cpu)
            ent = dist.entropy()
            return ent.unsqueeze(1).to(out_device)

        dist = Dirichlet(alpha)
        return dist.entropy().unsqueeze(1)

    def get_deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic proxy = mean of Dirichlet = alpha / alpha0."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        alpha = self.forward(state)
        return alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-12)
