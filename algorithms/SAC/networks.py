from __future__ import annotations

from typing import Callable, List

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
    if name == "gelu":
        return nn.GELU
    raise ValueError(f"Unknown activation: {name}. Use 'relu' or 'gelu'.")


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_size: int,
    num_layers: int,
    activation: str = "relu",
    layer_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    Act = _get_activation(activation)
    layers: List[nn.Module] = []

    if num_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    # first hidden
    layers.append(nn.Linear(in_dim, hidden_size))
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_size))
    layers.append(Act())
    if dropout and dropout > 0:
        layers.append(nn.Dropout(dropout))

    # middle hidden
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(Act())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))

    # output
    layers.append(nn.Linear(hidden_size, out_dim))
    return nn.Sequential(*layers)


def _xavier_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# =============================================================================
# Networks
# =============================================================================

class SoftQNetwork(nn.Module):
    """Critic Q(s,a)."""

    def __init__(self, state_dim: int, action_dim: int, net_cfg):
        super().__init__()
        self.net = build_mlp(
            in_dim=state_dim + action_dim,
            out_dim=1,
            hidden_size=int(net_cfg.hidden_size),
            num_layers=int(net_cfg.num_layers),
            activation=str(net_cfg.activation),
            layer_norm=bool(net_cfg.layer_norm),
            dropout=float(net_cfg.dropout),
        )
        self.apply(_xavier_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ValueNetwork(nn.Module):
    """Value V(s)."""

    def __init__(self, state_dim: int, net_cfg):
        super().__init__()
        self.net = build_mlp(
            in_dim=state_dim,
            out_dim=1,
            hidden_size=int(net_cfg.hidden_size),
            num_layers=int(net_cfg.num_layers),
            activation=str(net_cfg.activation),
            layer_norm=bool(net_cfg.layer_norm),
            dropout=float(net_cfg.dropout),
        )
        self.apply(_xavier_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PolicyNetwork(nn.Module):
    """Dirichlet policy over simplex weights."""

    def __init__(self, state_dim: int, action_dim: int, net_cfg):
        super().__init__()
        self.action_dim = int(action_dim)

        self.action_eps = float(net_cfg.action_eps)
        self.alpha_min = float(net_cfg.alpha_min)
        self.alpha_max = float(net_cfg.alpha_max)

        # feature net -> alpha head
        self.feature_net = build_mlp(
            in_dim=state_dim,
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
    def _ensure_device(x: torch.Tensor, device) -> torch.Tensor:
        if device is None:
            return x
        return x.to(device)

    @staticmethod
    def _safe_simplex(action: torch.Tensor, eps: float) -> torch.Tensor:
        action = torch.clamp(action, min=eps)
        action = action / action.sum(dim=-1, keepdim=True)
        return action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return alpha concentration params (strictly positive)."""
        h = self.feature_net(state)
        raw = self.alpha_head(h)

        alpha = F.softplus(raw) + self.alpha_min
        if self.alpha_max is not None and self.alpha_max > 0:
            alpha = torch.clamp(alpha, max=self.alpha_max)
        return alpha

    def sample(self, state: torch.Tensor, device=None):
        """Reparameterized sample + log_prob."""
        state = self._ensure_device(state, device)
        out_device = state.device

        alpha = self.forward(state)

        # MPS safety: Dirichlet gradients are often unsupported
        if out_device.type == "mps" and torch.is_grad_enabled():
            raise RuntimeError(
                "Dirichlet rsample/log_prob gradients may be unsupported on MPS. "
                "Use CPU or CUDA."
            )

        if out_device.type == "mps":
            # evaluation-only fallback: compute on CPU
            alpha_cpu = alpha.detach().cpu()
            dist = Dirichlet(alpha_cpu)
            action_cpu = dist.rsample()
            action_cpu = self._safe_simplex(action_cpu, self.action_eps)
            log_prob_cpu = dist.log_prob(action_cpu)
            return action_cpu.to(out_device), log_prob_cpu.to(out_device), alpha

        dist = Dirichlet(alpha)
        action = dist.rsample()
        action = self._safe_simplex(action, self.action_eps)
        log_prob = dist.log_prob(action)
        return action, log_prob, alpha

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, device=None):
        """Return log_prob and entropy for a provided action."""
        state = self._ensure_device(state, device)
        action = self._ensure_device(action, state.device)

        alpha = self.forward(state)
        out_device = state.device

        if out_device.type == "mps":
            alpha_cpu = alpha.detach().cpu()
            action_cpu = action.detach().cpu()
            dist = Dirichlet(alpha_cpu)
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
        alpha = self.forward(state)
        return alpha / alpha.sum(dim=-1, keepdim=True)