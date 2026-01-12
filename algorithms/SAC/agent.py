from __future__ import annotations

import copy
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import SoftQNetwork, PolicyNetwork
from replay_buffer import ReplayBuffer


class Agent:
    """Soft Actor-Critic (SAC v2) agent for portfolio management with Dirichlet policy.

    SAC v2:
      - Twin critics Q1, Q2 with target critics Q1_target, Q2_target
      - No Value network
      - Critic target uses: min(Q_target(s', a')) - alpha * log pi(a'|s')
    """

    def __init__(self, state_dim: int, action_dim: int, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        # -------------------------
        # Hyperparameters
        # -------------------------
        sac = cfg.sac
        self.gamma = float(sac.gamma)
        self.tau = float(sac.tau)

        self.batch_size = int(sac.batch_size)
        self.learning_starts = int(sac.learning_starts)
        self.update_frequency = int(sac.update_frequency)
        self.updates_per_step = int(getattr(sac, "updates_per_step", 1))
        self.grad_clip = float(getattr(sac, "gradient_clip_norm", 0.0))

        # Done handling
        self.treat_done_as_truncation = bool(getattr(cfg.env, "treat_done_as_truncation", True))

        # -------------------------
        # Temperature / entropy
        # -------------------------
        self.auto_entropy_tuning = bool(sac.auto_entropy_tuning)

        # We'll keep a numeric alpha for logging/use; if auto-tuning, it's derived from log_alpha.
        self.alpha = float(sac.init_alpha)

        # Target entropy heuristic (Dirichlet-specific heuristic from cfg)
        self.target_entropy = float(cfg.compute_target_entropy(self.action_dim))

        if self.auto_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=float(sac.alpha_lr))
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

        # -------------------------
        # Networks (SAC v2)
        # -------------------------
        net_cfg = cfg.network

        self.policy = PolicyNetwork(self.state_dim, self.action_dim, net_cfg).to(self.device)
        self.q1 = SoftQNetwork(self.state_dim, self.action_dim, net_cfg).to(self.device)
        self.q2 = SoftQNetwork(self.state_dim, self.action_dim, net_cfg).to(self.device)

        # Target critics
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # -------------------------
        # Optimizers
        # -------------------------
        wd = float(getattr(net_cfg, "weight_decay", 0.0))
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=float(sac.actor_lr), weight_decay=wd)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=float(sac.critic_lr), weight_decay=wd)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=float(sac.critic_lr), weight_decay=wd)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.state_dim,
            self.action_dim,
            max_size=int(sac.buffer_size),
            device=str(self.device),
        )

        self.global_step = 0

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Return portfolio weights action."""
        if isinstance(state, np.ndarray):
            s = state.flatten().astype(np.float32)
        else:
            s = np.asarray(state, dtype=np.float32).reshape(-1)

        state_tensor = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                action = self.policy.get_deterministic_action(state_tensor)
            else:
                action, _logp = self.policy.sample(state_tensor)

        return action.squeeze(0).cpu().numpy()

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Polyak averaging update: target = tau*source + (1-tau)*target."""
        tau = float(self.tau)
        with torch.no_grad():
            for p, tp in zip(source.parameters(), target.parameters()):
                tp.data.mul_(1.0 - tau)
                tp.data.add_(tau * p.data)

    def _clip_grads(self, module: nn.Module) -> None:
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)

    def _current_alpha_tensor(self) -> torch.Tensor:
        """Return alpha as a torch scalar tensor on device."""
        if self.auto_entropy_tuning and self.log_alpha is not None:
            return self.log_alpha.exp()
        return torch.tensor(self.alpha, device=self.device, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # Learning update (SAC v2)
    # -------------------------------------------------------------------------
    def update(self) -> Optional[Dict[str, float]]:
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch.get("dones", None)

        # Ensure shapes [B, 1]
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones is None:
            dones = torch.zeros_like(rewards)
        else:
            dones = dones.to(self.device)
            if dones.dim() == 1:
                dones = dones.unsqueeze(1)

        alpha_t = self._current_alpha_tensor().detach()  # used for targets & losses (detach unless tuning loss)

        # -------------------------
        # Critic target (SAC v2)
        # y = r + gamma * not_done * (min(Q_tgt(s',a')) - alpha * logp(a'|s'))
        # -------------------------
        with torch.no_grad():
            next_actions, next_logp = self.policy.sample(next_states)  # next_logp: [B,1]
            q1_t = self.q1_target(next_states, next_actions)
            q2_t = self.q2_target(next_states, next_actions)
            min_q_t = torch.min(q1_t, q2_t)

            if self.treat_done_as_truncation:
                not_done = torch.ones_like(dones)
            else:
                not_done = 1.0 - dones

            target_q = rewards + self.gamma * not_done * (min_q_t - alpha_t * next_logp)

        # -------------------------
        # Update critics
        # -------------------------
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        self._clip_grads(self.q1)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        self._clip_grads(self.q2)
        self.q2_optimizer.step()

        # -------------------------
        # Update policy
        # L_pi = E[ alpha*logp(a|s) - min(Q1,Q2)(s,a) ]
        # -------------------------
        new_actions, logp = self.policy.sample(states)  # logp: [B,1]
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        # Use alpha (detached) for actor loss; alpha gradient handled in alpha_loss
        policy_loss = (alpha_t * logp - min_q_new).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self._clip_grads(self.policy)
        self.policy_optimizer.step()

        # -------------------------
        # Temperature / alpha update (optional)
        # Standard SAC form:
        # L_alpha = -E[ log_alpha * (logp + target_entropy) ]
        # -------------------------
        alpha_loss = None
        if self.auto_entropy_tuning and self.log_alpha is not None and self.alpha_optimizer is not None:
            # Detach logp so alpha update doesn't backprop into policy
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = float(self.log_alpha.exp().item())

        # -------------------------
        # Soft update target critics
        # -------------------------
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        out: Dict[str, float] = {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha": float(self.alpha),
            "avg_logp": float(logp.mean().item()),
        }
        if alpha_loss is not None:
            out["alpha_loss"] = float(alpha_loss.item())
        return out

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def learn(self, env, total_timesteps: Optional[int] = None):
        """Train on env for total_timesteps. Episodes end on dataset truncation."""
        if total_timesteps is None:
            total_timesteps = int(self.cfg.training.total_timesteps)

        episode_returns: List[float] = []
        losses: List[Dict[str, float]] = []

        best_avg_return = -np.inf
        best_model_state = None

        obs = env.reset()
        episode_return = 0.0
        episode_count = 0

        start_time = time.time()

        for step in range(int(total_timesteps)):
            self.global_step = step

            # action selection
            if step < self.learning_starts:
                action = np.random.dirichlet(np.ones(self.action_dim)).astype(np.float32)
            else:
                action = self.select_action(obs, evaluate=False).astype(np.float32)

            next_obs, reward, done = env.step(action)
            episode_return += float(reward)

            # store transition
            self.replay_buffer.add(obs, action, reward, next_obs, float(done))

            # updates
            if step >= self.learning_starts and step % self.update_frequency == 0:
                for _ in range(max(1, self.updates_per_step)):
                    loss_dict = self.update()
                    if loss_dict is not None:
                        losses.append(loss_dict)

            # episode boundary: dataset end
            if done:
                episode_count += 1
                episode_returns.append(float(episode_return))

                if self.cfg.experiment.verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"global_step={step}  episode={episode_count}  "
                        f"episode_return={episode_return:.4f}  alpha={self.alpha:.4f}  "
                        f"elapsed={elapsed/60:.1f}m"
                    )

                # best model tracking (avg over last 10 episodes)
                if len(episode_returns) >= 10:
                    avg_return = float(np.mean(episode_returns[-10:]))
                    if avg_return > best_avg_return:
                        best_avg_return = avg_return
                        best_model_state = self.export_state(extra={
                            "episode": episode_count,
                            "global_step": step,
                            "avg_return": avg_return,
                            "all_returns": episode_returns.copy(),
                        })
                        # save best immediately
                        os.makedirs(self.cfg.training.model_dir, exist_ok=True)
                        torch.save(best_model_state, self.cfg.training.model_path_best)

                # periodic checkpoint save
                if (episode_count % int(self.cfg.training.save_interval_episodes)) == 0:
                    os.makedirs(self.cfg.training.model_dir, exist_ok=True)
                    ckpt = self.export_state(extra={"episode": episode_count, "global_step": step})
                    torch.save(ckpt, self.cfg.training.model_path_final.replace(".pth", f"_ep{episode_count}.pth"))

                # reset for next episode
                obs = env.reset()
                episode_return = 0.0
            else:
                obs = next_obs

        return episode_returns, losses, best_model_state

    # -------------------------------------------------------------------------
    # Checkpointing (SAC v2)
    # -------------------------------------------------------------------------
    def export_state(self, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        # Save online networks. Targets can be reconstructed at load.
        out: Dict[str, object] = {
            "sac_version": "v2",
            "policy_state_dict": {k: v.detach().cpu() for k, v in self.policy.state_dict().items()},
            "q1_state_dict": {k: v.detach().cpu() for k, v in self.q1.state_dict().items()},
            "q2_state_dict": {k: v.detach().cpu() for k, v in self.q2.state_dict().items()},
            "alpha": float(self.alpha),
            "auto_entropy_tuning": bool(self.auto_entropy_tuning),
            "target_entropy": float(self.target_entropy),
        }
        if self.auto_entropy_tuning and self.log_alpha is not None:
            out["log_alpha"] = self.log_alpha.detach().cpu()

        if extra:
            out.update(extra)
        return out

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.export_state(), path)

    def load_model(self, path: str, map_location: Optional[str] = None) -> None:
        loc = map_location if map_location is not None else str(self.device)
        ckpt = torch.load(path, map_location=loc)

        # Load online nets
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.q1.load_state_dict(ckpt["q1_state_dict"])
        self.q2.load_state_dict(ckpt["q2_state_dict"])

        # Rebuild targets from online nets for safety
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Alpha restore
        if ckpt.get("auto_entropy_tuning", False) and self.auto_entropy_tuning:
            if "log_alpha" in ckpt and self.log_alpha is not None:
                self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
                self.alpha = float(self.log_alpha.exp().item())
            else:
                # fallback to alpha scalar
                self.alpha = float(ckpt.get("alpha", self.alpha))
        else:
            self.alpha = float(ckpt.get("alpha", self.alpha))
