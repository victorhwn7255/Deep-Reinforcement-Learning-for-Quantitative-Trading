from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Dirichlet

from networks import SoftQNetwork, PolicyNetwork, ValueNetwork
from replay_buffer import ReplayBuffer

from config import Config


class Agent:
    """
    Soft Actor-Critic (SAC) agent for long-only portfolio management (Dirichlet policy).

    Repo conventions:
    - Action is simplex weights over (assets + cash), a >= 0, sum(a) = 1.
    - Env `done` is treated as time-limit truncation (end of dataset), not a failure terminal.
      => We DO NOT mask bootstrapping with (1 - done) unless you later define true terminals.
    - Env reward is usually reward_scale * log1p(net_return).
      For monitoring, we compute episode total return from env.last_net_return (equity-1).
    """

    def __init__(
        self,
        n_input: int,
        n_action: int,
        cfg: Optional["Config"] = None,

        # Backwards-compatible fallbacks if cfg is None:
        learning_rate: float = 1e-3,  # legacy single LR
        actor_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        value_lr: Optional[float] = None,
        alpha_lr: Optional[float] = None,

        gamma: float = 0.99,
        tau: float = 0.005,

        init_alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,

        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 10_000,
        update_frequency: int = 1,
        updates_per_step: int = 1,

        n_hidden: int = 256,
        alpha_min: float = 0.6,
        alpha_max: float = 100.0,
        action_eps: float = 1e-8,

        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = 1.0,

        device: Any = "cpu",
    ):
        # ---------------------------
        # Config-driven override
        # ---------------------------
        if cfg is not None:
            # If caller didn't explicitly pass device, use cfg auto-detect
            if device == "cpu":
                device = cfg.auto_detect_device()

            gamma = float(cfg.sac.gamma)
            tau = float(cfg.sac.tau)

            actor_lr = float(cfg.sac.actor_lr)
            critic_lr = float(cfg.sac.critic_lr)
            value_lr = float(cfg.sac.value_lr)
            alpha_lr = float(cfg.sac.alpha_lr)

            init_alpha = float(cfg.sac.init_alpha)
            auto_entropy_tuning = bool(cfg.sac.auto_entropy_tuning)
            target_entropy = cfg.sac.target_entropy  # may be None

            buffer_size = int(cfg.sac.buffer_size)
            batch_size = int(cfg.sac.batch_size)
            learning_starts = int(cfg.sac.learning_starts)
            update_frequency = int(cfg.sac.update_frequency)
            updates_per_step = int(cfg.sac.updates_per_step)

            n_hidden = int(cfg.network.hidden_size)
            alpha_min = float(cfg.network.alpha_min)
            alpha_max = float(cfg.network.alpha_max)
            action_eps = float(cfg.network.action_eps)

            # Your SACConfig has no weight_decay; use network.weight_decay
            weight_decay = float(cfg.network.weight_decay)

            # Your SACConfig uses gradient_clip_norm; 0 disables clipping
            gcn = float(cfg.sac.gradient_clip_norm)
            grad_clip_norm = None if gcn <= 0.0 else gcn

        # Backwards compatibility: if per-net LR not provided, use legacy learning_rate
        if actor_lr is None:
            actor_lr = float(learning_rate)
        if critic_lr is None:
            critic_lr = float(learning_rate)
        if value_lr is None:
            value_lr = float(learning_rate)
        if alpha_lr is None:
            alpha_lr = float(learning_rate)

        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.update_frequency = int(update_frequency)
        self.updates_per_step = int(max(1, updates_per_step))
        self.n_action = int(n_action)
        self.grad_clip_norm = grad_clip_norm

        # ---------------------------
        # Networks
        # ---------------------------
        self.q1 = SoftQNetwork(n_input, n_action, n_hidden).to(self.device)
        self.q2 = SoftQNetwork(n_input, n_action, n_hidden).to(self.device)

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        self.policy = PolicyNetwork(
            n_input=n_input,
            n_action=n_action,
            n_hidden=n_hidden,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            action_eps=action_eps,
        ).to(self.device)

        self.value = ValueNetwork(n_input, n_hidden).to(self.device)
        self.value_target = copy.deepcopy(self.value)
        for p in self.value_target.parameters():
            p.requires_grad = False

        # ---------------------------
        # Optimizers
        # ---------------------------
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=float(critic_lr), weight_decay=float(weight_decay))
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=float(critic_lr), weight_decay=float(weight_decay))
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=float(actor_lr), weight_decay=float(weight_decay))
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=float(value_lr), weight_decay=float(weight_decay))

        # ---------------------------
        # Entropy / temperature
        # ---------------------------
        self.auto_entropy_tuning = bool(auto_entropy_tuning)
        self.target_entropy: Optional[float]

        if self.auto_entropy_tuning:
            if target_entropy is None:
                # Consistent with Dirichlet: use its max entropy at alpha=1, minus a margin
                with torch.no_grad():
                    uniform_alpha = torch.ones(n_action, device=self.device)
                    h_max = Dirichlet(uniform_alpha).entropy().item()
                margin = float(cfg.sac.target_entropy_margin) if cfg is not None else 0.5
                self.target_entropy = float(h_max - margin)
            else:
                self.target_entropy = float(target_entropy)

            init_log_alpha = float(np.log(max(1e-12, float(init_alpha))))
            self.log_alpha = torch.tensor(
                init_log_alpha, dtype=torch.float32, device=self.device, requires_grad=True
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=float(alpha_lr), weight_decay=0.0)
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None
            self.alpha = float(init_alpha)

        # ---------------------------
        # Replay buffer
        # ---------------------------
        self.replay_buffer = ReplayBuffer(
            state_dim=n_input,
            action_dim=n_action,
            max_size=int(buffer_size),
            device=self.device,
        )

        self.global_step = 0

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _maybe_clip_grads(self, module: torch.nn.Module) -> None:
        if self.grad_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(module.parameters(), float(self.grad_clip_norm))

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ---------------------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------------------

    def select_action(self, state, evaluate: bool = False) -> np.ndarray:
        state_np = np.asarray(state, dtype=np.float32).reshape(-1)
        state_tensor = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                action = self.policy.get_deterministic_action(state_tensor)
            else:
                action, _, _ = self.policy.sample(state_tensor, device=self.device)

        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def choose_action_deterministic(self, observation):
        return self.select_action(observation, evaluate=True)

    def choose_action_stochastic(self, observation):
        return self.select_action(observation, evaluate=False)

    # ---------------------------------------------------------------------
    # SAC update
    # ---------------------------------------------------------------------

    def update(self) -> Optional[Dict[str, float]]:
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        # dones = batch["dones"]  # unused (time-limit truncation)

        # -------- Value update --------
        with torch.no_grad():
            cur_actions, cur_logp, _ = self.policy.sample(states, device=self.device)
            q1_cur = self.q1(states, cur_actions)
            q2_cur = self.q2(states, cur_actions)
            q_cur = torch.min(q1_cur, q2_cur)
            v_target = q_cur - self.alpha * cur_logp.unsqueeze(-1)

        v_pred = self.value(states)
        value_loss = F.mse_loss(v_pred, v_target)

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self._maybe_clip_grads(self.value)
        self.value_optimizer.step()

        # -------- Q update --------
        with torch.no_grad():
            v_next = self.value_target(next_states)
            q_target = rewards + self.gamma * v_next  # no (1-done) masking

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        self._maybe_clip_grads(self.q1)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        self._maybe_clip_grads(self.q2)
        self.q2_optimizer.step()

        # -------- Policy update --------
        new_actions, logp, _ = self.policy.sample(states, device=self.device)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * logp.unsqueeze(-1) - q_new).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self._maybe_clip_grads(self.policy)
        self.policy_optimizer.step()

        # -------- Alpha update --------
        alpha_loss = None
        if self.auto_entropy_tuning:
            with torch.no_grad():
                _, logp_a, _ = self.policy.sample(states, device=self.device)

            alpha_loss = -(self.log_alpha * (logp_a + float(self.target_entropy))).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.exp().item())

        # -------- Target updates --------
        self._soft_update(self.value, self.value_target)
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        out = {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "alpha": float(self.alpha),
        }
        if alpha_loss is not None:
            out["alpha_loss"] = float(alpha_loss.item())
        return out

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------

    def _pack_best_state(
        self,
        episode: int,
        global_step: int,
        avg_return: float,
        all_returns: List[float],
    ) -> Dict[str, Any]:
        return {
            "policy_state_dict": {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()},
            "q1_state_dict": {k: v.detach().cpu().clone() for k, v in self.q1.state_dict().items()},
            "q2_state_dict": {k: v.detach().cpu().clone() for k, v in self.q2.state_dict().items()},
            "value_state_dict": {k: v.detach().cpu().clone() for k, v in self.value.state_dict().items()},
            "q1_target_state_dict": {k: v.detach().cpu().clone() for k, v in self.q1_target.state_dict().items()},
            "q2_target_state_dict": {k: v.detach().cpu().clone() for k, v in self.q2_target.state_dict().items()},
            "value_target_state_dict": {k: v.detach().cpu().clone() for k, v in self.value_target.state_dict().items()},
            "episode": int(episode),
            "global_step": int(global_step),
            "avg_return": float(avg_return),
            "alpha": float(self.alpha),
            "log_alpha": float(self.log_alpha.detach().cpu().item()) if self.auto_entropy_tuning else None,
            "all_returns": list(all_returns),
        }

    def learn(
        self,
        env,
        total_timesteps: int,
        print_interval_steps: int = 1000,
        best_avg_lookback_episodes: int = 10,
        best_model_path: Optional[str] = None,
    ) -> Tuple[List[float], List[Dict[str, float]], Optional[Dict[str, Any]]]:
        """
        Returns:
            episode_returns: list of episode total net returns (equity - 1)
            losses: list of loss dicts
            best_model_state: cpu-safe dict or None
        """
        episode_returns: List[float] = []
        losses: List[Dict[str, float]] = []

        best_avg_return = -np.inf
        best_model_state: Optional[Dict[str, Any]] = None

        start_time = time.time()
        obs = env.reset()

        episode_equity = 1.0
        episode_count = 0

        for step in range(int(total_timesteps)):
            self.global_step = int(step)

            if step < self.learning_starts:
                action = np.random.dirichlet(np.ones(self.n_action, dtype=np.float32)).astype(np.float32)
            else:
                action = self.select_action(obs, evaluate=False)

            next_obs, reward, done = env.step(action)

            # Monitor with true net returns (not scaled log rewards)
            if hasattr(env, "last_net_return"):
                episode_equity *= float(1.0 + float(env.last_net_return))

            # Store transition
            obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
            next_obs_flat = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            self.replay_buffer.add(obs_flat, action, float(reward), next_obs_flat, float(done))

            # Updates
            if step >= self.learning_starts and (step % self.update_frequency == 0):
                for _ in range(self.updates_per_step):
                    ld = self.update()
                    if ld is not None:
                        losses.append(ld)

            # Logging
            if print_interval_steps > 0 and (step % int(print_interval_steps) == 0):
                elapsed = time.time() - start_time
                sps = step / elapsed if elapsed > 0 else 0.0
                print(f"Step {step:7d} | {sps:6.1f} steps/s | alpha={self.alpha:.4f} | device={self.device}")

            # Episode boundary
            if done:
                episode_count += 1
                ep_total_return = float(episode_equity - 1.0)
                episode_returns.append(ep_total_return)

                print(f"global_step={step}, episode={episode_count}, episode_total_return={ep_total_return:.6f}")

                if len(episode_returns) >= int(best_avg_lookback_episodes):
                    avg_return = float(np.mean(episode_returns[-int(best_avg_lookback_episodes):]))
                    if avg_return > best_avg_return:
                        best_avg_return = avg_return
                        best_model_state = self._pack_best_state(
                            episode=episode_count,
                            global_step=step,
                            avg_return=avg_return,
                            all_returns=episode_returns,
                        )
                        if best_model_path is not None:
                            torch.save(best_model_state, best_model_path)
                            print(
                                f"  ✓ New best! avg_return(last {best_avg_lookback_episodes})={avg_return:.6f}\n"
                                f"    → Saved best checkpoint to: {best_model_path}"
                            )

                obs = env.reset()
                episode_equity = 1.0
            else:
                obs = next_obs

        return episode_returns, losses, best_model_state

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        state = {
            "policy_state_dict": self.policy.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "value_target_state_dict": self.value_target.state_dict(),
            "alpha": float(self.alpha),
            "log_alpha": float(self.log_alpha.detach().cpu().item()) if self.auto_entropy_tuning else None,
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)

        # Backward compatibility: bare policy dict
        if isinstance(state, dict) and "policy_state_dict" not in state:
            self.policy.load_state_dict(state)
            print(f"Loaded policy-only checkpoint from {path}")
            return

        if not isinstance(state, dict) or "policy_state_dict" not in state:
            raise ValueError(f"Unrecognized checkpoint format at {path}")

        self.policy.load_state_dict(state["policy_state_dict"])
        if "q1_state_dict" in state:
            self.q1.load_state_dict(state["q1_state_dict"])
        if "q2_state_dict" in state:
            self.q2.load_state_dict(state["q2_state_dict"])
        if "value_state_dict" in state:
            self.value.load_state_dict(state["value_state_dict"])

        if "q1_target_state_dict" in state:
            self.q1_target.load_state_dict(state["q1_target_state_dict"])
        else:
            self.q1_target = copy.deepcopy(self.q1)
        if "q2_target_state_dict" in state:
            self.q2_target.load_state_dict(state["q2_target_state_dict"])
        else:
            self.q2_target = copy.deepcopy(self.q2)
        if "value_target_state_dict" in state:
            self.value_target.load_state_dict(state["value_target_state_dict"])
        else:
            self.value_target = copy.deepcopy(self.value)

        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
        for p in self.value_target.parameters():
            p.requires_grad = False

        if "alpha" in state:
            self.alpha = float(state["alpha"])
        if self.auto_entropy_tuning and self.log_alpha is not None and state.get("log_alpha") is not None:
            self.log_alpha.data.copy_(torch.tensor(float(state["log_alpha"]), device=self.device))
            self.alpha = float(self.log_alpha.exp().item())

        print(f"Model loaded from {path}")

    def load_best_model(self, best_model_state: Dict[str, Any]) -> None:
        self.policy.load_state_dict(best_model_state["policy_state_dict"])
        self.q1.load_state_dict(best_model_state["q1_state_dict"])
        self.q2.load_state_dict(best_model_state["q2_state_dict"])
        self.value.load_state_dict(best_model_state["value_state_dict"])

        if "q1_target_state_dict" in best_model_state:
            self.q1_target.load_state_dict(best_model_state["q1_target_state_dict"])
        if "q2_target_state_dict" in best_model_state:
            self.q2_target.load_state_dict(best_model_state["q2_target_state_dict"])
        if "value_target_state_dict" in best_model_state:
            self.value_target.load_state_dict(best_model_state["value_target_state_dict"])

        if self.auto_entropy_tuning and self.log_alpha is not None and best_model_state.get("log_alpha") is not None:
            self.log_alpha.data.copy_(torch.tensor(float(best_model_state["log_alpha"]), device=self.device))
            self.alpha = float(self.log_alpha.exp().item())
        elif "alpha" in best_model_state:
            self.alpha = float(best_model_state["alpha"])

        print(
            f"Best model loaded (Episode {best_model_state.get('episode')}, "
            f"Avg Return: {best_model_state.get('avg_return')})"
        )