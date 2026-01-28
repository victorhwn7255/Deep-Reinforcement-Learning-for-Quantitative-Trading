from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class Env:
    """Portfolio management environment with close-to-close mechanics.

    Action: target weights over (n_assets + cash) that sum to 1 (simplex).
            => action_dim = n_assets + 1
    Reward: reward_scale * log1p(net_return) where net_return includes transaction costs.
    """

    def __init__(self, df: pd.DataFrame, tickers: List[str], cfg):
        self.cfg = cfg
        self.tickers = list(tickers)
        self.n_assets = len(self.tickers)

        # --- env params from config ---
        self.lag = int(cfg.env.lag)
        self.include_position_in_state = bool(cfg.env.include_position_in_state)

        self.tc_rate = float(cfg.env.tc_rate)
        self.tc_fixed = float(cfg.env.tc_fixed)
        self.turnover_threshold = float(cfg.env.turnover_threshold)
        self.turnover_include_cash = bool(cfg.env.turnover_include_cash)
        self.turnover_use_half_factor = bool(cfg.env.turnover_use_half_factor)

        self.reward_type = str(cfg.env.reward_type)
        self.reward_scale = float(cfg.env.reward_scale)
        self.reward_clip_min = float(cfg.env.reward_clip_min)
        self.reward_clip_max = float(cfg.env.reward_clip_max)
        self.exp_risk_factor = float(cfg.env.exp_risk_factor)
        self.sharpe_window = int(cfg.env.sharpe_window)
        self.risk_free_rate = float(cfg.env.risk_free_rate)

        if self.lag <= 0:
            raise ValueError("lag must be a positive integer")
        if self.n_assets <= 0:
            raise ValueError("tickers must be non-empty")

        # --- feature columns (from config) ---
        self.columns = cfg.env.build_feature_columns(self.tickers, cfg.features)

        cleaned = df.copy()

        # Require price columns
        missing_prices = [t for t in self.tickers if t not in cleaned.columns]
        if missing_prices:
            raise ValueError(f"Missing price columns in df for tickers: {missing_prices}")

        # Require feature columns
        missing_features = [c for c in self.columns if c not in cleaned.columns]
        if missing_features:
            raise ValueError(
                "Missing feature columns in df. Missing: " + ", ".join(missing_features[:25])
                + (" ..." if len(missing_features) > 25 else "")
            )

        cleaned = cleaned.dropna(subset=self.tickers + self.columns).copy()

        # IMPORTANT safety: need at least `lag` rows to form first observation window
        if len(cleaned) < self.lag + 1:
            raise ValueError(
                f"Not enough rows after dropna to run env. "
                f"Need at least lag+1={self.lag + 1} rows, got {len(cleaned)}. "
                f"Consider lowering lag or improving feature NaN handling."
            )

        self.states = cleaned[self.columns].to_numpy(dtype=np.float32)
        self.prices = cleaned[self.tickers].to_numpy(dtype=np.float32)

        # --- internal state ---
        self.pos = self.lag - 1
        self.current_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0  # start in cash

        # monitoring stats
        self.last_turnover = 0.0
        self.last_turnover_total = 0.0
        self.last_tc_cost = 0.0
        self.last_gross_return = 0.0
        self.last_net_return = 0.0

        # For Sharpe reward: track recent returns
        self.recent_returns = []

    # ---------------------------
    # Public API
    # ---------------------------
    def reset(self) -> np.ndarray:
        self.pos = self.lag - 1
        self.current_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0

        self.last_turnover = 0.0
        self.last_turnover_total = 0.0
        self.last_tc_cost = 0.0
        self.last_gross_return = 0.0
        self.last_net_return = 0.0
        self.recent_returns = []
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        # Action includes cash => length = n_assets + 1
        if action.shape[0] != (self.n_assets + 1):
            raise ValueError(f"Action shape mismatch: expected {self.n_assets + 1}, got {action.shape[0]}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Invalid action: contains NaN/inf")

        # Project to simplex (long-only)
        w_tgt = self._project_to_simplex(action)

        ##########################################
        ### Rebalance every 'lag' days (eg. 5) ###
        ##########################################
        next_pos = self.pos + self.lag  
        
        if next_pos >= len(self.prices):
            # dataset end (truncation)
            return self._get_obs(), 0.0, True

        # t -> t+1 simple returns (cash return = 0)
        denom = self.prices[self.pos] + 1e-12
        asset_returns = (self.prices[next_pos] - self.prices[self.pos]) / denom
        asset_returns = np.concatenate([asset_returns, np.array([0.0], dtype=np.float32)])

        # Turnover + transaction cost
        turnover_oneway, turnover_total = self._compute_turnover(self.current_weights, w_tgt)

        if turnover_total < self.turnover_threshold:
            # no-trade region
            turnover_oneway = 0.0
            turnover_total = 0.0
            tc_cost = 0.0
            w_post_trade = self.current_weights
        else:
            tc_cost = self.tc_rate * turnover_oneway
            if self.tc_fixed > 0.0:
                tc_cost += self.tc_fixed
            w_post_trade = w_tgt

        gross_return = float(np.dot(w_post_trade, asset_returns))
        net_return = gross_return - float(tc_cost)

        # Track returns for Sharpe calculation
        self.recent_returns.append(net_return)
        if len(self.recent_returns) > self.sharpe_window:
            self.recent_returns.pop(0)

        # Reward calculation based on reward_type
        reward = self._compute_reward(net_return)

        # Drift weights to t+1 after returns
        w_drift = w_post_trade * (1.0 + asset_returns)
        w_sum = float(w_drift.sum())
        if w_sum <= 0.0 or not np.isfinite(w_sum):
            w_drift = np.zeros_like(w_drift, dtype=np.float32)
            w_drift[-1] = 1.0
        else:
            w_drift = (w_drift / w_sum).astype(np.float32)

        # advance
        self.pos = next_pos
        self.current_weights = w_drift

        # stats
        self.last_turnover = float(turnover_oneway)
        self.last_turnover_total = float(turnover_total)
        self.last_tc_cost = float(tc_cost)
        self.last_gross_return = float(gross_return)
        self.last_net_return = float(net_return)

        done = (self.pos + 1) >= len(self.states)
        return self._get_obs(), reward, bool(done)

    # ---------------------------
    # Dimensions & monitoring
    # ---------------------------
    def get_state_dim(self) -> int:
        base_dim = int(self.states.shape[1] * self.lag)
        return base_dim + (self.n_assets + 1 if self.include_position_in_state else 0)

    def get_action_dim(self) -> int:
        # MUST include cash
        return self.n_assets + 1

    def get_turnover_stats(self) -> Dict[str, object]:
        return {
            "position": int(self.pos),
            "current_weights": self.current_weights.copy(),
            "turnover_oneway": float(self.last_turnover),
            "turnover_total": float(self.last_turnover_total),
            "tc_cost": float(self.last_tc_cost),
            "gross_return": float(self.last_gross_return),
            "net_return": float(self.last_net_return),
        }

    # ---------------------------
    # Internals
    # ---------------------------
    def _get_obs(self) -> np.ndarray:
        start = self.pos - self.lag + 1
        end = self.pos + 1
        window = self.states[start:end].reshape(-1).astype(np.float32)

        if not self.include_position_in_state:
            return window

        return np.concatenate([window, self.current_weights.astype(np.float32)], axis=0)

    @staticmethod
    def _project_to_simplex(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        w = np.clip(w, eps, np.inf)
        s = float(w.sum())
        if s <= 0.0 or not np.isfinite(s):
            out = np.zeros_like(w, dtype=np.float32)
            out[-1] = 1.0
            return out
        return (w / s).astype(np.float32)

    def _compute_turnover(self, w_prev: np.ndarray, w_tgt: np.ndarray) -> Tuple[float, float]:
        if not self.turnover_include_cash:
            w_prev_use = w_prev[:-1]
            w_tgt_use = w_tgt[:-1]
        else:
            w_prev_use = w_prev
            w_tgt_use = w_tgt

        delta = np.abs(w_tgt_use - w_prev_use)
        total = float(delta.sum())
        oneway = 0.5 * total if self.turnover_use_half_factor else total
        return float(oneway), float(total)

    def _compute_reward(self, net_return: float) -> float:
        """Compute reward based on reward_type configuration.

        Args:
            net_return: Net return after transaction costs

        Returns:
            Scaled reward value
        """
        net_return_clipped = float(np.clip(net_return, self.reward_clip_min, self.reward_clip_max))

        if self.reward_type == "log":
            # Risk-averse (concave utility)
            # Penalizes losses more than it rewards gains
            raw_reward = np.log1p(net_return_clipped)

        elif self.reward_type == "linear":
            # Risk-neutral
            # Treats gains and losses symmetrically
            raw_reward = net_return_clipped

        elif self.reward_type == "exp":
            # Risk-seeking (convex utility)
            # Rewards large gains more than it penalizes losses
            # Higher exp_risk_factor = more aggressive
            raw_reward = (np.exp(net_return_clipped * self.exp_risk_factor) - 1.0) / self.exp_risk_factor

        elif self.reward_type == "sharpe":
            # Volatility-adjusted return (Sharpe-like)
            # Rewards high returns but penalizes when in high-volatility regime
            # Uses current return (not mean) for immediate RL feedback
            if len(self.recent_returns) < 2:
                # Not enough data for std calculation, use simple return
                raw_reward = net_return_clipped
            else:
                std_return = np.std(self.recent_returns)

                # Avoid division by zero
                if std_return < 1e-8:
                    raw_reward = net_return_clipped
                else:
                    # Volatility-adjusted reward: (return - risk_free) / rolling_volatility
                    raw_reward = (net_return_clipped - self.risk_free_rate / 252.0) / (std_return + 1e-8)
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        return float(self.reward_scale * raw_reward)
