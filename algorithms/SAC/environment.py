"""
Portfolio management environment (config-friendly, feature-flexible).

Long-only, simplex portfolio environment:
- Action: target weights over (assets + cash), w >= 0, sum(w)=1
- Step:
    1) Project action to simplex
    2) Compute turnover from current weights -> target weights
    3) Apply transaction costs
    4) Apply next-step price returns to obtain drifted weights
    5) Reward = reward_scale * log1p(net_return)  (net_return includes transaction costs)

`done` is treated as a time-limit truncation (end of dataset), not a terminal failure.
This matters for value bootstrapping during training: you typically should NOT force V=0 at done.
"""

from __future__ import annotations

from typing import Optional, Sequence, List

import numpy as np
import pandas as pd

from config import Config

class Env:
    def __init__(
        self,
        df: pd.DataFrame,
        tickers: Sequence[str],
        # If cfg is provided, it will override most knobs below unless explicitly passed.
        cfg: Optional["Config"] = None,
        # Core mechanics
        lag: int = 5,
        include_position_in_state: bool = True,
        feature_columns: Optional[Sequence[str]] = None,
        # Transaction costs
        tc_rate: float = 0.0005,
        tc_fixed: float = 0.0,
        turnover_threshold: float = 0.0,
        turnover_include_cash: bool = False,
        turnover_use_half_factor: bool = True,
        # Reward definition
        reward_scale: float = 100.0,
        net_return_clip_min: float = -0.95,
        net_return_clip_max: float = 10.0,
        # Initialization
        start_in_cash: bool = True,
    ):
        """
        If you pass cfg:
          - lag, include_position_in_state, costs, reward_scale, clipping, start_in_cash come from cfg.env
          - feature_columns defaults to cfg.feature_columns()
        """
        if cfg is not None:
            # Pull from cfg.env
            lag = int(cfg.env.lag)
            include_position_in_state = bool(cfg.env.include_position_in_state)

            tc_rate = float(cfg.env.tc_rate)
            tc_fixed = float(cfg.env.tc_fixed)
            turnover_threshold = float(cfg.env.turnover_threshold)
            turnover_include_cash = bool(cfg.env.turnover_include_cash)
            turnover_use_half_factor = bool(cfg.env.turnover_use_half_factor)

            reward_scale = float(cfg.env.reward_scale)
            net_return_clip_min = float(cfg.env.net_return_clip_min)
            net_return_clip_max = float(cfg.env.net_return_clip_max)

            start_in_cash = bool(cfg.env.start_in_cash)

            if feature_columns is None:
                feature_columns = cfg.feature_columns()

        # Validate inputs
        self.df = df.copy()
        if not isinstance(self.df.index, pd.DatetimeIndex):
            # Not required, but good to keep sorting consistent
            self.df = self.df.sort_index()
        else:
            self.df = self.df.sort_index()

        self.tickers: List[str] = list(tickers)
        if len(self.tickers) == 0:
            raise ValueError("tickers must be a non-empty list")

        self.lag = int(lag)
        if self.lag < 1:
            raise ValueError(f"lag must be >= 1 (got {self.lag})")

        self.include_position_in_state = bool(include_position_in_state)

        # Action dimension: assets + cash
        self.n_assets = len(self.tickers)
        self.action_dim = self.n_assets + 1

        # Costs / reward
        self.tc_rate = float(tc_rate)
        self.tc_fixed = float(tc_fixed)
        self.turnover_threshold = float(turnover_threshold)
        self.turnover_include_cash = bool(turnover_include_cash)
        self.turnover_use_half_factor = bool(turnover_use_half_factor)

        self.reward_scale = float(reward_scale)
        self.net_return_clip_min = float(net_return_clip_min)
        self.net_return_clip_max = float(net_return_clip_max)

        self.start_in_cash = bool(start_in_cash)

        # Determine feature columns
        if feature_columns is None:
            # Auto-infer: everything that's not a ticker price column
            feature_columns = [c for c in self.df.columns if c not in self.tickers]
        self.feature_columns = list(feature_columns)

        # Validate columns exist
        missing_prices = [t for t in self.tickers if t not in self.df.columns]
        if missing_prices:
            raise ValueError(f"Missing price columns for tickers: {missing_prices}")

        missing_feats = [c for c in self.feature_columns if c not in self.df.columns]
        if missing_feats:
            raise ValueError(f"Missing feature columns in df: {missing_feats}")

        # Build matrices
        self.prices = self.df[self.tickers].values.astype(np.float32)
        self.feats = self.df[self.feature_columns].values.astype(np.float32)

        if len(self.prices) != len(self.feats):
            raise ValueError("Price and feature matrices must have same number of rows")

        if len(self.df) <= self.lag:
            raise ValueError(f"Not enough rows for lag={self.lag}: got {len(self.df)}")

        # Runtime state
        self.pos = 0
        self.current_weights = np.zeros(self.action_dim, dtype=np.float32)

        # Trackers for evaluation/diagnostics
        self.last_turnover = 0.0        # one-way turnover used for tc_rate multiplication (if half-factor enabled)
        self.last_turnover_total = 0.0  # total L1 change (assets-only unless include_cash)
        self.last_tc_cost = 0.0
        self.last_gross_return = 0.0
        self.last_net_return = 0.0

        self.reset()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset(self):
        """Reset episode to start of data (after lag window)."""
        self.pos = self.lag - 1

        if self.start_in_cash:
            w0 = np.zeros(self.action_dim, dtype=np.float32)
            w0[-1] = 1.0
        else:
            w0 = np.ones(self.action_dim, dtype=np.float32) / float(self.action_dim)

        self.current_weights = w0

        self.last_turnover = 0.0
        self.last_turnover_total = 0.0
        self.last_tc_cost = 0.0
        self.last_gross_return = 0.0
        self.last_net_return = 0.0

        return self._get_obs()

    def step(self, action):
        """
        Returns: next_state, reward, done
        - done is True when you hit the end of the dataset (time-limit truncation)
        """
        if self.pos + 1 >= len(self.prices):
            raise RuntimeError("step() called after episode is done; call reset()")

        # 1) Project to simplex
        w_tgt = self._project_to_simplex(action)
        w_pre = self.current_weights

        # 2) Costs (based on weight changes)
        turnover_oneway, turnover_total, tc_cost = self._compute_costs(w_tgt, w_pre)

        # 3) Realized asset returns between t and t+1
        cur_prices = self.prices[self.pos]
        next_prices = self.prices[self.pos + 1]
        asset_returns = (next_prices / (cur_prices + 1e-12)) - 1.0

        # 4) Portfolio returns
        gross_return = float(np.dot(w_tgt[:-1], asset_returns))
        net_return = gross_return - float(tc_cost)

        # Safety clipping (avoid log1p blow-ups)
        net_return = float(np.clip(net_return, self.net_return_clip_min, self.net_return_clip_max))

        reward = float(self.reward_scale * np.log1p(net_return))

        # 5) Weight drift after market move (cash return = 0)
        growth = np.concatenate([1.0 + asset_returns, np.array([1.0], dtype=np.float32)])
        w_val = w_tgt * growth
        w_drift = w_val / (w_val.sum() + 1e-12)

        # Advance time
        self.pos += 1
        self.current_weights = w_drift.astype(np.float32)

        # Trackers
        self.last_turnover = float(turnover_oneway)
        self.last_turnover_total = float(turnover_total)
        self.last_tc_cost = float(tc_cost)
        self.last_gross_return = float(gross_return)
        self.last_net_return = float(net_return)

        done = self.pos >= (len(self.prices) - 1)
        next_state = self._get_obs()
        return next_state, reward, done

    def get_state_dim(self) -> int:
        base_dim = int(self.feats.shape[1] * self.lag)
        if self.include_position_in_state:
            base_dim += int(self.action_dim)
        return base_dim

    def get_action_dim(self) -> int:
        return int(self.action_dim)

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _get_obs(self):
        # lag window ending at current pos inclusive
        window = self.feats[self.pos - self.lag + 1 : self.pos + 1]
        base_flat = window.reshape(-1)

        if self.include_position_in_state:
            return np.concatenate([base_flat, self.current_weights.astype(np.float32)])
        return base_flat

    def _project_to_simplex(self, w) -> np.ndarray:
        """Project arbitrary vector to simplex with positivity and sum=1."""
        w = np.asarray(w, dtype=np.float32).reshape(-1)

        if w.size != self.action_dim:
            raise ValueError(f"Action dim mismatch: expected {self.action_dim}, got {w.size}")

        # Handle NaNs/Infs
        if not np.all(np.isfinite(w)):
            out = np.zeros(self.action_dim, dtype=np.float32)
            out[-1] = 1.0
            return out

        # Clamp and normalize
        w = np.clip(w, 1e-8, 1.0)
        s = float(w.sum())
        if (not np.isfinite(s)) or s <= 0.0:
            out = np.zeros(self.action_dim, dtype=np.float32)
            out[-1] = 1.0
            return out

        return (w / s).astype(np.float32)

    def _compute_costs(self, w_tgt: np.ndarray, w_pre: np.ndarray):
        """
        turnover_total: L1 change (assets-only unless turnover_include_cash=True)
        turnover_oneway: if half-factor enabled, = 0.5 * turnover_total
        tc_cost: tc_rate * turnover_oneway + tc_fixed (if above threshold)
        """
        if (self.tc_rate <= 0.0) and (self.tc_fixed <= 0.0):
            return 0.0, 0.0, 0.0

        if self.turnover_include_cash:
            diff = np.abs(w_tgt - w_pre)
        else:
            diff = np.abs(w_tgt[:-1] - w_pre[:-1])

        turnover_total = float(diff.sum())
        turnover_oneway = 0.5 * turnover_total if self.turnover_use_half_factor else turnover_total

        if turnover_oneway < self.turnover_threshold:
            return turnover_oneway, turnover_total, 0.0

        tc_cost = float(self.tc_rate * turnover_oneway)
        if turnover_oneway > 0.0:
            tc_cost += float(self.tc_fixed)

        return turnover_oneway, turnover_total, tc_cost