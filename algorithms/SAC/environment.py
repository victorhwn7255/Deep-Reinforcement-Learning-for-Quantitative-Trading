import numpy as np
import pandas as pd


class Env:
    """Portfolio management environment with tradable close-to-close mechanics.

    Key design choices:
    - Daily timestep (t -> t+1)
    - Observation at time t uses features up to t (with a `lag` lookback window)
    - Action = TARGET portfolio weights to hold over (t -> t+1), over (n_assets + 1) dims including cash
    - Long-only simplex enforcement (non-negative, sums to 1)
    - Transaction cost proportional to turnover, optionally with:
        - no-trade threshold (ignore tiny turnover)
        - fixed cost per rebalance event (adds a 'no-trade region')
    - Optional: include current holdings in the state (recommended)
    - Holdings are stored as DRIFTED weights after returns (so turnover is realistic)

    Timeline per step (close-to-close):
      1) At time t, you observe state_t and currently-held weights w_pre (already drifted)
      2) You choose action a = w_tgt (target weights for the NEXT holding period t->t+1)
      3) You pay transaction costs for rebalancing w_pre -> w_tgt
      4) You earn returns over (t->t+1) on w_tgt
      5) You compute drifted weights w_drift after returns and store them as holdings for next step

    Reward:
      - Uses reward_scale * log1p(net_return), where net_return = gross_return - tc_cost
      - reward_scale is a stability knob (typical 50–1000). It rescales the learning signal without changing the optimal policy in theory.
      - This treats tc_cost as a return drag (fraction of portfolio value).
        If you want dollar-based fixed fees, add an explicit portfolio value state.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: list[str],
        lag: int = 5,
        tc_rate: float = 0.0,
        include_position_in_state: bool = True,
        turnover_include_cash: bool = False,
        turnover_use_half_factor: bool = True,
        turnover_threshold: float = 0.0,
        tc_fixed: float = 0.0,
        reward_scale: float = 100.0,
    ):
        """Initialize environment.

        Args:
            df: DataFrame containing price columns for tickers + engineered feature columns.
            tickers: list of asset tickers (must exist as price columns in df).
            lag: lookback window length (days) for state.
            tc_rate: proportional transaction cost per unit **one-way turnover**
                     (e.g., 0.001 = 10 bps per 1.0 one-way turnover).
            include_position_in_state: if True, append current portfolio weights to state.
            turnover_include_cash: if True, include cash weight in turnover calc.
            turnover_use_half_factor: if True, turnover is computed as **one-way turnover** = 0.5 * sum(|Δw|).
                                     With this enabled, `tc_rate` is interpreted as bps per one-way turnover.
            turnover_threshold: if turnover < threshold, treat it as 0 (no-trade threshold).
            tc_fixed: fixed cost charged whenever turnover > threshold (per rebalance event), expressed
                      as a fraction of portfolio value (return drag).
            reward_scale: scales the per-step reward (log return). This improves SAC stability when
                          daily log returns are very small relative to entropy/log-prob terms.
                          Typical values: 50–1000. Start with 100.
        """
        self.tickers = list(tickers)
        self.n_assets = len(self.tickers)
        self.lag = int(lag)

        if self.n_assets <= 0:
            raise ValueError("tickers must be a non-empty list")
        if self.lag <= 0:
            raise ValueError("lag must be a positive integer")

        # --- Feature columns (same set as your current environment) ---
        self.columns: list[str] = []
        for ticker in self.tickers:
            self.columns += [
                f"{ticker}_RSI",
                f"{ticker}_volatility",
            ]

        self.columns += [
            "VIX_normalized",
            "VIX_regime",
            "VIX_term_structure",
        ]

        self.columns += [
            "Credit_Spread_normalized",
            "Credit_Spread_regime",
            "Credit_Spread_momentum",
            "Credit_Spread_zscore",
            "Credit_Spread_velocity",
            "Credit_VIX_divergence",
        ]

        cleaned = df.dropna().copy()

        missing_prices = [t for t in self.tickers if t not in cleaned.columns]
        if missing_prices:
            raise ValueError(f"Missing price columns in df for tickers: {missing_prices}")

        missing_features = [c for c in self.columns if c not in cleaned.columns]
        if missing_features:
            raise ValueError(
                "Missing feature columns in df. Missing: " + ", ".join(missing_features[:25])
                + (" ..." if len(missing_features) > 25 else "")
            )

        self.states = cleaned[self.columns].to_numpy(dtype=np.float32)
        self.prices = cleaned[self.tickers].to_numpy(dtype=np.float32)

        # --- Trading / cost params ---
        self.tc_rate = float(tc_rate)
        self.tc_fixed = float(tc_fixed)
        self.turnover_threshold = float(turnover_threshold)

        self.include_position_in_state = bool(include_position_in_state)
        self.turnover_include_cash = bool(turnover_include_cash)
        self.turnover_use_half_factor = bool(turnover_use_half_factor)

        # Reward scaling (stability for daily returns)
        self.reward_scale = float(reward_scale)
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be > 0")

        # Basic sanity checks
        if self.states.shape[0] != self.prices.shape[0]:
            raise ValueError("states and prices must have the same number of rows")
        if self.states.shape[0] <= self.lag:
            raise ValueError(f"Not enough data after dropna(): need > lag ({self.lag}) rows")

        # Runtime trackers (optional but helpful)
        self.pos: int = 0
        self.current_weights: np.ndarray = np.zeros(self.n_assets + 1, dtype=np.float32)

        # Turnover tracking:
        # - last_turnover: the value used for costing (one-way turnover when half-factor is enabled)
        # - last_turnover_total: sum(|Δw|) for reference/debug (buy+sell notional in weight space)
        self.last_turnover: float = 0.0
        self.last_turnover_total: float = 0.0

        self.last_tc_cost: float = 0.0
        self.last_gross_return: float = 0.0
        self.last_net_return: float = 0.0

        # Optional console info
        base_dim = self.states.shape[1] * self.lag
        aug_dim = base_dim + (self.n_assets + 1 if self.include_position_in_state else 0)
        print("Environment initialized:")
        print(f"  - Assets: {self.n_assets}")
        print(f"  - Features per timestep: {self.states.shape[1]}")
        print(f"  - Lag (lookback): {self.lag}")
        print(
            f"  - State dim: {aug_dim} (base {base_dim}"
            f"{' + position' if self.include_position_in_state else ''})"
        )
        turnover_type = "one-way" if self.turnover_use_half_factor else "two-way (total)"
        print(
            f"  - tc_rate: {self.tc_rate:.6f} ({self.tc_rate * 10000:.1f} bps per 1.0 {turnover_type} turnover)"
        )
        print(f"  - tc_fixed: {self.tc_fixed:.6f} (charged when turnover > threshold)")
        print(f"  - turnover_threshold: {self.turnover_threshold:.6f}")
        print(f"  - turnover_include_cash: {self.turnover_include_cash}")
        print(f"  - turnover_use_half_factor: {self.turnover_use_half_factor}")
        print(f"  - reward_scale: {self.reward_scale:.2f}")
        print(f"  - Number of data points: {len(self.states)}")

    # ---------------------------
    # Public API
    # ---------------------------
    def reset(self):
        """Reset environment to initial state."""
        self.pos = self.lag - 1

        # Start fully in cash
        self.current_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0

        self.last_turnover = 0.0
        self.last_turnover_total = 0.0
        self.last_tc_cost = 0.0
        self.last_gross_return = 0.0
        self.last_net_return = 0.0

        return self._get_obs()

    def step(self, action):
        """Advance environment by one day using tradable mechanics.

        Action is interpreted as TARGET weights to hold over (t -> t+1).

        Returns:
            next_state (np.ndarray), reward (float), done (bool)
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != (self.n_assets + 1):
            raise ValueError(
                f"Action shape mismatch: expected {self.n_assets + 1}, got {action.shape[0]}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Invalid action: contains NaN/inf")

        # Project to long-only simplex
        w_tgt = self._project_to_simplex(action)

        # Advance time index
        next_pos = self.pos + 1
        if next_pos >= len(self.prices):
            return self._get_obs(), 0.0, True

        # Next-period simple returns (t -> t+1), cash return = 0
        denom = self.prices[self.pos] + 1e-12
        asset_returns = (self.prices[next_pos] - self.prices[self.pos]) / denom
        asset_returns = np.concatenate([asset_returns, np.array([0.0], dtype=np.float32)])

        # Turnover and transaction costs for rebalancing CURRENT (already drifted) -> TARGET
        turnover_oneway, turnover_total, tc_cost = self._compute_costs(w_tgt, self.current_weights)

        # Earn gross return over (t -> t+1) from TARGET portfolio
        gross_return = float(w_tgt @ asset_returns)

        # Net return (treating costs as return drag)
        net_return = gross_return - float(tc_cost)

        # Reward: stable scaled log return
        # Daily log returns are tiny (often 1e-4 to 1e-3). Scaling improves SAC stability so the
        # critic/policy gradients are not dominated by entropy/log-prob terms.
        net_return_clipped = float(np.clip(net_return, -0.95, 10.0))
        reward = float(self.reward_scale * np.log1p(net_return_clipped))

        # Drift holdings after returns: w_drift ∝ w_tgt * (1 + r)
        w_grown = w_tgt * (1.0 + asset_returns)
        total = float(w_grown.sum())
        if total <= 1e-12 or (not np.isfinite(total)):
            w_drift = w_tgt.copy()
        else:
            w_drift = (w_grown / total).astype(np.float32)

        # Commit updates
        self.pos = next_pos
        self.current_weights = w_drift

        # Track stats
        self.last_turnover = float(turnover_oneway)
        self.last_turnover_total = float(turnover_total)
        self.last_tc_cost = float(tc_cost)
        self.last_gross_return = float(gross_return)
        self.last_net_return = float(net_return)

        done = (self.pos + 1) >= len(self.states)
        next_state = self._get_obs()
        return next_state, reward, done

    def get_state_dim(self):
        base_dim = self.states.shape[1] * self.lag
        return base_dim + (self.n_assets + 1 if self.include_position_in_state else 0)

    def get_action_dim(self):
        return self.n_assets + 1

    def get_turnover_stats(self):
        """Get recent trading stats for monitoring."""
        return {
            "position": int(self.pos),
            "current_weights": self.current_weights.copy(),
            "last_turnover": float(self.last_turnover),
            "last_turnover_total": float(self.last_turnover_total),
            "last_tc_cost": float(self.last_tc_cost),
            "last_gross_return": float(self.last_gross_return),
            "last_net_return": float(self.last_net_return),
        }

    # ---------------------------
    # Internals
    # ---------------------------
    def _get_obs(self):
        """Build observation as a 1D vector."""
        base_window = self.states[self.pos - self.lag + 1 : self.pos + 1]  # (lag, feat_dim)
        base_flat = base_window.reshape(-1)  # ALWAYS 1D

        if self.include_position_in_state:
            return np.concatenate([base_flat, self.current_weights.astype(np.float32)])
        return base_flat

    def _project_to_simplex(self, w: np.ndarray) -> np.ndarray:
        """Project raw weights to a long-only simplex: w_i >= 0, sum(w)=1.

        Note: This is a simple clip-and-renormalize which is fine for long-only SAC.
        """
        w = np.asarray(w, dtype=np.float32).reshape(-1)
        w = np.clip(w, 1e-8, 1.0)
        s = float(w.sum())
        if (not np.isfinite(s)) or (s <= 0.0):
            # Fallback to all cash
            out = np.zeros(self.n_assets + 1, dtype=np.float32)
            out[-1] = 1.0
            return out
        return (w / s).astype(np.float32)

    def _compute_costs(self, w_tgt: np.ndarray, w_pre: np.ndarray) -> tuple[float, float, float]:
        """Compute turnover and transaction cost from current holdings to target.

        Args:
            w_tgt: target weights (N+1 incl cash)
            w_pre: current weights (N+1 incl cash), should already be drifted

        Returns:
            turnover_oneway (float), turnover_total (float), tc_cost (float)

        Notes:
            - tc_cost is a fraction of portfolio value (return drag).
            - If turnover_include_cash is False, cash is excluded from turnover.
            - If turnover_use_half_factor is True, turnover_oneway = 0.5 * sum(|Δw|) and `tc_rate` is interpreted as bps per one-way turnover.
            - turnover_total is always sum(|Δw|) (buy+sell notional in weight space) and is returned for monitoring/debug.
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
            # Below threshold: treat as no-trade for costing, but preserve turnover stats for monitoring.
            return turnover_oneway, turnover_total, 0.0

        # Costing convention: tc_rate is bps per one-way turnover.
        tc_cost = float(self.tc_rate * turnover_oneway)
        if turnover_oneway > 0.0:
            tc_cost += float(self.tc_fixed)

        return turnover_oneway, turnover_total, tc_cost