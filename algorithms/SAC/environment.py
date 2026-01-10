import numpy as np
import pandas as pd

class Env:
    """
    Portfolio management environment with adaptive rebalancing.

    Key design choices:
    - Daily timestep (pos -> pos+1)
    - Action = target portfolio weights over (n_assets + 1) including cash
    - Long-only simplex enforcement (non-negative, sums to 1)
    - Transaction cost proportional to turnover, optionally with:
        - no-trade threshold (ignore tiny turnover)
        - fixed cost per rebalance event (adds a 'no-trade region')
    - Optional: include current holdings in the state (recommended)
    - State is ALWAYS returned as a 1D vector (flattened window + optional weights)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: list[str],
        lag: int = 5,
        tc_rate: float = 0.0,
        include_position_in_state: bool = True,
        turnover_include_cash: bool = False,
        turnover_use_half_factor: bool = False,
        turnover_threshold: float = 0.0,
        tc_fixed: float = 0.0,
    ):
        """
            df: DataFrame containing price columns for tickers + engineered feature columns
            tickers: list of asset tickers (must exist as price columns in df)
            lag: lookback window length (days) for state
            tc_rate: proportional transaction cost per unit turnover (e.g., 0.001 = 10 bps)
            include_position_in_state: if True, append current portfolio weights to state
            turnover_include_cash: if True, include cash weight in turnover calc
            turnover_use_half_factor: if True, turnover = 0.5 * sum(|Δw|). Often used if including cash.
            turnover_threshold: if turnover < threshold, treat it as 0 (no-trade threshold)
            tc_fixed: fixed cost charged whenever turnover > threshold (per rebalance event)
        """
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.lag = int(lag)

        # --- Feature columns (same set as your current environment) ---
        self.columns = []
        for ticker in tickers:
            self.columns += [
                ticker + "_RSI",
                ticker + "_volatility",
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
        self.states = cleaned[self.columns].to_numpy(dtype=np.float32)
        self.prices = cleaned[tickers].to_numpy(dtype=np.float32)

        # --- Trading / cost params ---
        self.tc_rate = float(tc_rate)
        self.tc_fixed = float(tc_fixed)
        self.turnover_threshold = float(turnover_threshold)

        self.include_position_in_state = bool(include_position_in_state)
        self.turnover_include_cash = bool(turnover_include_cash)
        self.turnover_use_half_factor = bool(turnover_use_half_factor)

        # Basic sanity checks
        if self.states.shape[0] != self.prices.shape[0]:
            raise ValueError("states and prices must have the same number of rows")
        if self.states.shape[0] <= self.lag:
            raise ValueError(f"Not enough data after dropna(): need > lag ({self.lag}) rows")

        # Optional console info
        base_dim = self.states.shape[1] * self.lag
        aug_dim = base_dim + (self.n_assets + 1 if self.include_position_in_state else 0)
        print("Environment initialized:")
        print(f"  - Assets: {self.n_assets}")
        print(f"  - Features per timestep: {self.states.shape[1]}")
        print(f"  - Lag (lookback): {self.lag}")
        print(f"  - State dim: {aug_dim} (base {base_dim}"
              f"{' + position' if self.include_position_in_state else ''})")
        print(f"  - tc_rate: {self.tc_rate:.6f} ({self.tc_rate * 10000:.1f} bps/unit turnover)")
        print(f"  - tc_fixed: {self.tc_fixed:.6f} (charged when turnover > threshold)")
        print(f"  - turnover_threshold: {self.turnover_threshold:.6f}")
        print(f"  - turnover_include_cash: {self.turnover_include_cash}")
        print(f"  - turnover_use_half_factor: {self.turnover_use_half_factor}")
        print(f"  - Number of data points: {len(self.states)}")

    def reset(self):
        """Reset environment to initial state."""
        self.pos = self.lag - 1

        # Current portfolio weights (start fully in cash)
        self.current_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0

        return self._get_obs()

    def step(self, action):
        """
        One daily step:
          - Apply (possibly updated) action weights
          - Compute 1-day portfolio return (gross)
          - Subtract transaction costs based on turnover from previous weights
          - Reward = log1p(net_return) (clipped for stability)
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != (self.n_assets + 1):
            raise ValueError(f"Action shape mismatch: expected {self.n_assets + 1}, got {action.shape[0]}")

        # Enforce long-only simplex
        if not np.all(np.isfinite(action)):
            raise ValueError("Invalid action: contains NaN/inf")

        action = np.clip(action, 1e-8, 1.0)
        s = float(action.sum())
        if (not np.isfinite(s)) or (s <= 0.0):
            raise ValueError(f"Invalid action: non-finite or non-positive sum (sum={s})")
        action = action / s

        # ----- Transaction cost based on turnover -----
        if self.tc_rate > 0.0 or self.tc_fixed > 0.0:
            if self.turnover_include_cash:
                diff = np.abs(action - self.current_weights)
                turnover = float(diff.sum())
                if self.turnover_use_half_factor:
                    turnover *= 0.5
            else:
                # Exclude cash by default (simpler; avoids double-counting)
                diff = np.abs(action[:-1] - self.current_weights[:-1])
                turnover = float(diff.sum())
                # If you want the half-turnover convention even excluding cash, set it here:
                if self.turnover_use_half_factor:
                    turnover *= 0.5

            # No-trade threshold
            if turnover < self.turnover_threshold:
                turnover = 0.0

            tc_cost = self.tc_rate * turnover
            if turnover > 0.0:
                tc_cost += self.tc_fixed  # fixed fee per rebalance event
        else:
            tc_cost = 0.0

        # ----- Advance time -----
        next_pos = self.pos + 1
        if next_pos >= len(self.prices):
            # Terminal: keep last obs, zero reward
            return self._get_obs(), 0.0, True

        # ----- Daily asset returns -----
        denom = self.prices[self.pos] + 1e-12
        asset_returns = (self.prices[next_pos] - self.prices[self.pos]) / denom
        asset_returns = np.concatenate([asset_returns, np.array([0.0], dtype=np.float32)])  # cash

        gross_return = float(action @ asset_returns)
        net_return = gross_return - float(tc_cost)

        # Reward as log return (stable & compounding-friendly)
        net_return = float(np.clip(net_return, -0.95, 10.0))
        reward = float(np.log1p(net_return))

        # ----- Update environment state -----
        self.pos = next_pos
        self.current_weights = action.copy()

        done = (self.pos + 1) >= len(self.states)
        next_state = self._get_obs()
        
        return next_state, reward, done

    def _get_obs(self):
        """Build observation as a 1D vector."""
        base_window = self.states[self.pos - self.lag + 1 : self.pos + 1]  # (lag, feat_dim)
        base_flat = base_window.reshape(-1)  # ALWAYS 1D

        if self.include_position_in_state:
            return np.concatenate([base_flat, self.current_weights.astype(np.float32)])
        return base_flat

    def get_state_dim(self):
        base_dim = self.states.shape[1] * self.lag
        return base_dim + (self.n_assets + 1 if self.include_position_in_state else 0)

    def get_action_dim(self):
        return self.n_assets + 1
      
    def get_turnover_stats(self):
        """Get statistics about recent trading behavior (for monitoring)"""
        return {
            'current_weights': self.current_weights.copy(),
            'position': self.pos,
        }