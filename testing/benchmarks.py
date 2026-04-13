
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Benchmark Agents
# ---------------------------------------------------------------------------

class FixedAgent:
    """Returns constant portfolio weights every step."""

    def __init__(self, weights: List[float]):
        self.weights = np.array(weights, dtype=np.float32)
        s = self.weights.sum()
        if s > 0:
            self.weights = self.weights / s

    def select_action(self, obs, evaluate: bool = True) -> np.ndarray:
        return self.weights.copy()


class BuyAndHoldAgent:
    """100% allocation to a single asset."""

    def __init__(self, asset_index: int, n_positions: int):
        self.weights = np.zeros(n_positions, dtype=np.float32)
        self.weights[asset_index] = 1.0

    def select_action(self, obs, evaluate: bool = True) -> np.ndarray:
        return self.weights.copy()


class MeanVarianceAgent:
    """Rolling Max-Sharpe portfolio with long-only constraints.

    Pre-computes the entire weight sequence from price data so that
    run_backtest() can call select_action() step by step.

    Args:
        prices: (T, N) array of asset prices (no cash column).
        lag: Rebalancing period (days between decisions).
        lookback: Rolling window for return estimation (default 252 = 1 year).
        n_positions: Total positions including cash (N + 1).
    """

    def __init__(
        self,
        prices: np.ndarray,
        lag: int = 5,
        lookback: int = 252,
        n_positions: int = 6,
    ):
        self.n_assets = prices.shape[1]
        self.n_positions = n_positions
        self.weight_sequence = self._compute_weights(prices, lag, lookback)
        self.step = 0

    def _compute_weights(
        self, prices: np.ndarray, lag: int, lookback: int,
    ) -> List[np.ndarray]:
        """Pre-compute weights at each rebalance point."""
        n_assets = prices.shape[1]
        T = prices.shape[0]
        weights = []

        # Daily log returns
        log_ret = np.diff(np.log(prices + 1e-12), axis=0)

        # Rebalance points (matching Env stepping: start at lag-1, step by lag)
        pos = lag - 1
        while pos + lag < T:
            end_idx = pos + 1  # use returns up to current position
            start_idx = max(0, end_idx - lookback)

            if end_idx - start_idx < 20:
                # Not enough data: equal weight
                w = np.ones(self.n_positions, dtype=np.float32)
                w[:n_assets] = 1.0 / n_assets
                w[n_assets:] = 0.0
                w = w / w.sum()
            else:
                window_ret = log_ret[start_idx:end_idx]
                w = self._max_sharpe(window_ret, n_assets)

            weights.append(w)
            pos += lag

        return weights

    def _max_sharpe(self, returns: np.ndarray, n_assets: int) -> np.ndarray:
        """Solve max-Sharpe with long-only constraint."""
        mu = returns.mean(axis=0)
        cov = np.cov(returns, rowvar=False)

        # Regularize covariance
        cov += np.eye(n_assets) * 1e-6

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w + 1e-12)
            return -port_ret / port_vol

        # Constraints: weights sum to <= 1 (rest goes to cash)
        constraints = [{"type": "ineq", "fun": lambda w: 1.0 - w.sum()}]
        bounds = [(0.0, 1.0)] * n_assets
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                neg_sharpe, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-10},
            )
            w_assets = np.clip(result.x, 0, 1).astype(np.float32)
        except Exception:
            w_assets = np.ones(n_assets, dtype=np.float32) / n_assets

        # Build full weight vector with cash
        w = np.zeros(self.n_positions, dtype=np.float32)
        w[:n_assets] = w_assets
        w[n_assets:] = max(0.0, 1.0 - w_assets.sum())
        w = w / (w.sum() + 1e-12)
        return w

    def select_action(self, obs, evaluate: bool = True) -> np.ndarray:
        idx = min(self.step, len(self.weight_sequence) - 1)
        w = self.weight_sequence[idx].copy()
        self.step += 1
        return w

    def reset(self):
        self.step = 0


# ---------------------------------------------------------------------------
# Convenience: create all benchmarks for a given dataset
# ---------------------------------------------------------------------------

# Default asset order: [VNQ, SPY, TLT, GLD, BTC-USD, CASH]
BENCHMARK_WEIGHTS = {
    "Equal-Weight": [0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
    "Fixed-Weight": [0.10, 0.50, 0.30, 0.05, 0.05, 0.0],
}


def create_benchmarks(
    prices_test: np.ndarray,
    prices_full: Optional[np.ndarray] = None,
    tickers: Optional[List[str]] = None,
    lag: int = 5,
    n_positions: int = 6,
) -> dict:
    """Create all benchmark agents.

    Args:
        prices_test: (T_test, N) test-period prices for Mean-Variance lookback.
        prices_full: (T_all, N) full prices (train+test) for Mean-Variance.
                     If None, uses prices_test only.
        tickers: Asset names (to find SPY index). Default assumes index 1.
        lag: Rebalancing period.
        n_positions: Total positions including cash.

    Returns:
        Dict[name -> agent]
    """
    spy_idx = 1  # default
    if tickers:
        try:
            spy_idx = list(tickers).index("SPY")
        except ValueError:
            spy_idx = 0

    benchmarks = {
        "SPY Buy-and-Hold": BuyAndHoldAgent(spy_idx, n_positions),
        "Equal-Weight": FixedAgent(BENCHMARK_WEIGHTS["Equal-Weight"]),
        "Fixed-Weight": FixedAgent(BENCHMARK_WEIGHTS["Fixed-Weight"]),
    }

    # Mean-Variance needs price history
    mv_prices = prices_full if prices_full is not None else prices_test
    benchmarks["Mean-Variance"] = MeanVarianceAgent(
        mv_prices, lag=lag, lookback=252, n_positions=n_positions,
    )

    return benchmarks
