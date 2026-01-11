"""
SAC portfolio evaluation/backtest (Config-driven).

Goals:
- No train/eval feature drift: feature engineering is centralized in data_utils.py.
- No hard-coded hyperparams: uses Config (config.py) everywhere.
- Same env mechanics as training: uses cfg.env.
- Loads best/final model based on cfg.eval.model_preference.

Run:
  python evaluate.py
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config, get_default_config
from data_utils import build_feature_dataframe, split_train_test
from environment import Env
from agent import Agent


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_makedirs(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def compute_performance_metrics(
    daily_net_returns: np.ndarray,
    daily_turnover: np.ndarray,
    daily_tc_cost: np.ndarray,
) -> Dict[str, float]:
    """
    daily_net_returns: arithmetic net returns (e.g. 0.001 for +0.1%)
    daily_turnover: env.last_turnover (one-way if env uses half-factor)
    daily_tc_cost: env.last_tc_cost (fraction drag applied that day)
    """
    eps = 1e-12
    n = len(daily_net_returns)
    if n == 0:
        return {}

    equity = np.cumprod(1.0 + daily_net_returns)
    total_return = float(equity[-1] - 1.0)

    ann_factor = 252.0
    years = n / ann_factor
    if years > 0:
        annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    else:
        annualized_return = float("nan")

    mean_daily = float(np.mean(daily_net_returns))
    std_daily = float(np.std(daily_net_returns) + eps)

    annualized_vol = float(std_daily * np.sqrt(ann_factor))
    sharpe = float((mean_daily / std_daily) * np.sqrt(ann_factor))

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / (running_max + eps)
    max_drawdown = float(np.min(drawdown))

    # Cost/turnover summaries
    avg_turnover = float(np.mean(daily_turnover)) if len(daily_turnover) else float("nan")
    total_turnover = float(np.sum(daily_turnover)) if len(daily_turnover) else float("nan")

    avg_tc = float(np.mean(daily_tc_cost)) if len(daily_tc_cost) else float("nan")
    total_tc = float(np.sum(daily_tc_cost)) if len(daily_tc_cost) else float("nan")

    return {
        "n_days": float(n),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
        "total_turnover": total_turnover,
        "avg_tc_cost": avg_tc,
        "total_tc_cost": total_tc,
    }


def run_backtest(
    env: Env,
    agent: Agent,
    deterministic_policy: bool,
) -> Dict[str, Any]:
    obs = env.reset()

    actions: List[np.ndarray] = []
    net_returns: List[float] = []
    gross_returns: List[float] = []
    tc_costs: List[float] = []
    turnovers: List[float] = []

    done = False
    while not done:
        a = agent.select_action(obs, evaluate=deterministic_policy)
        actions.append(a.copy())

        obs, reward, done = env.step(a)

        # Use environment trackers (more direct than inverting reward_scale * log1p)
        net_returns.append(float(env.last_net_return))
        gross_returns.append(float(env.last_gross_return))
        tc_costs.append(float(env.last_tc_cost))
        turnovers.append(float(env.last_turnover))

    actions_arr = np.asarray(actions, dtype=np.float32)
    net_arr = np.asarray(net_returns, dtype=np.float64)
    gross_arr = np.asarray(gross_returns, dtype=np.float64)
    tc_arr = np.asarray(tc_costs, dtype=np.float64)
    turn_arr = np.asarray(turnovers, dtype=np.float64)

    equity = np.cumprod(1.0 + net_arr)
    metrics = compute_performance_metrics(net_arr, turn_arr, tc_arr)

    return {
        "actions": actions_arr,
        "daily_net_returns": net_arr,
        "daily_gross_returns": gross_arr,
        "daily_tc_cost": tc_arr,
        "daily_turnover": turn_arr,
        "equity_curve": equity,
        "metrics": metrics,
    }


def plot_equity_and_drawdown(
    dates: np.ndarray,
    sac_equity: np.ndarray,
    ew_equity: Optional[np.ndarray],
    out_path: str,
    title: str,
) -> None:
    safe_makedirs(out_path)

    # Drawdown from equity
    eps = 1e-12
    sac_running_max = np.maximum.accumulate(sac_equity)
    sac_dd = (sac_equity - sac_running_max) / (sac_running_max + eps)

    if ew_equity is not None:
        ew_running_max = np.maximum.accumulate(ew_equity)
        ew_dd = (ew_equity - ew_running_max) / (ew_running_max + eps)
    else:
        ew_dd = None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Equity
    ax1.plot(dates, sac_equity, label="SAC")
    if ew_equity is not None:
        ax1.plot(dates, ew_equity, label="Equal-Weight (daily rebalance)", alpha=0.8)
    ax1.set_title(title)
    ax1.set_ylabel("Equity (start=1.0)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Drawdown
    ax2.plot(dates, sac_dd, label="SAC")
    if ew_dd is not None:
        ax2.plot(dates, ew_dd, label="Equal-Weight (daily rebalance)", alpha=0.8)
    ax2.axhline(0.0, linestyle="--", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"✓ Saved evaluation plot to: {out_path}")


def assert_env_feature_compat(cfg: Config) -> None:
    """
    Your Env currently expects RSI + volatility + VIX + credit features always.
    If you turn any family off in cfg.features, Env will raise missing columns.
    """
    req = {
        "include_rsi": cfg.features.include_rsi,
        "include_volatility": cfg.features.include_volatility,
        "include_vix": cfg.features.include_vix,
        "include_credit_spread": cfg.features.include_credit_spread,
    }
    missing = [k for k, v in req.items() if not v]
    if missing:
        raise ValueError(
            "Env requires these feature families to be enabled, but config has them disabled: "
            + ", ".join(missing)
            + "\nEither: (1) keep them enabled, or (2) refactor environment.py to be config-driven."
        )


def resolve_model_path(cfg: Config) -> str:
    prefer = (cfg.eval.model_preference or "best").lower().strip()
    best_path = cfg.training.model_path_best
    final_path = cfg.training.model_path_final

    if prefer == "best":
        if os.path.exists(best_path):
            return best_path
        if os.path.exists(final_path):
            print(f"⚠ Best model not found at {best_path}. Falling back to final model.")
            return final_path
        raise FileNotFoundError(f"No model found at {best_path} or {final_path}")

    if prefer == "final":
        if os.path.exists(final_path):
            return final_path
        if os.path.exists(best_path):
            print(f"⚠ Final model not found at {final_path}. Falling back to best model.")
            return best_path
        raise FileNotFoundError(f"No model found at {final_path} or {best_path}")

    raise ValueError(f"cfg.eval.model_preference must be 'best' or 'final'. Got: {cfg.eval.model_preference}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(cfg: Optional[Config] = None) -> None:
    cfg = cfg or get_default_config()

    print("=" * 80)
    print("SAC PORTFOLIO EVALUATION (CONFIG-DRIVEN)")
    print("=" * 80)

    cfg.ensure_output_dirs()
    assert_env_feature_compat(cfg)

    # Determinism for stochastic eval (if deterministic_policy=False, this still makes it repeatable)
    set_global_seeds(cfg.training.seed)

    # Device (same logic as training)
    device = cfg.auto_detect_device()
    print(f"✓ Device: {device}")

    # ------------------------------------------------------------------
    # Build dataset (same feature pipeline as training)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: BUILD FEATURE DATAFRAME")
    print("=" * 80)

    df = build_feature_dataframe(cfg)
    df_train, df_test = split_train_test(df, cfg)

    print(f"✓ Train rows: {len(df_train):,} | {df_train.index[0].date()} -> {df_train.index[-1].date()}")
    print(f"✓ Test rows:  {len(df_test):,}  | {df_test.index[0].date()} -> {df_test.index[-1].date()}")

    tickers = list(cfg.data.tickers or [])

    # ------------------------------------------------------------------
    # Create test environment with EXACT same mechanics as training
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: CREATE TEST ENVIRONMENT")
    print("=" * 80)

    env_test = Env(
        df=df_test,
        tickers=tickers,
        lag=cfg.env.lag,
        tc_rate=cfg.env.tc_rate,
        tc_fixed=cfg.env.tc_fixed,
        turnover_threshold=cfg.env.turnover_threshold,
        include_position_in_state=cfg.env.include_position_in_state,
        turnover_include_cash=cfg.env.turnover_include_cash,
        turnover_use_half_factor=cfg.env.turnover_use_half_factor,
        reward_scale=cfg.env.reward_scale,
    )

    state_dim = env_test.get_state_dim()
    action_dim = env_test.get_action_dim()
    print(f"✓ state_dim={state_dim} | action_dim={action_dim}")

    # ------------------------------------------------------------------
    # Init agent (hyperparams should match training)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: INIT AGENT + LOAD MODEL")
    print("=" * 80)

    agent = Agent(
        n_input=state_dim,
        n_action=action_dim,
        learning_rate=cfg.sac.actor_lr,         # Agent currently uses ONE shared LR
        gamma=cfg.sac.gamma,
        tau=cfg.sac.tau,
        alpha=cfg.sac.alpha_init,
        auto_entropy_tuning=cfg.sac.auto_entropy_tuning,
        target_entropy=cfg.sac.target_entropy,  # OK if None (Agent will set a default)
        buffer_size=cfg.sac.buffer_size,
        batch_size=cfg.sac.batch_size,
        learning_starts=cfg.sac.learning_starts,
        update_frequency=cfg.sac.update_frequency,
        n_hidden=cfg.network.n_hidden,
        device=device,
    )

    model_path = resolve_model_path(cfg)
    agent.load_model(model_path)

    # eval mode
    agent.policy.eval()
    agent.q1.eval()
    agent.q2.eval()
    agent.value.eval()

    mode = "DETERMINISTIC" if cfg.eval.deterministic_policy else "STOCHASTIC"
    print(f"✓ Loaded model: {model_path}")
    print(f"✓ Policy mode: {mode}")

    # ------------------------------------------------------------------
    # Run SAC backtest
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: RUN BACKTEST (SAC)")
    print("=" * 80)

    sac_out = run_backtest(env_test, agent, deterministic_policy=cfg.eval.deterministic_policy)
    sac_metrics = sac_out["metrics"]

    # ------------------------------------------------------------------
    # Run baseline: equal-weight daily rebalance (with same costs)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: RUN BASELINE (EQUAL-WEIGHT DAILY REBALANCE)")
    print("=" * 80)

    env_ew = Env(
        df=df_test,
        tickers=tickers,
        lag=cfg.env.lag,
        tc_rate=cfg.env.tc_rate,
        tc_fixed=cfg.env.tc_fixed,
        turnover_threshold=cfg.env.turnover_threshold,
        include_position_in_state=cfg.env.include_position_in_state,
        turnover_include_cash=cfg.env.turnover_include_cash,
        turnover_use_half_factor=cfg.env.turnover_use_half_factor,
        reward_scale=cfg.env.reward_scale,
    )

    # Equal weight across assets, zero cash
    n_assets = len(tickers)
    ew_action = np.zeros(n_assets + 1, dtype=np.float32)
    ew_action[:n_assets] = 1.0 / max(1, n_assets)
    ew_action[-1] = 0.0

    obs = env_ew.reset()
    ew_net: List[float] = []
    ew_turn: List[float] = []
    ew_tc: List[float] = []
    ew_done = False
    while not ew_done:
        obs, reward, ew_done = env_ew.step(ew_action)
        ew_net.append(float(env_ew.last_net_return))
        ew_turn.append(float(env_ew.last_turnover))
        ew_tc.append(float(env_ew.last_tc_cost))

    ew_net_arr = np.asarray(ew_net, dtype=np.float64)
    ew_equity = np.cumprod(1.0 + ew_net_arr)
    ew_metrics = compute_performance_metrics(
        ew_net_arr,
        np.asarray(ew_turn, dtype=np.float64),
        np.asarray(ew_tc, dtype=np.float64),
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    def _fmt(x: Any) -> str:
        if x is None:
            return "N/A"
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\nSAC:")
    for k in ["total_return", "annualized_return", "annualized_vol", "sharpe", "max_drawdown", "avg_turnover", "total_tc_cost"]:
        print(f"  {k:>16}: {_fmt(sac_metrics.get(k))}")

    print("\nEqual-Weight (daily rebalance):")
    for k in ["total_return", "annualized_return", "annualized_vol", "sharpe", "max_drawdown", "avg_turnover", "total_tc_cost"]:
        print(f"  {k:>16}: {_fmt(ew_metrics.get(k))}")

    # ------------------------------------------------------------------
    # Save artifacts (optional)
    # ------------------------------------------------------------------
    if cfg.eval.save_plots:
        print("\n" + "=" * 80)
        print("STEP 6: SAVE PLOT + JSON")
        print("=" * 80)

        dates = df_test.index.values[-len(sac_out["equity_curve"]):]  # align
        title = f"SAC Evaluation ({mode}) | Model: {os.path.basename(model_path)}"
        plot_equity_and_drawdown(
            dates=dates,
            sac_equity=sac_out["equity_curve"],
            ew_equity=ew_equity,
            out_path=cfg.eval.plots_path_eval,
            title=title,
        )

        # Save metrics JSON
        metrics_path = os.path.join(cfg.eval.plots_dir, "sac_evaluation_metrics.json")
        safe_makedirs(metrics_path)
        payload = {
            "model_path": model_path,
            "mode": mode,
            "tickers": tickers,
            "config": cfg.to_dict() if hasattr(cfg, "to_dict") else asdict(cfg),
            "sac_metrics": sac_metrics,
            "equal_weight_metrics": ew_metrics,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"✓ Saved evaluation metrics: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()