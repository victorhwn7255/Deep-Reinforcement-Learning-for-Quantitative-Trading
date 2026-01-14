from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from config import get_default_config, Config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent


def equity_curve(net_returns: np.ndarray) -> np.ndarray:
    r = np.asarray(net_returns, dtype=np.float64)
    if r.size == 0:
        return np.array([1.0], dtype=np.float64)
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def max_drawdown(eq: np.ndarray) -> float:
    eq = np.asarray(eq, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + 1e-12) - 1.0
    return float(dd.min())


def sharpe(net_returns: np.ndarray, ann: int = 252, step_size: int = 1) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        net_returns: Array of returns per step
        ann: Trading days per year (default 252)
        step_size: Days per step (default 1 for daily, 5 for weekly)
    """
    r = np.asarray(net_returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd < 1e-12:
        return 0.0
    periods_per_year = ann / step_size
    return (mu / sd) * float(np.sqrt(periods_per_year))


def ann_vol(net_returns: np.ndarray, ann: int = 252, step_size: int = 1) -> float:
    """Calculate annualized volatility.

    Args:
        net_returns: Array of returns per step
        ann: Trading days per year (default 252)
        step_size: Days per step (default 1 for daily, 5 for weekly)
    """
    r = np.asarray(net_returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    periods_per_year = ann / step_size
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def cagr(eq: np.ndarray, ann: int = 252, step_size: int = 1) -> float:
    """Calculate CAGR from equity curve.

    Args:
        eq: Equity curve array
        ann: Trading days per year (default 252)
        step_size: Days per step (default 1 for daily, 5 for weekly)
    """
    eq = np.asarray(eq, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    years = (eq.size - 1) * step_size / float(ann)
    if years <= 0:
        return 0.0
    return float(eq[-1] ** (1.0 / years) - 1.0)


def run_backtest(env: Env, agent: Agent, deterministic: bool) -> Dict[str, np.ndarray]:
    obs = env.reset()
    done = False

    net_rets: List[float] = []
    gross_rets: List[float] = []
    tc_costs: List[float] = []
    turnover_one: List[float] = []
    turnover_tot: List[float] = []
    weights: List[np.ndarray] = []

    while not done:
        action = agent.select_action(obs, evaluate=deterministic)
        obs, _reward, done = env.step(action)

        net_rets.append(float(getattr(env, "last_net_return", 0.0)))
        gross_rets.append(float(getattr(env, "last_gross_return", 0.0)))
        tc_costs.append(float(getattr(env, "last_tc_cost", 0.0)))
        turnover_one.append(float(getattr(env, "last_turnover", 0.0)))
        turnover_tot.append(float(getattr(env, "last_turnover_total", 0.0)))

        w = getattr(env, "current_weights", None)
        if w is None:
            w = action
        weights.append(np.asarray(w, dtype=np.float32).reshape(-1).copy())

    net = np.asarray(net_rets, dtype=np.float64)
    eq = equity_curve(net)

    return {
        "net_returns": net,
        "gross_returns": np.asarray(gross_rets, dtype=np.float64),
        "tc_costs": np.asarray(tc_costs, dtype=np.float64),
        "turnover_oneway": np.asarray(turnover_one, dtype=np.float64),
        "turnover_total": np.asarray(turnover_tot, dtype=np.float64),
        "equity": eq,
        "weights": np.vstack(weights) if weights else np.zeros((0, env.get_action_dim()), dtype=np.float32),
    }


def plot_results(cfg: Config, res: Dict[str, np.ndarray], tickers: List[str], out_dir: str) -> None:
    # Headless-safe backend selection
    import matplotlib
    headless = (os.environ.get("DISPLAY", "") == "")
    if headless or (not cfg.evaluation.render_plots):
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    equity = res["equity"]
    net = res["net_returns"]
    turnover_total = res["turnover_total"]
    turnover_one = res["turnover_oneway"]
    weights = res["weights"]

    plt.figure()
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.xlabel("Step")
    plt.ylabel("Equity ($1 start)")
    if cfg.evaluation.save_plots:
        plt.savefig(os.path.join(out_dir, "equity_curve.png"), dpi=150, bbox_inches="tight")
    if cfg.evaluation.render_plots:
        plt.show()
    else:
        plt.close()

    plt.figure()
    plt.plot(net)
    plt.title("Daily Net Returns")
    plt.xlabel("Step")
    plt.ylabel("Net Return")
    if cfg.evaluation.save_plots:
        plt.savefig(os.path.join(out_dir, "daily_net_returns.png"), dpi=150, bbox_inches="tight")
    if cfg.evaluation.render_plots:
        plt.show()
    else:
        plt.close()

    plt.figure()
    plt.plot(turnover_total, label="turnover_total")
    plt.plot(turnover_one, label="turnover_oneway")
    plt.title("Turnover")
    plt.xlabel("Step")
    plt.ylabel("Turnover")
    plt.legend()
    if cfg.evaluation.save_plots:
        plt.savefig(os.path.join(out_dir, "turnover.png"), dpi=150, bbox_inches="tight")
    if cfg.evaluation.render_plots:
        plt.show()
    else:
        plt.close()

    if weights.shape[0] > 0:
        labels = tickers + ["CASH"]
        avg_w = weights.mean(axis=0)
        plt.figure()
        plt.bar(np.arange(len(avg_w)), avg_w)
        plt.xticks(np.arange(len(avg_w)), labels, rotation=45, ha="right")
        plt.title("Average Weights")
        plt.ylabel("Weight")
        if cfg.evaluation.save_plots:
            plt.savefig(os.path.join(out_dir, "avg_weights.png"), dpi=150, bbox_inches="tight")
        if cfg.evaluation.render_plots:
            plt.show()
        else:
            plt.close()


def main() -> None:
    cfg = get_default_config()
    cfg.ensure_dirs()
    cfg.set_global_seeds()

    device = cfg.auto_detect_device()
    print(f"✓ Device: {device}")

    # Data
    _df_train, df_test, _feature_cols = load_and_prepare_data(cfg)
    print(f"✓ Test rows: {len(df_test)}")

    # Env + Agent
    env = Env(df_test, cfg.data.tickers, cfg)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    print(f"✓ Env dims: state_dim={state_dim}, action_dim={action_dim}")

    agent = Agent(state_dim, action_dim, cfg, device=device)

    if not os.path.exists(cfg.evaluation.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {cfg.evaluation.model_path}")

    agent.load_model(cfg.evaluation.model_path)
    print(f"✓ Loaded model: {cfg.evaluation.model_path}")

    # Run
    res = run_backtest(env, agent, deterministic=bool(cfg.evaluation.deterministic))

    eq = res["equity"]
    net = res["net_returns"]

    stats = {
        "CAGR": cagr(eq, step_size=cfg.env.lag),
        "Sharpe": sharpe(net, step_size=cfg.env.lag),
        "AnnVol": ann_vol(net, step_size=cfg.env.lag),
        "MaxDD": max_drawdown(eq),
        "FinalEquity": float(eq[-1]) if eq.size else 1.0,
        "AvgTurnoverOneWay": float(res["turnover_oneway"].mean()) if res["turnover_oneway"].size else 0.0,
        "AvgTurnoverTotal": float(res["turnover_total"].mean()) if res["turnover_total"].size else 0.0,
        "AvgTCCost": float(res["tc_costs"].mean()) if res["tc_costs"].size else 0.0,
    }

    print("\n" + "-" * 60)
    print("EVALUATION SUMMARY (SAC v2)")
    print("-" * 60)
    for k, v in stats.items():
        print(f"{k:>18}: {v:.6f}" if isinstance(v, float) else f"{k:>18}: {v}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.evaluation.output_dir, f"eval_{ts}")
    print(f"\n✓ Output dir: {out_dir}")

    if cfg.evaluation.save_plots or cfg.evaluation.render_plots:
        plot_results(cfg, res, cfg.data.tickers, out_dir)


if __name__ == "__main__":
    main()


