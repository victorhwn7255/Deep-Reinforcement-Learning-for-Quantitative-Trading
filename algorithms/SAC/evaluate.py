from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import get_default_config, Config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent


def equity_curve(daily_net_returns: np.ndarray) -> np.ndarray:
    r = np.asarray(daily_net_returns, dtype=np.float64)
    if r.size == 0:
        return np.array([1.0], dtype=np.float64)
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + 1e-12) - 1.0
    return float(dd.min())


def annualized_return(equity: np.ndarray, ann: int = 252) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    total = float(eq[-1])
    years = (eq.size - 1) / float(ann)
    if years <= 0:
        return 0.0
    return total ** (1.0 / years) - 1.0


def annualized_vol(daily_net_returns: np.ndarray, ann: int = 252) -> float:
    r = np.asarray(daily_net_returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(ann))


def sharpe_ratio(daily_net_returns: np.ndarray, ann: int = 252, rf_daily: float = 0.0) -> float:
    r = np.asarray(daily_net_returns, dtype=np.float64) - float(rf_daily)
    if r.size < 2:
        return 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd < 1e-12:
        return 0.0
    return (mu / sd) * np.sqrt(ann)


@dataclass
class BacktestResults:
    daily_rewards: np.ndarray
    daily_net_returns: np.ndarray
    daily_gross_returns: np.ndarray
    daily_turnover_oneway: np.ndarray
    daily_turnover_total: np.ndarray
    daily_tc_cost: np.ndarray
    equity: np.ndarray
    weights: np.ndarray  # [T, action_dim]


def summarize_results(res: BacktestResults, ann: int = 252) -> Dict[str, float]:
    eq = res.equity
    return {
        "CAGR": annualized_return(eq, ann=ann),
        "Sharpe": sharpe_ratio(res.daily_net_returns, ann=ann),
        "AnnVol": annualized_vol(res.daily_net_returns, ann=ann),
        "MaxDD": max_drawdown(eq),
        "FinalEquity": float(eq[-1]) if eq.size else 1.0,
        "AvgTurnoverOneWay": float(np.mean(res.daily_turnover_oneway)) if res.daily_turnover_oneway.size else 0.0,
        "AvgTurnoverTotal": float(np.mean(res.daily_turnover_total)) if res.daily_turnover_total.size else 0.0,
        "AvgTCCost": float(np.mean(res.daily_tc_cost)) if res.daily_tc_cost.size else 0.0,
    }


def run_backtest(env: Env, agent: Agent, deterministic: bool = True) -> BacktestResults:
    obs = env.reset()

    rewards: List[float] = []
    net_rets: List[float] = []
    gross_rets: List[float] = []
    to_one: List[float] = []
    to_tot: List[float] = []
    tc_costs: List[float] = []
    weights: List[np.ndarray] = []

    done = False
    while not done:
        action = agent.select_action(obs, evaluate=deterministic)
        next_obs, reward, done = env.step(action)

        rewards.append(float(reward))
        net_rets.append(float(getattr(env, "last_net_return", 0.0)))
        gross_rets.append(float(getattr(env, "last_gross_return", 0.0)))
        to_one.append(float(getattr(env, "last_turnover", 0.0)))
        to_tot.append(float(getattr(env, "last_turnover_total", 0.0)))
        tc_costs.append(float(getattr(env, "last_tc_cost", 0.0)))

        w = getattr(env, "current_weights", None)
        if w is None:
            w = action
        weights.append(np.asarray(w, dtype=np.float32).reshape(-1).copy())

        obs = next_obs

    daily_rewards = np.asarray(rewards, dtype=np.float64)
    daily_net_returns = np.asarray(net_rets, dtype=np.float64)
    daily_gross_returns = np.asarray(gross_rets, dtype=np.float64)
    daily_turnover_oneway = np.asarray(to_one, dtype=np.float64)
    daily_turnover_total = np.asarray(to_tot, dtype=np.float64)
    daily_tc_cost = np.asarray(tc_costs, dtype=np.float64)

    w_mat = np.vstack(weights) if len(weights) > 0 else np.zeros((0, env.get_action_dim()), dtype=np.float32)
    eq = equity_curve(daily_net_returns)

    return BacktestResults(
        daily_rewards=daily_rewards,
        daily_net_returns=daily_net_returns,
        daily_gross_returns=daily_gross_returns,
        daily_turnover_oneway=daily_turnover_oneway,
        daily_turnover_total=daily_turnover_total,
        daily_tc_cost=daily_tc_cost,
        equity=eq,
        weights=w_mat,
    )


def plot_results(
    res: BacktestResults,
    tickers: List[str],
    title: str,
    out_dir: Optional[str] = None,
    save_plots: bool = True,
) -> None:
    labels = list(tickers) + ["CASH"]
    Path(out_dir or ".").mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(res.equity)
    plt.title(f"{title} - Equity Curve")
    plt.xlabel("Step")
    plt.ylabel("Equity ($1 start)")
    if save_plots and out_dir:
        plt.savefig(os.path.join(out_dir, "equity_curve.png"), dpi=150, bbox_inches="tight")

    plt.figure()
    plt.plot(res.daily_net_returns)
    plt.title(f"{title} - Daily Net Returns")
    plt.xlabel("Step")
    plt.ylabel("Net Return")
    if save_plots and out_dir:
        plt.savefig(os.path.join(out_dir, "daily_net_returns.png"), dpi=150, bbox_inches="tight")

    plt.figure()
    plt.plot(res.daily_turnover_total, label="turnover_total")
    plt.plot(res.daily_turnover_oneway, label="turnover_oneway")
    plt.title(f"{title} - Turnover")
    plt.xlabel("Step")
    plt.ylabel("Turnover")
    plt.legend()
    if save_plots and out_dir:
        plt.savefig(os.path.join(out_dir, "turnover.png"), dpi=150, bbox_inches="tight")

    if res.weights.shape[0] > 0 and res.weights.shape[1] == len(labels):
        plt.figure()
        avg_w = np.mean(res.weights, axis=0)
        plt.bar(np.arange(len(avg_w)), avg_w)
        plt.xticks(np.arange(len(avg_w)), labels, rotation=45, ha="right")
        plt.title(f"{title} - Average Weights")
        plt.ylabel("Weight")
        if save_plots and out_dir:
            plt.savefig(os.path.join(out_dir, "avg_weights.png"), dpi=150, bbox_inches="tight")

    plt.show()


def main() -> None:
    cfg: Config = get_default_config()
    cfg.ensure_dirs()
    cfg.set_global_seeds()

    device = cfg.auto_detect_device()
    print(f"✓ Device: {device}")

    print("\n" + "=" * 60)
    print("STEP 1: LOAD + PREPARE DATA")
    print("=" * 60)

    _df_train, df_test, _feature_cols = load_and_prepare_data(cfg)
    print(f"✓ Test split rows: {len(df_test)}")

    print("\n" + "=" * 60)
    print("STEP 2: INIT ENV + AGENT")
    print("=" * 60)

    env = Env(df_test, cfg.data.tickers, cfg)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    print(f"✓ Env dims: state_dim={state_dim}, action_dim={action_dim}")

    agent = Agent(state_dim, action_dim, cfg, device=device)

    model_path = cfg.evaluation.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"✓ Loading model: {model_path}")
    agent.load_model(model_path)

    # Optional checkpoint sanity
    try:
        ckpt = torch.load(model_path, map_location=str(device))
        sac_version = ckpt.get("sac_version", None)
        if sac_version is not None and str(sac_version).lower() != "v2":
            print(f"⚠ Warning: checkpoint sac_version={sac_version}. This evaluator expects SAC v2.")
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("STEP 3: BACKTEST")
    print("=" * 60)

    deterministic = bool(cfg.evaluation.deterministic)
    res = run_backtest(env, agent, deterministic=deterministic)

    stats = summarize_results(res, ann=252)

    print("\n" + "-" * 60)
    print("EVALUATION SUMMARY (SAC v2)")
    print("-" * 60)
    for k, v in stats.items():
        if k in {"CAGR", "AnnVol", "MaxDD", "AvgTurnoverOneWay", "AvgTurnoverTotal", "AvgTCCost"}:
            print(f"{k:>18}: {v: .4f}")
        else:
            print(f"{k:>18}: {v}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.evaluation.output_dir, f"eval_{ts}")
    print(f"\n✓ Output dir: {out_dir}")

    if cfg.evaluation.render_plots or cfg.evaluation.save_plots:
        plot_results(
            res,
            tickers=cfg.data.tickers,
            title=f"SAC v2 Eval ({'det' if deterministic else 'stoch'})",
            out_dir=out_dir,
            save_plots=bool(cfg.evaluation.save_plots),
        )


if __name__ == "__main__":
    main()

