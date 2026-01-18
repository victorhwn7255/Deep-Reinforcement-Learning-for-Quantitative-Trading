from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from config import get_default_config, Config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent
from analysis.plotting import generate_evaluation_plots


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


def main() -> None:
    cfg = get_default_config()
    
    model_path = cfg.evaluation.model_path
    
    # Load the exact config used during training if present
    cfg_path = os.path.join(os.path.dirname(model_path), "config.json")
    if os.path.exists(cfg_path):
        cfg = Config.load_json(cfg_path)
        # Preserve the model_path you intended to evaluate
        cfg.evaluation.model_path = model_path
        print(f"✓ Loaded training config: {cfg_path}")
    else:
        print(f"⚠ No training config found at: {cfg_path}")
        print("  Using default config (may cause state_dim mismatch if features differ).")
    
    cfg.ensure_dirs()
    cfg.set_global_seeds()

    device = cfg.auto_detect_device()
    print(f"✓ Device: {device}")

    # Data
    _df_train, df_test, _feature_cols = load_and_prepare_data(cfg)
    print(f"✓ Test rows: {len(df_test)}")

    # Extract dates for time-based plots
    dates: Optional[pd.DatetimeIndex] = None
    if "date" in df_test.columns:
        dates = pd.DatetimeIndex(df_test["date"].values)
    elif df_test.index.name == "date" or isinstance(df_test.index, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(df_test.index)

    # Extract regime probabilities if HMM is enabled
    regime_probs: Optional[np.ndarray] = None
    if getattr(cfg.features, "use_regime_hmm", False):
        pcols = cfg.features.regime_prob_columns
        if all(c in df_test.columns for c in pcols):
            regime_probs = df_test[pcols].values
            print(f"✓ Regime probabilities available ({len(pcols)} states)")

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

    # Run backtest
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

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (SAC v2)")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:>18}: {v:.6f}" if isinstance(v, float) else f"  {k:>18}: {v}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.evaluation.output_dir, f"eval_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Generate thesis-quality plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    if cfg.evaluation.save_plots or cfg.evaluation.render_plots:
        # Align dates and regime_probs with backtest results (skip first row for initial state)
        plot_dates = dates[1:] if dates is not None and len(dates) > len(net) else dates
        plot_regime = regime_probs[1:] if regime_probs is not None and len(regime_probs) > len(net) else regime_probs

        saved_paths = generate_evaluation_plots(
            results=res,
            cfg=cfg,
            out_dir=out_dir,
            dates=plot_dates,
            regime_probs=plot_regime,
            show=cfg.evaluation.render_plots,
        )
        print(f"✓ Saved {len(saved_paths)} plots to: {out_dir}")
        for p in saved_paths:
            print(f"  - {os.path.basename(p)}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()


