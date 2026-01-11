from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import get_default_config, Config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass

def equity_curve(daily_net_returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + daily_net_returns)

def sharpe(daily_net_returns: np.ndarray, ann: int = 252) -> float:
    mu = float(np.mean(daily_net_returns))
    sd = float(np.std(daily_net_returns))
    return mu / (sd + 1e-12) * float(np.sqrt(ann))

def max_drawdown(equity: np.ndarray) -> float:
    rm = np.maximum.accumulate(equity)
    dd = (equity - rm) / (rm + 1e-12)
    return float(np.min(dd))

def evaluate_metrics(daily_net_returns: np.ndarray) -> Dict[str, float]:
    eq = equity_curve(daily_net_returns)
    return {
        "total_return": float(eq[-1] - 1.0) if len(eq) else 0.0,
        "sharpe": sharpe(daily_net_returns),
        "ann_vol": float(np.std(daily_net_returns) * np.sqrt(252)),
        "max_drawdown": max_drawdown(eq),
    }

def load_checkpoint(agent: Agent, path: str) -> None:
    ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        # best snapshot or full
        try:
            agent.load_best_model(ckpt)
        except Exception:
            agent.load_model(path)
    else:
        agent.load_model(path)

    agent.policy.eval()
    agent.q1.eval()
    agent.q2.eval()
    agent.value.eval()


# =============================================================================
# Rollout helpers
# =============================================================================

def run_policy_episode(env: Env, agent: Agent, deterministic: bool) -> Tuple[np.ndarray, np.ndarray]:
    obs = env.reset()
    done = False

    rets: List[float] = []
    acts: List[np.ndarray] = []

    while not done:
        a = agent.choose_action_deterministic(obs) if deterministic else agent.choose_action_stochastic(obs)
        acts.append(a.copy())
        obs, _, done = env.step(a)

        if not hasattr(env, "last_net_return"):
            raise RuntimeError("Env must expose env.last_net_return for evaluation.")
        rets.append(float(env.last_net_return))

    return np.asarray(rets, dtype=np.float64), np.asarray(acts, dtype=np.float64)

def run_benchmark_episode(env: Env, w: np.ndarray) -> np.ndarray:
    obs = env.reset()
    done = False
    rets: List[float] = []
    while not done:
        obs, _, done = env.step(w)
        rets.append(float(env.last_net_return))
    return np.asarray(rets, dtype=np.float64)


# =============================================================================
# Plotting
# =============================================================================

def plot_eval(
    cfg: Config,
    dates: pd.DatetimeIndex,
    sac_equity: np.ndarray,
    bench_equity: np.ndarray,
    sac_dd: np.ndarray,
    bench_dd: np.ndarray,
    sac_actions: np.ndarray,
    tickers: List[str],
    title: str,
) -> Optional[str]:
    if not cfg.evaluation.render_plots:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Equity
    axes[0, 0].plot(dates[: len(sac_equity)], sac_equity, label="SAC", linewidth=2)
    axes[0, 0].plot(dates[: len(bench_equity)], bench_equity, label="Benchmark", linestyle="--", linewidth=2)
    axes[0, 0].set_title("Cumulative Equity")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Drawdown
    axes[0, 1].plot(dates[: len(sac_dd)], sac_dd * 100.0, label="SAC DD")
    axes[0, 1].plot(dates[: len(bench_dd)], bench_dd * 100.0, label="Bench DD", linestyle="--")
    axes[0, 1].set_title("Drawdown (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # SAC daily returns dist
    sac_daily = np.diff(np.log(sac_equity + 1e-12))
    axes[1, 0].hist(sac_daily, bins=50, alpha=0.8)
    axes[1, 0].set_title("SAC Log-Return Distribution")
    axes[1, 0].grid(True, alpha=0.3)

    # Weights
    names = tickers + ["CASH"]
    for i in range(min(sac_actions.shape[1], len(names))):
        axes[1, 1].plot(dates[: sac_actions.shape[0]], sac_actions[:, i], label=names[i], alpha=0.85)
    axes[1, 1].set_title("Portfolio Weights")
    axes[1, 1].set_ylim([0.0, 1.0])
    axes[1, 1].legend(ncol=2, loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    out_path = None
    if cfg.evaluation.save_plots:
        ensure_dir(cfg.evaluation.output_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(Path(cfg.evaluation.output_dir) / f"evaluation_{ts}.png")
        plt.savefig(out_path, dpi=150)

    plt.show()
    return out_path


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = get_default_config()
    cfg.ensure_dirs()
    cfg.print_summary()

    device = cfg.auto_detect_device()

    df_train, df_test, feature_cols = load_and_prepare_data(cfg)

    # Build env (prefer config-driven env signature)
    try:
        env_test = Env(df=df_test, tickers=cfg.data.tickers, cfg=cfg, feature_columns=feature_cols)
    except TypeError:
        # fallback if your Env doesn't accept cfg/feature_columns yet
        env_test = Env(
            df_test,
            cfg.data.tickers,
            lag=cfg.env.lag,
            include_position_in_state=cfg.env.include_position_in_state,
        )

    # Infer dims
    obs0 = env_test.reset()
    state_dim = int(np.asarray(obs0).reshape(-1).shape[0])
    action_dim = int(len(cfg.data.tickers) + 1)

    agent = Agent(n_input=state_dim, n_action=action_dim, cfg=cfg, device=device)

    current_model_path = cfg.evaluation.model_path
    if os.path.exists(current_model_path):
        load_checkpoint(agent, current_model_path)
        print(f"\n✓ Loaded model: {current_model_path}")
    else:
        print(f"\n⚠ Model not found at cfg.evaluation.model_path: {current_model_path}")
        print("  Use menu option 4 to load a different checkpoint.")

    # Benchmark weights (must match (assets + cash))
    benchmark_weights = np.array([0.10, 0.50, 0.30, 0.05, 0.05, 0.00], dtype=np.float32)
    if benchmark_weights.shape[0] != action_dim:
        raise ValueError(f"Benchmark dim mismatch: expected {action_dim}, got {benchmark_weights.shape[0]}")

    dates = df_test.index

    while True:
        print("\n" + "=" * 60)
        print("EVALUATION MENU")
        print("=" * 60)
        print("\nWhat would you like to do?")
        print("  1. Run deterministic evaluation")
        print("  2. Run stochastic evaluation (with seed)")
        print("  3. Run multiple stochastic evaluations (statistical analysis)")
        print("  4. Switch to different model")
        print("  5. Exit")

        menu_choice = input("\nEnter choice (1-5): ").strip()

        if menu_choice == "5":
            print("\n" + "=" * 60)
            print("Exiting evaluation script. Goodbye!")
            print("=" * 60)
            break

        if menu_choice == "4":
            print("\n" + "=" * 60)
            print("SWITCH MODEL")
            print("=" * 60)
            print("\nWhich model do you want to load?")
            print(f"  1. Best model  ({cfg.training.model_path_best})")
            print(f"  2. Final model ({cfg.training.model_path_final})")
            print("  3. Custom path")

            ch = input("Enter choice (1-3): ").strip()
            if ch == "1":
                path = cfg.training.model_path_best
            elif ch == "2":
                path = cfg.training.model_path_final
            elif ch == "3":
                path = input("Enter model path: ").strip()
            else:
                print("✗ Invalid choice.")
                continue

            if not os.path.exists(path):
                print(f"⚠ Not found: {path}")
                continue

            load_checkpoint(agent, path)
            current_model_path = path
            print(f"✓ Switched model: {current_model_path}")
            continue

        if menu_choice not in ["1", "2", "3"]:
            print("✗ Invalid choice.")
            continue

        show_stats = False
        display_mode = ""
        sac_daily = None
        sac_actions = None

        if menu_choice == "1":
            display_mode = "Deterministic (Dirichlet mean)"
            sac_daily, sac_actions = run_policy_episode(env_test, agent, deterministic=True)

        elif menu_choice == "2":
            seed_in = input("Seed (default 42): ").strip()
            seed = int(seed_in) if seed_in else 42
            set_all_seeds(seed)
            display_mode = f"Stochastic (seed={seed})"
            sac_daily, sac_actions = run_policy_episode(env_test, agent, deterministic=False)

        elif menu_choice == "3":
            show_stats = True
            n_in = input("How many runs? (default 10, max 100): ").strip()
            try:
                n_runs = int(n_in) if n_in else 10
                n_runs = max(1, min(n_runs, 100))
            except ValueError:
                n_runs = 10

            display_mode = f"Multiple runs (n={n_runs})"
            base_seed = int(cfg.experiment.seed)

            all_returns, all_sharpes, all_dd, all_vol = [], [], [], []
            daily_runs, action_runs = [], []

            for r in range(n_runs):
                set_all_seeds(base_seed + r + 1)
                d, a = run_policy_episode(env_test, agent, deterministic=False)
                m = evaluate_metrics(d)
                all_returns.append(m["total_return"])
                all_sharpes.append(m["sharpe"])
                all_dd.append(m["max_drawdown"])
                all_vol.append(m["ann_vol"])
                daily_runs.append(d)
                action_runs.append(a)

            all_returns = np.asarray(all_returns, dtype=np.float64)
            all_sharpes = np.asarray(all_sharpes, dtype=np.float64)
            all_dd = np.asarray(all_dd, dtype=np.float64)
            all_vol = np.asarray(all_vol, dtype=np.float64)

            median_idx = int(np.argsort(all_returns)[len(all_returns) // 2])
            sac_daily = daily_runs[median_idx]
            sac_actions = action_runs[median_idx]

        # Benchmark
        bench_daily = run_benchmark_episode(env_test, benchmark_weights)

        # Metrics
        sac_m = evaluate_metrics(sac_daily)
        bench_m = evaluate_metrics(bench_daily)

        sac_eq = equity_curve(sac_daily)
        bench_eq = equity_curve(bench_daily)
        sac_dd_series = (sac_eq - np.maximum.accumulate(sac_eq)) / (np.maximum.accumulate(sac_eq) + 1e-12)
        bench_dd_series = (bench_eq - np.maximum.accumulate(bench_eq)) / (np.maximum.accumulate(bench_eq) + 1e-12)

        # Print
        print("\n" + "=" * 60)
        print("TEST SET PERFORMANCE")
        print("=" * 60)
        print(f"Model: {current_model_path}")
        print(f"Mode : {display_mode}")

        if show_stats:
            print("\n" + "-" * 60)
            print("SAC - STATISTICAL SUMMARY")
            print("-" * 60)
            print(f"Total Return (%): mean={np.mean(all_returns)*100:.2f}  std={np.std(all_returns)*100:.2f}  "
                  f"min={np.min(all_returns)*100:.2f}  max={np.max(all_returns)*100:.2f}  median={np.median(all_returns)*100:.2f}")
            print(f"Sharpe: mean={np.mean(all_sharpes):.4f}  std={np.std(all_sharpes):.4f}  "
                  f"min={np.min(all_sharpes):.4f}  max={np.max(all_sharpes):.4f}")
            print(f"Max DD (%): mean={np.mean(all_dd)*100:.2f}  worst={np.min(all_dd)*100:.2f}  best={np.max(all_dd)*100:.2f}")
            print(f"Ann Vol (%): mean={np.mean(all_vol)*100:.2f}  std={np.std(all_vol)*100:.2f}")

            prob_beat = float(np.mean(all_returns > bench_m["total_return"]) * 100.0)
            prob_pos = float(np.mean(all_returns > 0.0) * 100.0)
            print(f"P(beat benchmark)={prob_beat:.1f}%  P(positive)={prob_pos:.1f}%")

        else:
            print("\n" + "-" * 60)
            print("SAC (single run)")
            print("-" * 60)
            print(f"  Total Return:    {sac_m['total_return']*100:8.2f}%")
            print(f"  Sharpe Ratio:    {sac_m['sharpe']:8.4f}")
            print(f"  Max Drawdown:    {sac_m['max_drawdown']*100:8.2f}%")
            print(f"  Ann. Volatility: {sac_m['ann_vol']*100:8.2f}%")

        print("\n" + "-" * 60)
        print("BENCHMARK")
        print("-" * 60)
        print(f"  Total Return:    {bench_m['total_return']*100:8.2f}%")
        print(f"  Sharpe Ratio:    {bench_m['sharpe']:8.4f}")
        print(f"  Max Drawdown:    {bench_m['max_drawdown']*100:8.2f}%")
        print(f"  Ann. Volatility: {bench_m['ann_vol']*100:8.2f}%")

        print("\n" + "-" * 60)
        print("COMPARISON (SAC vs Benchmark)")
        print("-" * 60)
        print(f"  Return diff:   {(sac_m['total_return'] - bench_m['total_return'])*100:+8.2f}%")
        print(f"  Sharpe diff:   {(sac_m['sharpe'] - bench_m['sharpe']):+8.4f}")
        print(f"  MaxDD diff:    {(sac_m['max_drawdown'] - bench_m['max_drawdown'])*100:+8.2f}%")

        # Visualize
        visualize = input("\nVisualize results? (y/n): ").strip().lower()
        if visualize == "y":
            title = f"SAC Evaluation ({display_mode}) | {dates[0].date()} → {dates[-1].date()}"
            out = plot_eval(
                cfg=cfg,
                dates=dates,
                sac_equity=sac_eq,
                bench_equity=bench_eq,
                sac_dd=sac_dd_series,
                bench_dd=bench_dd_series,
                sac_actions=sac_actions,
                tickers=list(cfg.data.tickers),
                title=title,
            )
            if out is not None:
                print(f"✓ Plot saved: {out}")

    print("\nThank you for using the SAC Portfolio Evaluation System!")


if __name__ == "__main__":
    main()