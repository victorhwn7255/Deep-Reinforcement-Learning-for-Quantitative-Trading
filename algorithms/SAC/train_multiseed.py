"""
    python train_multiseed.py --no-hmm          # NON-HMM training
    python train_multiseed.py                   # Use default 5 seeds
    python train_multiseed.py --seeds 42 123    # Use specific seeds
    python train_multiseed.py --num_seeds 3     # Use 3 random seeds
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from config import get_default_config, Config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent
from evaluate import run_backtest, sharpe, cagr, max_drawdown, ann_vol
from analysis.plotting import (
    plot_training_returns,
    plot_training_losses,
    plot_multiseed_training_comparison,
    plot_multiseed_equity_comparison,
    plot_multiseed_metrics_summary,
)


# =============================================================================
# Multi-seed Training Results
# =============================================================================

@dataclass
class SeedResult:
    """Results from training with a single seed."""
    seed: int
    train_time_sec: float
    num_episodes: int
    final_episode_return: float
    best_episode_return: float

    # Evaluation metrics (on test set)
    test_sharpe: float
    test_cagr: float
    test_max_dd: float
    test_ann_vol: float
    test_final_equity: float

    model_path_final: str
    model_path_best: str

    # Data for plotting (stored separately to avoid huge dataclass)
    episode_returns: List[float] = field(default_factory=list)
    test_equity: np.ndarray = field(default_factory=lambda: np.array([]))


def print_header(title: str, char: str = "=", width: int = 80) -> None:
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_subheader(title: str, char: str = "-", width: int = 60) -> None:
    """Print a formatted subheader."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")


def train_single_seed(
    cfg: Config,
    seed: int,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    run_dir: str,
) -> SeedResult:
    """Train SAC agent with a single seed and evaluate on test set."""

    print_subheader(f"SEED {seed}")

    # Update seed in config
    cfg.experiment.seed = seed
    cfg.set_global_seeds()

    # Create seed-specific model paths
    model_dir = os.path.join(run_dir, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)

    model_path_final = os.path.join(model_dir, "sac_final.pth")
    model_path_best = os.path.join(model_dir, "sac_best.pth")

    # Save config for this seed
    cfg.save_json(os.path.join(model_dir, "config.json"))

    # Create environment and agent
    device = cfg.auto_detect_device()
    env = Env(df_train, cfg.data.tickers, cfg)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    print(f"  Device: {device}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")

    agent = Agent(state_dim, action_dim, cfg, device=device)

    # Train
    print(f"  Training with {cfg.training.total_timesteps:,} timesteps...")
    train_start = time.time()

    episode_returns, losses, best_model_state = agent.learn(
        env,
        total_timesteps=int(cfg.training.total_timesteps),
    )

    train_time = time.time() - train_start

    # Save models
    agent.save_model(model_path_final)
    if best_model_state is not None:
        torch.save(best_model_state, model_path_best)

    # Save per-seed training plots
    if len(episode_returns) > 0:
        path = os.path.join(model_dir, "episode_returns.png")
        plot_training_returns(episode_returns, path, title=f"Seed {seed} - Episode Returns")

    if len(losses) > 0:
        path = os.path.join(model_dir, "training_losses.png")
        plot_training_losses(losses, path)

    # Training summary
    num_episodes = len(episode_returns)
    final_return = episode_returns[-1] if episode_returns else 0.0
    best_return = max(episode_returns) if episode_returns else 0.0

    print(f"  Training complete in {train_time / 60:.1f} min")
    print(f"  Episodes: {num_episodes}, Final return: {final_return:.4f}, Best: {best_return:.4f}")

    # Evaluate on test set
    print(f"  Evaluating on test set...")

    # Load best model for evaluation
    if best_model_state is not None and os.path.exists(model_path_best):
        agent.load_model(model_path_best)

    test_env = Env(df_test, cfg.data.tickers, cfg)
    test_results = run_backtest(test_env, agent, deterministic=True)

    eq = test_results["equity"]
    net = test_results["net_returns"]
    step_size = cfg.env.lag

    test_sharpe = sharpe(net, step_size=step_size)
    test_cagr = cagr(eq, step_size=step_size)
    test_max_dd = max_drawdown(eq)
    test_ann_vol_val = ann_vol(net, step_size=step_size)
    test_final_eq = float(eq[-1]) if eq.size > 0 else 1.0

    print(f"  Test Sharpe: {test_sharpe:.3f}, CAGR: {test_cagr:.2%}, MaxDD: {test_max_dd:.2%}")

    return SeedResult(
        seed=seed,
        train_time_sec=train_time,
        num_episodes=num_episodes,
        final_episode_return=final_return,
        best_episode_return=best_return,
        test_sharpe=test_sharpe,
        test_cagr=test_cagr,
        test_max_dd=test_max_dd,
        test_ann_vol=test_ann_vol_val,
        test_final_equity=test_final_eq,
        model_path_final=model_path_final,
        model_path_best=model_path_best,
        episode_returns=list(episode_returns),
        test_equity=eq,
    )


def aggregate_results(results: List[SeedResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate results across seeds, computing mean and std."""
    metrics = {
        "test_sharpe": [r.test_sharpe for r in results],
        "test_cagr": [r.test_cagr for r in results],
        "test_max_dd": [r.test_max_dd for r in results],
        "test_ann_vol": [r.test_ann_vol for r in results],
        "test_final_equity": [r.test_final_equity for r in results],
        "train_time_sec": [r.train_time_sec for r in results],
        "num_episodes": [r.num_episodes for r in results],
    }

    aggregated = {}
    for name, values in metrics.items():
        arr = np.array(values)
        aggregated[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return aggregated


def print_results_table(results: List[SeedResult], aggregated: Dict) -> None:
    """Print a formatted results table."""

    print_header("MULTI-SEED TRAINING RESULTS")

    # Per-seed results
    print("\nPer-Seed Results:")
    print("-" * 90)
    print(f"{'Seed':>8} {'Time(min)':>10} {'Episodes':>10} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10} {'FinalEq':>10}")
    print("-" * 90)

    for r in results:
        print(f"{r.seed:>8} {r.train_time_sec/60:>10.1f} {r.num_episodes:>10} "
              f"{r.test_sharpe:>10.3f} {r.test_cagr:>10.2%} {r.test_max_dd:>10.2%} {r.test_final_equity:>10.3f}")

    print("-" * 90)

    # Aggregated results
    print("\nAggregated Results (Mean +/- Std):")
    print("-" * 60)

    fmt_metrics = [
        ("Test Sharpe", "test_sharpe", "{:.3f}"),
        ("Test CAGR", "test_cagr", "{:.2%}"),
        ("Test MaxDD", "test_max_dd", "{:.2%}"),
        ("Test Ann. Vol", "test_ann_vol", "{:.2%}"),
        ("Final Equity", "test_final_equity", "{:.3f}"),
        ("Train Time (min)", "train_time_sec", "{:.1f}"),
    ]

    for label, key, fmt in fmt_metrics:
        m = aggregated[key]
        mean_str = fmt.format(m["mean"] / 60 if "time" in key.lower() else m["mean"])
        std_str = fmt.format(m["std"] / 60 if "time" in key.lower() else m["std"])
        print(f"  {label:<20}: {mean_str} +/- {std_str}")

    print("-" * 60)

    # Best and worst seeds
    best_sharpe_result = max(results, key=lambda r: r.test_sharpe)
    worst_sharpe_result = min(results, key=lambda r: r.test_sharpe)

    print(f"\n  Best Sharpe:  Seed {best_sharpe_result.seed} ({best_sharpe_result.test_sharpe:.3f})")
    print(f"  Worst Sharpe: Seed {worst_sharpe_result.seed} ({worst_sharpe_result.test_sharpe:.3f})")

    # Consistency check
    sharpe_std = aggregated["test_sharpe"]["std"]
    sharpe_mean = aggregated["test_sharpe"]["mean"]
    cv = sharpe_std / abs(sharpe_mean) if sharpe_mean != 0 else float('inf')

    print(f"\n  Coefficient of Variation (Sharpe): {cv:.2f}")
    if cv < 0.2:
        print("  Assessment: LOW variance - Results are ROBUST")
    elif cv < 0.5:
        print("  Assessment: MODERATE variance - Results are reasonably stable")
    else:
        print("  Assessment: HIGH variance - Consider investigating hyperparameters")


def save_results_csv(results: List[SeedResult], aggregated: Dict, run_dir: str) -> str:
    """Save results to CSV file."""

    # Per-seed results
    rows = []
    for r in results:
        rows.append({
            "seed": r.seed,
            "train_time_min": r.train_time_sec / 60,
            "num_episodes": r.num_episodes,
            "final_episode_return": r.final_episode_return,
            "best_episode_return": r.best_episode_return,
            "test_sharpe": r.test_sharpe,
            "test_cagr": r.test_cagr,
            "test_max_dd": r.test_max_dd,
            "test_ann_vol": r.test_ann_vol,
            "test_final_equity": r.test_final_equity,
            "model_path_best": r.model_path_best,
        })

    # Add aggregated row
    rows.append({
        "seed": "MEAN",
        "train_time_min": aggregated["train_time_sec"]["mean"] / 60,
        "num_episodes": aggregated["num_episodes"]["mean"],
        "test_sharpe": aggregated["test_sharpe"]["mean"],
        "test_cagr": aggregated["test_cagr"]["mean"],
        "test_max_dd": aggregated["test_max_dd"]["mean"],
        "test_ann_vol": aggregated["test_ann_vol"]["mean"],
        "test_final_equity": aggregated["test_final_equity"]["mean"],
    })
    rows.append({
        "seed": "STD",
        "train_time_min": aggregated["train_time_sec"]["std"] / 60,
        "num_episodes": aggregated["num_episodes"]["std"],
        "test_sharpe": aggregated["test_sharpe"]["std"],
        "test_cagr": aggregated["test_cagr"]["std"],
        "test_max_dd": aggregated["test_max_dd"]["std"],
        "test_ann_vol": aggregated["test_ann_vol"]["std"],
        "test_final_equity": aggregated["test_final_equity"]["std"],
    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(run_dir, "multiseed_results.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


def generate_multiseed_plots(results: List[SeedResult], run_dir: str) -> None:
    """Generate multi-seed comparison plots."""

    print_subheader("GENERATING MULTI-SEED PLOTS")

    # 1. Training curves comparison (if we have episode returns)
    all_returns = {r.seed: r.episode_returns for r in results if len(r.episode_returns) > 0}
    if all_returns:
        path = os.path.join(run_dir, "multiseed_training_curves.png")
        plot_multiseed_training_comparison(all_returns, path)
        print(f"  Saved: {path}")

    # 2. Test equity curves comparison
    equity_dict = {r.seed: r.test_equity for r in results if r.test_equity.size > 0}
    if equity_dict:
        path = os.path.join(run_dir, "multiseed_equity_curves.png")
        plot_multiseed_equity_comparison(equity_dict, path)
        print(f"  Saved: {path}")

    # 3. Metrics summary bar chart
    metrics_df = pd.DataFrame([{
        "seed": r.seed,
        "sharpe": r.test_sharpe,
        "cagr": r.test_cagr,
        "max_dd": r.test_max_dd,
        "ann_vol": r.test_ann_vol,
    } for r in results])

    if len(metrics_df) > 0:
        path = os.path.join(run_dir, "multiseed_metrics_summary.png")
        plot_multiseed_metrics_summary(metrics_df, path)
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed SAC Training")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Specific seeds to use (e.g., --seeds 42 123 456)")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of seeds if --seeds not provided (default: 5)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps for each seed")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name (default: auto-generated)")
    parser.add_argument("--no-hmm", action="store_true",
                        help="Disable HMM regime features (for baseline comparison)")
    args = parser.parse_args()

    # Determine seeds
    if args.seeds is not None:
        seeds = args.seeds
    else:
        # Default 5 seeds
        seeds = [42, 123, 456, 789, 1024][:args.num_seeds]

    # Determine if HMM is enabled
    use_hmm = not args.no_hmm

    print_header("MULTI-SEED SAC PORTFOLIO TRAINING")
    print(f"\nSeeds: {seeds}")
    print(f"Number of seeds: {len(seeds)}")
    print(f"HMM Regime Features: {'ENABLED' if use_hmm else 'DISABLED'}")

    # Setup config
    cfg = get_default_config()
    cfg.features.use_regime_hmm = use_hmm

    if args.timesteps is not None:
        cfg.training.total_timesteps = args.timesteps

    print(f"Timesteps per seed: {cfg.training.total_timesteps:,}")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hmm_suffix = "hmm" if use_hmm else "no_hmm"
    run_name = args.run_name or f"multiseed_{len(seeds)}seeds_{hmm_suffix}_{timestamp}"
    run_dir = os.path.join(cfg.experiment.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}")

    # Load data once (shared across seeds)
    print_subheader("LOADING DATA")

    data_start = time.time()
    df_train, df_test, feature_cols = load_and_prepare_data(cfg)
    data_time = time.time() - data_start

    print(f"  Train rows: {len(df_train)}")
    print(f"  Test rows: {len(df_test)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Data load time: {data_time:.1f}s")

    # Verify HMM regime features
    if cfg.features.use_regime_hmm:
        pcols = cfg.features.regime_prob_columns
        prob_sum = df_train[pcols].sum(axis=1)
        print(f"  Regime probs sum: {prob_sum.min():.4f} - {prob_sum.max():.4f}")

    # Train with each seed
    print_header("TRAINING PHASE")

    total_start = time.time()
    results: List[SeedResult] = []

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Training with seed {seed}")

        result = train_single_seed(cfg, seed, df_train, df_test, run_dir)
        results.append(result)

    total_time = time.time() - total_start

    # Aggregate and display results
    aggregated = aggregate_results(results)
    print_results_table(results, aggregated)

    # Generate multi-seed plots
    generate_multiseed_plots(results, run_dir)

    # Save results
    csv_path = save_results_csv(results, aggregated, run_dir)
    print(f"\nResults saved to: {csv_path}")

    # Final summary
    print_header("TRAINING COMPLETE")
    print(f"\n  Total training time: {total_time / 60:.1f} minutes")
    print(f"  Average per seed: {total_time / len(seeds) / 60:.1f} minutes")
    print(f"  Output directory: {run_dir}")
    print(f"\n  Best model (by Sharpe): {max(results, key=lambda r: r.test_sharpe).model_path_best}")

    # Recommendation
    best = max(results, key=lambda r: r.test_sharpe)
    print(f"\n  Recommended for deployment: Seed {best.seed}")
    print(f"    Sharpe: {best.test_sharpe:.3f}, CAGR: {best.test_cagr:.2%}, MaxDD: {best.test_max_dd:.2%}")


if __name__ == "__main__":
    main()
