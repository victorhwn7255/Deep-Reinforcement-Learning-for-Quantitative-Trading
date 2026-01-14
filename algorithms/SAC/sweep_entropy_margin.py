"""
Hyperparameter sweep for target_entropy_margin in SAC.

Usage:
    python sweep_entropy_margin.py

Results are saved to: runs/sweep_entropy_margin/
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

from config import get_default_config, Config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent
from evaluate import run_backtest, cagr, sharpe, ann_vol, max_drawdown


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

ENTROPY_MARGINS = [0.5, 0.75, 1.0, 1.25, 1.5]  # Values to sweep

# Training settings (reduce if you want faster sweeps)
TOTAL_TIMESTEPS = 900_000  # Set to None to use default from config

# Output directory
SWEEP_OUTPUT_DIR = "runs/sweep_entropy_margin"


# =============================================================================
# SWEEP LOGIC
# =============================================================================

def run_single_experiment(
    margin: float,
    df_train,
    df_test,
    base_cfg: Config,
    device: torch.device,
    run_dir: str,
) -> Dict[str, Any]:
    """Train and evaluate a single entropy margin value."""

    print(f"\n{'='*60}")
    print(f"TRAINING: target_entropy_margin = {margin}")
    print(f"{'='*60}")

    # Create a fresh config with the new margin
    cfg = get_default_config()
    cfg.sac.target_entropy_margin = margin
    cfg.experiment.run_name = f"sweep_margin_{margin}"
    cfg.experiment.output_dir = run_dir
    cfg.training.model_dir = os.path.join(run_dir, "models")
    cfg.training.model_path_final = os.path.join(run_dir, "models", "final.pth")
    cfg.training.model_path_best = os.path.join(run_dir, "models", "best.pth")
    cfg.evaluation.output_dir = os.path.join(run_dir, "eval")
    cfg.evaluation.model_path = cfg.training.model_path_best
    cfg.evaluation.render_plots = False  # Don't show plots during sweep
    cfg.evaluation.save_plots = True

    if TOTAL_TIMESTEPS is not None:
        cfg.training.total_timesteps = TOTAL_TIMESTEPS

    cfg.ensure_dirs()
    cfg.set_global_seeds()

    # Save config
    cfg.save_json(os.path.join(run_dir, "config.json"))

    # Create env and agent
    env_train = Env(df_train, cfg.data.tickers, cfg)
    state_dim = env_train.get_state_dim()
    action_dim = env_train.get_action_dim()

    agent = Agent(state_dim, action_dim, cfg, device=device)

    # Train
    train_start = time.time()
    episode_returns, losses, best_model_state = agent.learn(
        env_train,
        total_timesteps=int(cfg.training.total_timesteps),
    )
    train_time = time.time() - train_start

    # Save models
    agent.save_model(cfg.training.model_path_final)
    if best_model_state is not None:
        torch.save(best_model_state, cfg.training.model_path_best)

    # Evaluate on test set
    print(f"\nEvaluating margin={margin}...")
    env_test = Env(df_test, cfg.data.tickers, cfg)

    # Load best model for evaluation
    if os.path.exists(cfg.training.model_path_best):
        agent.load_model(cfg.training.model_path_best)

    res = run_backtest(env_test, agent, deterministic=True)

    eq = res["equity"]
    net = res["net_returns"]

    stats = {
        "margin": margin,
        "CAGR": cagr(eq, step_size=cfg.env.lag),
        "Sharpe": sharpe(net, step_size=cfg.env.lag),
        "AnnVol": ann_vol(net, step_size=cfg.env.lag),
        "MaxDD": max_drawdown(eq),
        "FinalEquity": float(eq[-1]) if eq.size else 1.0,
        "AvgTurnoverOneWay": float(res["turnover_oneway"].mean()),
        "AvgTurnoverTotal": float(res["turnover_total"].mean()),
        "AvgTCCost": float(res["tc_costs"].mean()),
        "train_time_min": train_time / 60,
        "num_episodes": len(episode_returns),
    }

    # Save individual results
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    print("="*60)
    print("ENTROPY MARGIN HYPERPARAMETER SWEEP")
    print("="*60)
    print(f"Margins to test: {ENTROPY_MARGINS}")
    print(f"Output dir: {SWEEP_OUTPUT_DIR}")

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(SWEEP_OUTPUT_DIR, f"sweep_{timestamp}")
    Path(sweep_dir).mkdir(parents=True, exist_ok=True)

    # Load data once (shared across all experiments)
    base_cfg = get_default_config()
    base_cfg.set_global_seeds()
    device = base_cfg.auto_detect_device()
    print(f"\nDevice: {device}")

    print("\nLoading data (shared across all experiments)...")
    df_train, df_test, _ = load_and_prepare_data(base_cfg)
    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")

    # Run sweep
    all_results: List[Dict[str, Any]] = []
    sweep_start = time.time()

    for i, margin in enumerate(ENTROPY_MARGINS):
        run_dir = os.path.join(sweep_dir, f"margin_{margin}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n[{i+1}/{len(ENTROPY_MARGINS)}] Starting margin={margin}")

        try:
            stats = run_single_experiment(
                margin=margin,
                df_train=df_train,
                df_test=df_test,
                base_cfg=base_cfg,
                device=device,
                run_dir=run_dir,
            )
            all_results.append(stats)

            print(f"\n--- Results for margin={margin} ---")
            print(f"  CAGR:        {stats['CAGR']:.4f}")
            print(f"  Sharpe:      {stats['Sharpe']:.4f}")
            print(f"  MaxDD:       {stats['MaxDD']:.4f}")
            print(f"  FinalEquity: {stats['FinalEquity']:.4f}")

        except Exception as e:
            print(f"ERROR: margin={margin} failed: {e}")
            all_results.append({"margin": margin, "error": str(e)})

    sweep_time = time.time() - sweep_start

    # Save aggregated results
    summary_path = os.path.join(sweep_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "margins": ENTROPY_MARGINS,
            "results": all_results,
            "total_time_min": sweep_time / 60,
        }, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    print(f"{'Margin':>8} | {'CAGR':>8} | {'Sharpe':>8} | {'MaxDD':>8} | {'FinalEq':>10} | {'Turnover':>10}")
    print("-"*80)

    valid_results = [r for r in all_results if "error" not in r]
    for r in valid_results:
        print(f"{r['margin']:>8.2f} | {r['CAGR']:>8.4f} | {r['Sharpe']:>8.4f} | {r['MaxDD']:>8.4f} | {r['FinalEquity']:>10.4f} | {r['AvgTurnoverTotal']:>10.4f}")

    # Find best by Sharpe
    if valid_results:
        best = max(valid_results, key=lambda x: x["Sharpe"])
        print(f"\n★ Best by Sharpe: margin={best['margin']} (Sharpe={best['Sharpe']:.4f}, FinalEquity={best['FinalEquity']:.4f})")

        best_cagr = max(valid_results, key=lambda x: x["CAGR"])
        if best_cagr["margin"] != best["margin"]:
            print(f"★ Best by CAGR:   margin={best_cagr['margin']} (CAGR={best_cagr['CAGR']:.4f})")

    print(f"\nTotal sweep time: {sweep_time/60:.1f} minutes")
    print(f"Results saved to: {sweep_dir}")


if __name__ == "__main__":
    main()
