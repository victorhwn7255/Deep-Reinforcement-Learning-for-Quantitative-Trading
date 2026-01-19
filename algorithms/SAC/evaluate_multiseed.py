"""

python evaluate_multiseed.py --run_dir runs/multiseed_5seeds_20240101_120000
python evaluate_multiseed.py --run_dir runs/multiseed_5seeds_20260118_110501 --per_seed_plots
python evaluate_multiseed.py --model_paths models/seed_42/sac_best.pth models/seed_123/sac_best.pth

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from config import get_default_config, Config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent
from evaluate import (
    run_backtest,
    equity_curve,
    sharpe,
    cagr,
    max_drawdown,
    ann_vol,
)
from analysis.plotting import (
    generate_evaluation_plots,
    plot_multiseed_equity_comparison,
    plot_multiseed_metrics_summary,
)


@dataclass
class EvalResult:
    """Evaluation results for a single model."""
    seed: int
    model_path: str
    sharpe: float
    cagr: float
    max_dd: float
    ann_vol: float
    final_equity: float
    avg_turnover: float
    avg_tc_cost: float
    num_steps: int
    equity: np.ndarray = None  # Store for plotting
    backtest_results: Dict = None  # Store for detailed plots


def print_header(title: str, char: str = "=", width: int = 80) -> None:
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def extract_seed_from_path(path: str) -> int:
    """Extract seed number from model path.

    Expects paths like: .../seed_42/sac_best.pth
    Extracts seed from the parent directory name to avoid matching
    patterns like 'multiseed_5seeds' in the run directory name.
    """
    import re
    # Get the parent directory name (e.g., "seed_42" from ".../seed_42/sac_best.pth")
    parent_dir = os.path.basename(os.path.dirname(path))
    match = re.match(r'seed_(\d+)', parent_dir)
    if match:
        return int(match.group(1))
    # Fallback: try matching seed_XXX between path separators
    match = re.search(r'[/\\]seed_(\d+)[/\\]', path)
    if match:
        return int(match.group(1))
    # Final fallback: use hash of path
    return hash(path) % 10000


def evaluate_single_model(
    model_path: str,
    cfg: Config,
    df_test: pd.DataFrame,
    device: torch.device,
) -> EvalResult:
    """Evaluate a single model on test data."""

    seed = extract_seed_from_path(model_path)

    # Try to load config from model directory
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if os.path.exists(config_path):
        cfg = Config.load_json(config_path)

    # Create environment and agent
    env = Env(df_test, cfg.data.tickers, cfg)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    agent = Agent(state_dim, action_dim, cfg, device=device)
    agent.load_model(model_path)

    # Run backtest
    results = run_backtest(env, agent, deterministic=True)

    eq = results["equity"]
    net = results["net_returns"]
    step_size = cfg.env.lag

    return EvalResult(
        seed=seed,
        model_path=model_path,
        sharpe=sharpe(net, step_size=step_size),
        cagr=cagr(eq, step_size=step_size),
        max_dd=max_drawdown(eq),
        ann_vol=ann_vol(net, step_size=step_size),
        final_equity=float(eq[-1]) if eq.size > 0 else 1.0,
        avg_turnover=float(results["turnover_oneway"].mean()) if results["turnover_oneway"].size > 0 else 0.0,
        avg_tc_cost=float(results["tc_costs"].mean()) if results["tc_costs"].size > 0 else 0.0,
        num_steps=len(net),
        equity=eq,
        backtest_results=results,
    )


def aggregate_results(results: List[EvalResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate evaluation results across seeds."""
    metrics = {
        "sharpe": [r.sharpe for r in results],
        "cagr": [r.cagr for r in results],
        "max_dd": [r.max_dd for r in results],
        "ann_vol": [r.ann_vol for r in results],
        "final_equity": [r.final_equity for r in results],
        "avg_turnover": [r.avg_turnover for r in results],
        "avg_tc_cost": [r.avg_tc_cost for r in results],
    }

    aggregated = {}
    for name, values in metrics.items():
        arr = np.array(values)
        aggregated[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    return aggregated


def print_results_table(results: List[EvalResult], aggregated: Dict) -> None:
    """Print formatted results table."""

    print_header("MULTI-SEED EVALUATION RESULTS")

    # Per-seed results
    print("\nPer-Seed Results:")
    print("-" * 115)
    print(f"{'Seed':>8} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10} {'AnnVol':>10} {'FinalEq':>10} {'AccumRet':>12} {'Turnover':>10}")
    print("-" * 115)

    for r in sorted(results, key=lambda x: x.seed):
        accum_equity_pct = r.final_equity * 100  # Total value as percentage of initial (1.0 = 100%)
        print(f"{r.seed:>8} {r.sharpe:>10.3f} {r.cagr:>10.2%} {r.max_dd:>10.2%} "
              f"{r.ann_vol:>10.2%} {r.final_equity:>10.3f} {accum_equity_pct:>11.2f}% {r.avg_turnover:>10.4f}")

    print("-" * 115)

    # Aggregated results
    print("\nAggregated Results:")
    print("-" * 70)

    metrics_fmt = [
        ("Sharpe Ratio", "sharpe", "{:.3f}"),
        ("CAGR", "cagr", "{:.2%}"),
        ("Max Drawdown", "max_dd", "{:.2%}"),
        ("Annualized Vol", "ann_vol", "{:.2%}"),
        ("Final Equity", "final_equity", "{:.3f}"),
        ("Avg Turnover", "avg_turnover", "{:.4f}"),
        ("Avg TC Cost", "avg_tc_cost", "{:.6f}"),
    ]

    print(f"{'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)

    for label, key, fmt in metrics_fmt:
        m = aggregated[key]
        print(f"{label:<20} {fmt.format(m['mean']):>12} {fmt.format(m['std']):>12} "
              f"{fmt.format(m['min']):>12} {fmt.format(m['max']):>12}")

    print("-" * 70)

    # Best and worst
    best = max(results, key=lambda r: r.sharpe)
    worst = min(results, key=lambda r: r.sharpe)

    print(f"\nBest Sharpe:  Seed {best.seed} = {best.sharpe:.3f} (CAGR: {best.cagr:.2%}, MaxDD: {best.max_dd:.2%})")
    print(f"Worst Sharpe: Seed {worst.seed} = {worst.sharpe:.3f} (CAGR: {worst.cagr:.2%}, MaxDD: {worst.max_dd:.2%})")

    # Consistency assessment
    sharpe_cv = aggregated["sharpe"]["std"] / abs(aggregated["sharpe"]["mean"]) if aggregated["sharpe"]["mean"] != 0 else float('inf')

    print(f"\nConsistency Analysis:")
    print(f"  Sharpe CV (Std/Mean): {sharpe_cv:.2f}")

    if sharpe_cv < 0.15:
        print("  Assessment: EXCELLENT consistency - Very robust across seeds")
    elif sharpe_cv < 0.3:
        print("  Assessment: GOOD consistency - Reliable performance")
    elif sharpe_cv < 0.5:
        print("  Assessment: MODERATE consistency - Some seed sensitivity")
    else:
        print("  Assessment: HIGH variance - Results vary significantly by seed")

    # Statistical significance
    if len(results) >= 3:
        sharpe_mean = aggregated["sharpe"]["mean"]
        sharpe_se = aggregated["sharpe"]["std"] / np.sqrt(len(results))
        t_stat = sharpe_mean / sharpe_se if sharpe_se > 0 else float('inf')
        print(f"\n  Mean Sharpe: {sharpe_mean:.3f} +/- {sharpe_se:.3f} (SE)")
        print(f"  T-statistic vs 0: {t_stat:.2f}")
        if t_stat > 2.0:
            print("  Significance: Sharpe is statistically > 0 (p < 0.05)")
        elif t_stat > 1.5:
            print("  Significance: Sharpe is marginally > 0 (p < 0.10)")
        else:
            print("  Significance: Cannot reject Sharpe = 0")


def save_results_csv(results: List[EvalResult], aggregated: Dict, output_dir: str) -> str:
    """Save evaluation results to CSV."""

    rows = []
    for r in results:
        rows.append({
            "seed": r.seed,
            "model_path": r.model_path,
            "sharpe": r.sharpe,
            "cagr": r.cagr,
            "max_dd": r.max_dd,
            "ann_vol": r.ann_vol,
            "final_equity": r.final_equity,
            "avg_turnover": r.avg_turnover,
            "avg_tc_cost": r.avg_tc_cost,
            "num_steps": r.num_steps,
        })

    # Add summary rows
    rows.append({
        "seed": "MEAN",
        "sharpe": aggregated["sharpe"]["mean"],
        "cagr": aggregated["cagr"]["mean"],
        "max_dd": aggregated["max_dd"]["mean"],
        "ann_vol": aggregated["ann_vol"]["mean"],
        "final_equity": aggregated["final_equity"]["mean"],
        "avg_turnover": aggregated["avg_turnover"]["mean"],
        "avg_tc_cost": aggregated["avg_tc_cost"]["mean"],
    })
    rows.append({
        "seed": "STD",
        "sharpe": aggregated["sharpe"]["std"],
        "cagr": aggregated["cagr"]["std"],
        "max_dd": aggregated["max_dd"]["std"],
        "ann_vol": aggregated["ann_vol"]["std"],
        "final_equity": aggregated["final_equity"]["std"],
        "avg_turnover": aggregated["avg_turnover"]["std"],
        "avg_tc_cost": aggregated["avg_tc_cost"]["std"],
    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


def save_equity_curves_csv(results: List[EvalResult], output_dir: str) -> str:
    """Save equity curves for all seeds to a single CSV file.

    Creates a CSV with columns: step, seed_X, seed_Y, ..., mean, std
    This allows for easy plotting and analysis of equity progression.
    """
    # Find max length across all equity curves
    max_len = max(len(r.equity) for r in results if r.equity is not None)

    # Build DataFrame with step index
    data = {"step": np.arange(max_len)}

    # Add each seed's equity curve (pad shorter ones with NaN)
    equity_arrays = []
    for r in results:
        if r.equity is not None:
            col_name = f"seed_{r.seed}"
            padded = np.full(max_len, np.nan)
            padded[:len(r.equity)] = r.equity
            data[col_name] = padded
            equity_arrays.append(padded)

    # Compute mean and std across seeds
    if equity_arrays:
        arr = np.array(equity_arrays)
        data["mean"] = np.nanmean(arr, axis=0)
        data["std"] = np.nanstd(arr, axis=0)

    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "equity_curves.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


def generate_multiseed_eval_plots(
    results: List[EvalResult],
    aggregated: Dict,
    cfg: Config,
    output_dir: str,
    per_seed_plots: bool = False,
    dates: Optional[pd.DatetimeIndex] = None,
    regime_probs: Optional[np.ndarray] = None,
) -> List[str]:
    """Generate thesis-quality multi-seed evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # 1. Multi-seed equity comparison
    equity_dict = {r.seed: r.equity for r in results if r.equity is not None}
    if equity_dict:
        path = os.path.join(output_dir, "multiseed_equity_comparison.png")
        plot_multiseed_equity_comparison(equity_dict, path)
        saved_paths.append(path)

    # 2. Multi-seed metrics summary
    metrics_df = pd.DataFrame([
        {
            "seed": r.seed,
            "sharpe": r.sharpe,
            "cagr": r.cagr,
            "max_dd": r.max_dd,
            "ann_vol": r.ann_vol,
        }
        for r in results
    ])
    if not metrics_df.empty:
        path = os.path.join(output_dir, "multiseed_metrics_summary.png")
        plot_multiseed_metrics_summary(metrics_df, path)
        saved_paths.append(path)

    # 3. Per-seed detailed plots (optional)
    if per_seed_plots:
        for r in results:
            if r.backtest_results is not None:
                seed_dir = os.path.join(output_dir, f"seed_{r.seed}")
                os.makedirs(seed_dir, exist_ok=True)

                # Align dates and regime_probs with backtest results
                net_returns = r.backtest_results["net_returns"]
                plot_dates = dates[1:] if dates is not None and len(dates) > len(net_returns) else dates
                plot_regime = regime_probs[1:] if regime_probs is not None and len(regime_probs) > len(net_returns) else regime_probs

                seed_paths = generate_evaluation_plots(
                    results=r.backtest_results,
                    cfg=cfg,
                    out_dir=seed_dir,
                    dates=plot_dates,
                    regime_probs=plot_regime,
                )
                saved_paths.extend(seed_paths)

    return saved_paths


def find_models_in_run_dir(run_dir: str) -> List[str]:
    """Find all best model files in a multi-seed run directory.

    Searches for model files in priority order to avoid duplicates:
    1. sac_best.pth (preferred)
    2. *_best.pth (fallback)
    3. sac_final.pth (if no best models found)
    4. *_final.pth (last resort)
    """
    # Try specific pattern first (sac_best.pth)
    pattern = os.path.join(run_dir, "seed_*", "sac_best.pth")
    model_paths = glob(pattern)

    # If no sac_best.pth found, try wildcard *_best.pth
    if not model_paths:
        pattern = os.path.join(run_dir, "seed_*", "*_best.pth")
        model_paths = glob(pattern)

    # If still nothing, try final models
    if not model_paths:
        pattern = os.path.join(run_dir, "seed_*", "sac_final.pth")
        model_paths = glob(pattern)

    if not model_paths:
        pattern = os.path.join(run_dir, "seed_*", "*_final.pth")
        model_paths = glob(pattern)

    # Normalize paths for consistent deduplication (important on Windows)
    normalized = [os.path.normpath(p) for p in model_paths]
    return sorted(set(normalized))


def main():
    parser = argparse.ArgumentParser(description="Multi-seed SAC Evaluation")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Directory containing multi-seed training results")
    parser.add_argument("--model_paths", type=str, nargs="+", default=None,
                        help="Explicit list of model paths to evaluate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results")
    parser.add_argument("--save_plots", action="store_true", default=True,
                        help="Save thesis-quality comparison plots (default: True)")
    parser.add_argument("--per_seed_plots", action="store_true",
                        help="Generate detailed plots for each seed")
    args = parser.parse_args()

    # Determine model paths
    if args.model_paths:
        model_paths = args.model_paths
    elif args.run_dir:
        model_paths = find_models_in_run_dir(args.run_dir)
        if not model_paths:
            raise ValueError(f"No models found in {args.run_dir}")
    else:
        raise ValueError("Must provide either --run_dir or --model_paths")

    print_header("MULTI-SEED SAC EVALUATION")
    print(f"\nModels to evaluate: {len(model_paths)}")
    for p in model_paths:
        print(f"  - {p}")

    # Setup - load config from first model's directory (preserves training settings)
    cfg = get_default_config()
    first_config_path = os.path.join(os.path.dirname(model_paths[0]), "config.json")
    if os.path.exists(first_config_path):
        cfg = Config.load_json(first_config_path)
        print(f"\nLoaded config from: {first_config_path}")
    else:
        print(f"\nWarning: No config.json found at {first_config_path}")
        print("  Using default config (HMM disabled by default)")

    print(f"HMM Regime Features: {'ENABLED' if cfg.features.use_regime_hmm else 'DISABLED'}")

    device = cfg.auto_detect_device()
    print(f"Device: {device}")

    # Load test data
    print("\nLoading test data...")
    _df_train, df_test, _feature_cols = load_and_prepare_data(cfg)
    print(f"  Test rows: {len(df_test)}")

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
            print(f"  Regime probabilities available ({len(pcols)} states)")

    # Evaluate each model
    print_header("EVALUATING MODELS")

    results: List[EvalResult] = []
    for i, model_path in enumerate(model_paths):
        print(f"\n[{i+1}/{len(model_paths)}] Evaluating: {os.path.basename(os.path.dirname(model_path))}")

        result = evaluate_single_model(model_path, cfg, df_test, device)
        results.append(result)

        print(f"  Sharpe: {result.sharpe:.3f}, CAGR: {result.cagr:.2%}, MaxDD: {result.max_dd:.2%}")

    # Aggregate and display
    aggregated = aggregate_results(results)
    print_results_table(results, aggregated)

    # Save results
    output_dir = args.output_dir or args.run_dir or "eval_outputs"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = save_results_csv(results, aggregated, output_dir)
    print(f"\nResults saved to: {csv_path}")

    # Save equity curves
    equity_csv_path = save_equity_curves_csv(results, output_dir)
    print(f"Equity curves saved to: {equity_csv_path}")

    # Generate thesis-quality plots
    if args.save_plots:
        print_header("GENERATING PLOTS")

        saved_paths = generate_multiseed_eval_plots(
            results=results,
            aggregated=aggregated,
            cfg=cfg,
            output_dir=output_dir,
            per_seed_plots=args.per_seed_plots,
            dates=dates,
            regime_probs=regime_probs,
        )

        print(f"\nSaved {len(saved_paths)} plots to: {output_dir}")
        for p in saved_paths[:5]:  # Show first 5
            print(f"  - {os.path.basename(p)}")
        if len(saved_paths) > 5:
            print(f"  ... and {len(saved_paths) - 5} more")

    # Summary
    print_header("EVALUATION COMPLETE")

    best = max(results, key=lambda r: r.sharpe)
    print(f"\nRecommended model for deployment:")
    print(f"  Seed: {best.seed}")
    print(f"  Path: {best.model_path}")
    print(f"  Sharpe: {best.sharpe:.3f}")
    print(f"  CAGR: {best.cagr:.2%}")
    print(f"  MaxDD: {best.max_dd:.2%}")


if __name__ == "__main__":
    main()
