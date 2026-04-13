
from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAC_DIR = os.path.join(PROJECT_ROOT, "algorithms", "SAC")
sys.path.insert(0, SAC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "testing"))

from config import Config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent
from evaluate import run_backtest, sharpe, cagr, max_drawdown, ann_vol
from analysis.plotting import (
    plot_equity_curve,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_returns_distribution,
    plot_weights_evolution,
    plot_average_weights,
    plot_turnover,
    plot_regime_returns,
    plot_regime_timeline,
    plot_monthly_returns_heatmap,
)

from evaluate_clusters import (
    print_header, find_cluster_models, extract_seed,
    load_cluster_data,
)


# ---------------------------------------------------------------------------
# Per-seed detailed analysis
# ---------------------------------------------------------------------------

def analyze_seed(
    model_path: str,
    cfg: Config,
    df_test: pd.DataFrame,
    device,
    cluster_name: str,
    output_dir: str,
) -> dict:
    """Run full analysis for one seed and generate all plots."""
    seed = extract_seed(model_path)
    tickers = list(cfg.data.tickers)
    labels = tickers + ["CASH"]

    # Run backtest
    env = Env(df_test, tickers, cfg)
    agent = Agent(env.get_state_dim(), env.get_action_dim(), cfg, device=device)
    agent.load_model(model_path)
    results = run_backtest(env, agent, deterministic=True)

    eq = results["equity"]
    net = results["net_returns"]
    weights = results["weights"]
    step_size = cfg.env.lag

    metrics = {
        "cluster": cluster_name,
        "seed": seed,
        "sharpe": sharpe(net, step_size=step_size),
        "cagr": cagr(eq, step_size=step_size),
        "max_dd": max_drawdown(eq),
        "ann_vol": ann_vol(net, step_size=step_size),
        "final_equity": float(eq[-1]) if eq.size > 0 else 1.0,
        "avg_turnover": float(results["turnover_oneway"].mean()),
    }
    metrics["calmar"] = abs(metrics["cagr"] / metrics["max_dd"]) if metrics["max_dd"] != 0 else 0.0

    # Create output directory
    seed_dir = os.path.join(output_dir, cluster_name.replace(" ", "_").replace("(", "").replace(")", ""), f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    saved_plots = []

    # 1. Equity curve
    path = os.path.join(seed_dir, "equity_curve.png")
    plot_equity_curve(eq, path, title=f"{cluster_name} Seed {seed} — Equity Curve")
    saved_plots.append(path)

    # 2. Drawdown
    path = os.path.join(seed_dir, "drawdown.png")
    plot_drawdown(eq, path, title=f"{cluster_name} Seed {seed} — Drawdown")
    saved_plots.append(path)

    # 3. Rolling Sharpe
    window = min(60, len(net) // 4)
    if window >= 5:
        path = os.path.join(seed_dir, "rolling_sharpe.png")
        plot_rolling_sharpe(net, path, window=window, title=f"{cluster_name} Seed {seed} — Rolling Sharpe")
        saved_plots.append(path)

    # 4. Returns distribution
    path = os.path.join(seed_dir, "returns_distribution.png")
    plot_returns_distribution(net, path, title=f"{cluster_name} Seed {seed} — Returns Distribution")
    saved_plots.append(path)

    # 5. Weight evolution
    if weights.shape[0] > 0:
        path = os.path.join(seed_dir, "weight_evolution.png")
        plot_weights_evolution(weights, path, labels, title=f"{cluster_name} Seed {seed} — Weight Evolution")
        saved_plots.append(path)

    # 6. Average weights
    if weights.shape[0] > 0:
        path = os.path.join(seed_dir, "average_weights.png")
        plot_average_weights(weights, path, labels, title=f"{cluster_name} Seed {seed} — Average Weights")
        saved_plots.append(path)

    # 7. Turnover
    path = os.path.join(seed_dir, "turnover.png")
    plot_turnover(results["turnover_total"], results["turnover_oneway"], path,
                  title=f"{cluster_name} Seed {seed} — Turnover")
    saved_plots.append(path)

    # 8-9. Regime analysis (only if HMM is enabled and regime probs available)
    regime_probs = None
    if cfg.features.use_regime_hmm:
        pcols = cfg.features.regime_prob_columns
        if all(c in df_test.columns for c in pcols):
            regime_probs = df_test[pcols].values
            # Align: backtest skips the first observation, so regime_probs may need trimming
            if len(regime_probs) > len(net):
                regime_probs = regime_probs[1:]
            if len(regime_probs) > len(net):
                regime_probs = regime_probs[:len(net)]

    if regime_probs is not None and len(regime_probs) == len(net):
        path = os.path.join(seed_dir, "regime_returns.png")
        plot_regime_returns(net, regime_probs, path,
                          title=f"{cluster_name} Seed {seed} — Returns by Regime")
        saved_plots.append(path)

        path = os.path.join(seed_dir, "regime_timeline.png")
        plot_regime_timeline(regime_probs, eq, path,
                           title=f"{cluster_name} Seed {seed} — Regime Timeline")
        saved_plots.append(path)

    # 10. Monthly heatmap (if date index available)
    dates = None
    if isinstance(df_test.index, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(df_test.index)
        if len(dates) > len(net):
            dates = dates[1:]
        if len(dates) > len(net):
            dates = dates[:len(net)]

    if dates is not None and len(dates) == len(net):
        try:
            path = os.path.join(seed_dir, "monthly_returns_heatmap.png")
            plot_monthly_returns_heatmap(net, dates, path,
                                       title=f"{cluster_name} Seed {seed} — Monthly Returns")
            saved_plots.append(path)
        except Exception as e:
            print(f"    Warning: Monthly heatmap failed: {e}")

    metrics["n_plots"] = len(saved_plots)
    metrics["plot_dir"] = seed_dir

    return metrics


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def print_ablation_table(all_metrics: List[dict]) -> None:
    """Print ablation comparison across clusters."""
    if len(all_metrics) < 2:
        return

    print_header("ABLATION STUDY")

    # Group by cluster
    clusters = {}
    for m in all_metrics:
        cn = m["cluster"]
        if cn not in clusters:
            clusters[cn] = []
        clusters[cn].append(m)

    # Print per-cluster stats
    print(f"\n{'Cluster':<22} {'Sharpe':>12} {'CAGR':>12} {'MaxDD':>12} {'AnnVol':>12} {'Calmar':>12} {'Turnover':>12}")
    print("-" * 100)

    cluster_stats = {}
    for cn, mets in sorted(clusters.items()):
        s = np.array([m["sharpe"] for m in mets])
        c = np.array([m["cagr"] for m in mets])
        d = np.array([m["max_dd"] for m in mets])
        v = np.array([m["ann_vol"] for m in mets])
        cal = np.array([m["calmar"] for m in mets])
        t = np.array([m["avg_turnover"] for m in mets])

        cluster_stats[cn] = {"sharpe": s, "cagr": c, "max_dd": d, "ann_vol": v, "calmar": cal, "turnover": t}

        def f(arr):
            return f"{arr.mean():.3f}+/-{arr.std(ddof=1):.3f}" if len(arr) > 1 else f"{arr[0]:.3f}"
        def fp(arr):
            return f"{arr.mean():.2%}+/-{arr.std(ddof=1):.2%}" if len(arr) > 1 else f"{arr[0]:.2%}"

        print(f"{cn:<22} {f(s):>12} {fp(c):>12} {fp(d):>12} {fp(v):>12} {f(cal):>12} {fp(t):>12}")

    print("-" * 100)

    # Pairwise deltas
    cluster_names = sorted(clusters.keys())
    if len(cluster_names) >= 2:
        base = cluster_names[0]
        for other in cluster_names[1:]:
            bs = cluster_stats[base]
            os_ = cluster_stats[other]
            print(f"\nDelta ({base} - {other}):")
            for metric in ["sharpe", "cagr", "max_dd", "ann_vol", "turnover"]:
                delta = bs[metric].mean() - os_[metric].mean()
                sign = "+" if delta > 0 else ""
                if metric in ["cagr", "max_dd", "ann_vol", "turnover"]:
                    print(f"  {metric:<15}: {sign}{delta:.2%}")
                else:
                    print(f"  {metric:<15}: {sign}{delta:.3f}")


def save_ablation_csv(all_metrics: List[dict], output_dir: str) -> str:
    """Save ablation results to CSV."""
    rows = []
    for m in sorted(all_metrics, key=lambda x: (x["cluster"], x["seed"])):
        rows.append({
            "cluster": m["cluster"],
            "seed": m["seed"],
            "sharpe": m["sharpe"],
            "cagr": m["cagr"],
            "max_dd": m["max_dd"],
            "ann_vol": m["ann_vol"],
            "calmar": m["calmar"],
            "final_equity": m["final_equity"],
            "avg_turnover": m["avg_turnover"],
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "ablation_results.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CLUSTER_NAMES = {1: "Cluster 1 (HMM)", 2: "Cluster 2 (Base)", 3: "Cluster 3"}


def main():
    parser = argparse.ArgumentParser(description="Detailed per-seed analysis and ablation")
    parser.add_argument("--clusters", type=int, nargs="+", default=[1, 2, 3],
                        help="Clusters to analyze (default: 1 2 3)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Specific seeds to analyze (default: best per cluster)")
    parser.add_argument("--all_seeds", action="store_true",
                        help="Analyze all seeds (not just best)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "testing", "eval_results", "detailed")
    os.makedirs(output_dir, exist_ok=True)
    models_root = os.path.join(PROJECT_ROOT, "algorithms", "SAC", "models")

    print_header("DETAILED SAC ANALYSIS")

    all_metrics: List[dict] = []

    for cluster_id in args.clusters:
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        cluster_dir = os.path.join(models_root, f"cluster_{cluster_id}")

        if not os.path.isdir(cluster_dir):
            print(f"\nSkipping {cluster_name}: not found")
            continue

        model_paths = find_cluster_models(cluster_dir)
        if not model_paths:
            print(f"\nSkipping {cluster_name}: no models")
            continue

        print_header(f"ANALYZING {cluster_name.upper()}")

        # Load data
        cfg, df_train, df_test = load_cluster_data(cluster_dir)
        device = cfg.auto_detect_device()
        tickers = list(cfg.data.tickers)

        print(f"  Test rows: {len(df_test)}, HMM: {'ON' if cfg.features.use_regime_hmm else 'OFF'}")

        # Determine which seeds to analyze
        if args.seeds:
            # User specified seeds
            target_seeds = set(args.seeds)
            selected = [mp for mp in model_paths if extract_seed(mp) in target_seeds]
        elif args.all_seeds:
            selected = model_paths
        else:
            # Find best seed by running quick backtest on all
            print("  Finding best seed...")
            best_sharpe = -np.inf
            best_path = model_paths[0]
            for mp in model_paths:
                env = Env(df_test, tickers, cfg)
                agent = Agent(env.get_state_dim(), env.get_action_dim(), cfg, device=device)
                agent.load_model(mp)
                res = run_backtest(env, agent, deterministic=True)
                s = sharpe(res["net_returns"], step_size=cfg.env.lag)
                seed = extract_seed(mp)
                print(f"    Seed {seed}: Sharpe={s:.3f}")
                if s > best_sharpe:
                    best_sharpe = s
                    best_path = mp
            selected = [best_path]
            print(f"  Best seed: {extract_seed(best_path)} (Sharpe={best_sharpe:.3f})")

        # Analyze selected seeds
        for mp in selected:
            seed = extract_seed(mp)
            print(f"\n  Analyzing seed {seed}...")

            metrics = analyze_seed(mp, cfg, df_test, device, cluster_name, output_dir)
            all_metrics.append(metrics)

            print(f"    Sharpe={metrics['sharpe']:.3f}  CAGR={metrics['cagr']:.2%}  "
                  f"MaxDD={metrics['max_dd']:.2%}")
            print(f"    Generated {metrics['n_plots']} plots -> {metrics['plot_dir']}")

    # Ablation study (if multiple clusters analyzed with all seeds)
    if args.all_seeds and len(args.clusters) >= 2:
        print_ablation_table(all_metrics)
        abl_path = save_ablation_csv(all_metrics, output_dir)
        print(f"\nAblation CSV: {abl_path}")

    print_header("ANALYSIS COMPLETE")
    print(f"\nOutput: {output_dir}")
    print(f"Total models analyzed: {len(all_metrics)}")
    total_plots = sum(m.get("n_plots", 0) for m in all_metrics)
    print(f"Total plots generated: {total_plots}")


if __name__ == "__main__":
    main()
