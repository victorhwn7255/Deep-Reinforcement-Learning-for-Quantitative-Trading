
from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from typing import Dict, List, Optional

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

from benchmarks import create_benchmarks
from evaluate_clusters import (
    print_header, find_cluster_models, extract_seed,
    load_cluster_data, _verify_config_matches_model,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(env, agent, step_size: int, name: str, seed: int = 0):
    """Run backtest and compute metrics."""
    results = run_backtest(env, agent, deterministic=True)
    eq = results["equity"]
    net = results["net_returns"]
    to = results["turnover_oneway"]
    tc = results["tc_costs"]

    return {
        "name": name,
        "seed": seed,
        "sharpe": sharpe(net, step_size=step_size),
        "cagr": cagr(eq, step_size=step_size),
        "max_dd": max_drawdown(eq),
        "ann_vol": ann_vol(net, step_size=step_size),
        "calmar": abs(cagr(eq, step_size=step_size) / max_drawdown(eq)) if max_drawdown(eq) != 0 else 0.0,
        "final_equity": float(eq[-1]) if eq.size > 0 else 1.0,
        "avg_turnover": float(to.mean()) if to.size > 0 else 0.0,
        "avg_tc_cost": float(tc.mean()) if tc.size > 0 else 0.0,
        "equity": eq,
        "net_returns": net,
        "weights": results["weights"],
    }


def print_comparison_table(
    sac_results_by_cluster: Dict[str, List[dict]],
    benchmark_results: List[dict],
) -> None:
    """Print Table 5.1 style comparison."""
    print("\n" + "-" * 115)
    print(f"{'Strategy':<22} {'Sharpe':>12} {'CAGR':>12} {'MaxDD':>12} "
          f"{'AnnVol':>12} {'Calmar':>12} {'FinalEq':>12} {'Turnover':>12}")
    print("-" * 115)

    # SAC clusters (mean +/- std)
    for cluster_name, results in sac_results_by_cluster.items():
        vals = {k: np.array([r[k] for r in results]) for k in
                ["sharpe", "cagr", "max_dd", "ann_vol", "calmar", "final_equity", "avg_turnover"]}

        def fmt(arr):
            return f"{arr.mean():.2f}+/-{arr.std(ddof=1):.2f}" if len(arr) > 1 else f"{arr[0]:.3f}"

        def fmt_pct(arr):
            return f"{arr.mean():.1%}+/-{arr.std(ddof=1):.1%}" if len(arr) > 1 else f"{arr[0]:.2%}"

        print(f"{cluster_name:<22} "
              f"{fmt(vals['sharpe']):>12} "
              f"{fmt_pct(vals['cagr']):>12} "
              f"{fmt_pct(vals['max_dd']):>12} "
              f"{fmt_pct(vals['ann_vol']):>12} "
              f"{fmt(vals['calmar']):>12} "
              f"${vals['final_equity'].mean():.2f}{'':>6} "
              f"{fmt_pct(vals['avg_turnover']):>12}")

    print("-" * 115)

    # Benchmarks
    for r in benchmark_results:
        print(f"{r['name']:<22} "
              f"{r['sharpe']:>12.3f} "
              f"{r['cagr']:>12.2%} "
              f"{r['max_dd']:>12.2%} "
              f"{r['ann_vol']:>12.2%} "
              f"{r['calmar']:>12.3f} "
              f"${r['final_equity']:>11.2f} "
              f"{r['avg_turnover']:>12.4f}")

    print("-" * 115)


def save_comparison_csv(
    sac_results_by_cluster: Dict[str, List[dict]],
    benchmark_results: List[dict],
    output_dir: str,
) -> str:
    """Save comparison to CSV."""
    rows = []

    for cluster_name, results in sac_results_by_cluster.items():
        for r in results:
            rows.append({
                "strategy": cluster_name,
                "seed": r["seed"],
                "sharpe": r["sharpe"],
                "cagr": r["cagr"],
                "max_dd": r["max_dd"],
                "ann_vol": r["ann_vol"],
                "calmar": r["calmar"],
                "final_equity": r["final_equity"],
                "avg_turnover": r["avg_turnover"],
            })

        # Mean row
        vals = {k: np.mean([r[k] for r in results]) for k in
                ["sharpe", "cagr", "max_dd", "ann_vol", "calmar", "final_equity", "avg_turnover"]}
        rows.append({"strategy": cluster_name, "seed": "MEAN", **vals})

    for r in benchmark_results:
        rows.append({
            "strategy": r["name"],
            "seed": "-",
            "sharpe": r["sharpe"],
            "cagr": r["cagr"],
            "max_dd": r["max_dd"],
            "ann_vol": r["ann_vol"],
            "calmar": r["calmar"],
            "final_equity": r["final_equity"],
            "avg_turnover": r["avg_turnover"],
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "sac_vs_benchmarks.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_benchmark_plots(
    sac_results_by_cluster: Dict[str, List[dict]],
    benchmark_results: List[dict],
    output_dir: str,
) -> List[str]:
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []

    cluster_colors = {
        "Cluster 1 (HMM)": "#2563eb",
        "Cluster 2 (Base)": "#dc2626",
        "Cluster 3": "#16a34a",
    }
    bm_colors = {
        "SPY Buy-and-Hold": "#111827",
        "Equal-Weight": "#6C757D",
        "Fixed-Weight": "#9CA3AF",
        "Mean-Variance": "#D97706",
    }

    # --- 1. Equity curves: SAC mean+band vs benchmarks ---
    fig, ax = plt.subplots(figsize=(12, 6))

    for cluster_name, results in sac_results_by_cluster.items():
        color = cluster_colors.get(cluster_name, "blue")
        equities = [r["equity"] for r in results]
        max_len = max(len(e) for e in equities)
        padded = [np.pad(e, (0, max_len - len(e)), constant_values=np.nan) for e in equities]
        mean_eq = np.nanmean(padded, axis=0)
        std_eq = np.nanstd(padded, axis=0)

        steps = np.arange(max_len)
        ax.plot(steps, mean_eq, color=color, linewidth=2.5, label=cluster_name)
        ax.fill_between(steps, mean_eq - std_eq, mean_eq + std_eq, color=color, alpha=0.15)

    for r in benchmark_results:
        color = bm_colors.get(r["name"], "gray")
        ax.plot(r["equity"], color=color, linewidth=1.5, linestyle="--", label=r["name"])

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity ($1 start)")
    ax.set_title("SAC Agents vs Benchmark Strategies")
    ax.legend(loc="upper left", fontsize=8)
    path = os.path.join(output_dir, "equity_sac_vs_benchmarks.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # --- 2. Grouped bar chart ---
    metrics = ["sharpe", "cagr", "max_dd"]
    titles = ["Sharpe Ratio", "CAGR", "Max Drawdown"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    all_names = list(sac_results_by_cluster.keys()) + [r["name"] for r in benchmark_results]
    all_vals = {}
    for metric in metrics:
        vals = []
        for cluster_name, results in sac_results_by_cluster.items():
            vals.append(np.mean([r[metric] for r in results]))
        for r in benchmark_results:
            vals.append(r[metric])
        all_vals[metric] = vals

    all_colors = [cluster_colors.get(n, "blue") for n in sac_results_by_cluster.keys()]
    all_colors += [bm_colors.get(r["name"], "gray") for r in benchmark_results]

    for ax, metric, title in zip(axes, metrics, titles):
        vals = all_vals[metric]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=all_colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in all_names], fontsize=7, rotation=0)
        ax.set_title(title)
        if metric in ["cagr", "max_dd"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

    fig.suptitle("Performance Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "metrics_sac_vs_benchmarks.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # --- 3. Risk-return scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster_name, results in sac_results_by_cluster.items():
        color = cluster_colors.get(cluster_name, "blue")
        vols = [r["ann_vol"] for r in results]
        cagrs = [r["cagr"] for r in results]
        ax.scatter(vols, cagrs, color=color, s=60, alpha=0.7, label=cluster_name, zorder=3)

    for r in benchmark_results:
        color = bm_colors.get(r["name"], "gray")
        ax.scatter(r["ann_vol"], r["cagr"], color=color, s=100, marker="D",
                   edgecolors="black", linewidths=0.5, label=r["name"], zorder=4)

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("CAGR")
    ax.set_title("Risk-Return: SAC Seeds vs Benchmarks")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "risk_return_sac_vs_benchmarks.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CLUSTER_NAMES = {1: "Cluster 1 (HMM)", 2: "Cluster 2 (Base)", 3: "Cluster 3"}


def main():
    parser = argparse.ArgumentParser(description="SAC vs Benchmarks evaluation")
    parser.add_argument("--clusters", type=int, nargs="+", default=[1, 3],
                        help="Clusters to evaluate (default: 1 3)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "testing", "eval_results", "benchmarks")
    os.makedirs(output_dir, exist_ok=True)
    models_root = os.path.join(PROJECT_ROOT, "algorithms", "SAC", "models")

    print_header("SAC vs BENCHMARKS EVALUATION")

    sac_results_by_cluster: Dict[str, List[dict]] = {}
    benchmark_results: List[dict] = []
    benchmarks_run = False

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

        print_header(f"EVALUATING {cluster_name.upper()}")

        # Load data
        cfg, df_train, df_test = load_cluster_data(cluster_dir)
        device = cfg.auto_detect_device()
        tickers = list(cfg.data.tickers)
        step_size = cfg.env.lag

        # --- Evaluate SAC seeds ---
        cluster_results = []
        for mp in model_paths:
            seed = extract_seed(mp)
            env = Env(df_test, tickers, cfg)
            agent = Agent(env.get_state_dim(), env.get_action_dim(), cfg, device=device)
            agent.load_model(mp)
            r = evaluate_agent(env, agent, step_size, cluster_name, seed)
            cluster_results.append(r)
            print(f"  Seed {seed}: Sharpe={r['sharpe']:.3f}  CAGR={r['cagr']:.2%}")

        sac_results_by_cluster[cluster_name] = cluster_results

        # --- Evaluate benchmarks (once, using the first cluster's data) ---
        if not benchmarks_run:
            print_header("EVALUATING BENCHMARKS")

            # Get full price data for Mean-Variance lookback
            prices_train = df_train[tickers].values
            prices_test = df_test[tickers].values
            prices_full = np.vstack([prices_train, prices_test])

            bm_agents = create_benchmarks(
                prices_test=prices_test,
                prices_full=prices_full,
                tickers=tickers,
                lag=cfg.env.lag,
                n_positions=len(tickers) + 1,
            )

            for bm_name, bm_agent in bm_agents.items():
                env = Env(df_test, tickers, cfg)
                r = evaluate_agent(env, bm_agent, step_size, bm_name)
                benchmark_results.append(r)
                print(f"  {bm_name}: Sharpe={r['sharpe']:.3f}  CAGR={r['cagr']:.2%}")

            benchmarks_run = True

    # --- Results ---
    print_header("RESULTS: SAC vs BENCHMARKS")
    print_comparison_table(sac_results_by_cluster, benchmark_results)

    csv_path = save_comparison_csv(sac_results_by_cluster, benchmark_results, output_dir)
    print(f"\nCSV: {csv_path}")

    # Plots
    print_header("GENERATING PLOTS")
    saved = generate_benchmark_plots(sac_results_by_cluster, benchmark_results, output_dir)
    for p in saved:
        print(f"  Saved: {os.path.basename(p)}")

    print_header("COMPLETE")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
