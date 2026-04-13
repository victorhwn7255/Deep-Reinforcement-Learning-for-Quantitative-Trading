
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path setup: import from algorithms/SAC without modifying it
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAC_DIR = os.path.join(PROJECT_ROOT, "algorithms", "SAC")
sys.path.insert(0, SAC_DIR)

from config import Config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent
from evaluate import run_backtest, sharpe, cagr, max_drawdown, ann_vol


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SeedResult:
    """Evaluation result for a single model."""
    cluster: str
    seed: int
    model_path: str
    sharpe: float
    cagr: float
    max_dd: float
    ann_vol: float
    calmar: float
    final_equity: float
    avg_turnover: float
    avg_tc_cost: float
    num_steps: int
    equity: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ClusterResult:
    """Aggregated results for one cluster."""
    cluster_name: str
    seed_results: List[SeedResult]
    mean_sharpe: float
    std_sharpe: float
    mean_cagr: float
    std_cagr: float
    mean_max_dd: float
    std_max_dd: float
    mean_ann_vol: float
    std_ann_vol: float
    mean_calmar: float
    std_calmar: float
    mean_final_equity: float
    std_final_equity: float
    mean_turnover: float
    std_turnover: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str, char: str = "=", width: int = 80) -> None:
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def find_cluster_models(cluster_dir: str) -> List[str]:
    """Find all sac_best.pth files in a cluster directory."""
    pattern = os.path.join(cluster_dir, "seed_*", "sac_best.pth")
    paths = sorted(glob(pattern))
    return [os.path.normpath(p) for p in paths]


def extract_seed(model_path: str) -> int:
    """Extract seed from path like .../seed_42/sac_best.pth"""
    import re
    parent = os.path.basename(os.path.dirname(model_path))
    match = re.match(r"seed_(\d+)", parent)
    return int(match.group(1)) if match else 0


def _verify_config_matches_model(cfg: Config, cluster_dir: str) -> Config:
    """Verify that the config produces the correct state_dim for the saved models.

    Cluster 2's saved configs are stale (8 macro features giving state_dim=96),
    but the models were actually trained with 9 macro features (state_dim=101).
    When a mismatch is detected, fall back to Cluster 3's config which has
    the correct feature set for state_dim=101.
    """
    import torch

    # Check first model's actual input dimension
    models = find_cluster_models(cluster_dir)
    if not models:
        return cfg
    ckpt = torch.load(models[0], map_location="cpu")
    model_input_dim = ckpt["policy_state_dict"]["feature_net.0.weight"].shape[1]

    # Compute expected state_dim from config
    tickers = list(cfg.data.tickers)
    feature_cols = cfg.env.build_feature_columns(tickers, cfg.features)
    n_features = len(feature_cols)
    lag = cfg.env.lag
    n_positions = len(tickers) + 1
    expected_dim = lag * n_features + (n_positions if cfg.env.include_position_in_state else 0)

    if expected_dim == model_input_dim:
        return cfg

    print(f"  WARNING: Config state_dim ({expected_dim}) != model input_dim ({model_input_dim})")
    print(f"  Searching for a compatible config in sibling clusters...")

    # Try sibling cluster configs
    models_root = os.path.dirname(cluster_dir)
    for sibling in sorted(os.listdir(models_root)):
        sibling_dir = os.path.join(models_root, sibling)
        if sibling_dir == cluster_dir or not os.path.isdir(sibling_dir):
            continue
        sibling_configs = sorted(glob(os.path.join(sibling_dir, "seed_*", "config.json")))
        if not sibling_configs:
            continue
        alt_cfg = Config.load_json(sibling_configs[0])
        alt_cols = alt_cfg.env.build_feature_columns(list(alt_cfg.data.tickers), alt_cfg.features)
        alt_dim = lag * len(alt_cols) + (n_positions if alt_cfg.env.include_position_in_state else 0)
        if alt_dim == model_input_dim:
            print(f"  Found compatible config in {sibling} (state_dim={alt_dim})")
            return alt_cfg

    print(f"  ERROR: No compatible config found. Proceeding with original (may fail).")
    return cfg


def load_cluster_data(cluster_dir: str) -> Tuple[Config, pd.DataFrame, pd.DataFrame]:
    """Load config from first seed in cluster and prepare train/test data.

    Verifies config matches model dimensions. Falls back to a sibling
    cluster's config if there's a mismatch (handles stale configs).

    Runs from algorithms/SAC/ working directory context so that relative
    paths in config.json (like ../../data/) resolve correctly.
    """
    # Load config from first seed
    seeds = sorted(glob(os.path.join(cluster_dir, "seed_*", "config.json")))
    if not seeds:
        raise FileNotFoundError(f"No config.json found in {cluster_dir}")

    cfg = Config.load_json(seeds[0])

    # Verify config matches saved model dimensions
    cfg = _verify_config_matches_model(cfg, cluster_dir)

    # Save and change to SAC directory so relative paths resolve
    original_cwd = os.getcwd()
    os.chdir(SAC_DIR)

    try:
        df_train, df_test, feature_cols = load_and_prepare_data(cfg)
    finally:
        os.chdir(original_cwd)

    return cfg, df_train, df_test


def evaluate_single_model(
    model_path: str,
    cfg: Config,
    df_test: pd.DataFrame,
    device: torch.device,
    cluster_name: str,
) -> SeedResult:
    """Evaluate one model on test data.

    Uses the cluster-level cfg (already verified to match model dimensions)
    rather than per-seed configs, which may be stale.
    """
    seed = extract_seed(model_path)

    env = Env(df_test, cfg.data.tickers, cfg)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    agent = Agent(state_dim, action_dim, cfg, device=device)
    agent.load_model(model_path)

    results = run_backtest(env, agent, deterministic=True)

    eq = results["equity"]
    net = results["net_returns"]
    step_size = cfg.env.lag

    s = sharpe(net, step_size=step_size)
    c = cagr(eq, step_size=step_size)
    mdd = max_drawdown(eq)
    calmar_val = abs(c / mdd) if mdd != 0 else 0.0

    return SeedResult(
        cluster=cluster_name,
        seed=seed,
        model_path=model_path,
        sharpe=s,
        cagr=c,
        max_dd=mdd,
        ann_vol=ann_vol(net, step_size=step_size),
        calmar=calmar_val,
        final_equity=float(eq[-1]) if eq.size > 0 else 1.0,
        avg_turnover=float(results["turnover_oneway"].mean()) if results["turnover_oneway"].size > 0 else 0.0,
        avg_tc_cost=float(results["tc_costs"].mean()) if results["tc_costs"].size > 0 else 0.0,
        num_steps=len(net),
        equity=eq,
        weights=results["weights"],
    )


def aggregate_cluster(cluster_name: str, results: List[SeedResult]) -> ClusterResult:
    """Compute mean/std statistics for a cluster."""
    def _stats(vals):
        arr = np.array(vals)
        return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    ms, ss = _stats([r.sharpe for r in results])
    mc, sc = _stats([r.cagr for r in results])
    md, sd = _stats([r.max_dd for r in results])
    mv, sv = _stats([r.ann_vol for r in results])
    mcal, scal = _stats([r.calmar for r in results])
    me, se = _stats([r.final_equity for r in results])
    mt, st = _stats([r.avg_turnover for r in results])

    return ClusterResult(
        cluster_name=cluster_name,
        seed_results=results,
        mean_sharpe=ms, std_sharpe=ss,
        mean_cagr=mc, std_cagr=sc,
        mean_max_dd=md, std_max_dd=sd,
        mean_ann_vol=mv, std_ann_vol=sv,
        mean_calmar=mcal, std_calmar=scal,
        mean_final_equity=me, std_final_equity=se,
        mean_turnover=mt, std_turnover=st,
    )


# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------

def print_per_seed_table(all_results: List[SeedResult]) -> None:
    """Print per-seed results table."""
    print("\nPer-Seed Results:")
    print("-" * 120)
    print(f"{'Cluster':<18} {'Seed':>6} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10} "
          f"{'AnnVol':>10} {'Calmar':>10} {'FinalEq':>10} {'Turnover':>10}")
    print("-" * 120)

    for r in sorted(all_results, key=lambda x: (x.cluster, x.seed)):
        print(f"{r.cluster:<18} {r.seed:>6} {r.sharpe:>10.3f} {r.cagr:>10.2%} {r.max_dd:>10.2%} "
              f"{r.ann_vol:>10.2%} {r.calmar:>10.3f} {r.final_equity:>10.3f} {r.avg_turnover:>10.4f}")

    print("-" * 120)


def print_cluster_summary(clusters: List[ClusterResult]) -> None:
    """Print cluster comparison table."""
    print("\nCluster Comparison (Mean +/- Std):")
    print("-" * 100)
    print(f"{'Cluster':<18} {'Sharpe':>14} {'CAGR':>14} {'MaxDD':>14} "
          f"{'AnnVol':>14} {'Calmar':>14} {'FinalEq':>14}")
    print("-" * 100)

    for c in clusters:
        print(f"{c.cluster_name:<18} "
              f"{c.mean_sharpe:>6.3f}+/-{c.std_sharpe:<5.3f} "
              f"{c.mean_cagr:>6.2%}+/-{c.std_cagr:<5.2%} "
              f"{c.mean_max_dd:>7.2%}+/-{c.std_max_dd:<5.2%} "
              f"{c.mean_ann_vol:>6.2%}+/-{c.std_ann_vol:<5.2%} "
              f"{c.mean_calmar:>6.3f}+/-{c.std_calmar:<5.3f} "
              f"{c.mean_final_equity:>6.3f}+/-{c.std_final_equity:<5.3f}")

    print("-" * 100)

    # Best cluster
    best = max(clusters, key=lambda c: c.mean_sharpe)
    print(f"\nBest cluster: {best.cluster_name} (Sharpe: {best.mean_sharpe:.3f})")

    # Consistency
    for c in clusters:
        cv = c.std_sharpe / abs(c.mean_sharpe) if c.mean_sharpe != 0 else float("inf")
        print(f"  {c.cluster_name} CV(Sharpe): {cv:.3f}")


def save_results_csv(all_results: List[SeedResult], clusters: List[ClusterResult], output_dir: str) -> str:
    """Save all results to CSV."""
    rows = []
    for r in sorted(all_results, key=lambda x: (x.cluster, x.seed)):
        rows.append({
            "cluster": r.cluster,
            "seed": r.seed,
            "sharpe": r.sharpe,
            "cagr": r.cagr,
            "max_dd": r.max_dd,
            "ann_vol": r.ann_vol,
            "calmar": r.calmar,
            "final_equity": r.final_equity,
            "avg_turnover": r.avg_turnover,
            "avg_tc_cost": r.avg_tc_cost,
            "num_steps": r.num_steps,
            "model_path": r.model_path,
        })

    # Add cluster summary rows
    for c in clusters:
        rows.append({
            "cluster": c.cluster_name,
            "seed": "MEAN",
            "sharpe": c.mean_sharpe,
            "cagr": c.mean_cagr,
            "max_dd": c.mean_max_dd,
            "ann_vol": c.mean_ann_vol,
            "calmar": c.mean_calmar,
            "final_equity": c.mean_final_equity,
            "avg_turnover": c.mean_turnover,
        })
        rows.append({
            "cluster": c.cluster_name,
            "seed": "STD",
            "sharpe": c.std_sharpe,
            "cagr": c.std_cagr,
            "max_dd": c.std_max_dd,
            "ann_vol": c.std_ann_vol,
            "calmar": c.std_calmar,
            "final_equity": c.std_final_equity,
            "avg_turnover": c.std_turnover,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def save_equity_curves_csv(all_results: List[SeedResult], output_dir: str) -> str:
    """Save equity curves for all models to CSV."""
    max_len = max(len(r.equity) for r in all_results)
    data = {"step": np.arange(max_len)}

    for r in all_results:
        col = f"{r.cluster}_seed_{r.seed}"
        padded = np.full(max_len, np.nan)
        padded[: len(r.equity)] = r.equity
        data[col] = padded

    df = pd.DataFrame(data)
    path = os.path.join(output_dir, "equity_curves.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(
    all_results: List[SeedResult],
    clusters: List[ClusterResult],
    output_dir: str,
    tickers: List[str],
) -> List[str]:
    """Generate evaluation plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []

    # --- 1. Equity curves by cluster ---
    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_colors = {"Cluster 1 (HMM)": "#2563eb", "Cluster 2 (Base)": "#dc2626", "Cluster 3": "#16a34a"}

    for cr in clusters:
        color = cluster_colors.get(cr.cluster_name, "gray")
        equities = []
        for r in cr.seed_results:
            ax.plot(r.equity, color=color, alpha=0.2, linewidth=0.8)
            equities.append(r.equity)

        # Mean equity
        max_len = max(len(e) for e in equities)
        padded = [np.pad(e, (0, max_len - len(e)), constant_values=np.nan) for e in equities]
        mean_eq = np.nanmean(padded, axis=0)
        ax.plot(mean_eq, color=color, linewidth=2.5, label=f"{cr.cluster_name} (mean)")

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity ($1 start)")
    ax.set_title("Equity Curves: All Clusters")
    ax.legend(loc="upper left")
    path = os.path.join(output_dir, "equity_curves_all_clusters.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # --- 2. Metrics comparison bar chart ---
    metrics = ["sharpe", "cagr", "max_dd", "ann_vol"]
    titles = ["Sharpe Ratio", "CAGR", "Max Drawdown", "Ann. Volatility"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, metric, title in zip(axes, metrics, titles):
        for i, cr in enumerate(clusters):
            vals = [getattr(r, metric) for r in cr.seed_results]
            color = list(cluster_colors.values())[i] if i < len(cluster_colors) else "gray"
            x = np.arange(len(vals)) + i * 0.25
            ax.bar(x, vals, width=0.22, color=color, alpha=0.8, label=cr.cluster_name)

        ax.set_title(title)
        ax.set_xticks(np.arange(5) + 0.25)
        ax.set_xticklabels([f"s{s}" for s in [42, 123, 456, 789, 1024]], fontsize=8)

        if metric in ["cagr", "max_dd", "ann_vol"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

    axes[0].legend(fontsize=7)
    fig.suptitle("Per-Seed Metrics by Cluster", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "metrics_by_cluster.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # --- 3. Risk-return scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cr in enumerate(clusters):
        color = list(cluster_colors.values())[i] if i < len(cluster_colors) else "gray"
        vols = [r.ann_vol for r in cr.seed_results]
        cagrs = [r.cagr for r in cr.seed_results]
        ax.scatter(vols, cagrs, color=color, s=60, alpha=0.7, label=cr.cluster_name, zorder=3)

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("CAGR")
    ax.set_title("Risk-Return Scatter: All Seeds")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "risk_return_scatter.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # --- 4. Average weights per cluster (best seed from each) ---
    fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 4))
    if len(clusters) == 1:
        axes = [axes]
    labels = tickers + ["CASH"]

    for ax, cr in zip(axes, clusters):
        best = max(cr.seed_results, key=lambda r: r.sharpe)
        if best.weights.size > 0:
            avg_w = best.weights.mean(axis=0)
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            bars = ax.bar(np.arange(len(labels)), avg_w, color=colors, alpha=0.8)
            for bar, val in zip(bars, avg_w):
                ax.annotate(f"{val:.1%}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{cr.cluster_name}\n(Best: seed {best.seed}, SR={best.sharpe:.3f})")
        ax.set_ylabel("Avg Weight")

    fig.suptitle("Average Portfolio Weights (Best Seed per Cluster)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "avg_weights_by_cluster.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CLUSTER_NAMES = {
    1: "Cluster 1 (HMM)",
    2: "Cluster 2 (Base)",
    3: "Cluster 3",
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC clusters on 2023-2024 test set")
    parser.add_argument("--clusters", type=int, nargs="+", default=[1, 2, 3],
                        help="Which clusters to evaluate (default: 1 2 3)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: testing/eval_results)")
    parser.add_argument("--per_seed_plots", action="store_true",
                        help="Generate detailed per-seed plots")
    args = parser.parse_args()

    models_root = os.path.join(PROJECT_ROOT, "algorithms", "SAC", "models")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "testing", "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    print_header("SAC CLUSTER EVALUATION (2023-2024 TEST SET)")

    all_results: List[SeedResult] = []
    cluster_results: List[ClusterResult] = []
    tickers = []

    for cluster_id in args.clusters:
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        cluster_dir = os.path.join(models_root, f"cluster_{cluster_id}")

        if not os.path.isdir(cluster_dir):
            print(f"\nSkipping {cluster_name}: directory not found ({cluster_dir})")
            continue

        model_paths = find_cluster_models(cluster_dir)
        if not model_paths:
            print(f"\nSkipping {cluster_name}: no models found")
            continue

        print_header(f"EVALUATING {cluster_name.upper()}")
        print(f"Models: {len(model_paths)}")

        # Load data using this cluster's config
        print(f"Loading data for {cluster_name}...")
        cfg, df_train, df_test = load_cluster_data(cluster_dir)
        tickers = list(cfg.data.tickers)

        print(f"  Train rows: {len(df_train)}")
        print(f"  Test rows:  {len(df_test)}")
        print(f"  HMM: {'ON' if cfg.features.use_regime_hmm else 'OFF'}")
        print(f"  Macro features: {len(cfg.features.macro_feature_columns)}")

        device = cfg.auto_detect_device()
        print(f"  Device: {device}")

        # Evaluate each seed
        seed_results = []
        for i, mp in enumerate(model_paths):
            seed = extract_seed(mp)
            print(f"\n  [{i+1}/{len(model_paths)}] Seed {seed}...", end=" ")

            result = evaluate_single_model(mp, cfg, df_test, device, cluster_name)
            seed_results.append(result)
            all_results.append(result)

            print(f"Sharpe={result.sharpe:.3f}  CAGR={result.cagr:.2%}  MaxDD={result.max_dd:.2%}")

        # Aggregate
        cr = aggregate_cluster(cluster_name, seed_results)
        cluster_results.append(cr)

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print_header("RESULTS")
    print_per_seed_table(all_results)
    print_cluster_summary(cluster_results)

    # Save
    csv_path = save_results_csv(all_results, cluster_results, output_dir)
    print(f"\nResults CSV: {csv_path}")

    eq_path = save_equity_curves_csv(all_results, output_dir)
    print(f"Equity CSV:  {eq_path}")

    # Plots
    print_header("GENERATING PLOTS")
    saved = generate_plots(all_results, cluster_results, output_dir, tickers)
    for p in saved:
        print(f"  Saved: {os.path.basename(p)}")

    print_header("EVALUATION COMPLETE")
    print(f"\nOutput: {output_dir}")

    best_overall = max(all_results, key=lambda r: r.sharpe)
    print(f"\nBest overall model:")
    print(f"  {best_overall.cluster}, Seed {best_overall.seed}")
    print(f"  Sharpe: {best_overall.sharpe:.3f}")
    print(f"  CAGR:   {best_overall.cagr:.2%}")
    print(f"  MaxDD:  {best_overall.max_dd:.2%}")


if __name__ == "__main__":
    main()
