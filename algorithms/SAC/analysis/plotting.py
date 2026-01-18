"""
Thesis-Quality Plotting Module for SAC Portfolio Management

Provides consistent, publication-ready visualizations for:
- Training progress (single and multi-seed)
- Evaluation results (single and multi-seed)
- Regime analysis
- Performance attribution

Uses thesis_style.py for consistent styling.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Headless-safe imports
import matplotlib
matplotlib.use("Agg")  # Set before importing pyplot
import matplotlib.pyplot as plt

# Import thesis style (same folder)
try:
    from .thesis_style import apply_thesis_style, thesis_palette
except ImportError:
    # Fallback for direct script execution
    from thesis_style import apply_thesis_style, thesis_palette


def _ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


def _save_and_close(fig: plt.Figure, path: str, show: bool = False) -> None:
    """Save figure and optionally show."""
    _ensure_dir(path)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# TRAINING PLOTS
# =============================================================================

def plot_training_returns(
    episode_returns: List[float],
    out_path: str,
    title: str = "Episode Returns",
    window: int = 20,
    show: bool = False,
) -> None:
    """Plot episode returns with smoothed trend line."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(len(episode_returns))
    returns = np.array(episode_returns)

    # Raw returns (light)
    ax.plot(episodes, returns, alpha=0.3, color=palette["agent"], linewidth=1, label="Raw")

    # Smoothed returns (bold)
    if len(returns) >= window:
        smoothed = pd.Series(returns).rolling(window, min_periods=1).mean().values
        ax.plot(episodes, smoothed, color=palette["agent"], linewidth=2.5,
                label=f"Smoothed ({window}-ep)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    _save_and_close(fig, out_path, show)


def plot_training_losses(
    losses: List[Dict],
    out_path: str,
    show: bool = False,
) -> None:
    """
    Plot combined training losses (policy + critic) on single figure.

    Note: Q1 and Q2 losses are averaged into a single "Critic Loss" since
    they track the same objective and individual plots add noise without insight.
    """
    apply_thesis_style()
    palette = thesis_palette()

    if not losses:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Extract loss components
    updates = np.arange(len(losses))

    q1_vals = np.array([d.get("q1_loss", np.nan) for d in losses])
    q2_vals = np.array([d.get("q2_loss", np.nan) for d in losses])
    policy_vals = np.array([d.get("policy_loss", np.nan) for d in losses])
    alpha_vals = np.array([d.get("alpha_loss", np.nan) for d in losses])
    logp_vals = np.array([d.get("avg_logp", np.nan) for d in losses])

    # Combined critic loss (more meaningful than separate Q1/Q2)
    critic_loss = (q1_vals + q2_vals) / 2

    window = max(10, len(losses) // 50)

    # Plot 1: Critic Loss (combined Q1+Q2)
    ax = axes[0, 0]
    ax.plot(updates, critic_loss, alpha=0.3, color=palette["agent"], linewidth=1)
    if len(critic_loss) >= window:
        smoothed = pd.Series(critic_loss).rolling(window, min_periods=1).mean()
        ax.plot(updates, smoothed, color=palette["agent"], linewidth=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Critic Loss (avg Q1+Q2)")

    # Plot 2: Policy Loss
    ax = axes[0, 1]
    ax.plot(updates, policy_vals, alpha=0.3, color=palette["crisis"], linewidth=1)
    if len(policy_vals) >= window:
        smoothed = pd.Series(policy_vals).rolling(window, min_periods=1).mean()
        ax.plot(updates, smoothed, color=palette["crisis"], linewidth=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Policy Loss")

    # Plot 3: Alpha Loss (entropy temperature)
    ax = axes[1, 0]
    ax.plot(updates, alpha_vals, alpha=0.3, color=palette["trans"], linewidth=1)
    if len(alpha_vals) >= window:
        smoothed = pd.Series(alpha_vals).rolling(window, min_periods=1).mean()
        ax.plot(updates, smoothed, color=palette["trans"], linewidth=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Alpha Loss (Entropy Tuning)")

    # Plot 4: Average Log Probability
    ax = axes[1, 1]
    ax.plot(updates, logp_vals, alpha=0.3, color=palette["stable"], linewidth=1)
    if len(logp_vals) >= window:
        smoothed = pd.Series(logp_vals).rolling(window, min_periods=1).mean()
        ax.plot(updates, smoothed, color=palette["stable"], linewidth=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Log Probability")
    ax.set_title("Average Log Probability")

    fig.suptitle("Training Losses", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    _save_and_close(fig, out_path, show)


def plot_multiseed_training_comparison(
    all_returns: Dict[int, List[float]],
    out_path: str,
    window: int = 20,
    show: bool = False,
) -> None:
    """Plot training returns for multiple seeds with confidence band."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Find max length and pad shorter sequences
    max_len = max(len(r) for r in all_returns.values())

    # Plot individual seeds (light)
    for seed, returns in all_returns.items():
        episodes = np.arange(len(returns))
        ax.plot(episodes, returns, alpha=0.2, linewidth=1, label=f"Seed {seed}")

    # Compute mean and std across seeds (align by padding)
    padded = []
    for returns in all_returns.values():
        padded_r = np.full(max_len, np.nan)
        padded_r[:len(returns)] = returns
        padded.append(padded_r)

    arr = np.array(padded)
    mean_returns = np.nanmean(arr, axis=0)
    std_returns = np.nanstd(arr, axis=0)

    episodes = np.arange(max_len)

    # Smooth mean
    if max_len >= window:
        mean_smooth = pd.Series(mean_returns).rolling(window, min_periods=1).mean().values
        std_smooth = pd.Series(std_returns).rolling(window, min_periods=1).mean().values
    else:
        mean_smooth = mean_returns
        std_smooth = std_returns

    # Plot mean with confidence band
    ax.plot(episodes, mean_smooth, color=palette["agent"], linewidth=2.5, label="Mean")
    ax.fill_between(episodes, mean_smooth - std_smooth, mean_smooth + std_smooth,
                    color=palette["agent"], alpha=0.2, label="±1 Std")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return")
    ax.set_title("Multi-Seed Training Progress")
    ax.legend(loc="lower right", fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    _save_and_close(fig, out_path, show)


# =============================================================================
# EVALUATION PLOTS
# =============================================================================

def plot_equity_curve(
    equity: np.ndarray,
    out_path: str,
    benchmark_equity: Optional[np.ndarray] = None,
    benchmark_name: str = "SPY",
    title: str = "Equity Curve",
    show: bool = False,
) -> None:
    """Plot equity curve with optional benchmark comparison."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(10, 5))

    steps = np.arange(len(equity))
    ax.plot(steps, equity, color=palette["agent"], linewidth=2, label="SAC Agent")

    if benchmark_equity is not None:
        ax.plot(steps[:len(benchmark_equity)], benchmark_equity,
                color=palette["spx"], linewidth=1.5, linestyle="--", label=benchmark_name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Equity ($1 start)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    _save_and_close(fig, out_path, show)


def plot_drawdown(
    equity: np.ndarray,
    out_path: str,
    title: str = "Drawdown (Underwater Plot)",
    show: bool = False,
) -> None:
    """Plot drawdown underwater chart."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(10, 4))

    equity = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100  # As percentage

    steps = np.arange(len(drawdown))
    ax.fill_between(steps, drawdown, 0, color=palette["crisis"], alpha=0.6)
    ax.plot(steps, drawdown, color=palette["crisis"], linewidth=1)

    max_dd = drawdown.min()
    max_dd_idx = np.argmin(drawdown)
    ax.axhline(y=max_dd, color=palette["crisis"], linestyle="--", alpha=0.7, linewidth=1)
    ax.annotate(f"Max DD: {max_dd:.1f}%", xy=(max_dd_idx, max_dd),
                xytext=(max_dd_idx + len(drawdown)*0.05, max_dd + 2),
                fontsize=10, color=palette["crisis"])

    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title)
    ax.set_ylim(min(max_dd * 1.2, -5), 2)

    _save_and_close(fig, out_path, show)


def plot_rolling_sharpe(
    net_returns: np.ndarray,
    out_path: str,
    window: int = 60,
    ann_factor: int = 252,
    title: str = "Rolling Sharpe Ratio",
    show: bool = False,
) -> None:
    """Plot rolling Sharpe ratio over time."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(10, 4))

    returns = pd.Series(net_returns)
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(ann_factor)

    steps = np.arange(len(rolling_sharpe))
    ax.plot(steps, rolling_sharpe, color=palette["agent"], linewidth=1.5)

    # Color regions
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    ax.axhline(y=1, color=palette["stable"], linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=2, color=palette["stable"], linestyle="--", alpha=0.3, linewidth=1)
    ax.fill_between(steps, 0, rolling_sharpe.values,
                    where=rolling_sharpe.values > 0, color=palette["stable"], alpha=0.2)
    ax.fill_between(steps, 0, rolling_sharpe.values,
                    where=rolling_sharpe.values < 0, color=palette["crisis"], alpha=0.2)

    ax.set_xlabel("Step")
    ax.set_ylabel(f"Sharpe Ratio ({window}-day rolling)")
    ax.set_title(title)

    _save_and_close(fig, out_path, show)


def plot_weights_evolution(
    weights: np.ndarray,
    out_path: str,
    labels: List[str],
    title: str = "Portfolio Weight Evolution",
    show: bool = False,
) -> None:
    """Plot stacked area chart of portfolio weights over time."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(12, 5))

    steps = np.arange(weights.shape[0])

    # Use a colormap for multiple assets
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    ax.stackplot(steps, weights.T, labels=labels, colors=colors, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=9)
    ax.set_ylim(0, 1)

    _save_and_close(fig, out_path, show)


def plot_average_weights(
    weights: np.ndarray,
    out_path: str,
    labels: List[str],
    title: str = "Average Portfolio Weights",
    show: bool = False,
) -> None:
    """Plot bar chart of average weights."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(8, 5))

    avg_weights = weights.mean(axis=0)
    x = np.arange(len(labels))

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    bars = ax.bar(x, avg_weights, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, avg_weights):
        ax.annotate(f"{val:.1%}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Average Weight")
    ax.set_title(title)

    _save_and_close(fig, out_path, show)


def plot_returns_distribution(
    net_returns: np.ndarray,
    out_path: str,
    title: str = "Returns Distribution",
    show: bool = False,
) -> None:
    """Plot histogram of returns with statistics."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(8, 5))

    returns = np.asarray(net_returns, dtype=np.float64) * 100  # Convert to percentage

    ax.hist(returns, bins=50, color=palette["agent"], alpha=0.7, edgecolor="white", linewidth=0.5)

    # Statistics
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    skew = pd.Series(returns).skew()
    kurt = pd.Series(returns).kurtosis()

    ax.axvline(x=mean_ret, color=palette["trans"], linestyle="--", linewidth=2, label=f"Mean: {mean_ret:.2f}%")
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

    stats_text = f"Std: {std_ret:.2f}%\nSkew: {skew:.2f}\nKurtosis: {kurt:.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")

    _save_and_close(fig, out_path, show)


def plot_regime_returns(
    net_returns: np.ndarray,
    regime_probs: np.ndarray,
    out_path: str,
    regime_names: List[str] = ["Stable", "Transition", "Crisis"],
    title: str = "Returns by Market Regime",
    show: bool = False,
) -> None:
    """Plot box plot of returns conditioned on regime."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Assign regime based on highest probability
    regime_idx = np.argmax(regime_probs, axis=1)

    # Collect returns by regime
    returns_by_regime = []
    colors = [palette["stable"], palette["trans"], palette["crisis"]]

    for k in range(len(regime_names)):
        mask = regime_idx == k
        if mask.sum() > 0:
            returns_by_regime.append(net_returns[mask] * 100)  # To percentage
        else:
            returns_by_regime.append(np.array([0]))

    bp = ax.boxplot(returns_by_regime, patch_artist=True, labels=regime_names)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_ylabel("Return (%)")
    ax.set_title(title)

    # Add count labels
    for i, (name, data) in enumerate(zip(regime_names, returns_by_regime)):
        ax.annotate(f"n={len(data)}", xy=(i+1, ax.get_ylim()[1]),
                    xytext=(0, -5), textcoords="offset points",
                    ha="center", fontsize=9, color="gray")

    _save_and_close(fig, out_path, show)


def plot_regime_timeline(
    regime_probs: np.ndarray,
    equity: np.ndarray,
    out_path: str,
    regime_names: List[str] = ["Stable", "Transition", "Crisis"],
    title: str = "Regime Probabilities with Equity",
    show: bool = False,
) -> None:
    """Plot regime probability timeline with equity overlay."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    steps = np.arange(len(regime_probs))
    colors = [palette["stable"], palette["trans"], palette["crisis"]]

    # Top: Equity with regime shading
    ax1.plot(steps[:len(equity)], equity, color=palette["agent"], linewidth=1.5, label="Equity")

    # Shade by dominant regime
    regime_idx = np.argmax(regime_probs, axis=1)
    for k, color in enumerate(colors):
        mask = regime_idx == k
        if mask.any():
            ax1.fill_between(steps, ax1.get_ylim()[0], equity,
                           where=mask[:len(equity)], color=color, alpha=0.15)

    ax1.set_ylabel("Equity ($1 start)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")

    # Bottom: Stacked regime probabilities
    ax2.stackplot(steps, regime_probs.T, colors=colors, alpha=0.8, labels=regime_names)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save_and_close(fig, out_path, show)


def plot_monthly_returns_heatmap(
    net_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    out_path: str,
    title: str = "Monthly Returns Heatmap",
    show: bool = False,
) -> None:
    """Plot monthly returns heatmap (year x month)."""
    apply_thesis_style()

    # Create DataFrame with dates
    df = pd.DataFrame({"return": net_returns}, index=dates)

    # Resample to monthly returns
    monthly = df["return"].resample("M").apply(lambda x: (1 + x).prod() - 1) * 100

    # Pivot to year x month
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="return")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom colormap (red for negative, green for positive)
    cmap = plt.cm.RdYlGn
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    # Labels
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.iloc[i, j] if j < pivot.shape[1] else np.nan
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                       fontsize=8, color=color)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Return (%)", shrink=0.8)

    _save_and_close(fig, out_path, show)


def plot_turnover(
    turnover_total: np.ndarray,
    turnover_oneway: np.ndarray,
    out_path: str,
    title: str = "Portfolio Turnover",
    show: bool = False,
) -> None:
    """Plot turnover over time."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(10, 4))

    steps = np.arange(len(turnover_total))
    ax.plot(steps, turnover_total, color=palette["trans"], linewidth=1.5,
            label=f"Total (avg: {turnover_total.mean():.4f})", alpha=0.8)
    ax.plot(steps, turnover_oneway, color=palette["agent"], linewidth=1.5,
            label=f"One-way (avg: {turnover_oneway.mean():.4f})", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Turnover")
    ax.set_title(title)
    ax.legend(loc="upper right")

    _save_and_close(fig, out_path, show)


def plot_multiseed_equity_comparison(
    equity_dict: Dict[int, np.ndarray],
    out_path: str,
    title: str = "Multi-Seed Equity Curves",
    show: bool = False,
) -> None:
    """Plot equity curves for multiple seeds with mean and std band."""
    apply_thesis_style()
    palette = thesis_palette()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Find max length
    max_len = max(len(eq) for eq in equity_dict.values())

    # Plot individual seeds
    for seed, equity in equity_dict.items():
        steps = np.arange(len(equity))
        ax.plot(steps, equity, alpha=0.3, linewidth=1, label=f"Seed {seed}")

    # Compute mean and std
    padded = []
    for equity in equity_dict.values():
        padded_eq = np.full(max_len, np.nan)
        padded_eq[:len(equity)] = equity
        padded.append(padded_eq)

    arr = np.array(padded)
    mean_eq = np.nanmean(arr, axis=0)
    std_eq = np.nanstd(arr, axis=0)

    steps = np.arange(max_len)
    ax.plot(steps, mean_eq, color=palette["agent"], linewidth=2.5, label="Mean")
    ax.fill_between(steps, mean_eq - std_eq, mean_eq + std_eq,
                    color=palette["agent"], alpha=0.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Equity ($1 start)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    _save_and_close(fig, out_path, show)


def plot_multiseed_metrics_summary(
    metrics_df: pd.DataFrame,
    out_path: str,
    title: str = "Multi-Seed Performance Summary",
    show: bool = False,
) -> None:
    """Plot bar chart comparing metrics across seeds."""
    apply_thesis_style()
    palette = thesis_palette()

    # Select key metrics
    key_metrics = ["sharpe", "cagr", "max_dd", "ann_vol"]
    available = [m for m in key_metrics if m in metrics_df.columns]

    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(3.5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    seeds = metrics_df["seed"].astype(str).tolist() if "seed" in metrics_df.columns else [str(i) for i in range(len(metrics_df))]

    for ax, metric in zip(axes, available):
        values = metrics_df[metric].values
        x = np.arange(len(values))

        # Color bars
        colors = [palette["agent"] if v >= np.mean(values) else palette["eqw"] for v in values]
        if metric == "max_dd":
            colors = [palette["agent"] if v >= np.mean(values) else palette["crisis"] for v in values]

        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor="white")

        # Add mean line
        ax.axhline(y=np.mean(values), color=palette["trans"], linestyle="--", linewidth=2, label="Mean")

        ax.set_xticks(x)
        ax.set_xticklabels(seeds, rotation=45, ha="right")
        ax.set_xlabel("Seed")

        # Format based on metric
        if metric in ["cagr", "max_dd", "ann_vol"]:
            ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))
        else:
            ax.set_ylabel(metric.replace("_", " ").title())

        ax.set_title(metric.replace("_", " ").title())

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    _save_and_close(fig, out_path, show)


# =============================================================================
# CONVENIENCE FUNCTION FOR FULL EVALUATION SUITE
# =============================================================================

def generate_evaluation_plots(
    results: Dict[str, np.ndarray],
    cfg,
    out_dir: str,
    dates: Optional[pd.DatetimeIndex] = None,
    regime_probs: Optional[np.ndarray] = None,
    show: bool = False,
) -> List[str]:
    """
    Generate all evaluation plots and return list of saved paths.

    Args:
        results: Dict with keys: equity, net_returns, weights, turnover_oneway, turnover_total, etc.
        cfg: Config object
        out_dir: Output directory
        dates: Optional DatetimeIndex for time-based plots
        regime_probs: Optional regime probabilities array (T, 3)
        show: Whether to display plots

    Returns:
        List of saved file paths
    """
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    tickers = list(cfg.data.tickers) + ["CASH"]
    equity = results["equity"]
    net_returns = results["net_returns"]
    weights = results["weights"]

    # 1. Equity Curve
    path = os.path.join(out_dir, "equity_curve.png")
    plot_equity_curve(equity, path, show=show)
    saved_paths.append(path)

    # 2. Drawdown
    path = os.path.join(out_dir, "drawdown.png")
    plot_drawdown(equity, path, show=show)
    saved_paths.append(path)

    # 3. Rolling Sharpe
    path = os.path.join(out_dir, "rolling_sharpe.png")
    window = min(60, len(net_returns) // 4)
    plot_rolling_sharpe(net_returns, path, window=window, show=show)
    saved_paths.append(path)

    # 4. Returns Distribution
    path = os.path.join(out_dir, "returns_distribution.png")
    plot_returns_distribution(net_returns, path, show=show)
    saved_paths.append(path)

    # 5. Average Weights
    if weights.shape[0] > 0:
        path = os.path.join(out_dir, "average_weights.png")
        plot_average_weights(weights, path, tickers, show=show)
        saved_paths.append(path)

        # 6. Weight Evolution
        path = os.path.join(out_dir, "weight_evolution.png")
        plot_weights_evolution(weights, path, tickers, show=show)
        saved_paths.append(path)

    # 7. Turnover
    path = os.path.join(out_dir, "turnover.png")
    plot_turnover(results["turnover_total"], results["turnover_oneway"], path, show=show)
    saved_paths.append(path)

    # 8. Regime-conditioned plots (if regime probs available)
    if regime_probs is not None and len(regime_probs) == len(net_returns):
        path = os.path.join(out_dir, "regime_returns.png")
        plot_regime_returns(net_returns, regime_probs, path, show=show)
        saved_paths.append(path)

        path = os.path.join(out_dir, "regime_timeline.png")
        plot_regime_timeline(regime_probs, equity, path, show=show)
        saved_paths.append(path)

    # 9. Monthly heatmap (if dates available)
    if dates is not None and len(dates) == len(net_returns):
        try:
            path = os.path.join(out_dir, "monthly_returns_heatmap.png")
            plot_monthly_returns_heatmap(net_returns, dates, path, show=show)
            saved_paths.append(path)
        except Exception as e:
            print(f"  Warning: Could not generate monthly heatmap: {e}")

    return saved_paths
