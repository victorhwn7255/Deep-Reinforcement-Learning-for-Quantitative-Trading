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
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy import stats as scipy_stats

from config import get_default_config, Config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent
from evaluate import run_backtest, cagr, sharpe, ann_vol, max_drawdown, equity_curve


# =============================================================================
# PLOT STYLE CONFIGURATION (Academic/Professional)
# =============================================================================

# Use a clean, publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette (colorblind-friendly)
COLORS = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']

# Font settings for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

ENTROPY_MARGINS = [0.5, 0.75, 1.0, 1.25, 1.5]  # Values to sweep

# Training settings (reduce if you want faster sweeps)
TOTAL_TIMESTEPS = 900_000  # Set to None to use default from config

# Output directory
SWEEP_OUTPUT_DIR = "runs/sweep_entropy_margin"


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_equity_curves(
    equity_data: Dict[float, np.ndarray],
    sweep_dir: str,
    step_size: int = 5,
) -> None:
    """
    Plot overlaid equity curves for all entropy margin values.

    Academic style: clean lines, proper legend, log scale option.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Linear scale
    ax1 = axes[0]
    for i, (margin, eq) in enumerate(sorted(equity_data.items())):
        color = COLORS[i % len(COLORS)]
        steps = np.arange(len(eq)) * step_size  # Convert to trading days
        ax1.plot(steps, eq, label=f'$\\tau_{{margin}}$={margin}', color=color, linewidth=1.5)

    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Portfolio Value ($1 initial)')
    ax1.set_title('(a) Equity Curves (Linear Scale)')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Right: Log scale
    ax2 = axes[1]
    for i, (margin, eq) in enumerate(sorted(equity_data.items())):
        color = COLORS[i % len(COLORS)]
        steps = np.arange(len(eq)) * step_size
        ax2.semilogy(steps, eq, label=f'$\\tau_{{margin}}$={margin}', color=color, linewidth=1.5)

    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Portfolio Value (log scale)')
    ax2.set_title('(b) Equity Curves (Log Scale)')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, 'equity_curves_comparison.png'))
    plt.savefig(os.path.join(sweep_dir, 'equity_curves_comparison.pdf'))  # Vector format for papers
    plt.close()
    print(f"  Saved: equity_curves_comparison.png/pdf")


def plot_drawdown_curves(
    equity_data: Dict[float, np.ndarray],
    sweep_dir: str,
    step_size: int = 5,
) -> None:
    """
    Plot underwater (drawdown) curves for each margin value.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (margin, eq) in enumerate(sorted(equity_data.items())):
        color = COLORS[i % len(COLORS)]
        # Calculate drawdown
        peak = np.maximum.accumulate(eq)
        drawdown = (eq / peak - 1.0) * 100  # Percentage
        steps = np.arange(len(eq)) * step_size
        ax.fill_between(steps, drawdown, 0, alpha=0.3, color=color)
        ax.plot(steps, drawdown, label=f'$\\tau_{{margin}}$={margin}', color=color, linewidth=1.2)

    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Underwater Curves by Entropy Margin')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_ylim(top=0)

    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, 'drawdown_curves.png'))
    plt.savefig(os.path.join(sweep_dir, 'drawdown_curves.pdf'))
    plt.close()
    print(f"  Saved: drawdown_curves.png/pdf")


def plot_metrics_comparison(
    results: List[Dict[str, Any]],
    sweep_dir: str,
) -> None:
    """
    Bar charts comparing key performance metrics across entropy margins.
    """
    valid = [r for r in results if 'error' not in r]
    if len(valid) < 2:
        return

    margins = [r['margin'] for r in valid]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    metrics = [
        ('Sharpe', 'Sharpe Ratio', '{:.2f}'),
        ('CAGR', 'CAGR', '{:.1%}'),
        ('AnnVol', 'Annualized Volatility', '{:.1%}'),
        ('MaxDD', 'Maximum Drawdown', '{:.1%}'),
        ('FinalEquity', 'Final Equity ($1 initial)', '${:.2f}'),
        ('AvgTurnoverTotal', 'Avg. Daily Turnover', '{:.1%}'),
    ]

    for idx, (key, title, fmt) in enumerate(metrics):
        ax = axes[idx]
        values = [r[key] for r in valid]

        # Highlight best value
        if key == 'MaxDD':
            best_idx = np.argmax(values)  # Least negative
        elif key == 'AnnVol' or key == 'AvgTurnoverTotal':
            best_idx = np.argmin(values)  # Lower is better
        else:
            best_idx = np.argmax(values)  # Higher is better

        colors_bar = [COLORS[0] if i != best_idx else '#2ECC71' for i in range(len(values))]

        bars = ax.bar(range(len(margins)), values, color=colors_bar, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(margins)))
        ax.set_xticklabels([f'{m}' for m in margins])
        ax.set_xlabel('Entropy Margin ($\\tau_{margin}$)')
        ax.set_ylabel(title)
        ax.set_title(title)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label = fmt.format(val) if '%' not in fmt else fmt.format(val)
            va = 'bottom' if height >= 0 else 'top'
            offset = 0.01 * max(abs(v) for v in values) if height >= 0 else -0.01 * max(abs(v) for v in values)
            ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                       ha='center', va=va, fontsize=8, fontweight='bold')

    plt.suptitle('Performance Metrics by Entropy Margin', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, 'metrics_comparison.png'))
    plt.savefig(os.path.join(sweep_dir, 'metrics_comparison.pdf'))
    plt.close()
    print(f"  Saved: metrics_comparison.png/pdf")


def plot_risk_return_scatter(
    results: List[Dict[str, Any]],
    sweep_dir: str,
) -> None:
    """
    Risk-return scatter plot with Sharpe ratio isolines.
    """
    valid = [r for r in results if 'error' not in r]
    if len(valid) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Sharpe ratio isolines
    vol_range = np.linspace(0.05, 0.35, 100)
    for sr in [0.5, 1.0, 1.5, 2.0]:
        returns = sr * vol_range
        ax.plot(vol_range * 100, returns * 100, '--', color='gray', alpha=0.4, linewidth=0.8)
        ax.annotate(f'SR={sr}', xy=(vol_range[-1]*100, returns[-1]*100),
                   fontsize=7, color='gray', alpha=0.7)

    # Plot each margin
    for i, r in enumerate(sorted(valid, key=lambda x: x['margin'])):
        color = COLORS[i % len(COLORS)]
        ax.scatter(r['AnnVol'] * 100, r['CAGR'] * 100,
                  s=150, c=color, edgecolors='black', linewidth=1, zorder=5,
                  label=f"$\\tau_{{margin}}$={r['margin']}")
        ax.annotate(f"{r['margin']}",
                   xy=(r['AnnVol']*100 + 0.5, r['CAGR']*100 + 0.5),
                   fontsize=9, fontweight='bold')

    ax.set_xlabel('Annualized Volatility (%)')
    ax.set_ylabel('CAGR (%)')
    ax.set_title('Risk-Return Profile by Entropy Margin')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, 'risk_return_scatter.png'))
    plt.savefig(os.path.join(sweep_dir, 'risk_return_scatter.pdf'))
    plt.close()
    print(f"  Saved: risk_return_scatter.png/pdf")


def plot_sensitivity_analysis(
    results: List[Dict[str, Any]],
    sweep_dir: str,
) -> None:
    """
    Line plots showing how each metric changes with entropy margin.
    Useful for understanding the sensitivity of performance to this hyperparameter.
    """
    valid = sorted([r for r in results if 'error' not in r], key=lambda x: x['margin'])
    if len(valid) < 2:
        return

    margins = [r['margin'] for r in valid]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Sharpe Ratio
    ax = axes[0, 0]
    sharpes = [r['Sharpe'] for r in valid]
    ax.plot(margins, sharpes, 'o-', color=COLORS[0], linewidth=2, markersize=8)
    ax.fill_between(margins, sharpes, alpha=0.2, color=COLORS[0])
    ax.set_xlabel('Entropy Margin ($\\tau_{margin}$)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio Sensitivity')
    best_idx = np.argmax(sharpes)
    ax.axvline(x=margins[best_idx], color='green', linestyle='--', alpha=0.7, label=f'Optimal: {margins[best_idx]}')
    ax.legend()

    # CAGR
    ax = axes[0, 1]
    cagrs = [r['CAGR'] * 100 for r in valid]
    ax.plot(margins, cagrs, 's-', color=COLORS[1], linewidth=2, markersize=8)
    ax.fill_between(margins, cagrs, alpha=0.2, color=COLORS[1])
    ax.set_xlabel('Entropy Margin ($\\tau_{margin}$)')
    ax.set_ylabel('CAGR (%)')
    ax.set_title('CAGR Sensitivity')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))

    # Max Drawdown
    ax = axes[1, 0]
    mdd = [r['MaxDD'] * 100 for r in valid]
    ax.plot(margins, mdd, '^-', color=COLORS[4], linewidth=2, markersize=8)
    ax.fill_between(margins, mdd, alpha=0.2, color=COLORS[4])
    ax.set_xlabel('Entropy Margin ($\\tau_{margin}$)')
    ax.set_ylabel('Maximum Drawdown (%)')
    ax.set_title('Maximum Drawdown Sensitivity')

    # Turnover
    ax = axes[1, 1]
    turnover = [r['AvgTurnoverTotal'] * 100 for r in valid]
    ax.plot(margins, turnover, 'D-', color=COLORS[3], linewidth=2, markersize=8)
    ax.fill_between(margins, turnover, alpha=0.2, color=COLORS[3])
    ax.set_xlabel('Entropy Margin ($\\tau_{margin}$)')
    ax.set_ylabel('Avg. Daily Turnover (%)')
    ax.set_title('Turnover Sensitivity')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))

    plt.suptitle('Hyperparameter Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, 'sensitivity_analysis.png'))
    plt.savefig(os.path.join(sweep_dir, 'sensitivity_analysis.pdf'))
    plt.close()
    print(f"  Saved: sensitivity_analysis.png/pdf")


def plot_returns_distribution(
    returns_data: Dict[float, np.ndarray],
    sweep_dir: str,
) -> None:
    """
    Violin plots or histograms showing return distributions for each margin.
    """
    valid_margins = sorted(returns_data.keys())
    if len(valid_margins) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Overlaid histograms
    ax1 = axes[0]
    for i, margin in enumerate(valid_margins):
        color = COLORS[i % len(COLORS)]
        rets = returns_data[margin] * 100  # Convert to percentage
        ax1.hist(rets, bins=50, alpha=0.4, color=color, label=f'$\\tau_{{margin}}$={margin}', density=True)
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Return Distributions')
    ax1.legend(loc='upper right')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # Right: Box plots
    ax2 = axes[1]
    data_for_box = [returns_data[m] * 100 for m in valid_margins]
    bp = ax2.boxplot(data_for_box, labels=[f'{m}' for m in valid_margins], patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.6)
    ax2.set_xlabel('Entropy Margin ($\\tau_{margin}$)')
    ax2.set_ylabel('Daily Return (%)')
    ax2.set_title('(b) Return Distribution Box Plots')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, 'returns_distribution.png'))
    plt.savefig(os.path.join(sweep_dir, 'returns_distribution.pdf'))
    plt.close()
    print(f"  Saved: returns_distribution.png/pdf")


def plot_radar_chart(
    results: List[Dict[str, Any]],
    sweep_dir: str,
) -> None:
    """
    Radar/spider chart comparing normalized metrics across margin values.
    """
    valid = [r for r in results if 'error' not in r]
    if len(valid) < 2:
        return

    # Metrics to include (normalized so higher is always better)
    metrics = ['Sharpe', 'CAGR', 'MaxDD_inv', 'Turnover_inv', 'FinalEquity']
    metric_labels = ['Sharpe\nRatio', 'CAGR', 'Drawdown\n(inverted)', 'Turnover\n(inverted)', 'Final\nEquity']

    # Prepare data
    data = []
    for r in valid:
        row = [
            r['Sharpe'],
            r['CAGR'],
            -r['MaxDD'],  # Invert so higher is better
            1 - r['AvgTurnoverTotal'],  # Invert
            r['FinalEquity'] - 1,  # Normalize from 1
        ]
        data.append(row)
    data = np.array(data)

    # Normalize to 0-1 range
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1  # Avoid division by zero
    data_norm = (data - data_min) / data_range

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, r in enumerate(valid):
        values = data_norm[i].tolist()
        values += values[:1]  # Close the polygon
        color = COLORS[i % len(COLORS)]
        ax.plot(angles, values, 'o-', linewidth=2, label=f"$\\tau_{{margin}}$={r['margin']}", color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title('Normalized Performance Comparison', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, 'radar_comparison.png'))
    plt.savefig(os.path.join(sweep_dir, 'radar_comparison.pdf'))
    plt.close()
    print(f"  Saved: radar_comparison.png/pdf")


def plot_summary_dashboard(
    results: List[Dict[str, Any]],
    equity_data: Dict[float, np.ndarray],
    sweep_dir: str,
    step_size: int = 5,
) -> None:
    """
    Single-page summary dashboard combining key visualizations.
    Suitable for presentations or paper figures.
    """
    valid = sorted([r for r in results if 'error' not in r], key=lambda x: x['margin'])
    if len(valid) < 2:
        return

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1.5, 1, 1])

    # 1. Equity curves (large, left side)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (margin, eq) in enumerate(sorted(equity_data.items())):
        color = COLORS[i % len(COLORS)]
        steps = np.arange(len(eq)) * step_size
        ax1.plot(steps, eq, label=f'$\\tau_{{m}}$={margin}', color=color, linewidth=1.5)
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Portfolio Value')
    ax1.set_title('Equity Curves')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # 2. Sharpe comparison (bar)
    ax2 = fig.add_subplot(gs[0, 1])
    margins = [r['margin'] for r in valid]
    sharpes = [r['Sharpe'] for r in valid]
    best_idx = np.argmax(sharpes)
    colors_bar = [COLORS[0] if i != best_idx else '#2ECC71' for i in range(len(sharpes))]
    ax2.bar(range(len(margins)), sharpes, color=colors_bar, edgecolor='black')
    ax2.set_xticks(range(len(margins)))
    ax2.set_xticklabels([f'{m}' for m in margins])
    ax2.set_xlabel('$\\tau_{margin}$')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio Comparison')

    # 3. Risk-Return scatter
    ax3 = fig.add_subplot(gs[0, 2])
    for i, r in enumerate(valid):
        color = COLORS[i % len(COLORS)]
        ax3.scatter(r['AnnVol']*100, r['CAGR']*100, s=100, c=color, edgecolors='black', zorder=5)
        ax3.annotate(f"{r['margin']}", xy=(r['AnnVol']*100+0.3, r['CAGR']*100+0.3), fontsize=8)
    ax3.set_xlabel('Volatility (%)')
    ax3.set_ylabel('CAGR (%)')
    ax3.set_title('Risk-Return Trade-off')
    ax3.grid(True, alpha=0.3)

    # 4. Drawdown curves
    ax4 = fig.add_subplot(gs[1, 0])
    for i, (margin, eq) in enumerate(sorted(equity_data.items())):
        color = COLORS[i % len(COLORS)]
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak - 1.0) * 100
        steps = np.arange(len(eq)) * step_size
        ax4.fill_between(steps, dd, 0, alpha=0.3, color=color)
        ax4.plot(steps, dd, color=color, linewidth=1)
    ax4.set_xlabel('Trading Days')
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_title('Underwater Curves')
    ax4.set_ylim(top=0)

    # 5. Sensitivity plot
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(margins, sharpes, 'o-', color=COLORS[0], linewidth=2, markersize=8, label='Sharpe')
    ax5.set_xlabel('$\\tau_{margin}$')
    ax5.set_ylabel('Sharpe Ratio', color=COLORS[0])
    ax5.tick_params(axis='y', labelcolor=COLORS[0])
    ax5_twin = ax5.twinx()
    cagrs = [r['CAGR']*100 for r in valid]
    ax5_twin.plot(margins, cagrs, 's--', color=COLORS[3], linewidth=2, markersize=8, label='CAGR')
    ax5_twin.set_ylabel('CAGR (%)', color=COLORS[3])
    ax5_twin.tick_params(axis='y', labelcolor=COLORS[3])
    ax5.set_title('Sensitivity Analysis')

    # 6. Summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Find best
    best = max(valid, key=lambda x: x['Sharpe'])

    table_data = [
        ['Metric', 'Best Value', 'Optimal $\\tau_{margin}$'],
        ['Sharpe', f"{best['Sharpe']:.3f}", f"{best['margin']}"],
        ['CAGR', f"{best['CAGR']:.1%}", f"{max(valid, key=lambda x: x['CAGR'])['margin']}"],
        ['Max DD', f"{min(valid, key=lambda x: x['MaxDD'])['MaxDD']:.1%}",
         f"{min(valid, key=lambda x: x['MaxDD'])['margin']}"],
        ['Final Eq.', f"${best['FinalEquity']:.2f}", f"{best['margin']}"],
    ]

    table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.35, 0.35, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax6.set_title('Optimal Hyperparameters', pad=20)

    plt.suptitle('Entropy Margin Hyperparameter Sweep Summary',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(sweep_dir, 'summary_dashboard.png'))
    plt.savefig(os.path.join(sweep_dir, 'summary_dashboard.pdf'))
    plt.close()
    print(f"  Saved: summary_dashboard.png/pdf")


def generate_latex_table(
    results: List[Dict[str, Any]],
    sweep_dir: str,
) -> None:
    """
    Generate a LaTeX table for academic papers.
    """
    valid = sorted([r for r in results if 'error' not in r], key=lambda x: x['margin'])
    if not valid:
        return

    # Find best for each metric
    best_sharpe_margin = max(valid, key=lambda x: x['Sharpe'])['margin']
    best_cagr_margin = max(valid, key=lambda x: x['CAGR'])['margin']
    best_mdd_margin = min(valid, key=lambda x: x['MaxDD'])['margin']

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Performance Metrics by Target Entropy Margin ($\tau_{margin}$)}
\label{tab:entropy_margin_sweep}
\begin{tabular}{lccccc}
\toprule
$\tau_{margin}$ & Sharpe & CAGR (\%) & Vol. (\%) & Max DD (\%) & Final Equity \\
\midrule
"""

    for r in valid:
        sharpe_fmt = f"\\textbf{{{r['Sharpe']:.3f}}}" if r['margin'] == best_sharpe_margin else f"{r['Sharpe']:.3f}"
        cagr_fmt = f"\\textbf{{{r['CAGR']*100:.1f}}}" if r['margin'] == best_cagr_margin else f"{r['CAGR']*100:.1f}"
        mdd_fmt = f"\\textbf{{{r['MaxDD']*100:.1f}}}" if r['margin'] == best_mdd_margin else f"{r['MaxDD']*100:.1f}"

        latex += f"{r['margin']:.2f} & {sharpe_fmt} & {cagr_fmt} & {r['AnnVol']*100:.1f} & {mdd_fmt} & \\${r['FinalEquity']:.2f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(os.path.join(sweep_dir, 'results_table.tex'), 'w') as f:
        f.write(latex)
    print(f"  Saved: results_table.tex")


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

    # Return stats along with equity curve and returns for plotting
    return {
        "stats": stats,
        "equity": eq,
        "net_returns": net,
    }


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
    step_size = base_cfg.env.lag  # For proper time scaling in plots
    print(f"\nDevice: {device}")

    print("\nLoading data (shared across all experiments)...")
    df_train, df_test, _ = load_and_prepare_data(base_cfg)
    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")

    # Run sweep - collect results and data for plotting
    all_results: List[Dict[str, Any]] = []
    equity_data: Dict[float, np.ndarray] = {}
    returns_data: Dict[float, np.ndarray] = {}
    sweep_start = time.time()

    for i, margin in enumerate(ENTROPY_MARGINS):
        run_dir = os.path.join(sweep_dir, f"margin_{margin}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n[{i+1}/{len(ENTROPY_MARGINS)}] Starting margin={margin}")

        try:
            result = run_single_experiment(
                margin=margin,
                df_train=df_train,
                df_test=df_test,
                base_cfg=base_cfg,
                device=device,
                run_dir=run_dir,
            )

            stats = result["stats"]
            all_results.append(stats)
            equity_data[margin] = result["equity"]
            returns_data[margin] = result["net_returns"]

            print(f"\n--- Results for margin={margin} ---")
            print(f"  CAGR:        {stats['CAGR']:.4f}")
            print(f"  Sharpe:      {stats['Sharpe']:.4f}")
            print(f"  MaxDD:       {stats['MaxDD']:.4f}")
            print(f"  FinalEquity: {stats['FinalEquity']:.4f}")

        except Exception as e:
            print(f"ERROR: margin={margin} failed: {e}")
            import traceback
            traceback.print_exc()
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

    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    if len(valid_results) >= 2 and len(equity_data) >= 2:
        try:
            # 1. Equity curves comparison (linear + log scale)
            plot_equity_curves(equity_data, sweep_dir, step_size=step_size)

            # 2. Drawdown (underwater) curves
            plot_drawdown_curves(equity_data, sweep_dir, step_size=step_size)

            # 3. Bar charts for each metric
            plot_metrics_comparison(valid_results, sweep_dir)

            # 4. Risk-return scatter with Sharpe isolines
            plot_risk_return_scatter(valid_results, sweep_dir)

            # 5. Sensitivity analysis (how metrics change with margin)
            plot_sensitivity_analysis(valid_results, sweep_dir)

            # 6. Return distributions (histograms + box plots)
            plot_returns_distribution(returns_data, sweep_dir)

            # 7. Radar/spider chart for normalized comparison
            plot_radar_chart(valid_results, sweep_dir)

            # 8. Summary dashboard (single-page overview)
            plot_summary_dashboard(valid_results, equity_data, sweep_dir, step_size=step_size)

            # 9. LaTeX table for papers
            generate_latex_table(valid_results, sweep_dir)

            print("\n" + "="*60)
            print("VISUALIZATION COMPLETE")
            print("="*60)
            print(f"\nAll plots saved to: {sweep_dir}")
            print("\nGenerated files:")
            print("  - equity_curves_comparison.png/pdf")
            print("  - drawdown_curves.png/pdf")
            print("  - metrics_comparison.png/pdf")
            print("  - risk_return_scatter.png/pdf")
            print("  - sensitivity_analysis.png/pdf")
            print("  - returns_distribution.png/pdf")
            print("  - radar_comparison.png/pdf")
            print("  - summary_dashboard.png/pdf")
            print("  - results_table.tex")

        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Not enough valid results for visualization (need at least 2).")


if __name__ == "__main__":
    main()
