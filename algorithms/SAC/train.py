"""
SAC portfolio training entrypoint (Config-driven).

Key goals:
- ALL knobs live in config.py (Config dataclass).
- Feature engineering is centralized in data_utils.py to avoid train/eval drift.
- Environment mechanics + transaction costs come from cfg.env.
- Saves: final model, best model (also saved inside agent.learn), interrupted model, config snapshot.
- Optional training plots controlled by cfg.training.plot_after_training.

Run:
  python train.py
"""

from __future__ import annotations

import os
import time
import json
import random
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import Dirichlet

from config import Config, get_default_config
from data_utils import build_feature_dataframe, split_train_test
from environment import Env
from agent import Agent


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_global_seeds(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism can slow training; use only when you really need reproducibility
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ema_smooth(x: List[float], alpha: float = 0.05) -> List[float]:
    if len(x) == 0:
        return []
    y = [x[0]]
    for xi in x[1:]:
        y.append(alpha * xi + (1.0 - alpha) * y[-1])
    return y


def save_training_metrics(
    path: str,
    episode_returns: List[float],
    losses: List[Dict[str, Any]],
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "episode_returns": episode_returns,
        "losses": losses,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def plot_training(
    episode_returns: List[float],
    losses: List[Dict[str, Any]],
    out_path: str,
) -> None:
    if len(episode_returns) == 0:
        print("No episode returns to plot.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Extract loss curves if present
    q1 = [d["q1_loss"] for d in losses if isinstance(d, dict) and "q1_loss" in d]
    q2 = [d["q2_loss"] for d in losses if isinstance(d, dict) and "q2_loss" in d]
    pol = [d["policy_loss"] for d in losses if isinstance(d, dict) and "policy_loss" in d]
    alp = [d["alpha"] for d in losses if isinstance(d, dict) and "alpha" in d]

    ep_ret_pct = [r * 100.0 for r in episode_returns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode returns
    ax = axes[0, 0]
    ax.plot(ep_ret_pct, alpha=0.25, linewidth=0.7)
    ax.plot(ema_smooth(ep_ret_pct, alpha=0.05), linewidth=2.0)
    ax.axhline(0, linestyle="--", alpha=0.4)
    ax.set_title("Episode Returns (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3)

    # Q losses
    ax = axes[0, 1]
    if q1:
        ax.plot(ema_smooth(q1, alpha=0.02), linewidth=1.2, label="Q1 loss")
    if q2:
        ax.plot(ema_smooth(q2, alpha=0.02), linewidth=1.2, label="Q2 loss")
    ax.set_title("Critic Losses (EMA)")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Policy loss
    ax = axes[1, 0]
    if pol:
        ax.plot(ema_smooth(pol, alpha=0.02), linewidth=1.2)
    ax.set_title("Policy Loss (EMA)")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Alpha
    ax = axes[1, 1]
    if alp:
        ax.plot(alp, linewidth=1.2)
    ax.set_title("Entropy Temperature (alpha)")
    ax.set_xlabel("Update step")
    ax.set_ylabel("alpha")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"✓ Saved training plot to: {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(cfg: Optional[Config] = None) -> None:
    cfg = cfg or get_default_config()

    print("=" * 80)
    print("SAC PORTFOLIO TRAINING (CONFIG-DRIVEN)")
    print("=" * 80)

    # Output dirs + snapshot config
    cfg.ensure_output_dirs()
    cfg_path = cfg.save_json()
    print(f"✓ Config snapshot saved: {cfg_path}")

    # Seeds
    set_global_seeds(cfg.training.seed, cfg.training.deterministic_torch)
    print(f"✓ Seed set: {cfg.training.seed} | deterministic_torch={cfg.training.deterministic_torch}")

    # Device
    device = cfg.auto_detect_device()
    print(f"✓ Device: {device}")

    # ------------------------------------------------------------------
    # Build dataset (prices + engineered features)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: BUILD FEATURE DATAFRAME")
    print("=" * 80)

    df = build_feature_dataframe(cfg)
    df_train, df_test = split_train_test(df, cfg)

    print(f"✓ Full data rows:  {len(df):,}")
    print(f"✓ Train rows:      {len(df_train):,}  ({df_train.index[0].date()} -> {df_train.index[-1].date()})")
    print(f"✓ Test rows:       {len(df_test):,}   ({df_test.index[0].date()} -> {df_test.index[-1].date()})")

    # ------------------------------------------------------------------
    # Create environment (training)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: CREATE TRAIN ENVIRONMENT")
    print("=" * 80)

    tickers = list(cfg.data.tickers or [])
    env = Env(
        df=df_train,
        tickers=tickers,
        lag=cfg.env.lag,
        tc_rate=cfg.env.tc_rate,
        include_position_in_state=cfg.env.include_position_in_state,
        turnover_include_cash=cfg.env.turnover_include_cash,
        turnover_use_half_factor=cfg.env.turnover_use_half_factor,
        turnover_threshold=cfg.env.turnover_threshold,
        tc_fixed=cfg.env.tc_fixed,
        reward_scale=cfg.env.reward_scale,
    )

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    print(f"✓ state_dim={state_dim} | action_dim={action_dim}")

    # ------------------------------------------------------------------
    # Target entropy (Dirichlet)
    # ------------------------------------------------------------------
    target_entropy: Optional[float] = None
    if cfg.sac.auto_entropy_tuning:
        if cfg.sac.target_entropy is not None:
            target_entropy = float(cfg.sac.target_entropy)
        else:
            # Max entropy on simplex is at alpha=1 (uniform Dirichlet); use margin below max
            with torch.no_grad():
                uniform_alpha = torch.ones(action_dim, device=device)
                h_max = Dirichlet(uniform_alpha).entropy().item()
            target_entropy = float(h_max - cfg.sac.target_entropy_margin)

        print(f"✓ Auto-entropy tuning enabled | target_entropy={target_entropy:.6f}")

    # ------------------------------------------------------------------
    # Initialize agent
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: INIT SAC AGENT")
    print("=" * 80)

    # NOTE: current Agent takes ONE learning_rate for all nets.
    # We pass cfg.sac.actor_lr as the single LR knob (you can later refactor Agent to accept per-net LRs).
    agent = Agent(
        n_input=state_dim,
        n_action=action_dim,
        learning_rate=cfg.sac.actor_lr,
        gamma=cfg.sac.gamma,
        tau=cfg.sac.tau,
        alpha=cfg.sac.alpha_init,
        auto_entropy_tuning=cfg.sac.auto_entropy_tuning,
        target_entropy=target_entropy,
        buffer_size=cfg.sac.buffer_size,
        batch_size=cfg.sac.batch_size,
        learning_starts=cfg.sac.learning_starts,
        update_frequency=cfg.sac.update_frequency,
        n_hidden=cfg.network.n_hidden,
        device=device,
    )

    print("✓ Agent initialized")
    print(f"  total_timesteps: {cfg.training.total_timesteps:,}")
    print(f"  gamma: {cfg.sac.gamma} | tau: {cfg.sac.tau}")
    print(f"  alpha_init: {cfg.sac.alpha_init} | auto_entropy_tuning: {cfg.sac.auto_entropy_tuning}")
    print(f"  buffer_size: {cfg.sac.buffer_size:,} | batch_size: {cfg.sac.batch_size}")
    print(f"  learning_starts: {cfg.sac.learning_starts:,} | update_frequency: {cfg.sac.update_frequency}")
    print(f"  learning_rate (shared): {cfg.sac.actor_lr}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: TRAIN")
    print("=" * 80)

    t0 = time.time()
    try:
        episode_returns, losses, best_model_state = agent.learn(env, cfg.training.total_timesteps)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted. Saving interrupted checkpoint...")
        agent.save_model(cfg.training.model_path_interrupted)
        print(f"✓ Saved: {cfg.training.model_path_interrupted}")
        return

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("TRAINING DONE")
    print("=" * 80)
    print(f"Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.2f} hours)")

    # ------------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: SAVE CHECKPOINTS")
    print("=" * 80)

    agent.save_model(cfg.training.model_path_final)
    print(f"✓ Saved final: {cfg.training.model_path_final}")

    # agent.learn already saves best to "models/sac_portfolio_best.pth",
    # but we also save to cfg.training.model_path_best to ensure consistency.
    if best_model_state is not None:
        os.makedirs(os.path.dirname(cfg.training.model_path_best) or ".", exist_ok=True)
        torch.save(best_model_state, cfg.training.model_path_best)
        print(f"✓ Saved best:  {cfg.training.model_path_best}")
        print(
            f"  best episode={best_model_state.get('episode')} | "
            f"global_step={best_model_state.get('global_step')} | "
            f"avg_return(last10)={best_model_state.get('avg_return')}"
        )
    else:
        print("⚠ No best_model_state returned (maybe too few episodes).")

    # Save metrics JSON (helpful for later analysis)
    metrics_path = os.path.join(cfg.training.model_dir, "sac_training_metrics.json")
    save_training_metrics(metrics_path, episode_returns, losses)
    print(f"✓ Saved metrics: {metrics_path}")

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    if cfg.training.plot_after_training:
        print("\n" + "=" * 80)
        print("STEP 6: PLOT")
        print("=" * 80)
        plot_training(episode_returns, losses, cfg.training.plots_path_training)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if episode_returns:
        print(f"Episodes: {len(episode_returns)}")
        print(f"Avg episode return: {float(np.mean(episode_returns)):.6f}")
        print(f"Best episode return: {float(np.max(episode_returns)):.6f}")
        print(f"Last 10 avg return: {float(np.mean(episode_returns[-10:])):.6f}" if len(episode_returns) >= 10 else "")
    print(f"Best model:  {cfg.training.model_path_best}")
    print(f"Final model: {cfg.training.model_path_final}")
    print("\nNext: run evaluation/backtest with evaluate.py (also config-driven).")


if __name__ == "__main__":
    main()
