from __future__ import annotations

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from config import get_default_config
from data_utils import load_and_prepare_data
from environment import Env
from agent import Agent


def _maybe_save_fig(cfg, filename: str) -> None:
    if not cfg.experiment.save_resolved_config:
        return
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.experiment.output_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved plot: {out_path}")


def main():
    print("=" * 60)
    print("SAC v2 PORTFOLIO TRAINING")
    print("=" * 60)

    cfg = get_default_config()

    # Enable HMM regime features
    cfg.features.use_regime_hmm = True

    cfg.ensure_dirs()
    cfg.set_global_seeds()
    device = cfg.auto_detect_device()
    cfg.print_summary()
    print(f"\n✓ Using device: {device}\n")

    # Save config next to models for reproducibility
    if cfg.experiment.save_resolved_config:
        cfg.save_json(os.path.join(cfg.training.model_dir, "config.json"))

    # -------------------------
    # Data
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 1: LOAD + PREPARE DATA")
    print("=" * 60)

    t0 = time.time()
    df_train, df_test, _feature_cols = load_and_prepare_data(cfg)
    print(f"✓ Train rows: {len(df_train)}")
    print(f"✓ Test  rows: {len(df_test)}")
    print(f"✓ Data prep time: {(time.time() - t0):.1f}s")
    
    # ---- sanity check for HMM regime features ----
    if getattr(cfg.features, "use_regime_hmm", False):
        pcols = cfg.features.regime_prob_columns
        missing = [c for c in pcols if c not in df_train.columns]
        if missing:
            raise ValueError(f"Missing HMM regime columns in df_train: {missing}")
        s = df_train[pcols].sum(axis=1)
        print(f"✓ Regime prob sum min/max: {float(s.min()):.6f} / {float(s.max()):.6f}")


    # -------------------------
    # Env + Agent
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 2: INIT ENV + AGENT")
    print("=" * 60)

    env = Env(df_train, cfg.data.tickers, cfg)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    print(f"✓ Env dims: state_dim={state_dim}, action_dim={action_dim}")

    agent = Agent(state_dim, action_dim, cfg, device=device)

    # Resume / warm start
    if cfg.training.resume_from:
        print(f"✓ Resuming from checkpoint: {cfg.training.resume_from}")
        if not os.path.exists(cfg.training.resume_from):
            raise FileNotFoundError(f"resume_from checkpoint not found: {cfg.training.resume_from}")
        try:
            agent.load_model(cfg.training.resume_from)
            print("✓ Resume successful.")
        except KeyError as e:
            raise RuntimeError(
                "Failed to load checkpoint. Likely a SAC v1 checkpoint being loaded into SAC v2.\n"
                "Fix: point resume_from to a SAC v2 checkpoint or retrain from scratch.\n"
                f"Original error: {e}"
            ) from e

    # -------------------------
    # Train
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 3: TRAIN")
    print("=" * 60)

    train_start = time.time()
    episode_returns, losses, best_model_state = agent.learn(
        env,
        total_timesteps=int(cfg.training.total_timesteps),
    )
    train_time = time.time() - train_start
    print(f"\n✓ Training complete in {train_time / 60:.1f} minutes")

    # -------------------------
    # Save
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 4: SAVE MODELS")
    print("=" * 60)

    agent.save_model(cfg.training.model_path_final)
    print(f"✓ Saved final model: {cfg.training.model_path_final}")

    if best_model_state is not None:
        os.makedirs(cfg.training.model_dir, exist_ok=True)
        torch.save(best_model_state, cfg.training.model_path_best)
        print(f"✓ Saved best model:  {cfg.training.model_path_best}")
    else:
        print("ℹ No best_model_state returned (may occur if <10 episodes or no improvement).")

    # -------------------------
    # Plots
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 5: PLOTS")
    print("=" * 60)

    if len(episode_returns) > 0:
        plt.figure()
        plt.plot(episode_returns)
        plt.title("Episode Return (Reward Sum)")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        _maybe_save_fig(cfg, f"{cfg.experiment.run_name}_episode_returns.png")
        plt.show()

    if len(losses) > 0:
        keys = ["q1_loss", "q2_loss", "policy_loss", "alpha_loss", "avg_logp"]
        for k in keys:
            vals = [d[k] for d in losses if k in d]
            if not vals:
                continue
            plt.figure()
            plt.plot(vals)
            plt.title(k)
            plt.xlabel("Update")
            plt.ylabel(k)
            _maybe_save_fig(cfg, f"{cfg.experiment.run_name}_{k}.png")
            plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()

