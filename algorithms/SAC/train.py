from __future__ import annotations

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from config import get_default_config
from data_utils import build_feature_dataframe
from environment import Env
from agent import Agent


def main():
    print("=" * 60)
    print("SAC PORTFOLIO TRAINING")
    print("=" * 60)

    # -------------------------
    # Load config (all knobs in one place)
    # -------------------------
    cfg = get_default_config()

    cfg.ensure_dirs()
    cfg.set_global_seeds()
    device = cfg.auto_detect_device()
    cfg.print_summary()
    print(f"\n✓ Using device: {device}\n")

    # Save config next to checkpoints for reproducibility
    if cfg.experiment.save_resolved_config:
        cfg.save_json(os.path.join(cfg.training.model_dir, "config.json"))

    # -------------------------
    # Build dataset + features
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 1: BUILD FEATURE DATAFRAME")
    print("=" * 60)

    t0 = time.time()
    df = build_feature_dataframe(cfg)
    print(f"✓ Feature dataframe ready: {df.shape} (built in {(time.time()-t0):.1f}s)")

    # Train/test split (chronological)
    split_idx = int(len(df) * float(cfg.data.train_split_ratio))
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    print(f"✓ Split: train={len(df_train)} rows, test={len(df_test)} rows")

    # -------------------------
    # Environment + Agent
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 2: INIT ENV + AGENT")
    print("=" * 60)

    env = Env(df_train, cfg.data.tickers, cfg)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    print(f"✓ Env dims: state_dim={state_dim}, action_dim={action_dim}")

    agent = Agent(state_dim, action_dim, cfg, device=device)

    # Optional warm start
    if cfg.training.resume_from:
        print(f"✓ Resuming from checkpoint: {cfg.training.resume_from}")
        agent.load_model(cfg.training.resume_from)

    # -------------------------
    # Train
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 3: TRAIN")
    print("=" * 60)

    train_start = time.time()
    episode_returns, losses, best_model_state = agent.learn(env, total_timesteps=cfg.training.total_timesteps)
    train_time = time.time() - train_start
    print(f"\n✓ Training complete in {train_time/60:.1f} minutes")

    # -------------------------
    # Save final / best
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 4: SAVE MODELS")
    print("=" * 60)

    agent.save_model(cfg.training.model_path_final)
    print(f"✓ Final model saved: {cfg.training.model_path_final}")

    if best_model_state is not None:
        torch.save(best_model_state, cfg.training.model_path_best)
        print(f"✓ Best model saved: {cfg.training.model_path_best}")
    else:
        print("⚠ No best model captured (need >= 10 episodes to compute rolling best).")

    # -------------------------
    # Simple plots
    # -------------------------
    if len(episode_returns) > 0:
        plt.figure()
        plt.plot(episode_returns)
        plt.title("Episode Returns (reward units)")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        if cfg.experiment.save_resolved_config:
            out_path = os.path.join(cfg.experiment.output_dir, f"{cfg.experiment.run_name}_episode_returns.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved plot: {out_path}")
        plt.show()

    if len(losses) > 0:
        # Plot a few common losses
        keys = ["q1_loss", "q2_loss", "policy_loss", "value_loss"]
        for k in keys:
            vals = [d[k] for d in losses if k in d]
            if len(vals) > 0:
                plt.figure()
                plt.plot(vals)
                plt.title(k)
                plt.xlabel("Update")
                plt.ylabel(k)
                if cfg.experiment.save_resolved_config:
                    out_path = os.path.join(cfg.experiment.output_dir, f"{cfg.experiment.run_name}_{k}.png")
                    plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()