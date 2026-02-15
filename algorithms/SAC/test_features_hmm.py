"""
Comprehensive test script to verify all features and HMM calculations.
Run from the SAC directory: python test_features_hmm.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# Change to the script's directory (SAC folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add current directory to path
sys.path.insert(0, ".")

from config import Config
from data_utils import (
    build_feature_dataframe,
    split_train_test,
    add_regime_hmm_probabilities,
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_feature_bounds(df: pd.DataFrame, col: str, expected_min: float, expected_max: float, tolerance: float = 0.01):
    """Check if a feature is within expected bounds."""
    actual_min = df[col].min()
    actual_max = df[col].max()

    in_bounds = (actual_min >= expected_min - tolerance) and (actual_max <= expected_max + tolerance)
    status = "OK" if in_bounds else "FAIL"

    print(f"  {col:<30} [{actual_min:>8.4f}, {actual_max:>8.4f}]  expected [{expected_min}, {expected_max}]  {status}")

    return in_bounds


def main():
    print_section("LOADING CONFIG")
    cfg = Config()

    # Enable HMM for testing
    cfg.features.use_regime_hmm = True

    print(f"Tickers: {cfg.data.tickers}")
    print(f"Date range: {cfg.data.start_date} to {cfg.data.end_date}")
    print(f"HMM enabled: {cfg.features.use_regime_hmm}")
    print(f"HMM states: {cfg.features.hmm_n_states}")
    print(f"HMM sticky_diag: {cfg.features.hmm_sticky_diag}")
    print(f"HMM n_iter: {cfg.features.hmm_n_iter}")
    print(f"HMM min_var: {cfg.features.hmm_min_var}")

    # =========================================================================
    # STEP 1: Build feature dataframe (without HMM first)
    # =========================================================================
    print_section("STEP 1: BUILD FEATURE DATAFRAME")

    df_all = build_feature_dataframe(cfg)
    print(f"Total rows: {len(df_all)}")
    print(f"Date range: {df_all.index.min()} to {df_all.index.max()}")
    print(f"Columns: {len(df_all.columns)}")

    # =========================================================================
    # STEP 2: Split train/test
    # =========================================================================
    print_section("STEP 2: TRAIN/TEST SPLIT")

    df_train, df_test = split_train_test(df_all, cfg.data.train_split_ratio)
    split_idx = len(df_train)

    print(f"Train rows: {len(df_train)} ({len(df_train)/len(df_all)*100:.1f}%)")
    print(f"Test rows: {len(df_test)} ({len(df_test)/len(df_all)*100:.1f}%)")
    print(f"Train date range: {df_train.index.min()} to {df_train.index.max()}")
    print(f"Test date range: {df_test.index.min()} to {df_test.index.max()}")

    # =========================================================================
    # STEP 3: Check per-asset features
    # =========================================================================
    print_section("STEP 3: PER-ASSET FEATURE BOUNDS")

    all_ok = True
    for ticker in cfg.data.tickers:
        print(f"\n  --- {ticker} ---")
        # RSI: should be in [-1, 1] (normalized from 0-100)
        all_ok &= check_feature_bounds(df_all, f"{ticker}_RSI", -1.0, 1.0)
        # Volatility: should be in [0, 1]
        all_ok &= check_feature_bounds(df_all, f"{ticker}_volatility", 0.0, 1.0)

    # =========================================================================
    # STEP 4: Check macro features
    # =========================================================================
    print_section("STEP 4: MACRO FEATURE BOUNDS")

    # VIX_normalized: [-1, 1]
    all_ok &= check_feature_bounds(df_all, "VIX_normalized", -1.0, 1.0)

    # VIX_term_structure: [-1, 1]
    all_ok &= check_feature_bounds(df_all, "VIX_term_structure", -1.0, 1.0)

    # Credit_Spread_zscore: [-1, 1] (rescaled)
    all_ok &= check_feature_bounds(df_all, "Credit_Spread_zscore", -1.0, 1.0)

    # Credit_Spread_momentum: [-1, 1]
    all_ok &= check_feature_bounds(df_all, "Credit_Spread_momentum", -1.0, 1.0)

    # Credit_Spread_velocity: [-1, 1]
    all_ok &= check_feature_bounds(df_all, "Credit_Spread_velocity", -1.0, 1.0)

    # Credit_VIX_divergence: [-1, 1]
    all_ok &= check_feature_bounds(df_all, "Credit_VIX_divergence", -1.0, 1.0)

    # YieldCurve_10Y3M: [-1, 1]
    all_ok &= check_feature_bounds(df_all, "YieldCurve_10Y3M", -1.0, 1.0)

    # YieldCurve_10Y3M_change: [-1, 1]
    all_ok &= check_feature_bounds(df_all, "YieldCurve_10Y3M_change", -1.0, 1.0)

    # =========================================================================
    # STEP 5: Add HMM regime probabilities
    # =========================================================================
    print_section("STEP 5: HMM REGIME PROBABILITIES")

    df_with_hmm = add_regime_hmm_probabilities(df_all, cfg, split_idx=split_idx)

    # Check regime probability columns exist
    regime_cols = cfg.features.regime_prob_columns
    print(f"\nRegime columns: {regime_cols}")

    for col in regime_cols:
        if col not in df_with_hmm.columns:
            print(f"  ERROR: Missing column {col}")
            all_ok = False
        else:
            # Each probability should be in [0, 1]
            all_ok &= check_feature_bounds(df_with_hmm, col, 0.0, 1.0)

    # =========================================================================
    # STEP 6: Verify regime probabilities sum to 1
    # =========================================================================
    print_section("STEP 6: REGIME PROBABILITY SUM CHECK")

    prob_sum = df_with_hmm[regime_cols].sum(axis=1)
    sum_min = prob_sum.min()
    sum_max = prob_sum.max()
    sum_mean = prob_sum.mean()

    print(f"  Probability sum: min={sum_min:.6f}, max={sum_max:.6f}, mean={sum_mean:.6f}")

    if abs(sum_min - 1.0) > 0.001 or abs(sum_max - 1.0) > 0.001:
        print("  ERROR: Probabilities do not sum to 1.0!")
        all_ok = False
    else:
        print("  OK: All probabilities sum to 1.0")

    # =========================================================================
    # STEP 7: Regime distribution analysis
    # =========================================================================
    print_section("STEP 7: REGIME DISTRIBUTION ANALYSIS")

    # Get most likely regime for each day
    regime_probs = df_with_hmm[regime_cols].values
    regime_assignments = np.argmax(regime_probs, axis=1)

    state_names = ["Bull", "Caution", "Stress", "Crisis"]

    print("\n  --- Full Dataset ---")
    total = len(regime_assignments)
    for k, name in enumerate(state_names):
        count = np.sum(regime_assignments == k)
        pct = count / total * 100
        print(f"    {name:<10}: {count:>5} days ({pct:>5.1f}%)")

    # Train only
    print("\n  --- Training Data ---")
    train_assignments = regime_assignments[:split_idx]
    total_train = len(train_assignments)
    for k, name in enumerate(state_names):
        count = np.sum(train_assignments == k)
        pct = count / total_train * 100
        print(f"    {name:<10}: {count:>5} days ({pct:>5.1f}%)")

    # Test only
    print("\n  --- Test Data ---")
    test_assignments = regime_assignments[split_idx:]
    total_test = len(test_assignments)
    for k, name in enumerate(state_names):
        count = np.sum(test_assignments == k)
        pct = count / total_test * 100
        print(f"    {name:<10}: {count:>5} days ({pct:>5.1f}%)")

    # =========================================================================
    # STEP 8: Check for NaN/Inf in features
    # =========================================================================
    print_section("STEP 8: NaN/Inf CHECK")

    # Get all feature columns
    feature_cols = cfg.env.build_feature_columns(cfg.data.tickers, cfg.features)

    print(f"\n  Total feature columns: {len(feature_cols)}")

    nan_cols = []
    inf_cols = []

    for col in feature_cols:
        if col in df_with_hmm.columns:
            if df_with_hmm[col].isna().any():
                nan_cols.append(col)
            if np.isinf(df_with_hmm[col]).any():
                inf_cols.append(col)

    if nan_cols:
        print(f"  WARNING: Columns with NaN: {nan_cols}")
        all_ok = False
    else:
        print("  OK: No NaN values in feature columns")

    if inf_cols:
        print(f"  WARNING: Columns with Inf: {inf_cols}")
        all_ok = False
    else:
        print("  OK: No Inf values in feature columns")

    # =========================================================================
    # STEP 9: Feature statistics summary
    # =========================================================================
    print_section("STEP 9: FEATURE STATISTICS SUMMARY")

    print(f"\n  {'Feature':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 70)

    for col in feature_cols:
        if col in df_with_hmm.columns:
            mean = df_with_hmm[col].mean()
            std = df_with_hmm[col].std()
            min_val = df_with_hmm[col].min()
            max_val = df_with_hmm[col].max()
            print(f"  {col:<30} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")

    # =========================================================================
    # STEP 10: State dimension verification
    # =========================================================================
    print_section("STEP 10: STATE DIMENSION VERIFICATION")

    from environment import Env

    # Create environment with train data
    df_train_hmm = df_with_hmm.iloc[:split_idx].copy()
    env = Env(df_train_hmm, cfg.data.tickers, cfg)

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")

    # Calculate expected state dimension
    n_assets = len(cfg.data.tickers)
    n_per_asset_features = len(cfg.features.per_asset_feature_names)
    n_macro_features = len(cfg.features.macro_feature_columns)
    # Add discrete regime features if enabled
    n_vix_regime = 1 if getattr(cfg.features, "use_vix_regime", False) else 0
    n_credit_regime = 1 if getattr(cfg.features, "use_credit_regime", False) else 0
    n_hmm_features = len(cfg.features.regime_prob_columns) if cfg.features.use_regime_hmm else 0
    lag = cfg.env.lag
    include_position = cfg.env.include_position_in_state

    total_macro = n_macro_features + n_vix_regime + n_credit_regime
    expected_base = (n_assets * n_per_asset_features + total_macro + n_hmm_features) * lag
    expected_position = (n_assets + 1) if include_position else 0
    expected_total = expected_base + expected_position

    print(f"\n  Expected calculation:")
    print(f"    Per-asset features: {n_assets} tickers x {n_per_asset_features} features = {n_assets * n_per_asset_features}")
    print(f"    Macro features: {n_macro_features} base + {n_vix_regime} VIX_regime + {n_credit_regime} Credit_regime = {total_macro}")
    print(f"    HMM features: {n_hmm_features}")
    print(f"    Features per timestep: {n_assets * n_per_asset_features + total_macro + n_hmm_features}")
    print(f"    Lag window: {lag}")
    print(f"    Base state dim: {expected_base}")
    print(f"    Position features: {expected_position}")
    print(f"    Expected total: {expected_total}")

    if state_dim == expected_total:
        print(f"\n  OK: State dimension matches expected ({state_dim})")
    else:
        print(f"\n  ERROR: State dimension mismatch! Got {state_dim}, expected {expected_total}")
        all_ok = False

    # =========================================================================
    # STEP 11: Test environment step
    # =========================================================================
    print_section("STEP 11: ENVIRONMENT STEP TEST")

    obs = env.reset()
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Initial observation range: [{obs.min():.4f}, {obs.max():.4f}]")

    # Take a random action (uniform weights)
    action = np.ones(action_dim) / action_dim
    next_obs, reward, done = env.step(action)

    print(f"  After step:")
    print(f"    Next observation shape: {next_obs.shape}")
    print(f"    Reward: {reward:.6f}")
    print(f"    Done: {done}")

    stats = env.get_turnover_stats()
    print(f"    Turnover stats: {stats}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("FINAL SUMMARY")

    if all_ok:
        print("\n  ALL TESTS PASSED!")
        print("\n  The 4-state HMM and all features are working correctly.")
    else:
        print("\n  SOME TESTS FAILED - Please review the errors above.")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
