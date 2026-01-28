from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

# NOTE:
# - This module is intentionally "config-aware" but does NOT import Config directly
#   to avoid circular imports. Callers should pass the relevant sub-config objects.


# =============================================================================
# Market data
# =============================================================================

def load_market_data(tickers, start, end, auto_adjust=True, progress=False) -> pd.DataFrame:
    """Download and clean market data from yfinance.

    Returns:
        DataFrame indexed by date with columns = tickers (Close prices, adjusted if auto_adjust=True)
    """
    print(f"Downloading data for {tickers}...")
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise ImportError(
            "yfinance is required for load_market_data (pip install yfinance). "
            f"Import error: {e}"
        )

    df_raw = yf.download(tickers, start=start, end=end, progress=progress, auto_adjust=auto_adjust)

    if df_raw.empty:
        raise ValueError("Downloaded data is empty. Check tickers/date range/internet connectivity.")

    # yfinance may return MultiIndex columns: ('Close', 'SPY') etc
    if isinstance(df_raw.columns, pd.MultiIndex):
        # Prefer "Close" if present, else take the first level heuristically
        if "Close" in df_raw.columns.get_level_values(0):
            df = df_raw["Close"].copy()
        else:
            # best effort: take the first column group
            first_group = df_raw.columns.get_level_values(0)[0]
            df = df_raw[first_group].copy()
    else:
        # Single ticker may come as Series-like DF with OHLC columns; prefer 'Close'
        if "Close" in df_raw.columns:
            df = df_raw[["Close"]].copy()
            df.columns = [tickers] if isinstance(tickers, str) else [tickers[0]]
        else:
            raise ValueError("Unexpected yfinance schema: expected MultiIndex or a 'Close' column.")

    # Ensure DatetimeIndex
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Ensure expected columns exist
    tickers_list = [tickers] if isinstance(tickers, str) else list(tickers)
    missing = [t for t in tickers_list if t not in df.columns]
    if missing:
        raise ValueError(f"Missing tickers in downloaded data: {missing}")

    # Align start at first date where all tickers are non-NA
    first_valid = df.dropna(how="any").index.min()
    if first_valid is None:
        raise ValueError("No date found where all tickers have valid prices.")
    df = df.loc[first_valid:].copy()

    # Forward-fill gaps conservatively, then drop remaining NAs
    df = df.ffill().dropna(how="any")

    print(f"  ✓ Downloaded {len(df)} rows")
    print(f"  ✓ Date range: {df.index.min().date()} to {df.index.max().date()}")

    MIN_DAYS_REQUIRED = 100
    if len(df) < MIN_DAYS_REQUIRED:
        raise ValueError(
            f"Insufficient data after cleaning: {len(df)} days "
            f"(minimum required: {MIN_DAYS_REQUIRED})."
        )

    return df


# =============================================================================
# Macro data loading (VIX / VIX3M / Credit Spread)
# =============================================================================

def _read_macro_csv(
    path: str,
    date_col: str = "observation_date",
    value_col_candidates: Sequence[str] = (),
    rename_to: str = "value",
) -> pd.DataFrame:
    """Read a macro CSV with a date column and one value column (candidate list)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Macro CSV not found: {path}")

    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Expected date column '{date_col}' in {path}. Columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    value_col = None
    for c in value_col_candidates:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        # fallback: if only one non-date column exists, use it
        non_date_cols = [c for c in df.columns]
        if len(non_date_cols) == 1:
            value_col = non_date_cols[0]
        else:
            raise ValueError(
                f"Could not infer value column in {path}. "
                f"Tried: {list(value_col_candidates)}. Available: {list(df.columns)}"
            )

    out = df[[value_col]].rename(columns={value_col: rename_to})
    return out


def load_macro_data(data_cfg) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load VIX, VIX3M, and Credit Spread series using DataConfig candidate columns."""
    vix = _read_macro_csv(
        data_cfg.vix_path,
        value_col_candidates=getattr(data_cfg, "vix_col_candidates", ["VIXCLS", "VIX"]),
        rename_to="VIX",
    )
    vix3m = _read_macro_csv(
        data_cfg.vix3m_path,
        value_col_candidates=getattr(data_cfg, "vix3m_col_candidates", ["VXVCLS", "VIX3M"]),
        rename_to="VIX3M",
    )
    credit = _read_macro_csv(
        data_cfg.credit_spread_path,
        value_col_candidates=getattr(data_cfg, "credit_col_candidates", ["Credit_Spread"]),
        rename_to="Credit_Spread",
    )
    return vix, vix3m, credit


def _to_percent_points(series: pd.Series) -> pd.Series:
    """Convert yield-like series to percentage points.
    FRED is usually already in percent points (3.77). Some vendors use decimals (0.0377).
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    med = float(s.dropna().median()) if s.dropna().shape[0] > 0 else 0.0
    if med != 0.0 and abs(med) < 0.5:
        s = s * 100.0
    return s


def load_yield_curve_data(data_cfg) -> pd.DataFrame:
    """Load yield curve (10Y - 3M) term spread.
    Returns a DF indexed by date with 'YieldCurve_10Y3M_raw' (percent points).
    """
    path = getattr(data_cfg, "yield_curve_path", None)
    if path is None:
        raise ValueError("DataConfig.yield_curve_path is required for yield curve features")

    yc = _read_macro_csv(
        path,
        value_col_candidates=getattr(data_cfg, "yieldcurve_col_candidates", ["T10Y3M"]),
        rename_to="YieldCurve_10Y3M_raw",
    )
    yc["YieldCurve_10Y3M_raw"] = _to_percent_points(yc["YieldCurve_10Y3M_raw"])
    return yc


def load_dxy_data(data_cfg) -> pd.DataFrame:
    """Load Dollar Index (DXY) data.
    Returns a DF indexed by date with 'DXY_raw' column.
    """
    path = getattr(data_cfg, "dxy_path", None)
    if path is None:
        raise ValueError("DataConfig.dxy_path is required for DXY features")

    dxy = _read_macro_csv(
        path,
        value_col_candidates=getattr(data_cfg, "dxy_col_candidates", ["DTWEXBGS", "DXY", "Dollar_Index", "USD"]),
        rename_to="DXY_raw",
    )
    return dxy


# =============================================================================
# Feature engineering
# =============================================================================

def add_technical_features(df_prices: pd.DataFrame, tickers: List[str], feat_cfg) -> pd.DataFrame:
    """Add per-asset technical features required by the environment."""
    df = df_prices.copy()

    try:
        import talib  # type: ignore
    except Exception as e:
        raise ImportError(
            "talib is required for RSI features (pip install TA-Lib / ta-lib). "
            f"Import error: {e}"
        )

    for t in tickers:
        close = df[t].astype(float)
        log_price = np.log(close)

        # RSI
        rsi = talib.RSI(close.values, timeperiod=int(feat_cfg.rsi_period))
        df[f"{t}_RSI"] = (rsi - 50.0) / 50.0

        # Volatility (rolling std of log returns)
        log_ret = log_price.diff()
        vol = log_ret.rolling(int(feat_cfg.volatility_window)).std()
        df[f"{t}_volatility"] = (vol * np.sqrt(252)).clip(0.0, 2.0)

        # 20-day log return (momentum)
        # log(P_t / P_{t-20}) = log(P_t) - log(P_{t-20})
        momentum_window = int(getattr(feat_cfg, "momentum_window", 20))
        ret_20d = log_price.diff(momentum_window)
        # Clip to [-0.5, 0.5] (~40% move in 20 days is extreme)
        df[f"{t}_ret_20d"] = ret_20d.clip(-0.5, 0.5)

    return df


def add_macro_features(df: pd.DataFrame, data_cfg, feat_cfg) -> pd.DataFrame:
    """Join macro series and compute macro features required by the environment."""
    vix, vix3m, credit = load_macro_data(data_cfg)
    yc = load_yield_curve_data(data_cfg)

    macro = vix.join(vix3m, how="outer").join(credit, how="outer").join(yc, how="outer")

    # Load DXY if path is configured
    dxy_path = getattr(data_cfg, "dxy_path", None)
    if dxy_path and os.path.exists(dxy_path):
        try:
            dxy = load_dxy_data(data_cfg)
            macro = macro.join(dxy, how="outer")
        except Exception as e:
            print(f"  Failed to load DXY data: {e}")

    df2 = df.join(macro, how=getattr(data_cfg, "macro_join_how", "left"))

    if getattr(data_cfg, "macro_ffill", True):
        df2["VIX"] = df2["VIX"].ffill()
        df2["VIX3M"] = df2["VIX3M"].ffill()
        df2["Credit_Spread"] = df2["Credit_Spread"].ffill()
        df2["YieldCurve_10Y3M_raw"] = df2["YieldCurve_10Y3M_raw"].ffill()
        if "DXY_raw" in df2.columns:
            df2["DXY_raw"] = df2["DXY_raw"].ffill()

    # -----------------
    # VIX features
    # -----------------
    vix_baseline = float(feat_cfg.vix_baseline)
    df2["VIX_normalized"] = (df2["VIX"] - vix_baseline) / vix_baseline

    low = float(feat_cfg.vix_regime_low)
    high = float(feat_cfg.vix_regime_high)
    df2["VIX_regime"] = np.where(df2["VIX"] < low, -1.0, np.where(df2["VIX"] < high, 0.0, 1.0))

    # term structure: (VIX3M - VIX) / VIX, clipped
    ts = (df2["VIX3M"] - df2["VIX"]) / (df2["VIX"] + 1e-8)
    clip = float(getattr(feat_cfg, "vix_term_structure_clip", 1.0))
    df2["VIX_term_structure"] = np.clip(ts, -clip, clip)

    # -----------------
    # Credit spread features
    # -----------------
    credit_baseline = float(feat_cfg.credit_baseline)
    df2["Credit_Spread_normalized"] = (df2["Credit_Spread"] - credit_baseline) / (credit_baseline + 1e-8)

    c_low = float(feat_cfg.credit_regime_low)
    c_high = float(feat_cfg.credit_regime_high)
    df2["Credit_Spread_regime"] = np.where(df2["Credit_Spread"] < c_low, -1.0, np.where(df2["Credit_Spread"] < c_high, 0.0, 1.0))

    # momentum
    mom_win = int(feat_cfg.credit_momentum_window)
    momentum = df2["Credit_Spread"].pct_change(mom_win)
    df2["Credit_Spread_momentum"] = np.clip(momentum, -float(feat_cfg.credit_momentum_clip), float(feat_cfg.credit_momentum_clip))

    # z-score
    z_win = int(feat_cfg.credit_zscore_window)
    roll_mean = df2["Credit_Spread"].rolling(z_win).mean()
    roll_std = df2["Credit_Spread"].rolling(z_win).std()
    z = (df2["Credit_Spread"] - roll_mean) / (roll_std + 1e-8)
    df2["Credit_Spread_zscore"] = np.clip(z, -float(feat_cfg.credit_zscore_clip), float(feat_cfg.credit_zscore_clip))

    # velocity
    lag = int(feat_cfg.credit_velocity_lag)
    vel = df2["Credit_Spread_momentum"].diff(lag)
    df2["Credit_Spread_velocity"] = np.clip(vel, -float(feat_cfg.credit_velocity_clip), float(feat_cfg.credit_velocity_clip))

    # credit-vix divergence (z-score style on shorter window)
    d_win = int(feat_cfg.credit_divergence_window)
    vix_norm = (df2["VIX"] - df2["VIX"].rolling(d_win).mean()) / (df2["VIX"].rolling(d_win).std() + 1e-8)
    cred_norm = (df2["Credit_Spread"] - df2["Credit_Spread"].rolling(d_win).mean()) / (df2["Credit_Spread"].rolling(d_win).std() + 1e-8)
    div = vix_norm - cred_norm
    df2["Credit_VIX_divergence"] = np.clip(div, -float(feat_cfg.credit_divergence_clip), float(feat_cfg.credit_divergence_clip))
    
    # -----------------
    # Yield curve features
    # -----------------
    slope = pd.to_numeric(df2["YieldCurve_10Y3M_raw"], errors="coerce").astype(float)

    slope_scale = max(float(getattr(feat_cfg, "yield_curve_slope_scale", 3.0)), 1e-8)
    df2["YieldCurve_10Y3M"] = np.clip(slope / slope_scale, -1.0, 1.0)

    chg_lag = int(getattr(feat_cfg, "yield_curve_change_lag", 5))
    chg_scale = max(float(getattr(feat_cfg, "yield_curve_change_scale", 1.0)), 1e-8)
    delta = slope - slope.shift(chg_lag)
    df2["YieldCurve_10Y3M_change"] = np.clip(delta / chg_scale, -1.0, 1.0)

    return df2


def build_feature_dataframe(cfg) -> pd.DataFrame:
    """End-to-end: download prices, add technical + macro features, and clean NAs."""
    tickers = list(cfg.data.tickers)
    df_prices = load_market_data(tickers, cfg.data.start_date, cfg.data.end_date, auto_adjust=True, progress=False)

    df = add_technical_features(df_prices, tickers, cfg.features)
    df = add_macro_features(df, cfg.data, cfg.features)

    # Drop NaNs for features that exist at THIS stage (technical + macro).
    base_cols = []
    for t in tickers:
        for name in cfg.features.per_asset_feature_names:
            base_cols.append(f"{t}_{name}")

    base_cols.extend(cfg.features.macro_feature_columns)

    required_cols = tickers + base_cols
    df = df.dropna(subset=required_cols).copy()

    return df

##################
### Regime HMM ###
##################

def _print_hmm_summary(
    params,
    feature_names: List[str],
    state_names: List[str],
    probs_train: np.ndarray,
    verbose: bool = True,
) -> None:
    """Print a summary of the fitted HMM model."""
    if not verbose:
        return

    n_states = params.n_states
    n_features = len(feature_names)

    print("\n" + "=" * 70)
    print("HMM MODEL SUMMARY")
    print("=" * 70)

    # Features used
    print(f"\nObservation Features ({n_features}):")
    for i, name in enumerate(feature_names):
        print(f"  [{i}] {name}")

    # State means (in scaled space, but we can show interpretation)
    print(f"\nState Means (scaled space):")
    print("-" * 70)
    header = f"{'Feature':<20}" + "".join(f"{name:>12}" for name in state_names)
    print(header)
    print("-" * 70)
    for i, fname in enumerate(feature_names):
        row = f"{fname:<20}"
        for k in range(n_states):
            row += f"{params.means[k, i]:>12.4f}"
        print(row)

    # Transition matrix
    print(f"\nTransition Matrix (row = from, col = to):")
    print("-" * 50)
    header = f"{'From/To':<12}" + "".join(f"{name:>12}" for name in state_names)
    print(header)
    print("-" * 50)
    for i, from_name in enumerate(state_names):
        row = f"{from_name:<12}"
        for j in range(n_states):
            row += f"{params.A[i, j]:>12.1%}"
        print(row)

    # Stickiness (diagonal elements)
    stickiness = np.mean(np.diag(params.A))
    print(f"\nAverage Stickiness: {stickiness:.1%}")

    # Initial state distribution
    print(f"\nInitial State Distribution:")
    for i, name in enumerate(state_names):
        print(f"  {name}: {params.pi[i]:.1%}")

    # Regime distribution in training data
    regime_assignments = np.argmax(probs_train, axis=1)
    print(f"\nRegime Distribution (Training Data):")
    total = len(regime_assignments)
    for k, name in enumerate(state_names):
        count = np.sum(regime_assignments == k)
        pct = count / total * 100
        print(f"  {name}: {count:,} days ({pct:.1f}%)")

    print("=" * 70 + "\n")


def add_regime_hmm_probabilities(df_all: pd.DataFrame, cfg, split_idx: int) -> pd.DataFrame:
    """
    Fit HMM on train slice only ([:split_idx]) and append filtered regime probabilities for all rows.
    """
    if not getattr(cfg.features, "use_regime_hmm", False):
        return df_all

    from regime_hmm import fit_gaussian_hmm_em, forward_filter, reorder_states_by_feature

    verbose = getattr(cfg.experiment, "verbose", True)

    tickers = list(cfg.data.tickers)
    obs_ticker = getattr(cfg.features, "hmm_obs_ticker", "SPY")
    if obs_ticker not in df_all.columns:
        # fallback: pick first ticker (but ideally keep SPY)
        obs_ticker = tickers[0]

    # --- core obs: log return (always included)
    px = df_all[obs_ticker].astype(float)
    ret = np.log(px).diff().fillna(0.0)

    # --- build observation columns list with feature names
    cols = [ret.values]
    feature_names = [f"{obs_ticker}_logret"]

    # --- optional: realized volatility (disabled by default - redundant with VIX)
    if getattr(cfg.features, "hmm_include_rvol", False):
        rvol_win = int(getattr(cfg.features, "hmm_rvol_window", 20))
        rvol = ret.rolling(rvol_win).std().fillna(0.0)
        cols.append(rvol.values)
        feature_names.append(f"{obs_ticker}_rvol_{rvol_win}d")

    # --- optional macro drivers
    if getattr(cfg.features, "hmm_include_vix", True) and "VIX" in df_all.columns:
        cols.append(df_all["VIX"].astype(float).ffill().fillna(0.0).values)
        feature_names.append("VIX")

    if getattr(cfg.features, "hmm_include_credit_spread", True) and "Credit_Spread" in df_all.columns:
        cols.append(df_all["Credit_Spread"].astype(float).ffill().fillna(0.0).values)
        feature_names.append("Credit_Spread")

    if getattr(cfg.features, "hmm_include_vix_term", True) and "VIX_term_structure" in df_all.columns:
        cols.append(df_all["VIX_term_structure"].astype(float).ffill().fillna(0.0).values)
        feature_names.append("VIX_term_struct")

    # --- yield curve change: Δ(T10Y3M) over N days
    if getattr(cfg.features, "hmm_include_yc_change", True):
        if "YieldCurve_10Y3M_raw" in df_all.columns:
            yc_lag = int(getattr(cfg.features, "hmm_yc_change_lag", 5))
            yc_raw = df_all["YieldCurve_10Y3M_raw"].astype(float).ffill()
            yc_change = yc_raw.diff(yc_lag).fillna(0.0).values
            cols.append(yc_change)
            feature_names.append(f"YC_change_{yc_lag}d")
        else:
            print("  ⚠ hmm_include_yc_change=True but YieldCurve_10Y3M_raw not found. "
                  "Ensure use_yield_curve=True in config.")

    # --- DXY log return
    if getattr(cfg.features, "hmm_include_dxy", True):
        if "DXY_raw" in df_all.columns:
            dxy_px = df_all["DXY_raw"].astype(float).ffill()
            dxy_logret = np.log(dxy_px).diff().fillna(0.0).values
            cols.append(dxy_logret)
            feature_names.append("DXY_logret")
        else:
            print("  ⚠ hmm_include_dxy=True but DXY_raw not found in dataframe. "
                  "Ensure dxy_path is correctly set in config.")

    X_all = np.column_stack(cols).astype(np.float64)

    # --- Fit strictly on train slice
    X_train = X_all[:split_idx]

    if verbose:
        print(f"\nFitting HMM on {split_idx:,} training samples with {len(feature_names)} features...")

    params, scaler = fit_gaussian_hmm_em(
        X_train,
        n_states=int(getattr(cfg.features, "hmm_n_states", 3)),
        n_iter=int(getattr(cfg.features, "hmm_n_iter", 50)),
        tol=float(getattr(cfg.features, "hmm_tol", 1e-4)),
        min_var=float(getattr(cfg.features, "hmm_min_var", 1e-4)),
        seed=int(cfg.experiment.seed),
    )

    # --- Reorder states by volatility dimension (index=1 => VIX by default), low->stable
    # Feature order: [ret, (rvol if enabled), VIX, Credit, VIX_term, YC_change, DXY]
    # With hmm_include_rvol=False (default), VIX is at index=1
    params, _perm = reorder_states_by_feature(params, feature_index=1, ascending=True)

    # --- Forward filter on full series in scaled space (filtered probs)
    X_all_scaled = scaler.transform(X_all)
    probs, _ll = forward_filter(params, X_all_scaled)

    # --- Print HMM summary
    prob_cols = list(getattr(cfg.features, "regime_prob_columns", [
        "RegimeP_stable", "RegimeP_trans", "RegimeP_crisis"
    ]))
    state_names = ["Stable", "Transition", "Crisis"]

    probs_train = probs[:split_idx]
    _print_hmm_summary(params, feature_names, state_names, probs_train, verbose=verbose)

    if len(prob_cols) != probs.shape[1]:
        raise ValueError(f"regime_prob_columns length must be {probs.shape[1]} got {len(prob_cols)}")

    out = df_all.copy()
    for i, c in enumerate(prob_cols):
        out[c] = probs[:, i].astype(np.float32)

    return out


# =============================================================================
# Train / test split + canonical entrypoint for train/eval
# =============================================================================

def split_train_test(df: pd.DataFrame, train_split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series split (no shuffling)."""
    if not (0.0 < train_split_ratio < 1.0):
        raise ValueError(f"train_split_ratio must be in (0,1). Got: {train_split_ratio}")

    n = len(df)
    if n < 10:
        raise ValueError(f"Not enough rows to split: {n}")

    split_idx = int(np.floor(n * train_split_ratio))
    split_idx = max(1, min(split_idx, n - 1))

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    return df_train, df_test


def load_and_prepare_data(cfg) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Canonical entrypoint used by BOTH train.py and evaluate.py.

    Returns:
        df_train: DataFrame with prices + features
        df_test:  DataFrame with prices + features
        feature_cols: list of feature columns (NOT including raw price columns)
    """
    df_all = build_feature_dataframe(cfg)

    # split first (so HMM fit uses train only)
    df_train, df_test = split_train_test(df_all, float(cfg.data.train_split_ratio))
    split_idx = len(df_train)

    # append regime probs causally
    if getattr(cfg.features, "use_regime_hmm", False):
        df_all = add_regime_hmm_probabilities(df_all, cfg, split_idx=split_idx)
        df_train = df_all.iloc[:split_idx].copy()
        df_test = df_all.iloc[split_idx:].copy()

    tickers = list(cfg.data.tickers)
    feature_cols = cfg.env.build_feature_columns(tickers, cfg.features)

    # final sanity + dropna including regime columns (if enabled)
    required = tickers + feature_cols
    df_train = df_train.dropna(subset=required).copy()
    df_test = df_test.dropna(subset=required).copy()

    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing required columns after pipeline: {missing}")

    return df_train, df_test, feature_cols