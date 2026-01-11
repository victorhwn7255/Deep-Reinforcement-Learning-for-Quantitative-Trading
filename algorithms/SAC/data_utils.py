"""Data loading + feature engineering.

This module is shared by train.py and evaluate.py to avoid *train/eval feature drift*.

It is intentionally driven by `Config` (config.py) so that all knobs live in one place.

Key functions:
- load_market_data(...): downloads Close prices for tickers from yfinance.
- build_feature_dataframe(cfg): returns a single DataFrame containing:
    * price columns (tickers)
    * macro columns used for features (e.g., VIX, VIX3M, raw credit series)
    * engineered feature columns matching cfg.feature_columns()

Notes:
- This code expects `talib` to be installed (you already use it in train.py).
- Macro CSVs are expected to have a date column specified by cfg.data.macro_date_column.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import talib

from config import Config


def _try_alt_paths(path: str) -> List[str]:
    """Return a few candidate paths to handle common case / naming variants."""
    cands = [path]
    cands.append(path.replace("Credit_Spread", "CREDIT_SPREAD"))
    cands.append(path.replace("CREDIT_SPREAD", "Credit_Spread"))
    return list(dict.fromkeys(cands))


def _read_macro_csv(path: str, date_col: str) -> pd.DataFrame:
    """Read a macro CSV, parse its date column, and set the date as index."""
    last_err: Optional[Exception] = None
    for p in _try_alt_paths(path):
        try:
            df = pd.read_csv(p)
            if date_col not in df.columns:
                raise ValueError(
                    f"Expected date column '{date_col}' not found in {p}. "
                    f"Columns={df.columns.tolist()}"
                )
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            return df
        except Exception as e:
            last_err = e
            continue

    raise FileNotFoundError(f"Failed to read macro csv at '{path}'. Last error: {last_err}")


def load_market_data(
    tickers: List[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
    progress: bool = False,
    min_days_required: int = 100,
) -> pd.DataFrame:
    """Download and clean Close price data from yfinance.

    Handles:
    - MultiIndex extraction (yfinance newer versions)
    - Index de-duplication
    - Ticker verification
    - Aligning histories to the latest first-valid date
    - Strict dropna for clean environment input
    """
    tick_list = list(tickers)
    print(f"Downloading data for {tick_list}...")

    df_raw = yf.download(
        tick_list,
        start=start,
        end=end,
        progress=progress,
        auto_adjust=auto_adjust,
    )

    if df_raw.empty:
        raise ValueError("Downloaded data is empty. Check tickers and date range.")

    # Extract Close prices robustly
    if isinstance(df_raw.columns, pd.MultiIndex):
        lv0 = df_raw.columns.get_level_values(0)
        lv1 = df_raw.columns.get_level_values(1)

        if "Close" in lv0:
            df = df_raw.xs("Close", axis=1, level=0).copy()
        elif "Close" in lv1:
            df = df_raw.xs("Close", axis=1, level=1).copy()
        else:
            raise ValueError("MultiIndex detected but cannot locate 'Close' level.")
    else:
        # Single ticker or older yfinance
        if "Close" in df_raw.columns:
            df = df_raw[["Close"]].copy()
            df.columns = [tick_list[0]]
        else:
            df = df_raw.copy()

    # Remove duplicate timestamps and sort
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Verify expected tickers
    missing = [t for t in tick_list if t not in df.columns]
    if missing:
        raise ValueError(
            f"Missing tickers in downloaded data: {missing}\n"
            f"Expected: {tick_list}\n"
            f"Available: {df.columns.tolist()}"
        )

    # Align all tickers to the latest first valid index (assets can start later, e.g. BTC-USD)
    first_valid = df.apply(pd.Series.first_valid_index).max()
    if pd.isna(first_valid):
        raise ValueError("No valid data found for any ticker")
    df = df.loc[first_valid:].copy()

    # Strict NA removal: each day must have data for all tickers
    df = df.dropna(how="any").copy()

    if len(df) < int(min_days_required):
        raise ValueError(
            f"Insufficient data after cleaning: {len(df)} days (min required: {min_days_required})."
        )

    return df


def build_feature_dataframe(cfg: Config) -> pd.DataFrame:
    """Download prices + join macro series + engineer all features defined by cfg."""
    tickers = list(cfg.data.tickers or [])
    if not tickers:
        raise ValueError("cfg.data.tickers is empty")

    df = load_market_data(
        tickers=tickers,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        auto_adjust=cfg.data.auto_adjust,
        progress=cfg.data.progress,
        min_days_required=cfg.data.min_days_required,
    )

    # ------------------------------------------------------------------
    # VIX / VIX3M features
    # ------------------------------------------------------------------
    if cfg.features.include_vix:
        vix_df = _read_macro_csv(cfg.data.vix_path, cfg.data.macro_date_column)
        if cfg.data.vix_value_column not in vix_df.columns:
            raise ValueError(
                f"Expected VIX column '{cfg.data.vix_value_column}' not found in {cfg.data.vix_path}. "
                f"Columns={vix_df.columns.tolist()}"
            )
        vix_df = vix_df.rename(columns={cfg.data.vix_value_column: "VIX"})

        vix3m_df = _read_macro_csv(cfg.data.vix3m_path, cfg.data.macro_date_column)
        if cfg.data.vix3m_value_column not in vix3m_df.columns:
            raise ValueError(
                f"Expected VIX3M column '{cfg.data.vix3m_value_column}' not found in {cfg.data.vix3m_path}. "
                f"Columns={vix3m_df.columns.tolist()}"
            )
        vix3m_df = vix3m_df.rename(columns={cfg.data.vix3m_value_column: "VIX3M"})

        vix_combined = vix_df.join(vix3m_df, how="outer")
        df = df.join(vix_combined, how="left")
        df["VIX"] = df["VIX"].ffill()
        df["VIX3M"] = df["VIX3M"].ffill()

        df["VIX_normalized"] = (df["VIX"] - cfg.features.vix_baseline) / cfg.features.vix_baseline

        def _vix_regime(v: float) -> float:
            if v < cfg.features.vix_regime_low:
                return -1.0
            if v < cfg.features.vix_regime_high:
                return 0.0
            return 1.0

        df["VIX_regime"] = df["VIX"].apply(_vix_regime)

        ts = (df["VIX3M"] - df["VIX"]) / (df["VIX"] + 1e-12)
        df["VIX_term_structure"] = np.clip(
            ts,
            cfg.features.vix_term_structure_clip_min,
            cfg.features.vix_term_structure_clip_max,
        )

    # ------------------------------------------------------------------
    # Credit spread features
    # ------------------------------------------------------------------
    if cfg.features.include_credit_spread:
        credit_df = _read_macro_csv(cfg.data.credit_spread_path, cfg.data.macro_date_column)
        if cfg.data.credit_value_column not in credit_df.columns:
            raise ValueError(
                f"Expected credit spread column '{cfg.data.credit_value_column}' not found in {cfg.data.credit_spread_path}. "
                f"Columns={credit_df.columns.tolist()}"
            )

        df = df.join(credit_df[[cfg.data.credit_value_column]], how="left")
        df[cfg.data.credit_value_column] = df[cfg.data.credit_value_column].ffill()

        credit_raw = df[cfg.data.credit_value_column].astype(float)
        if cfg.features.credit_is_percent:
            credit_raw = credit_raw / 100.0
        df["Credit_Spread"] = credit_raw

        df["Credit_Spread_normalized"] = (
            (df["Credit_Spread"] - cfg.features.credit_baseline) / cfg.features.credit_scale
        )

        def _credit_regime(sp: float) -> float:
            if sp < cfg.features.credit_regime_low:
                return -1.0
            if sp < cfg.features.credit_regime_high:
                return 0.0
            return 1.0

        df["Credit_Spread_regime"] = df["Credit_Spread"].apply(_credit_regime)

        mom = df["Credit_Spread"].pct_change(cfg.features.credit_momentum_window)
        df["Credit_Spread_momentum"] = np.clip(
            mom,
            cfg.features.credit_momentum_clip_min,
            cfg.features.credit_momentum_clip_max,
        )

        roll = df["Credit_Spread"].rolling(cfg.features.credit_zscore_lookback)
        z = (df["Credit_Spread"] - roll.mean()) / (roll.std() + 1e-8)
        df["Credit_Spread_zscore"] = np.clip(
            z,
            cfg.features.credit_zscore_clip_min,
            cfg.features.credit_zscore_clip_max,
        )

        vel = df["Credit_Spread_momentum"].diff(cfg.features.credit_velocity_lag)
        df["Credit_Spread_velocity"] = np.clip(
            vel,
            cfg.features.credit_velocity_clip_min,
            cfg.features.credit_velocity_clip_max,
        )

        # Divergence feature: compare z-scored VIX vs z-scored credit
        if cfg.features.include_vix:
            vix_z = (df["VIX"] - df["VIX"].rolling(cfg.features.divergence_window).mean()) / (
                df["VIX"].rolling(cfg.features.divergence_window).std() + 1e-8
            )
        else:
            vix_z = 0.0

        credit_z = (
            (df["Credit_Spread"] - df["Credit_Spread"].rolling(cfg.features.divergence_window).mean())
            / (df["Credit_Spread"].rolling(cfg.features.divergence_window).std() + 1e-8)
        )
        div = vix_z - credit_z
        df["Credit_VIX_divergence"] = np.clip(
            div,
            cfg.features.divergence_clip_min,
            cfg.features.divergence_clip_max,
        )

    # ------------------------------------------------------------------
    # RSI features
    # ------------------------------------------------------------------
    if cfg.features.include_rsi:
        for t in tickers:
            rsi = talib.RSI(df[t].values.astype(float), timeperiod=cfg.features.rsi_period)
            df[f"{t}_RSI"] = (rsi / cfg.features.rsi_divisor) - cfg.features.rsi_shift

    # ------------------------------------------------------------------
    # Realized volatility features
    # ------------------------------------------------------------------
    if cfg.features.include_volatility:
        ann = np.sqrt(cfg.features.volatility_annualization)
        for t in tickers:
            ret = df[t].pct_change()
            vol = ret.rolling(cfg.features.volatility_window).std() * ann
            df[f"{t}_volatility"] = (vol - cfg.features.volatility_baseline) / cfg.features.volatility_scale

    # Drop rows with any NA from rolling computations
    df = df.dropna(how="any").copy()

    # Validate that all expected feature columns exist
    required = cfg.feature_columns()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Feature engineering produced missing columns: {missing}")

    return df


def split_train_test(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience helper to split by cfg.data.train_split_ratio."""
    if not (0.0 < cfg.data.train_split_ratio < 1.0):
        raise ValueError(f"train_split_ratio must be in (0,1). Got {cfg.data.train_split_ratio}")

    n_train = int(cfg.data.train_split_ratio * len(df))
    if n_train <= 0 or n_train >= len(df):
        raise ValueError(
            f"train/test split results in empty set: n_train={n_train}, n_total={len(df)}"
        )

    df_train = df.iloc[:n_train].copy()
    df_test = df.iloc[n_train:].copy()
    return df_train, df_test