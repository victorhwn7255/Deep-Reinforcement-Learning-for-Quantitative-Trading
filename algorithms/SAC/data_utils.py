import pandas as pd
import yfinance as yf


def load_market_data(tickers, start, end, auto_adjust=True, progress=False):
    """Download and clean market data from yfinance.

    Handles:
    - MultiIndex extraction (yfinance >= 0.2.0)
    - Index deduplication
    - Ticker verification
    - First valid index alignment
    - NA cleaning

    Args:
        tickers: List of ticker symbols or single ticker string
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        auto_adjust: Use adjusted prices (default True)
        progress: Show download progress (default False)

    Returns:
        DataFrame with Close prices, single-level columns (tickers as column names)

    Raises:
        ValueError: If data download fails or tickers missing

    Example:
        >>> tickers = ['SPY', 'TLT', 'GLD']
        >>> df = load_market_data(tickers, '2020-01-01', '2024-01-01')
        >>> print(df.columns.tolist())
        ['SPY', 'TLT', 'GLD']
    """
    print(f"Downloading data for {tickers}...")
    df_raw = yf.download(tickers, start=start, end=end, progress=progress, auto_adjust=auto_adjust)

    if df_raw.empty:
        raise ValueError("Downloaded data is empty. Check tickers and date range.")

    print(f"✓ Downloaded {len(df_raw)} days of data")

    # Extract Close prices robustly
    if isinstance(df_raw.columns, pd.MultiIndex):
        print("  → Detected MultiIndex columns (yfinance >= 0.2.0)")
        lv0 = df_raw.columns.get_level_values(0)
        lv1 = df_raw.columns.get_level_values(1)

        if "Close" in lv0:
            df = df_raw.xs("Close", axis=1, level=0).copy()
            print("  → Extracted Close prices from level 0")
        elif "Close" in lv1:
            df = df_raw.xs("Close", axis=1, level=1).copy()
            print("  → Extracted Close prices from level 1")
        else:
            raise ValueError("MultiIndex detected but cannot locate 'Close' level.")
    else:
        print("  → Single-level columns (single ticker or old yfinance)")
        if "Close" in df_raw.columns:
            # Single ticker case - yfinance returns simple columns
            df = df_raw[["Close"]].copy()
            ticker_name = tickers[0] if isinstance(tickers, list) else tickers
            df.columns = [ticker_name]
        else:
            # Already has ticker names as columns
            df = df_raw.copy()

    # Basic index cleanup - remove duplicate timestamps
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        print(f"  → Removing {dup_count} duplicate timestamps")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Verify all expected tickers are present
    tick_list = list(tickers) if isinstance(tickers, list) else [tickers]
    missing = [t for t in tick_list if t not in df.columns]
    if missing:
        available = df.columns.tolist()
        raise ValueError(
            f"Missing tickers in downloaded data: {missing}\n"
            f"  Expected: {tick_list}\n"
            f"  Available: {available}"
        )

    print(f"  ✓ Verified all {len(tick_list)} tickers present: {tick_list}")

    # Finance-aware NA handling
    # Different assets start trading at different times (e.g., BTC-USD much later than SPY)
    # Align all tickers to the latest first valid index
    first_valid = df.apply(pd.Series.first_valid_index).max()
    if pd.isna(first_valid):
        raise ValueError("No valid data found for any ticker")

    days_trimmed = (first_valid - df.index.min()).days
    if days_trimmed > 0:
        print(f"  → Trimming {days_trimmed} days from start (aligning ticker histories)")
    df = df.loc[first_valid:].copy()

    # Strict dropna - all tickers must have data on each day
    # (ensures no missing values in environment)
    df = df.dropna(how="any").copy()

    print(f"  ✓ After cleaning: {len(df)} days remaining")
    print(f"    Date range: {df.index.min().date()} to {df.index.max().date()}")

    # Verify minimum data length
    MIN_DAYS_REQUIRED = 100  # Minimum for any meaningful training/eval
    if len(df) < MIN_DAYS_REQUIRED:
        raise ValueError(
            f"Insufficient data after cleaning: {len(df)} days "
            f"(minimum required: {MIN_DAYS_REQUIRED}). "
            f"Try adjusting start date or reviewing data quality."
        )

    return df
