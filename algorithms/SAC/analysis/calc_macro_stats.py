"""Calculate VIX and Credit Spread statistics from 2010-2024 data."""
import pandas as pd

# =============================================================================
# VIX Statistics
# =============================================================================
df = pd.read_csv('../../../data/VIX_CLS_2010_2024.csv', parse_dates=['observation_date'])
df['VIXCLS'] = pd.to_numeric(df['VIXCLS'], errors='coerce')
df = df.dropna()

print('=' * 50)
print('VIX Statistics (2010-2024):')
print('=' * 50)
print(f'Count:    {len(df)} days')
print(f'Mean:     {df["VIXCLS"].mean():.2f}')
print(f'Std Dev:  {df["VIXCLS"].std():.2f}')
print(f'Median:   {df["VIXCLS"].median():.2f}')
print(f'Min:      {df["VIXCLS"].min():.2f}')
print(f'Max:      {df["VIXCLS"].max():.2f}')
print()
print('Percentiles:')
print(f'  10th:   {df["VIXCLS"].quantile(0.10):.2f}')
print(f'  25th:   {df["VIXCLS"].quantile(0.25):.2f}')
print(f'  50th:   {df["VIXCLS"].quantile(0.50):.2f}')
print(f'  75th:   {df["VIXCLS"].quantile(0.75):.2f}')
print(f'  90th:   {df["VIXCLS"].quantile(0.90):.2f}')
print()
print('Regime distribution with current thresholds (low=15, high=30):')
low_count = (df['VIXCLS'] < 15).sum()
mid_count = ((df['VIXCLS'] >= 15) & (df['VIXCLS'] < 30)).sum()
high_count = (df['VIXCLS'] >= 30).sum()
print(f'  VIX < 15:       {low_count} days ({100*low_count/len(df):.1f}%)')
print(f'  15 <= VIX < 30: {mid_count} days ({100*mid_count/len(df):.1f}%)')
print(f'  VIX >= 30:      {high_count} days ({100*high_count/len(df):.1f}%)')

# =============================================================================
# Credit Spread Statistics
# =============================================================================
print()
print('=' * 50)
print('Credit Spread Statistics (2010-2024):')
print('=' * 50)

try:
    df_credit = pd.read_csv('../../../data/CREDIT_SPREAD_2010_2024.csv', parse_dates=True, index_col=0)
    # Try common column names
    col = None
    for c in ['Credit_Spread', 'CREDIT_SPREAD', 'credit_spread', 'spread', df_credit.columns[0]]:
        if c in df_credit.columns:
            col = c
            break
    if col is None:
        col = df_credit.columns[0]

    df_credit[col] = pd.to_numeric(df_credit[col], errors='coerce')
    df_credit = df_credit.dropna()
    spread = df_credit[col]

    # Check if values are in percentage or decimal
    if spread.mean() > 1:
        print(f'(Data appears to be in percentage points, converting to decimal)')
        spread = spread / 100

    print(f'Count:    {len(spread)} days')
    print(f'Mean:     {spread.mean():.4f} ({spread.mean()*100:.2f}%)')
    print(f'Std Dev:  {spread.std():.4f} ({spread.std()*100:.2f}%)')
    print(f'Median:   {spread.median():.4f} ({spread.median()*100:.2f}%)')
    print(f'Min:      {spread.min():.4f} ({spread.min()*100:.2f}%)')
    print(f'Max:      {spread.max():.4f} ({spread.max()*100:.2f}%)')
    print()
    print('Percentiles:')
    print(f'  10th:   {spread.quantile(0.10):.4f} ({spread.quantile(0.10)*100:.2f}%)')
    print(f'  25th:   {spread.quantile(0.25):.4f} ({spread.quantile(0.25)*100:.2f}%)')
    print(f'  50th:   {spread.quantile(0.50):.4f} ({spread.quantile(0.50)*100:.2f}%)')
    print(f'  75th:   {spread.quantile(0.75):.4f} ({spread.quantile(0.75)*100:.2f}%)')
    print(f'  90th:   {spread.quantile(0.90):.4f} ({spread.quantile(0.90)*100:.2f}%)')
    print()
    print('Regime distribution with current thresholds (low=0.02, high=0.04):')
    low_count = (spread < 0.02).sum()
    mid_count = ((spread >= 0.02) & (spread < 0.04)).sum()
    high_count = (spread >= 0.04).sum()
    print(f'  Spread < 2%:      {low_count} days ({100*low_count/len(spread):.1f}%)')
    print(f'  2% <= Spread < 4%: {mid_count} days ({100*mid_count/len(spread):.1f}%)')
    print(f'  Spread >= 4%:      {high_count} days ({100*high_count/len(spread):.1f}%)')

except Exception as e:
    print(f'Error loading credit spread data: {e}')

# =============================================================================
# Yield Curve Statistics
# =============================================================================
print()
print('=' * 50)
print('Yield Curve (T10Y3M) Statistics (2010-2024):')
print('=' * 50)

try:
    df_yc = pd.read_csv('../../../data/YIELD_CURVE_10Y3M_2010_2024.csv', parse_dates=True, index_col=0)
    # Try common column names
    col = None
    for c in ['T10Y3M', '10Y3M', '10Y_3M', 'YieldCurve', 'yield_curve', 'slope', df_yc.columns[0]]:
        if c in df_yc.columns:
            col = c
            break
    if col is None:
        col = df_yc.columns[0]

    df_yc[col] = pd.to_numeric(df_yc[col], errors='coerce')
    df_yc = df_yc.dropna()
    yc = df_yc[col]

    print(f'Count:    {len(yc)} days')
    print(f'Mean:     {yc.mean():.2f}%')
    print(f'Std Dev:  {yc.std():.2f}%')
    print(f'Median:   {yc.median():.2f}%')
    print(f'Min:      {yc.min():.2f}%')
    print(f'Max:      {yc.max():.2f}%')
    print()
    print('Percentiles:')
    print(f'  10th:   {yc.quantile(0.10):.2f}%')
    print(f'  25th:   {yc.quantile(0.25):.2f}%')
    print(f'  50th:   {yc.quantile(0.50):.2f}%')
    print(f'  75th:   {yc.quantile(0.75):.2f}%')
    print(f'  90th:   {yc.quantile(0.90):.2f}%')
    print()
    print('Yield Curve Regimes:')
    inverted = (yc < 0).sum()
    flat = ((yc >= 0) & (yc < 1)).sum()
    normal = ((yc >= 1) & (yc < 2)).sum()
    steep = (yc >= 2).sum()
    print(f'  Inverted (< 0%):   {inverted} days ({100*inverted/len(yc):.1f}%)')
    print(f'  Flat (0-1%):       {flat} days ({100*flat/len(yc):.1f}%)')
    print(f'  Normal (1-2%):     {normal} days ({100*normal/len(yc):.1f}%)')
    print(f'  Steep (>= 2%):     {steep} days ({100*steep/len(yc):.1f}%)')
    print()
    print(f'Normalized by scale=3.0:')
    print(f'  Range: [{yc.min()/3:.2f}, {yc.max()/3:.2f}]')
    print(f'  Mean:  {yc.mean()/3:.2f}')

except Exception as e:
    print(f'Error loading yield curve data: {e}')
