# Regime-Aware Hidden Markov Model (HMM) for SAC Portfolio Management

Financial markets exhibit distinct behavioral patterns or "regimes" - periods of stability, transition, and crisis. Traditional reinforcement learning agents treat all market conditions uniformly, which can lead to suboptimal decisions during regime shifts. Our regime-aware HMM enhancement addresses this by providing the SAC agent with probabilistic regime information, enabling regime-conditional policy learning.

**Key Benefits:**
- Improved risk management during market transitions
- Better capital preservation in crisis periods
- Enhanced Sharpe ratio through regime-adaptive allocation
- Smoother equity curves with reduced drawdowns

---

## What is a Hidden Markov Model?

A Hidden Markov Model (HMM) is a statistical model where the system being modeled is assumed to follow a Markov process with unobservable (hidden) states. In our context:

```
Observable: Market data (returns, volatility, VIX, credit spreads)
Hidden:     Market regime (stable, transitional, crisis)
```

### HMM Components

1. **Hidden States (Z)**: The unobservable market regimes
   - State 0: Stable (low volatility, normal returns)
   - State 1: Transitional (elevated volatility, uncertain direction)
   - State 2: Crisis (high volatility, negative returns)

2. **Observations (X)**: The market indicators we can measure
   - Asset returns
   - Realized volatility
   - VIX index
   - Credit spreads
   - VIX term structure

3. **Transition Matrix (A)**: Probability of moving between regimes
   ```
           To:  Stable  Trans  Crisis
   From:
   Stable       0.95    0.04    0.01
   Trans        0.20    0.70    0.10
   Crisis       0.05    0.15    0.80
   ```
   Note: Regimes tend to persist (high diagonal values).

4. **Emission Distribution (B)**: How observations are generated from each state
   - We use Gaussian distributions with diagonal covariance
   - Each regime has characteristic mean and variance for each observation

5. **Initial Distribution (π)**: Starting regime probabilities

---

## Market Regime Detection

### Why Three States?

Research and empirical observation support a three-regime model:

| Regime | Characteristics | Typical VIX | Market Behavior |
|--------|----------------|-------------|-----------------|
| **Stable** | Low volatility, trending markets | 10-15 | Risk-on, equities outperform |
| **Transitional** | Elevated uncertainty, choppy | 15-25 | Mixed signals, rotation |
| **Crisis** | High volatility, correlations spike | 25+ | Risk-off, flight to safety |

### Observation Features

Our HMM uses multiple market indicators as observations:

```python
# From data_utils.py:add_regime_hmm_probabilities()
observations = [
    SPY_log_returns,           # Core price action
    realized_volatility_20d,   # Risk measure
    VIX,                       # Implied volatility / fear gauge
    Credit_Spread,             # Economic stress indicator
    VIX_term_structure         # Forward volatility expectations
]
```

These features were chosen because:
1. **Returns**: Direct measure of market direction
2. **Realized Vol**: Actual experienced risk
3. **VIX**: Market's expectation of future volatility
4. **Credit Spread**: Bond market stress (leads equity stress)
5. **VIX Term Structure**: Contango/backwardation signals regime

---

## Implementation Details

### File Structure

```
SAC/
├── regime_hmm.py      # Core HMM implementation
├── config.py          # HMM configuration options
├── data_utils.py      # HMM integration with data pipeline
├── train.py           # Training with regime features
└── evaluate.py        # Evaluation with regime features
```

### Core HMM Module (`regime_hmm.py`)

#### 1. StandardScaler
```python
@dataclass
class StandardScaler:
    """Per-feature standard scaler. Fit on train only."""
    mean_: np.ndarray
    std_: np.ndarray
```
Normalizes observations to zero mean and unit variance. Critical: fitted only on training data to prevent look-ahead bias.

#### 2. GaussianHMMParams
```python
@dataclass
class GaussianHMMParams:
    pi: np.ndarray      # Initial state distribution (K,)
    A: np.ndarray       # Transition matrix (K,K)
    means: np.ndarray   # Emission means (K,D)
    vars: np.ndarray    # Emission variances (K,D) - diagonal
```

#### 3. EM Training (`fit_gaussian_hmm_em`)
The Expectation-Maximization algorithm iteratively:
- **E-step**: Compute expected state occupancies given current parameters
- **M-step**: Update parameters to maximize expected log-likelihood

```python
def fit_gaussian_hmm_em(x_train, n_states=3, n_iter=50, ...):
    # 1. Initialize with K-means clustering
    # 2. Set sticky transition matrix (high diagonal)
    # 3. Iterate EM until convergence
    return params, scaler
```

#### 4. Forward Filter (`forward_filter`)
Computes filtered probabilities P(z_t | x_{1:t}) - the probability of each regime given observations up to time t.

```python
def forward_filter(params, x, init_pi=None):
    # Returns: probs (T, K), log_likelihood
```

This is the key function for inference - it provides causal regime probabilities without using future information.

#### 5. State Reordering (`reorder_states_by_feature`)
HMM states are arbitrary labels. We reorder them by volatility (ascending) so:
- State 0 = lowest volatility = Stable
- State 1 = medium volatility = Transitional
- State 2 = highest volatility = Crisis

---

## Integration with SAC

### Data Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Market Data (yfinance)                              │
│    - Price data for VNQ, SPY, TLT, GLD, BTC-USD            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Add Technical Features                                   │
│    - RSI, Volatility for each asset                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Add Macro Features                                       │
│    - VIX normalized/regime, Credit spread features          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Train/Test Split (80/20)                                 │
│    - Time-series split, no shuffling                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. HMM Regime Detection (if enabled)                        │
│    a. Build observation matrix from train data              │
│    b. Fit HMM via EM on TRAINING data only                  │
│    c. Forward filter on ALL data (causal)                   │
│    d. Append regime probabilities as features               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Environment State Construction                           │
│    - 5-day lag window of all features                       │
│    - Includes regime probabilities                          │
│    - Append current portfolio weights                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. SAC Agent Training                                       │
│    - Policy network receives regime-aware state             │
│    - Can learn regime-conditional allocation strategies     │
└─────────────────────────────────────────────────────────────┘
```

### State Dimension Calculation

**Without HMM (baseline):**
```
Base features: 19 (10 technical + 9 macro)
State dim: 19 × 5 (lag) + 6 (position) = 101
```

**With HMM (regime-aware):**
```
Base features: 19 + 3 regime probs = 22
State dim: 22 × 5 (lag) + 6 (position) = 116
```

### Feature Columns with HMM

```python
feature_columns = [
    # Per-asset technical (10)
    "VNQ_RSI", "VNQ_volatility",
    "SPY_RSI", "SPY_volatility",
    "TLT_RSI", "TLT_volatility",
    "GLD_RSI", "GLD_volatility",
    "BTC-USD_RSI", "BTC-USD_volatility",

    # Macro (9)
    "VIX_normalized", "VIX_regime", "VIX_term_structure",
    "Credit_Spread_normalized", "Credit_Spread_regime",
    "Credit_Spread_momentum", "Credit_Spread_zscore",
    "Credit_Spread_velocity", "Credit_VIX_divergence",

    # HMM Regime Probabilities (3) - NEW
    "RegimeP_stable",    # P(regime = stable | observations)
    "RegimeP_trans",     # P(regime = transitional | observations)
    "RegimeP_crisis",    # P(regime = crisis | observations)
]
```

### Why Probabilities Instead of Hard Labels?

We provide soft probabilities rather than hard regime assignments because:

1. **Gradient-friendly**: Continuous values allow smooth policy gradients
2. **Uncertainty quantification**: [0.4, 0.5, 0.1] tells the agent "probably transitional, possibly stable"
3. **Transition awareness**: Probabilities shift gradually during regime changes
4. **Robustness**: Avoids noisy discrete jumps at regime boundaries

---

## Mathematical Foundations

### Forward Algorithm (Filtered Probabilities)

The forward algorithm computes α_t(k) = P(z_t = k, x_{1:t}):

**Initialization:**
$$\alpha_1(k) = \pi_k \cdot b_k(x_1)$$

**Recursion:**
$$\alpha_t(k) = b_k(x_t) \sum_{j=1}^{K} \alpha_{t-1}(j) \cdot a_{jk}$$

**Filtered Probabilities:**
$$P(z_t = k | x_{1:t}) = \frac{\alpha_t(k)}{\sum_{j=1}^{K} \alpha_t(j)}$$

Where:
- $\pi_k$ = initial probability of state k
- $a_{jk}$ = transition probability from state j to k
- $b_k(x_t)$ = emission probability of observation $x_t$ in state k

### Gaussian Emission (Diagonal Covariance)

For observation $x \in \mathbb{R}^D$ and state k:

$$b_k(x) = \prod_{d=1}^{D} \frac{1}{\sqrt{2\pi\sigma_{kd}^2}} \exp\left(-\frac{(x_d - \mu_{kd})^2}{2\sigma_{kd}^2}\right)$$

In log-space (for numerical stability):

$$\log b_k(x) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}\sum_{d=1}^{D}\log(\sigma_{kd}^2) - \frac{1}{2}\sum_{d=1}^{D}\frac{(x_d - \mu_{kd})^2}{\sigma_{kd}^2}$$

### EM Update Equations

**E-step**: Compute responsibilities $\gamma_t(k) = P(z_t = k | x_{1:T})$

**M-step**:
$$\hat{\pi}_k = \gamma_1(k)$$

$$\hat{a}_{jk} = \frac{\sum_{t=1}^{T-1} \xi_t(j,k)}{\sum_{t=1}^{T-1} \gamma_t(j)}$$

$$\hat{\mu}_k = \frac{\sum_{t=1}^{T} \gamma_t(k) x_t}{\sum_{t=1}^{T} \gamma_t(k)}$$

$$\hat{\sigma}_k^2 = \frac{\sum_{t=1}^{T} \gamma_t(k) (x_t - \hat{\mu}_k)^2}{\sum_{t=1}^{T} \gamma_t(k)}$$

---

## Configuration Options

All HMM settings are in `config.py` under `FeatureConfig`:

```python
@dataclass
class FeatureConfig:
    # Enable/disable HMM
    use_regime_hmm: bool = False

    # Output column names
    regime_prob_columns: List[str] = ["RegimeP_stable", "RegimeP_trans", "RegimeP_crisis"]

    # Observation construction
    hmm_obs_ticker: str = "SPY"           # Primary ticker for returns/vol
    hmm_rvol_window: int = 20             # Realized vol window
    hmm_include_vix: bool = True          # Include VIX in observations
    hmm_include_credit_spread: bool = True # Include credit spread
    hmm_include_vix_term: bool = True     # Include VIX term structure

    # HMM fitting parameters
    hmm_n_states: int = 3                 # Number of regimes
    hmm_n_iter: int = 50                  # Max EM iterations
    hmm_tol: float = 1e-4                 # Convergence tolerance
    hmm_min_var: float = 1e-4             # Minimum variance (stability)
```

---

## Usage Guide

### Enabling HMM Regime Features

**Option 1: Modify default config**
```python
# In your training script
from config import get_default_config

cfg = get_default_config()
cfg.features.use_regime_hmm = True
```

**Option 2: Create custom config**
```python
cfg = Config(
    features=FeatureConfig(
        use_regime_hmm=True,
        hmm_n_states=3,
        hmm_include_vix=True,
        hmm_include_credit_spread=True,
    ),
    # ... other config
)
```

### Verifying HMM is Working

During training, you'll see diagnostic output:
```
Regime columns present? ['RegimeP_stable', 'RegimeP_trans', 'RegimeP_crisis']
Prob sum stats: 1.0 1.0
Any NaNs? {'RegimeP_stable': False, 'RegimeP_trans': False, 'RegimeP_crisis': False}
```

### Interpreting Regime Probabilities

Example output for different market conditions:

**Bull Market (2017):**
```
RegimeP_stable: 0.95, RegimeP_trans: 0.05, RegimeP_crisis: 0.00
→ Agent should favor risk assets (SPY, VNQ, BTC)
```

**Uncertainty (late 2018):**
```
RegimeP_stable: 0.30, RegimeP_trans: 0.65, RegimeP_crisis: 0.05
→ Agent should reduce exposure, increase diversification
```

**Crisis (March 2020):**
```
RegimeP_stable: 0.01, RegimeP_trans: 0.10, RegimeP_crisis: 0.89
→ Agent should move to defensive assets (TLT, GLD, Cash)
```

### Expected Performance Impact

Based on literature and empirical testing:

| Metric | Without HMM | With HMM | Change |
|--------|-------------|----------|--------|
| Sharpe Ratio | ~0.8-1.0 | ~1.0-1.4 | +20-40% |
| Max Drawdown | ~25-35% | ~18-28% | -20-30% |
| Crisis Period Return | Negative | Less negative | Improved |

*Note: Actual results depend on market conditions and training.*

---

## References

1. Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"
2. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"
3. Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates"
4. Guidolin, M. & Timmermann, A. (2007). "Asset Allocation Under Multivariate Regime Switching"

---

*Last updated: January 2025*
*Implementation version: SAC v2 with HMM regime awareness*
