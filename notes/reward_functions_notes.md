# Reward Functions Guide

## Overview
The reward function fundamentally shapes your agent's risk preference. This guide explains the 4 reward types now available in the SAC implementation.

---

## 1. Log Reward (Default - Risk Averse)

**Config:**
```python
reward_type: str = "log"
reward_scale: float = 10.0
```

**Formula:** `reward = reward_scale * log(1 + net_return)`

**Characteristics:**
- **Risk Preference:** Risk-averse (concave utility)
- **Behavior:** Prefers stable, diversified portfolios
- **Math:** Diminishing marginal utility of gains, heavy penalty on losses

**Example Returns:**
| Net Return | Raw Reward (scale=10) |
|------------|----------------------|
| +10%       | 0.95                |
| +30%       | 2.62                |
| +50%       | 4.05                |
| -10%       | -1.05               |
| -30%       | -3.57               |

**When to Use:**
- Conservative portfolio management
- Wealth preservation focus
- Real-world institutional investing
- When you want the agent to avoid large drawdowns

---

## 2. Linear Reward (Risk Neutral)

**Config:**
```python
reward_type: str = "linear"
reward_scale: float = 10.0
```

**Formula:** `reward = reward_scale * net_return`

**Characteristics:**
- **Risk Preference:** Risk-neutral
- **Behavior:** Maximizes expected return regardless of volatility
- **Math:** Linear utility - treats +10% and -10% symmetrically

**Example Returns:**
| Net Return | Raw Reward (scale=10) |
|------------|----------------------|
| +10%       | 1.0                 |
| +30%       | 3.0                 |
| +50%       | 5.0                 |
| -10%       | -1.0                |
| -30%       | -3.0                |

**When to Use:**
- Maximize absolute returns
- You accept higher volatility for higher returns
- Short-term trading strategies
- Benchmarking against simple return maximization

---

## 3. Exponential Reward (Risk Seeking)

**Config:**
```python
reward_type: str = "exp"
reward_scale: float = 10.0
exp_risk_factor: float = 2.0    # Higher = more aggressive
```

**Formula:** `reward = reward_scale * (exp(net_return * risk_factor) - 1) / risk_factor`

**Characteristics:**
- **Risk Preference:** Risk-seeking (convex utility)
- **Behavior:** Actively pursues concentrated, high-volatility strategies
- **Math:** Exponential amplification of gains > penalty on losses

**Example Returns (risk_factor=2.0):**
| Net Return | Raw Reward (scale=10) |
|------------|----------------------|
| +10%       | 1.11                |
| +30%       | 3.73                |
| +50%       | 8.59                |
| -10%       | -0.95               |
| -30%       | -2.64               |

**When to Use:**
- **THIS IS WHAT YOU WANT FOR RISKY STRATEGIES**
- Aggressive growth portfolios
- Cryptocurrency/high-volatility asset trading
- When you want concentrated bets on high-conviction ideas
- Maximize upside potential

**Tuning:**
- `exp_risk_factor = 1.0`: Mild risk-seeking
- `exp_risk_factor = 2.0`: Moderate risk-seeking (default)
- `exp_risk_factor = 5.0`: Very aggressive

---

## 4. Sharpe Reward (Risk-Adjusted)

**Config:**
```python
reward_type: str = "sharpe"
reward_scale: float = 10.0
sharpe_window: int = 20         # Rolling window for volatility
risk_free_rate: float = 0.02    # 2% annualized
```

**Formula:** `reward = reward_scale * (net_return - rf) / volatility`

**Characteristics:**
- **Risk Preference:** Risk-adjusted
- **Behavior:** Maximizes return per unit of risk
- **Math:** Penalizes volatility even if returns are high

**When to Use:**
- Professional portfolio management
- You want consistent returns with controlled volatility
- Benchmarking against Sharpe ratio
- Institutional constraints on risk

**Note:** This is the most sophisticated but also the most unstable during early training (requires sufficient return history).

---

## Recommended Settings for Different Goals

### Conservative (Low Risk)
```python
reward_type: str = "log"
reward_scale: float = 10.0
alpha_min: float = 1.0      # Force diversification
alpha_max: float = 60.0
tc_rate: float = 0.0005     # Keep realistic costs
```

### Balanced (Moderate Risk)
```python
reward_type: str = "linear"
reward_scale: float = 10.0
alpha_min: float = 0.6
alpha_max: float = 60.0
tc_rate: float = 0.0005
```

### Aggressive (High Risk) - **RECOMMENDED FOR YOUR GOAL**
```python
reward_type: str = "exp"
reward_scale: float = 10.0
exp_risk_factor: float = 2.0    # Start here, increase to 3-5 if still too conservative
alpha_min: float = 0.1          # Allow concentrated positions
alpha_max: float = 60.0
tc_rate: float = 0.0002         # Lower costs to enable active trading
reward_clip_max: float = 2.0    # Allow larger gains to be rewarded
```

### Professional (Risk-Adjusted)
```python
reward_type: str = "sharpe"
reward_scale: float = 10.0
sharpe_window: int = 20
risk_free_rate: float = 0.02
alpha_min: float = 0.6
alpha_max: float = 60.0
tc_rate: float = 0.0005
```

---

## How to Change Reward Function

Edit `algorithms/SAC/config.py`:

```python
@dataclass
class EnvironmentConfig:
    # Change this line:
    reward_type: str = "exp"  # Change from "log" to "exp", "linear", or "sharpe"

    # For exponential reward:
    exp_risk_factor: float = 2.0  # Increase for more risk-seeking

    # Also consider:
    reward_clip_max: float = 2.0  # Allow larger gains
```

---

## Expected Behavioral Changes

### From Log → Linear:
- More tolerance for volatility
- Less diversified portfolios
- Higher turnover
- Larger drawdowns but potentially higher returns

### From Log → Exponential:
- **Concentrated positions** (60-90% in 1-2 assets)
- **Active rebalancing** to chase momentum
- **Much higher volatility**
- **Potential for 100%+ gains and 30%+ losses**
- Agent learns to "go all-in" on high-conviction signals

### From Log → Sharpe:
- Most consistent returns
- Controlled drawdowns
- Lower absolute returns but better risk-adjusted
- Very stable portfolio weights

---

## Testing Your Changes

After changing the reward function:

1. **Train a new agent:**
   ```bash
   python algorithms/SAC/train.py
   ```

2. **Monitor during training:**
   - Check `turnover_oneway` - should increase with risk-seeking rewards
   - Check portfolio concentration - exponential should show 1-2 dominant assets
   - Check episode returns - should see higher variance with exp/linear

3. **Evaluate:**
   ```bash
   python algorithms/SAC/evaluate.py
   ```

4. **Compare metrics:**
   - Total return (should increase with risk-seeking)
   - Sharpe ratio (may decrease with risk-seeking)
   - Max drawdown (will increase with risk-seeking)
   - Average concentration (% in top asset)

---

## Important Notes

1. **Changing reward_type alone is not enough** - you must also:
   - Lower `alpha_min` (e.g., 0.1 or 0.2)
   - Consider lowering `tc_rate` to enable active trading
   - Increase `reward_clip_max` to allow large gains

2. **Training will be different:**
   - Risk-seeking agents take longer to converge
   - You may see wild swings in early training
   - Final performance may be more unstable

3. **Real-world considerations:**
   - Exponential utility is NOT standard in finance
   - Linear/Sharpe are more academically accepted
   - Log utility is based on Kelly criterion

4. **Combine with alpha_min:**
   - The reward function controls WHAT the agent wants
   - `alpha_min` controls HOW concentrated it CAN be
   - Both must be changed together for aggressive strategies
