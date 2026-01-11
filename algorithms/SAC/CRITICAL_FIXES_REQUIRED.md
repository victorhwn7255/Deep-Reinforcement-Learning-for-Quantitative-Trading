# Critical Issues to Fix Before Real-World Training

## Status: ⚠️ 4 CRITICAL BUGS IN EVALUATE.PY

After comprehensive review, **train.py is ready**, but **evaluate.py has critical bugs** that will cause runtime crashes.

---

## 🔴 CRITICAL ISSUE 1: Undefined Methods in evaluate.py

**File:** evaluate.py
**Lines:** 66, 90
**Severity:** CRITICAL - Will crash at runtime

### Problem

evaluate.py calls methods that **don't exist** in agent.py:

```python
# Line 66: ❌ This method doesn't exist
agent.load_best_model(ckpt)

# Line 90: ❌ These methods don't exist
agent.choose_action_deterministic(obs)
agent.choose_action_stochastic(obs)
```

### What agent.py Actually Has

```python
# agent.py only has:
agent.select_action(state, evaluate=False)  # ✓ This exists
agent.load_model(path)                       # ✓ This exists
```

### Fix Required

**evaluate.py:66** - Replace:
```python
# WRONG:
try:
    agent.load_best_model(ckpt)
except Exception:
    agent.load_model(path)

# CORRECT:
agent.load_model(path)
```

**evaluate.py:90** - Replace:
```python
# WRONG:
a = agent.choose_action_deterministic(obs) if deterministic else agent.choose_action_stochastic(obs)

# CORRECT:
a = agent.select_action(obs, evaluate=deterministic)
```

---

## 🔴 CRITICAL ISSUE 2: Wrong Agent Constructor Signature

**File:** evaluate.py
**Line:** 203
**Severity:** CRITICAL - Will crash at runtime

### Problem

```python
# Line 203 - WRONG parameter names:
agent = Agent(n_input=state_dim, n_action=action_dim, cfg=cfg, device=device)
```

### What agent.py Actually Expects

```python
# agent.py line 21:
def __init__(self, state_dim: int, action_dim: int, cfg, device: torch.device):
```

### Fix Required

**evaluate.py:203** - Replace:
```python
# WRONG:
agent = Agent(n_input=state_dim, n_action=action_dim, cfg=cfg, device=device)

# CORRECT:
agent = Agent(state_dim, action_dim, cfg, device)
```

---

## 🟡 IMPORTANT ISSUE 3: Missing load_checkpoint Error Handling

**File:** evaluate.py
**Lines:** 61-75
**Severity:** IMPORTANT - Will crash if checkpoint format unexpected

### Problem

The `load_checkpoint()` function tries to handle different checkpoint formats but has flawed logic:

```python
def load_checkpoint(agent: Agent, path: str) -> None:
    ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        try:
            agent.load_best_model(ckpt)  # ❌ This method doesn't exist!
        except Exception:
            agent.load_model(path)
    else:
        agent.load_model(path)
```

### Fix Required

**evaluate.py:61-75** - Replace entire function:
```python
def load_checkpoint(agent: Agent, path: str) -> None:
    """Load model checkpoint and set to eval mode."""
    agent.load_model(path)

    # Set networks to eval mode
    agent.policy.eval()
    agent.q1.eval()
    agent.q2.eval()
    agent.value.eval()
```

---

## 🟡 IMPORTANT ISSUE 4: Environment Signature Mismatch Risk

**File:** evaluate.py
**Lines:** 187-196
**Severity:** IMPORTANT - Fallback code may not work

### Problem

evaluate.py has fallback code for different environment signatures, but the fallback signature doesn't match what Env actually expects:

```python
try:
    env_test = Env(df=df_test, tickers=cfg.data.tickers, cfg=cfg, feature_columns=feature_cols)
except TypeError:
    # This fallback is WRONG - missing cfg parameter
    env_test = Env(
        df_test,
        cfg.data.tickers,
        lag=cfg.env.lag,
        include_position_in_state=cfg.env.include_position_in_state,
    )
```

### What environment.py Actually Expects

```python
# environment.py line 58:
def __init__(self, df: pd.DataFrame, tickers: List[str], cfg, **kwargs):
```

### Fix Required

**evaluate.py:187-196** - Replace:
```python
# CORRECT (matching environment.py signature):
env_test = Env(df_test, cfg.data.tickers, cfg)
```

Remove the try/except entirely - it's unnecessary complexity.

---

## ✅ NON-ISSUES (Verified as Correct)

### 1. Config Attributes ✅
All config attributes used in agent.py exist and are correct:
- `cfg.experiment.verbose` ✓
- `cfg.sac.gradient_clip_norm` ✓
- `cfg.network.weight_decay` ✓
- `cfg.training.save_interval_episodes` ✓

### 2. SAC Mathematics ✅
All SAC update equations are mathematically correct.

### 3. Environment Mechanics ✅
Cost calculation and reward shaping are correct.

### 4. Data Files ✅
All required macro data CSV files exist:
```
✓ ../../data/VIX_CLS_2010_2024.csv
✓ ../../data/VIX3M_CLS_2010_2024.csv
✓ ../../data/CREDIT_SPREAD_2010_2024.csv
```

### 5. train.py ✅
Training script is correct and ready to use.

### 6. data_utils.py ✅
Data loading and feature engineering are correct.

---

## Summary of Required Fixes

### Files to Fix

| File | Issue | Priority | Lines |
|------|-------|----------|-------|
| evaluate.py | Wrong method names | 🔴 CRITICAL | 66, 90 |
| evaluate.py | Wrong constructor params | 🔴 CRITICAL | 203 |
| evaluate.py | Unnecessary complexity | 🟡 IMPORTANT | 61-75, 187-196 |

### Files That Are Ready

| File | Status |
|------|--------|
| config.py | ✅ Ready |
| agent.py | ✅ Ready |
| environment.py | ✅ Ready |
| networks.py | ✅ Ready |
| replay_buffer.py | ✅ Ready |
| data_utils.py | ✅ Ready |
| train.py | ✅ Ready |

---

## Recommended Actions

### Option 1: Skip evaluate.py for now (RECOMMENDED)
**Just run training first:**
```bash
python train.py
```

evaluate.py is only needed **after** training completes. You can fix it later.

### Option 2: Fix evaluate.py now
Apply the 4 fixes above before running any evaluation.

---

## Training Readiness Checklist

Before running `python train.py`:

- [x] Config attributes all correct
- [x] SAC mathematics correct
- [x] Data files exist
- [x] Dependencies installed (torch, numpy, pandas, yfinance, talib, matplotlib)
- [x] train.py script is correct
- [x] No critical bugs in training pipeline

**You can safely run training now!** ✅

---

## Post-Training Checklist

After training completes:

- [ ] Fix evaluate.py (4 issues above)
- [ ] Run evaluation on test set
- [ ] Analyze backtest results
- [ ] Compare SAC vs benchmark

---

## Exact Fixes Needed in evaluate.py

### Fix 1: Line 66
```python
# BEFORE:
try:
    agent.load_best_model(ckpt)
except Exception:
    agent.load_model(path)

# AFTER:
agent.load_model(path)
```

### Fix 2: Line 90
```python
# BEFORE:
a = agent.choose_action_deterministic(obs) if deterministic else agent.choose_action_stochastic(obs)

# AFTER:
a = agent.select_action(obs, evaluate=deterministic)
```

### Fix 3: Line 203
```python
# BEFORE:
agent = Agent(n_input=state_dim, n_action=action_dim, cfg=cfg, device=device)

# AFTER:
agent = Agent(state_dim, action_dim, cfg, device)
```

### Fix 4: Lines 187-196
```python
# BEFORE:
try:
    env_test = Env(df=df_test, tickers=cfg.data.tickers, cfg=cfg, feature_columns=feature_cols)
except TypeError:
    env_test = Env(
        df_test,
        cfg.data.tickers,
        lag=cfg.env.lag,
        include_position_in_state=cfg.env.include_position_in_state,
    )

# AFTER:
env_test = Env(df_test, cfg.data.tickers, cfg)
```

---

## Final Recommendation

🚀 **Proceed with training immediately:**

```bash
cd /Users/victor_he/Downloads/MSAI_CCDS/quant_trading/algorithms/SAC
python train.py
```

**Training pipeline is 100% ready.** Fix evaluate.py later when you need it.

---

**Generated:** 2026-01-12
**Priority:** Fix evaluate.py before evaluation, but training can proceed now
