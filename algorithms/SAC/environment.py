import numpy as np
import pandas as pd

class Env:
  def __init__(self, df, tickers, lag=5):
    # Build feature columns list
    self.columns = []
    
    # Asset-specific features (per ticker)
    for ticker in tickers:
      self.columns += [
        ticker + '_RSI',
        ticker + '_volatility',
      ]
      
    # VIX-based features
    self.columns += [
      'VIX_normalized',
      'VIX_regime',
      'VIX_term_structure',
    ]
    
    # Credit Spread-based features
    self.columns += [
      'Credit_Spread_normalized',     
      'Credit_Spread_regime',         
      'Credit_Spread_momentum',       
      'Credit_Spread_zscore',         
      'Credit_Spread_velocity',       
      'Credit_VIX_divergence',        
    ]
      
    cleaned_data = df.dropna()
    self.states = cleaned_data[self.columns].to_numpy()
    self.prices = cleaned_data[tickers].to_numpy() # used for computing returns
    self.lag = lag
    
    print(f"Environment initialized:")
    print(f"  - State features: {len(self.columns)}")
    print(f"  - Features per ticker: 2 (RSI, volatility)")
    print(f"  - Market features: 3 (VIX_normalized, VIX_regime, VIX_term_structure)")
    print(f"  - Credit Spread features: 6 (normalized, regime, momentum, zscore, velocity, VIX_divergence)")
    print(f"  - Total state dimension per timestep: {self.states.shape[1]}")
    print(f"  - Lag (temporal window): {lag}")
    print(f"  - Final input dimension: {self.states.shape[1] * lag}")
    print(f"  - Number of data points: {len(self.states)}")

  # to initialize the environment at the beginning of a simulation or training episode
  # the initial state will be the first 5 data points
  # then moving as a sliding window of the last 5 data points
  def reset(self):
    self.pos = self.lag
    return self.states[:self.pos]
    # later: self.states[self.pos - self.lag:self.pos]

  def step(self, action):
    # action space has dimension = #assets + 1 (for cash position)
    if action.shape[-1] != self.prices.shape[-1] + 1:
      raise Exception(
        f"action has the wrong shape, expected: {self.prices.shape[-1] + 1}, got: {action.shape[-1]}")

    # compute reward(pct_change) for this period
    next_pos = self.pos + self.lag
    
    if next_pos >= len(self.prices):
      return np.array([]), 0.0, True
    
    asset_returns = (self.prices[next_pos] - self.prices[self.pos]) / self.prices[self.pos]
    asset_returns = np.concatenate((asset_returns, [0])) # zero for cash
    
    ##############
    ### REWARD ###
    ##############
    pct_change = action @ asset_returns # reward = total portfolio return

    # update pointer
    self.pos = next_pos

    # are we done?
    # done if the next-next position doesn't exist in states array
    # (because there will be no next state)
    done = self.pos + self.lag >= len(self.states)
    
    # get next state (sliding window)
    next_state = self.states[self.pos - self.lag:self.pos]

    # return next state, reward, done
    return next_state, pct_change, done
