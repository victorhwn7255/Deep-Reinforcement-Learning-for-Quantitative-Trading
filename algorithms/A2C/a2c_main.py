import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Dirichlet

from environment import Env
from networks import AdvantageActorCritic
from agent import Agent

start = '2010-01-01'
end = '2024-12-31'

tickers = ['VNQ', 'SPY', 'GLD', 'BTC-USD']

df = yf.download(tickers, start=start, end=end)
df = df['Close']
df = df.dropna().copy()

####################
### VIX Features ###
####################
vix_df = pd.read_csv('../../data/VIX_CLS_2010_2024.csv')
vix_df['observation_date'] = pd.to_datetime(vix_df['observation_date'])
vix_df = vix_df.set_index('observation_date')
vix_df = vix_df.rename(columns={'VIXCLS': 'VIX'})

vix3m_df = pd.read_csv('../../data/VIX3M_CLS_2010_2024.csv')
vix3m_df['observation_date'] = pd.to_datetime(vix3m_df['observation_date'])
vix3m_df = vix3m_df.set_index('observation_date')
vix3m_df = vix3m_df.rename(columns={'VXVCLS': 'VIX3M'})

vix_combined = vix_df.join(vix3m_df, how='outer')
df = df.join(vix_combined, how='left')
df['VIX'] = df['VIX'].ffill()
df['VIX3M'] = df['VIX3M'].ffill()

# VIX Feature 1: VIX_normalized: Normalized VIX level
df['VIX_normalized'] = (df['VIX'] - 20) / 20  # Range roughly [-1, 1]

# VIX Feature 2: VIX_regime: Categorical regime (Low/Normal/High volatility)
def get_vix_regime(vix):
    if vix < 15:
        return -1.0  # Low vol / complacent
    elif vix < 25:
        return 0.0   # Normal vol
    else:
        return 1.0   # High vol / stress
df['VIX_regime'] = df['VIX'].apply(get_vix_regime)

# 3. VIX_term_structure: REAL term structure using VIX3M
# Positive = Contango (normal markets, VIX3M > VIX)
# Negative = Backwardation (stressed markets, VIX3M < VIX)
df['VIX_term_structure'] = (df['VIX3M'] - df['VIX']) / df['VIX']
df['VIX_term_structure'] = np.clip(df['VIX_term_structure'], -1, 1)  # Clip extreme values

####################
### RSI Features ###
####################
for ticker in tickers:
  df[ticker + '_RSI'] = talib.RSI(df[ticker], timeperiod=14) / 50 - 1 # change RSI range to [-1, +1]
  
#####################
### MACD Features ###
#####################
for ticker in tickers:
  macd, signal, hist = talib.MACD(df[ticker], fastperiod=12, slowperiod=26, signalperiod=9)
  df[ticker + '_MACD'] = macd / df[ticker] * 10 # scale by close price
  df[ticker + '_MACD_Signal'] = signal / df[ticker] * 10 # scale by close price

###########################
### Realized Volatility ###
###########################
for ticker in tickers:
    returns = df[ticker].pct_change()
    df[ticker + '_volatility'] = returns.rolling(20).std() * np.sqrt(252)  # 20-day rolling vol
    # Normalize: typical stock vol is ~0.20 (20%), so divide by 0.25 to get roughly [-1, 1] range
    df[ticker + '_volatility'] = (df[ticker + '_volatility'] - 0.25) / 0.25

df = df.dropna()
print(f"\nData shape after feature engineering: {df.shape}")
print(f"Final date range: {df.index[0]} to {df.index[-1]}")

print("\nAll features created:")
feature_cols = [col for col in df.columns if col not in tickers + ['VIX', 'VIX3M']]
for col in sorted(feature_cols):
    print(f"  - {col}")

##########################
### SPLIT Train & Test ###
##########################
n_train = int(0.8 * len(df))
df_train = df.iloc[:n_train]
df_test = df.iloc[n_train:]
print(f"\nTrain set: {len(df_train)} days ({df_train.index[0]} to {df_train.index[-1]})")
print(f"Test set: {len(df_test)} days ({df_test.index[0]} to {df_test.index[-1]})")

###################
### ENVIRONMENT ###
###################
env = Env(df_train, tickers, lag=5)
print(f"\nEnvironment state dimension: {env.states.shape[1] * env.lag}")

#################
### Benchmark ###
#################
weights = np.array([0.1, 0.8, 0.05, 0.05, 0]) # constant weight portfolio
done = False
obs = env.reset()
rewards = []
while not done:
  obs, reward, done = env.step(weights)
  rewards.append(reward)
  
benchmark_return = np.sum(rewards)
benchmark_sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)
print(f"Benchmark cumulative return: {benchmark_return:.4f}")
print(f"Benchmark Sharpe ratio: {benchmark_sharpe:.4f}")

##############
### Device ###
##############
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu"
                      )

#######################
### Hyperparameters ###
#######################
total_timesteps = 500_000       # Total number of training steps
learning_rate = 0.001           # Adam optimizer learning rate
gamma = 0.99                    # Discount factor for future rewards
entropy_weight = 0.001          # Weight for entropy regularization (exploration)
value_weight = 0.25             # Weight for value function loss in total loss
max_norm = 0.5                  # Gradient clipping threshold to prevent 

model_path = "models/a2c_portfolio.pth"


#############
### AGENT ###
#############
agent = Agent(
    n_input=env.states.shape[1] * env.lag,
    n_action=len(tickers) + 1,
    learning_rate=learning_rate,
    gamma=gamma,
    entropy_weight=entropy_weight,
    value_weight=value_weight,
    max_norm=max_norm,
    device=device
)

# Start training
print("\n" + "="*50)
print("Starting A2C training...")
print("="*50)
episode_returns, losses = agent.learn(env, total_timesteps)

# Save the trained model
agent.save_model(model_path)
print("\n" + "="*50)
print("Training completed!")
print("="*50)