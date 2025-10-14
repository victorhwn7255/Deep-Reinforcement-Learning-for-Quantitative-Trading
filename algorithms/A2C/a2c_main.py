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

# compute RSI
for ticker in tickers:
  df[ticker + '_RSI'] = talib.RSI(df[ticker], timeperiod=14) / 50 - 1 # change RSI range to [-1, +1]
  
# compute MACD
for ticker in tickers:
  macd, signal, hist = talib.MACD(df[ticker], fastperiod=12, slowperiod=26, signalperiod=9)
  df[ticker + '_MACD'] = macd / df[ticker] * 10 # scale by close price
  df[ticker + '_MACD_Signal'] = signal / df[ticker] * 10 # scale by close price
  
# split into train and test
n_train = int(0.8 * len(df))

df_train = df.iloc[:n_train]
df_test = df.iloc[n_train:]

###################
### ENVIRONMENT ###
###################
env = Env(df_train, tickers, lag=5)


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
    n_input=len(tickers) * 3 * env.lag,
    n_action=len(tickers) + 1,
    learning_rate=learning_rate,
    gamma=gamma,
    entropy_weight=entropy_weight,
    value_weight=value_weight,
    max_norm=max_norm,
    device=device
)

# Start training
print("Starting A2C training...")
episode_returns, losses = agent.learn(env, total_timesteps)

# Save the trained model
agent.save_model(model_path)
print("Training completed!")