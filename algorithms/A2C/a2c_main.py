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
total_timesteps = 600000      # Total number of training steps
learning_rate = 0.001           # Adam optimizer learning rate
gamma = 0.99                    # Discount factor for future rewards
entropy_weight = 0.001          # Weight for entropy regularization (exploration)
value_weight = 0.25             # Weight for value function loss in total loss
max_norm = 0.5                  # Gradient clipping threshold to prevent 

model_path_final = "models/a2c_portfolio_final.pth"
model_path_best = "models/a2c_portfolio_best.pth"


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

#######################
### Training Starts ###
#######################
print("\n" + "="*50)
print("Starting A2C training...")
print("="*50)
episode_returns, losses, best_model_state = agent.learn(env, total_timesteps)

#######################
### Save the Models ###
#######################
agent.save_model(model_path_final)
print(f"\n✓ Final model saved to {model_path_final}")

if best_model_state is not None:
    torch.save(best_model_state, model_path_best)
    print(f"✓ Best model confirmed at {model_path_best}")
    print(f"  - Episode: {best_model_state['episode']}")
    print(f"  - Global step: {best_model_state['global_step']}")
    print(f"  - Avg return (last 10 episodes): {best_model_state['avg_return']:.4f}")
else:
    print("⚠ No best model was saved (training may have been too short)")

print("\n" + "="*50)
print("Training completed!")
print("="*50)

print(f"\nTraining Summary:")
print(f"  - Total episodes completed: {len(episode_returns)}")
print(f"  - Average episode return: {np.mean(episode_returns):.4f}")
print(f"  - Best episode return: {np.max(episode_returns):.4f}")
print(f"  - Worst episode return: {np.min(episode_returns):.4f}")
if len(episode_returns) >= 10:
    print(f"  - Final 10 episodes avg: {np.mean(episode_returns[-10:]):.4f}")
    
##########################
### Print Return Chart ###
##########################
print("\n" + "="*50)
while True:
    print("\nOptions:")
    print("  1. Evaluate model on test set")
    print("  2. Plot training results")
    print("  3. Quit")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        print("\n" + "="*50)
        print("EVALUATING ON TEST SET")
        print("="*50)
        
        # Choose which model to evaluate
        print("\nWhich model do you want to evaluate?")
        print("  1. Best model")
        print("  2. Final model")
        model_choice = input("Enter choice (1 or 2): ").strip()
        
        if model_choice == '1':
            eval_agent = Agent(
                n_input=env.states.shape[1] * env.lag,
                n_action=len(tickers) + 1,
                learning_rate=learning_rate,
                gamma=gamma,
                entropy_weight=entropy_weight,
                value_weight=value_weight,
                max_norm=max_norm,
                device=device
            )
            eval_agent.load_best_model(best_model_state)
            print("\n✓ Loaded best model for evaluation")
        elif model_choice == '2':
            eval_agent = agent  # Use the current agent (final model)
            print("\n✓ Using final model for evaluation")
        else:
            print("Invalid choice, skipping evaluation")
            continue
        
        eval_agent.ac_network.eval()  # Set to evaluation mode
        
        # Create test environment
        env_test = Env(df_test, tickers, lag=5)
        print(f"Test set: {len(df_test)} days ({df_test.index[0]} to {df_test.index[-1]})")
        
        ##############################
        ### EVALUATE TRAINED MODEL ###
        ##############################
        print("\nEvaluating A2C Agent...")
        obs = env_test.reset()
        a2c_rewards = []
        a2c_actions = []
        done = False
        
        while not done:
            with torch.no_grad():
                alpha, _ = eval_agent.ac_network(eval_agent.np2torch(obs))
                actions = eval_agent.sample_action(alpha)
                actions_np = actions.detach().cpu().numpy().flatten()
            
            a2c_actions.append(actions_np.copy())
            obs, reward, done = env_test.step(actions_np)
            a2c_rewards.append(reward)
        
        a2c_rewards = np.array(a2c_rewards)
        a2c_actions = np.array(a2c_actions)
        
        # Calculate metrics for A2C
        a2c_cumulative = np.cumprod(1 + a2c_rewards)
        a2c_total_return = a2c_cumulative[-1] - 1
        a2c_sharpe = np.mean(a2c_rewards) / (np.std(a2c_rewards) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        a2c_running_max = np.maximum.accumulate(a2c_cumulative)
        a2c_drawdown = (a2c_cumulative - a2c_running_max) / a2c_running_max
        a2c_max_drawdown = np.min(a2c_drawdown)
        
        ###################################
        ### EVALUATE BENCHMARK PORTFOLIO ###
        ###################################
        print("Evaluating Benchmark...")
        benchmark_weights = np.array([0.1, 0.8, 0.05, 0.05, 0])
        obs = env_test.reset()
        benchmark_rewards = []
        done = False
        
        while not done:
            obs, reward, done = env_test.step(benchmark_weights)
            benchmark_rewards.append(reward)
        
        benchmark_rewards = np.array(benchmark_rewards)
        benchmark_cumulative = np.cumprod(1 + benchmark_rewards)
        benchmark_total_return = benchmark_cumulative[-1] - 1
        benchmark_sharpe = np.mean(benchmark_rewards) / (np.std(benchmark_rewards) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        benchmark_running_max = np.maximum.accumulate(benchmark_cumulative)
        benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
        benchmark_max_drawdown = np.min(benchmark_drawdown)
        
        ########################
        ### PRINT COMPARISON ###
        ########################
        print("\n" + "="*50)
        print("TEST SET PERFORMANCE")
        print("="*50)
        
        print(f"\nA2C Agent:")
        print(f"  - Total Return: {a2c_total_return*100:.2f}%")
        print(f"  - Sharpe Ratio: {a2c_sharpe:.4f}")
        print(f"  - Max Drawdown: {a2c_max_drawdown*100:.2f}%")
        print(f"  - Ann. Volatility: {np.std(a2c_rewards)*np.sqrt(252)*100:.2f}%")
        
        print(f"\nBenchmark (10/80/5/5/0):")
        print(f"  - Total Return: {benchmark_total_return*100:.2f}%")
        print(f"  - Sharpe Ratio: {benchmark_sharpe:.4f}")
        print(f"  - Max Drawdown: {benchmark_max_drawdown*100:.2f}%")
        print(f"  - Ann. Volatility: {np.std(benchmark_rewards)*np.sqrt(252)*100:.2f}%")
        
        print(f"\nA2C vs Benchmark:")
        print(f"  - Return difference: {(a2c_total_return - benchmark_total_return)*100:+.2f}%")
        print(f"  - Sharpe difference: {a2c_sharpe - benchmark_sharpe:+.4f}")
        
        #####################
        ### VISUALIZATION ###
        #####################
        visualize = input("\nWould you like to visualize the results? (y/n): ").strip().lower()
        
        if visualize == 'y':
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Cumulative Returns
            ax1 = axes[0, 0]
            ax1.plot(a2c_cumulative, label='A2C Agent', linewidth=2)
            ax1.plot(benchmark_cumulative, label='Benchmark', linewidth=2, linestyle='--')
            ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Trading Days')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Drawdown
            ax2 = axes[0, 1]
            ax2.fill_between(range(len(a2c_drawdown)), a2c_drawdown*100, 0, 
                              alpha=0.3, label='A2C Drawdown')
            ax2.fill_between(range(len(benchmark_drawdown)), benchmark_drawdown*100, 0, 
                              alpha=0.3, label='Benchmark Drawdown')
            ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Trading Days')
            ax2.set_ylabel('Drawdown (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Daily Returns Distribution
            ax3 = axes[1, 0]
            ax3.hist(a2c_rewards*100, bins=50, alpha=0.5, label='A2C', density=True)
            ax3.hist(benchmark_rewards*100, bins=50, alpha=0.5, label='Benchmark', density=True)
            ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Daily Return (%)')
            ax3.set_ylabel('Density')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Portfolio Weights Over Time
            ax4 = axes[1, 1]
            asset_names = ['VNQ', 'SPY', 'GLD', 'BTC', 'Cash']
            for i, name in enumerate(asset_names):
                ax4.plot(a2c_actions[:, i], label=name, alpha=0.7)
            ax4.set_title('A2C Portfolio Weights Over Time', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Trading Days')
            ax4.set_ylabel('Weight')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
            
            plt.tight_layout()
            plt.show()
            print("\n✓ Visualization displayed!")
    
    elif choice == '2':
        def smooth(x, a=0.1):
            """Exponential moving average smoothing"""
            y = [x[0]]
            for xi in x[1:]:
                yi = a * xi + (1 - a) * y[-1]
                y.append(yi)
            return y
        
        # Convert episode returns to percentage
        episode_returns_pct = [ret * 100 for ret in episode_returns]
        
        plt.figure(figsize=(10, 6))
        plt.plot(episode_returns_pct, alpha=0.2, label='Raw Returns', linewidth=0.5)
        plt.plot(smooth(episode_returns_pct), label='Smoothed Returns', linewidth=2)
        plt.title("Episode Returns During Training", fontsize=14, fontweight='bold')
        plt.xlabel("Episode Number", fontsize=12)
        plt.ylabel("Total Return (%)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("\n✓ Plot displayed!")
        
    elif choice == '3':
        print("\nExiting... Goodbye! 👋")
        break
    
    else:
        print("\n⚠️  Invalid choice. Please enter 1, 2, or 3.")

print("="*50)