import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import torch
import os

from environment import Env
from networks import AdvantageActorCritic
from agent import Agent

##############################################
### CONFIGURATION - CUSTOMIZE THIS SECTION ###
##############################################
print("="*50)
print("A2C PORTFOLIO EVALUATION")
print("="*50)

# Option 1: Use custom date range
print("\nHow do you want to specify the evaluation period?")
print("  1. Use default test split (same as training: 2022-2024)")
print("  2. Enter custom date range")
print("  3. Use completely new future data")

date_choice = input("\nEnter choice (1, 2, or 3): ").strip()

if date_choice == '1':
    # Default: same as training split
    start_date = '2010-01-01'
    end_date = '2024-12-31'
    use_test_split = True
    print(f"\n✓ Using training data with 80/20 split")
    
elif date_choice == '2':
    # Custom date range
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    use_test_split = False
    print(f"\n✓ Evaluating on custom period: {start_date} to {end_date}")
    
elif date_choice == '3':
    # Future data (e.g., 2025 onwards)
    start_date = '2025-01-01'
    end_date = '2025-12-31'
    use_test_split = False
    print(f"\n✓ Evaluating on future data: {start_date} to {end_date}")
    
else:
    print("Invalid choice, using defaults")
    start_date = '2010-01-01'
    end_date = '2024-12-31'
    use_test_split = True

# Assets (must match training!)
tickers = ['VNQ', 'SPY', 'GLD', 'BTC-USD']

##############################################
### DATA LOADING & PREPROCESSING ###
##############################################
print("\n" + "="*50)
print("Loading and preprocessing data...")
print("="*50)

df = yf.download(tickers, start=start_date, end=end_date)
df = df['Close']
df = df.dropna().copy()

print(f"Downloaded {len(df)} days of data")

# VIX Features
print("Adding VIX features...")
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

df['VIX_normalized'] = (df['VIX'] - 20) / 20

def get_vix_regime(vix):
    if vix < 15:
        return -1.0
    elif vix < 25:
        return 0.0
    else:
        return 1.0

df['VIX_regime'] = df['VIX'].apply(get_vix_regime)
df['VIX_term_structure'] = (df['VIX3M'] - df['VIX']) / df['VIX']
df['VIX_term_structure'] = np.clip(df['VIX_term_structure'], -1, 1)

# RSI Features
print("Calculating RSI...")
for ticker in tickers:
    df[ticker + '_RSI'] = talib.RSI(df[ticker], timeperiod=14) / 50 - 1

# MACD Features
print("Calculating MACD...")
for ticker in tickers:
    macd, signal, hist = talib.MACD(df[ticker], fastperiod=12, slowperiod=26, signalperiod=9)
    df[ticker + '_MACD'] = macd / df[ticker] * 10
    df[ticker + '_MACD_Signal'] = signal / df[ticker] * 10

# Realized Volatility
print("Calculating volatility...")
for ticker in tickers:
    returns = df[ticker].pct_change()
    df[ticker + '_volatility'] = returns.rolling(20).std() * np.sqrt(252)
    df[ticker + '_volatility'] = (df[ticker + '_volatility'] - 0.25) / 0.25

df = df.dropna()
print(f"After preprocessing: {len(df)} days remaining")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# If using test split, take last 20%
if use_test_split:
    n_train = int(0.8 * len(df))
    df_eval = df.iloc[n_train:]
    print(f"\n✓ Using test split: {len(df_eval)} days ({df_eval.index[0]} to {df_eval.index[-1]})")
else:
    df_eval = df
    print(f"\n✓ Using entire dataset: {len(df_eval)} days")

##############
### DEVICE ###
##############
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
print(f"Using device: {device}")

##########################
### LOAD TRAINED MODEL ###
##########################
print("\n" + "="*50)
print("Available models:")
print("="*50)

model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
if not model_files:
    print("❌ No models found in models/ folder!")
    print("Please train a model first by running a2c_main.py")
    exit(1)

for i, f in enumerate(model_files, 1):
    file_size = os.path.getsize(os.path.join('models', f)) / 1024  # KB
    print(f"  {i}. {f} ({file_size:.1f} KB)")

choice = input("\nEnter the number of the model to evaluate: ").strip()

try:
    model_idx = int(choice) - 1
    model_file = model_files[model_idx]
    model_path = os.path.join('models', model_file)
    print(f"\n✓ Selected: {model_file}")
except (ValueError, IndexError):
    print("Invalid choice. Exiting...")
    exit(1)

# Create environment
env_eval = Env(df_eval, tickers, lag=5)

# Initialize agent with same hyperparameters as training
agent = Agent(
    n_input=env_eval.states.shape[1] * env_eval.lag,
    n_action=len(tickers) + 1,
    learning_rate=0.001,  # Not used for evaluation
    gamma=0.99,
    entropy_weight=0.001,
    value_weight=0.25,
    max_norm=0.5,
    device=device
)

# Load the model
print(f"Loading model weights...")
if 'best' in model_file:
    # Load best model (has metadata)
    best_state = torch.load(model_path, map_location=device)
    agent.load_best_model(best_state)
    if 'episode' in best_state:
        print(f"  - Model from episode: {best_state['episode']}")
        print(f"  - Training step: {best_state['global_step']}")
        print(f"  - Training avg return: {best_state['avg_return']:.4f}")
else:
    # Load final model
    agent.load_model(model_path)

agent.ac_network.eval()  # Set to evaluation mode
print("✓ Model loaded successfully!")

###################################
### EVALUATE A2C AGENT ###
###################################
print("\n" + "="*50)
print("EVALUATING A2C AGENT")
print("="*50)

obs = env_eval.reset()
a2c_rewards = []
a2c_actions = []
done = False

print("Running evaluation...")
while not done:
    with torch.no_grad():
        alpha, _ = agent.ac_network(agent.np2torch(obs))
        actions = agent.sample_action(alpha)
        actions_np = actions.detach().cpu().numpy().flatten()
    
    a2c_actions.append(actions_np.copy())
    obs, reward, done = env_eval.step(actions_np)
    a2c_rewards.append(reward)

a2c_rewards = np.array(a2c_rewards)
a2c_actions = np.array(a2c_actions)

# Calculate metrics
a2c_cumulative = np.cumprod(1 + a2c_rewards)
a2c_total_return = a2c_cumulative[-1] - 1
a2c_sharpe = np.mean(a2c_rewards) / (np.std(a2c_rewards) + 1e-8) * np.sqrt(252)

# Maximum drawdown
a2c_running_max = np.maximum.accumulate(a2c_cumulative)
a2c_drawdown = (a2c_cumulative - a2c_running_max) / a2c_running_max
a2c_max_drawdown = np.min(a2c_drawdown)

# Average portfolio weights
avg_weights = np.mean(a2c_actions, axis=0)

###################################
### EVALUATE BENCHMARK ###
###################################
print("Running benchmark evaluation...")
benchmark_weights = np.array([0.1, 0.8, 0.05, 0.05, 0])
obs = env_eval.reset()
benchmark_rewards = []
done = False

while not done:
    obs, reward, done = env_eval.step(benchmark_weights)
    benchmark_rewards.append(reward)

benchmark_rewards = np.array(benchmark_rewards)
benchmark_cumulative = np.cumprod(1 + benchmark_rewards)
benchmark_total_return = benchmark_cumulative[-1] - 1
benchmark_sharpe = np.mean(benchmark_rewards) / (np.std(benchmark_rewards) + 1e-8) * np.sqrt(252)

# Maximum drawdown
benchmark_running_max = np.maximum.accumulate(benchmark_cumulative)
benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
benchmark_max_drawdown = np.min(benchmark_drawdown)

###################################
### RESULTS ###
###################################
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

print(f"\nEvaluation Period: {df_eval.index[0]} to {df_eval.index[-1]}")
print(f"Number of trading days: {len(df_eval)}")

print(f"\n{'='*50}")
print("A2C AGENT PERFORMANCE")
print(f"{'='*50}")
print(f"Total Return:        {a2c_total_return*100:>8.2f}%")
print(f"Sharpe Ratio:        {a2c_sharpe:>8.4f}")
print(f"Max Drawdown:        {a2c_max_drawdown*100:>8.2f}%")
print(f"Annualized Vol:      {np.std(a2c_rewards)*np.sqrt(252)*100:>8.2f}%")
print(f"\nAverage Portfolio Allocation:")
asset_names = ['VNQ', 'SPY', 'GLD', 'BTC', 'Cash']
for i, name in enumerate(asset_names):
    print(f"  {name:>4}: {avg_weights[i]*100:>6.2f}%")

print(f"\n{'='*50}")
print("BENCHMARK PERFORMANCE (10/80/5/5/0)")
print(f"{'='*50}")
print(f"Total Return:        {benchmark_total_return*100:>8.2f}%")
print(f"Sharpe Ratio:        {benchmark_sharpe:>8.4f}")
print(f"Max Drawdown:        {benchmark_max_drawdown*100:>8.2f}%")
print(f"Annualized Vol:      {np.std(benchmark_rewards)*np.sqrt(252)*100:>8.2f}%")

print(f"\n{'='*50}")
print("COMPARISON (A2C vs Benchmark)")
print(f"{'='*50}")
return_diff = (a2c_total_return - benchmark_total_return) * 100
sharpe_diff = a2c_sharpe - benchmark_sharpe
print(f"Return Difference:   {return_diff:>+8.2f}%")
print(f"Sharpe Difference:   {sharpe_diff:>+8.4f}")
print(f"Drawdown Difference: {(a2c_max_drawdown - benchmark_max_drawdown)*100:>+8.2f}%")

if a2c_total_return > benchmark_total_return:
    print(f"\n✓ A2C outperformed benchmark by {return_diff:.2f}%")
else:
    print(f"\n⚠️  A2C underperformed benchmark by {abs(return_diff):.2f}%")

###################################
### VISUALIZATION ###
###################################
visualize = input("\nWould you like to see visualizations? (y/n): ").strip().lower()

if visualize == 'y':
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    ax1.plot(a2c_cumulative, label='A2C Agent', linewidth=2, color='blue')
    ax1.plot(benchmark_cumulative, label='Benchmark', linewidth=2, linestyle='--', color='orange')
    ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    ax2.fill_between(range(len(a2c_drawdown)), a2c_drawdown*100, 0, 
                      alpha=0.4, label='A2C Drawdown', color='blue')
    ax2.fill_between(range(len(benchmark_drawdown)), benchmark_drawdown*100, 0, 
                      alpha=0.4, label='Benchmark Drawdown', color='orange')
    ax2.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Daily Returns Distribution
    ax3 = axes[1, 0]
    ax3.hist(a2c_rewards*100, bins=50, alpha=0.5, label='A2C', density=True, color='blue')
    ax3.hist(benchmark_rewards*100, bins=50, alpha=0.5, label='Benchmark', density=True, color='orange')
    ax3.axvline(np.mean(a2c_rewards)*100, color='blue', linestyle='--', linewidth=2, label='A2C Mean')
    ax3.axvline(np.mean(benchmark_rewards)*100, color='orange', linestyle='--', linewidth=2, label='Bench Mean')
    ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Daily Return (%)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Portfolio Weights Over Time
    ax4 = axes[1, 1]
    colors = ['brown', 'green', 'gold', 'orange', 'gray']
    for i, (name, color) in enumerate(zip(asset_names, colors)):
        ax4.plot(a2c_actions[:, i], label=name, alpha=0.7, linewidth=1.5, color=color)
    ax4.set_title('A2C Portfolio Weights Over Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trading Days')
    ax4.set_ylabel('Weight')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.suptitle(f'Evaluation Results: {df_eval.index[0].strftime("%Y-%m-%d")} to {df_eval.index[-1].strftime("%Y-%m-%d")}', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Visualizations displayed!")

print("\n" + "="*50)
print("Evaluation complete!")
print("="*50)