import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import time
import os

import torch

from environment import Env
from agent import Agent

print("="*60)
print("A2C PORTFOLIO TRAINING SCRIPT")
print("="*60)

####################
### CONFIGURATION ###
####################
start = '2010-01-01'
end = '2024-12-31'
tickers = ['VNQ', 'SPY', 'TLT', 'GLD', 'BTC-USD']

# Hyperparameters
total_timesteps = 1800000
learning_rate = 0.001
gamma = 0.99
entropy_weight = 0.001
value_weight = 0.25
max_norm = 0.5

# Paths
model_path_final = "models/a2c_portfolio_final.pth"
model_path_best = "models/a2c_portfolio_best.pth"

# Create models directory
os.makedirs("models", exist_ok=True)

####################
### LOAD DATA ###
####################
print("\n" + "="*60)
print("STEP 1: DOWNLOADING MARKET DATA")
print("="*60)

try:
    print(f"Downloading data for {tickers}...")
    df = yf.download(tickers, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("Downloaded data is empty. Check tickers and date range.")
    print(f"✓ Downloaded {len(df)} days of data")
except Exception as e:
    print(f"✗ Error downloading data: {e}")
    print(f"  Tickers: {tickers}")
    print(f"  Date range: {start} to {end}")
    exit(1)

df = df['Close']
df = df.dropna().copy()

####################
### VIX Features ###
####################
print("\n" + "="*60)
print("STEP 2: LOADING VIX DATA")
print("="*60)

try:
    vix_df = pd.read_csv('../../data/VIX_CLS_2010_2024.csv')
    vix_df['observation_date'] = pd.to_datetime(vix_df['observation_date'])
    vix_df = vix_df.set_index('observation_date')
    vix_df = vix_df.rename(columns={'VIXCLS': 'VIX'})
    
    vix3m_df = pd.read_csv('../../data/VIX3M_CLS_2010_2024.csv')
    vix3m_df['observation_date'] = pd.to_datetime(vix3m_df['observation_date'])
    vix3m_df = vix3m_df.set_index('observation_date')
    vix3m_df = vix3m_df.rename(columns={'VXVCLS': 'VIX3M'})
    
    print("✓ VIX data loaded successfully")
except Exception as e:
    print(f"✗ Error loading VIX data: {e}")
    exit(1)

vix_combined = vix_df.join(vix3m_df, how='outer')
df = df.join(vix_combined, how='left')
df['VIX'] = df['VIX'].ffill()
df['VIX3M'] = df['VIX3M'].ffill()

# VIX Feature 1 - VIX_normalized
df['VIX_normalized'] = (df['VIX'] - 20) / 20

# VIX Feature 2 - VIX_regime
def get_vix_regime(vix):
    if vix < 15:
        return -1.0
    elif vix < 25:
        return 0.0
    else:
        return 1.0
df['VIX_regime'] = df['VIX'].apply(get_vix_regime)

# VIX Feature 3 - VIX_term_structure
df['VIX_term_structure'] = (df['VIX3M'] - df['VIX']) / df['VIX']
df['VIX_term_structure'] = np.clip(df['VIX_term_structure'], -1, 1)

print("✓ VIX features created: 3 features")

##############################
### Credit Spread Features ###
##############################
print("\n" + "="*60)
print("STEP 3: LOADING CREDIT SPREAD DATA")
print("="*60)

try:
    credit_df = pd.read_csv('../../data/CREDIT_SPREAD_2010_2024.csv')
    credit_df['observation_date'] = pd.to_datetime(credit_df['observation_date'])
    credit_df = credit_df.set_index('observation_date')
    print("✓ Credit spread data loaded successfully")
except Exception as e:
    print(f"✗ Error loading credit spread data: {e}")
    exit(1)

df = df.join(credit_df, how='left')

if 'BAMLC0A4CBBB' not in df.columns:
    print(f"✗ Expected column 'BAMLC0A4CBBB' not found!")
    print(f"  Available columns: {credit_df.columns.tolist()}")
    exit(1)

df['BAMLC0A4CBBB'] = df['BAMLC0A4CBBB'].ffill()
df['Credit_Spread'] = df['BAMLC0A4CBBB'] / 100

# CS Feature 1 - Credit_Spread_normalized
df['Credit_Spread_normalized'] = (df['Credit_Spread'] - 0.02) / 0.02

# CS Feature 2 - Credit_Spread_regime
def get_credit_regime(spread):
    if spread < 0.02:
        return -1.0
    elif spread < 0.04:
        return 0.0
    else:
        return 1.0

df['Credit_Spread_regime'] = df['Credit_Spread'].apply(get_credit_regime)

# CS Feature 3 - Credit_Spread_momentum
df['Credit_Spread_momentum'] = df['Credit_Spread'].pct_change(30)
df['Credit_Spread_momentum'] = np.clip(df['Credit_Spread_momentum'], -1, 1)

# CS Feature 4 - Credit_Spread_zscore
df['Credit_Spread_zscore'] = (
    (df['Credit_Spread'] - df['Credit_Spread'].rolling(252).mean()) / 
    (df['Credit_Spread'].rolling(252).std() + 1e-8)
)
df['Credit_Spread_zscore'] = np.clip(df['Credit_Spread_zscore'], -3, 3)

# CS Feature 5 - Credit_Spread_velocity
df['Credit_Spread_velocity'] = df['Credit_Spread_momentum'].diff(5)
df['Credit_Spread_velocity'] = np.clip(df['Credit_Spread_velocity'], -1, 1)

# CS Feature 6 - Credit_VIX_divergence
vix_normalized = (df['VIX'] - df['VIX'].rolling(60).mean()) / (df['VIX'].rolling(60).std() + 1e-8)
credit_normalized = (df['Credit_Spread'] - df['Credit_Spread'].rolling(60).mean()) / (df['Credit_Spread'].rolling(60).std() + 1e-8)
df['Credit_VIX_divergence'] = vix_normalized - credit_normalized
df['Credit_VIX_divergence'] = np.clip(df['Credit_VIX_divergence'], -3, 3)

print("✓ Credit spread features created: 6 features")

####################
### RSI Features ###
####################
print("\n" + "="*60)
print("STEP 4: CREATING RSI FEATURES")
print("="*60)

for ticker in tickers:
    df[ticker + '_RSI'] = talib.RSI(df[ticker], timeperiod=14) / 50 - 1

print(f"✓ RSI features created: {len(tickers)} features")

###########################
### Realized Volatility ###
###########################
print("\n" + "="*60)
print("STEP 5: CREATING VOLATILITY FEATURES")
print("="*60)

for ticker in tickers:
    returns = df[ticker].pct_change()
    df[ticker + '_volatility'] = returns.rolling(20).std() * np.sqrt(252)
    df[ticker + '_volatility'] = (df[ticker + '_volatility'] - 0.25) / 0.25

print(f"✓ Volatility features created: {len(tickers)} features")

# Drop NaN rows
df = df.dropna()

print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"Data shape after feature engineering: {df.shape}")
print(f"Final date range: {df.index[0]} to {df.index[-1]}")

feature_cols = [col for col in df.columns if col not in tickers + ['VIX', 'VIX3M', 'BAMLC0A4CBBB', 'Credit_Spread']]
print(f"\nTotal features: {len(feature_cols)}")
print("  - VIX features: 3")
print("  - Credit spread features: 6")
print(f"  - RSI features: {len(tickers)}")
print(f"  - Volatility features: {len(tickers)}")

##########################
### SPLIT Train & Test ###
##########################
print("\n" + "="*60)
print("STEP 6: SPLITTING TRAIN/TEST DATA")
print("="*60)

n_train = int(0.8 * len(df))
df_train = df.iloc[:n_train]
df_test = df.iloc[n_train:]

print(f"Train set: {len(df_train)} days ({df_train.index[0]} to {df_train.index[-1]})")
print(f"Test set:  {len(df_test)} days ({df_test.index[0]} to {df_test.index[-1]})")

###################
### ENVIRONMENT ###
###################
print("\n" + "="*60)
print("STEP 7: CREATING TRAINING ENVIRONMENT")
print("="*60)

env = Env(df_train, tickers, lag=5)
print(f"✓ Environment state dimension: {env.states.shape[1] * env.lag}")

##############
### Device ###
##############
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
print(f"✓ Using device: {device}")

#############
### AGENT ###
#############
print("\n" + "="*60)
print("STEP 8: INITIALIZING AGENT")
print("="*60)

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

print("✓ Agent initialized with hyperparameters:")
print(f"  - Total timesteps: {total_timesteps:,}")
print(f"  - Learning rate: {learning_rate}")
print(f"  - Gamma: {gamma}")
print(f"  - Entropy weight: {entropy_weight}")
print(f"  - Value weight: {value_weight}")
print(f"  - Max norm: {max_norm}")

#######################
### Training Starts ###
#######################
print("\n" + "="*60)
print("STEP 9: STARTING TRAINING")
print("="*60)
print("This will take approximately 2 hours on GPU...")
print("You can monitor progress in the output below.")
print("="*60 + "\n")

training_start_time = time.time()

try:
    episode_returns, losses, best_model_state = agent.learn(env, total_timesteps)
except KeyboardInterrupt:
    print("\n" + "="*60)
    print("⚠ Training interrupted by user!")
    print("="*60)
    print("Saving current model state...")
    agent.save_model("models/a2c_portfolio_interrupted.pth")
    print("✓ Model saved to: models/a2c_portfolio_interrupted.pth")
    print("  You can resume or evaluate this model later.")
    exit(0)
except Exception as e:
    print("\n" + "="*60)
    print("✗ Training failed with error!")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

training_time = time.time() - training_start_time

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Total training time: {training_time/60:.1f} minutes ({training_time/3600:.2f} hours)")

#######################
### Save the Models ###
#######################
print("\n" + "="*60)
print("STEP 10: SAVING MODELS")
print("="*60)

agent.save_model(model_path_final)
print(f"✓ Final model saved to: {model_path_final}")

if best_model_state is not None:
    torch.save(best_model_state, model_path_best)
    print(f"✓ Best model saved to: {model_path_best}")
    print(f"  - Episode: {best_model_state['episode']}")
    print(f"  - Global step: {best_model_state['global_step']}")
    print(f"  - Avg return (last 10 episodes): {best_model_state['avg_return']:.4f}")
else:
    print("⚠ No best model was saved (training may have been too short)")

##########################
### Training Statistics ###
##########################
print("\n" + "="*60)
print("TRAINING STATISTICS")
print("="*60)

if len(episode_returns) > 0:
    print(f"Total episodes completed: {len(episode_returns)}")
    print(f"Average episode return:   {np.mean(episode_returns):.4f}")
    print(f"Best episode return:      {np.max(episode_returns):.4f}")
    print(f"Worst episode return:     {np.min(episode_returns):.4f}")
    if len(episode_returns) >= 10:
        print(f"Final 10 episodes avg:    {np.mean(episode_returns[-10:]):.4f}")
    if len(episode_returns) >= 50:
        print(f"Final 50 episodes avg:    {np.mean(episode_returns[-50:]):.4f}")

######################
### Plot Training ###
######################
print("\n" + "="*60)
print("VISUALIZING TRAINING PROGRESS")
print("="*60)

if len(episode_returns) > 0:
    plot_choice = input("Would you like to plot training results? (y/n): ").strip().lower()
    
    if plot_choice == 'y':
        def smooth(x, a=0.1):
            """Exponential moving average smoothing"""
            y = [x[0]]
            for xi in x[1:]:
                yi = a * xi + (1 - a) * y[-1]
                y.append(yi)
            return y
        
        episode_returns_pct = [ret * 100 for ret in episode_returns]
        
        plt.figure(figsize=(12, 6))
        plt.plot(episode_returns_pct, alpha=0.2, label='Raw Returns', linewidth=0.5, color='blue')
        plt.plot(smooth(episode_returns_pct, a=0.05), label='Smoothed Returns (α=0.05)', 
                linewidth=2, color='red')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.title("Episode Returns During Training", fontsize=14, fontweight='bold')
        plt.xlabel("Episode Number", fontsize=12)
        plt.ylabel("Total Return (%)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        os.makedirs("plots", exist_ok=True)
        plot_filename = "plots/training_progress.png"
        plt.savefig(plot_filename, dpi=150)
        print(f"✓ Plot saved to: {plot_filename}")
        
        plt.show()
        print("✓ Plot displayed!")

###############
### Summary ###
###############
print("\n" + "="*60)
print("TRAINING COMPLETE - NEXT STEPS")
print("="*60)
print("\nYour trained model is ready!")
print(f"  - Best model:  {model_path_best}")
print(f"  - Final model: {model_path_final}")
print("\nTo evaluate your model on the test set:")
print("  python evaluate.py")
print("\n" + "="*60)