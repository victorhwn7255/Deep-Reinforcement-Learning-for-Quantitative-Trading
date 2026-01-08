import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import random
import time
from datetime import datetime

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

tickers = ['VNQ', 'SPY', 'TLT', 'GLD', 'BTC-USD']

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

# VIX Feature 1 - VIX_normalized: Normalized VIX level
# Range roughly [-1, 1]
df['VIX_normalized'] = (df['VIX'] - 20) / 20

# VIX Feature 2 - VIX_regime: Categorical regime (Low/Normal/High volatility)
def get_vix_regime(vix):
    if vix < 15:
        return -1.0  # Low vol / complacent
    elif vix < 25:
        return 0.0   # Normal vol
    else:
        return 1.0   # High vol / stress
df['VIX_regime'] = df['VIX'].apply(get_vix_regime)

# VIX Feature 3 - VIX_term_structure: REAL term structure using VIX3M
# Positive = Contango (normal markets, VIX3M > VIX)
# Negative = Backwardation (stressed markets, VIX3M < VIX)
df['VIX_term_structure'] = (df['VIX3M'] - df['VIX']) / df['VIX']
df['VIX_term_structure'] = np.clip(df['VIX_term_structure'], -1, 1)  # Clip extreme values


##############################
### Credit Spread Features ###
##############################
credit_df = pd.read_csv('../../data/CREDIT_SPREAD_2010_2024.csv')
credit_df['observation_date'] = pd.to_datetime(credit_df['observation_date'])
credit_df = credit_df.set_index('observation_date')

df = df.join(credit_df, how='left')
df['BAMLC0A4CBBB'] = df['BAMLC0A4CBBB'].ffill()
df['Credit_Spread'] = df['BAMLC0A4CBBB'] / 100

# CS Feature 1 - Credit_Spread_normalized
# Normalize around typical value of 2% (0.02). 
# Normal range: 1-2.5% → normalized ≈ [-0.5, 0.25]
# Crisis: 5%+ → normalized ≈ 1.5+
df['Credit_Spread_normalized'] = (df['Credit_Spread'] - 0.02) / 0.02

# CS Feature 2 - Credit_Spread_regime (categorical)
def get_credit_regime(spread):
    if spread < 0.02:      # <2%
        return -1.0  # Risk-on (normal credit conditions)
    elif spread < 0.04:    # 2-4%
        return 0.0   # Elevated risk
    else:                  # >4%
        return 1.0   # Risk-off (crisis/severe stress)

df['Credit_Spread_regime'] = df['Credit_Spread'].apply(get_credit_regime)

# count regime distribution
regime_counts = df['Credit_Spread_regime'].value_counts().sort_index()
print(f"  - Credit_Spread_regime distribution:")
regime_names = {-1.0: 'Risk-On', 0.0: 'Elevated', 1.0: 'Crisis'}
for regime, count in regime_counts.items():
    pct = count / len(df) * 100
    print(f"    {regime_names[regime]:>10} ({regime:+.0f}): {count:4d} days ({pct:.1f}%)")

# CS Feature 3 - Credit_Spread_momentum (RoC)
# 30-day momentum captures credit deterioration/improvement
df['Credit_Spread_momentum'] = df['Credit_Spread'].pct_change(30)
df['Credit_Spread_momentum'] = np.clip(df['Credit_Spread_momentum'], -1, 1)

# CS Feature 4 - Credit_Spread_zscore
# How many standard deviations from rolling mean?
# Identifies extreme moves (leading indicator of regime change)
df['Credit_Spread_zscore'] = (
    (df['Credit_Spread'] - df['Credit_Spread'].rolling(252).mean()) / 
    (df['Credit_Spread'].rolling(252).std() + 1e-8)
)
df['Credit_Spread_zscore'] = np.clip(df['Credit_Spread_zscore'], -3, 3)

# CS Feature 5 - Credit_Spread_velocity (acceleration)
# Second derivative - how fast is spread changing?
# Rapid widening = panic; rapid tightening = recovery
df['Credit_Spread_velocity'] = df['Credit_Spread_momentum'].diff(5)
df['Credit_Spread_velocity'] = np.clip(df['Credit_Spread_velocity'], -1, 1)

# CS Feature 6 - Credit_VIX_divergence
# When credit spreads and VIX diverge, it signals regime transition
# Positive = VIX rising faster (equity fear)
# Negative = Credit spreads rising faster (credit fear)
vix_normalized = (df['VIX'] - df['VIX'].rolling(60).mean()) / (df['VIX'].rolling(60).std() + 1e-8)
credit_normalized = (df['Credit_Spread'] - df['Credit_Spread'].rolling(60).mean()) / (df['Credit_Spread'].rolling(60).std() + 1e-8)
df['Credit_VIX_divergence'] = vix_normalized - credit_normalized
df['Credit_VIX_divergence'] = np.clip(df['Credit_VIX_divergence'], -3, 3)


####################
### RSI Features ###
####################
for ticker in tickers:
  df[ticker + '_RSI'] = talib.RSI(df[ticker], timeperiod=14) / 50 - 1 # change RSI range to [-1, +1]


###########################
### Realized Volatility ###
###########################
for ticker in tickers:
    returns = df[ticker].pct_change()
    df[ticker + '_volatility'] = returns.rolling(20).std() * np.sqrt(252)  # 20-day rolling vol
    # Normalize: typical stock vol is ~0.20 (20%), so divide by 0.25 to get roughly [-1, 1] range
    df[ticker + '_volatility'] = (df[ticker + '_volatility'] - 0.25) / 0.25


########################
### Features Summary ###
########################
df = df.dropna()
print(f"\nData shape after feature engineering: {df.shape}")
print(f"Final date range: {df.index[0]} to {df.index[-1]}")

print("\nAll features created:")
feature_cols = [col for col in df.columns if col not in tickers + ['VIX', 'VIX3M']]
for col in sorted(feature_cols):
    print(f"  - {col}")

print("\nFeature Range Validation:")
feature_cols = [col for col in df.columns if col not in tickers + ['VIX', 'VIX3M', 'BAMLC0A4CBBB', 'Credit_Spread']]

out_of_range = []
for col in feature_cols:
    min_val, max_val = df[col].min(), df[col].max()
    if min_val < -10 or max_val > 10:
        out_of_range.append(f"  {col}: [{min_val:.2f}, {max_val:.2f}]")

if out_of_range:
    print("  ⚠ Features with unusual ranges:")
    for item in out_of_range:
        print(item)
else:
    print("  ✓ All features within expected ranges")

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
weights = np.array([0.1, 0.5, 0.3, 0.05, 0.05, 0]) # constant weight portfolio
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
total_timesteps = 1800000      # Total number of training steps
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
    
################################
### Post-Training Evaluation ###
################################
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
        
        #############################
        ### CHOOSE EVALUATION MODE ###
        #############################
        print("\n" + "="*50)
        print("EVALUATION MODE SELECTION")
        print("="*50)
        print("\nHow would you like to evaluate the model?")
        print("  1. Deterministic (mean allocation - reproducible, single run)")
        print("  2. Stochastic with seed (reproducible, single run)")
        print("  3. Multiple stochastic runs (statistical analysis)")
        
        eval_mode = input("\nEnter choice (1, 2, or 3): ").strip()
        
        # Initialize variables for all modes
        n_runs = 1  # Default for modes 1 and 2
        show_statistics = False
        
        ##############################
        ### MODE 1: DETERMINISTIC ###
        ##############################
        if eval_mode == '1':
            print("\n✓ Using deterministic evaluation (mean of Dirichlet)")
            print("  This gives reproducible results by using expected allocation")
            
            obs = env_test.reset()
            a2c_rewards = []
            a2c_actions = []
            done = False
            
            while not done:
                actions_np = eval_agent.choose_action_deterministic(obs)
                a2c_actions.append(actions_np.copy())
                obs, reward, done = env_test.step(actions_np)
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
            
            # For display
            display_mode = "Deterministic"
        
        ##################################
        ### MODE 2: STOCHASTIC + SEED ###
        ##################################
        elif eval_mode == '2':
            print("\n✓ Using stochastic evaluation with fixed seed")
            print("  This gives reproducible results with sampling randomness")
            
            # Set seed for reproducibility
            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)
            
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
            
            # Calculate metrics
            a2c_cumulative = np.cumprod(1 + a2c_rewards)
            a2c_total_return = a2c_cumulative[-1] - 1
            a2c_sharpe = np.mean(a2c_rewards) / (np.std(a2c_rewards) + 1e-8) * np.sqrt(252)
            
            # Maximum drawdown
            a2c_running_max = np.maximum.accumulate(a2c_cumulative)
            a2c_drawdown = (a2c_cumulative - a2c_running_max) / a2c_running_max
            a2c_max_drawdown = np.min(a2c_drawdown)
            
            # For display
            display_mode = "Stochastic (seed=42)"
        
        #################################
        ### MODE 3: MULTIPLE RUNS ###
        #################################
        elif eval_mode == '3':
            # Get number of runs with validation
            n_runs_input = input("\nHow many evaluation runs? (recommended: 10-20, default: 10): ").strip()
            try:
                n_runs = int(n_runs_input) if n_runs_input else 10
                n_runs = max(1, min(n_runs, 100))  # Clamp between 1-100
                if n_runs != int(n_runs_input or 10):
                    print(f"  Note: Adjusted to {n_runs} runs (valid range: 1-100)")
            except ValueError:
                print("  Invalid input. Using default of 10 runs.")
                n_runs = 10
            
            print(f"\n✓ Running {n_runs} stochastic evaluations...")
            print("  This may take a few minutes...")
            
            all_returns = []
            all_sharpes = []
            all_drawdowns = []
            all_volatilities = []
            all_actions_list = []
            all_rewards_list = []
            
            # Progress indicator
            print("  Progress: ", end="", flush=True)
            progress_interval = max(1, n_runs // 10)
            
            for run in range(n_runs):
                obs = env_test.reset()
                run_rewards = []
                run_actions = []
                done = False
                
                while not done:
                    with torch.no_grad():
                        alpha, _ = eval_agent.ac_network(eval_agent.np2torch(obs))
                        actions = eval_agent.sample_action(alpha)
                        actions_np = actions.detach().cpu().numpy().flatten()
                    
                    run_actions.append(actions_np.copy())
                    obs, reward, done = env_test.step(actions_np)
                    run_rewards.append(reward)
                
                run_rewards = np.array(run_rewards)
                run_actions = np.array(run_actions)
                
                # Calculate metrics for this run
                run_cumulative = np.cumprod(1 + run_rewards)
                run_return = run_cumulative[-1] - 1
                run_sharpe = np.mean(run_rewards) / (np.std(run_rewards) + 1e-8) * np.sqrt(252)
                run_volatility = np.std(run_rewards) * np.sqrt(252)
                
                run_running_max = np.maximum.accumulate(run_cumulative)
                run_drawdown = (run_cumulative - run_running_max) / run_running_max
                run_max_drawdown = np.min(run_drawdown)
                
                all_returns.append(run_return)
                all_sharpes.append(run_sharpe)
                all_drawdowns.append(run_max_drawdown)
                all_volatilities.append(run_volatility)
                all_actions_list.append(run_actions)
                all_rewards_list.append(run_rewards)
                
                # Progress indicator
                if (run + 1) % progress_interval == 0 or run == n_runs - 1:
                    progress_pct = int((run + 1) * 100 / n_runs)
                    print(f"{progress_pct}%", end=" ", flush=True)
            
            print("\n  Complete!\n")
            
            # Print individual run results
            print("  Individual Run Results:")
            for run in range(n_runs):
                print(f"    Run {run+1:2d}/{n_runs}: Return = {all_returns[run]*100:7.2f}%, "
                      f"Sharpe = {all_sharpes[run]:.4f}, Max DD = {all_drawdowns[run]*100:.2f}%")
            
            # Convert to arrays
            all_returns = np.array(all_returns)
            all_sharpes = np.array(all_sharpes)
            all_drawdowns = np.array(all_drawdowns)
            all_volatilities = np.array(all_volatilities)
            
            # Use median run for visualization
            median_idx = np.argsort(all_returns)[len(all_returns)//2]
            a2c_rewards = all_rewards_list[median_idx]
            a2c_actions = all_actions_list[median_idx]
            
            # Recalculate for median run (for visualization)
            a2c_cumulative = np.cumprod(1 + a2c_rewards)
            a2c_total_return = a2c_cumulative[-1] - 1
            a2c_sharpe = np.mean(a2c_rewards) / (np.std(a2c_rewards) + 1e-8) * np.sqrt(252)
            a2c_running_max = np.maximum.accumulate(a2c_cumulative)
            a2c_drawdown = (a2c_cumulative - a2c_running_max) / a2c_running_max
            a2c_max_drawdown = np.min(a2c_drawdown)
            
            # For display
            display_mode = f"Multiple Runs (n={n_runs})"
            show_statistics = True
        
        else:
            print("Invalid choice, skipping evaluation")
            continue
        
        ###################################
        ### EVALUATE BENCHMARK PORTFOLIO ###
        ###################################
        print("\nEvaluating Benchmark...")
        benchmark_weights = np.array([0.1, 0.5, 0.3, 0.05, 0.05, 0])
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
        ### PRINT RESULTS ###
        ########################
        print("\n" + "="*50)
        print("TEST SET PERFORMANCE")
        print("="*50)
        print(f"Evaluation Mode: {display_mode}")
        
        if show_statistics:
            # Show aggregated statistics
            print("\n" + "-"*50)
            print("A2C AGENT - STATISTICAL SUMMARY")
            print("-"*50)
            print(f"  Total Return:")
            print(f"    Mean:        {np.mean(all_returns)*100:>8.2f}%")
            print(f"    Std Dev:     {np.std(all_returns)*100:>8.2f}%")
            print(f"    Min:         {np.min(all_returns)*100:>8.2f}%")
            print(f"    Max:         {np.max(all_returns)*100:>8.2f}%")
            print(f"    Median:      {np.median(all_returns)*100:>8.2f}%")
            print(f"    95% CI:      [{np.percentile(all_returns, 2.5)*100:.2f}%, {np.percentile(all_returns, 97.5)*100:.2f}%]")
            
            print(f"\n  Sharpe Ratio:")
            print(f"    Mean:        {np.mean(all_sharpes):>8.4f}")
            print(f"    Std Dev:     {np.std(all_sharpes):>8.4f}")
            print(f"    Min:         {np.min(all_sharpes):>8.4f}")
            print(f"    Max:         {np.max(all_sharpes):>8.4f}")
            
            print(f"\n  Max Drawdown:")
            print(f"    Mean:        {np.mean(all_drawdowns)*100:>8.2f}%")
            print(f"    Std Dev:     {np.std(all_drawdowns)*100:>8.2f}%")
            print(f"    Best (min):  {np.max(all_drawdowns)*100:>8.2f}%")
            print(f"    Worst (max): {np.min(all_drawdowns)*100:>8.2f}%")
            
            print(f"\n  Ann. Volatility:")
            print(f"    Mean:        {np.mean(all_volatilities)*100:>8.2f}%")
            print(f"    Std Dev:     {np.std(all_volatilities)*100:>8.2f}%")
            
            # Probability metrics
            prob_beat_benchmark = np.mean(all_returns > benchmark_total_return) * 100
            prob_positive = np.mean(all_returns > 0) * 100
            prob_2x_benchmark = np.mean(all_returns > 2 * benchmark_total_return) * 100
            
            print(f"\n  Probability Metrics:")
            print(f"    P(beat benchmark):   {prob_beat_benchmark:>6.1f}%")
            print(f"    P(positive return):  {prob_positive:>6.1f}%")
            print(f"    P(>2x benchmark):    {prob_2x_benchmark:>6.1f}%")
            
            # Coefficient of variation
            cv_return = np.std(all_returns) / np.mean(all_returns) if np.mean(all_returns) != 0 else 0
            print(f"\n  Stability Metric:")
            print(f"    Coefficient of Variation: {cv_return:.4f}")
            if cv_return < 0.05:
                stability = "VERY STABLE ✓✓✓"
            elif cv_return < 0.10:
                stability = "STABLE ✓✓"
            elif cv_return < 0.15:
                stability = "MODERATE ✓"
            else:
                stability = "UNSTABLE ⚠"
            print(f"    Assessment: {stability}")
            
            # Option to save results to CSV
            save_csv = input("\n  Would you like to save detailed results to CSV? (y/n): ").strip().lower()
            if save_csv == 'y':
                results_df = pd.DataFrame({
                    'run': range(1, n_runs + 1),
                    'total_return_pct': all_returns * 100,
                    'sharpe_ratio': all_sharpes,
                    'max_drawdown_pct': all_drawdowns * 100,
                    'ann_volatility_pct': all_volatilities * 100
                })
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"evaluation_results_{timestamp}.csv"
                results_df.to_csv(filename, index=False)
                print(f"  ✓ Results saved to: {filename}")
            
        else:
            # Show single run results
            print("\n" + "-"*50)
            print("A2C AGENT")
            print("-"*50)
            print(f"  Total Return:      {a2c_total_return*100:>8.2f}%")
            print(f"  Sharpe Ratio:      {a2c_sharpe:>8.4f}")
            print(f"  Max Drawdown:      {a2c_max_drawdown*100:>8.2f}%")
            print(f"  Ann. Volatility:   {np.std(a2c_rewards)*np.sqrt(252)*100:>8.2f}%")
        
        print("\n" + "-"*50)
        print("BENCHMARK (VNQ:10% SPY:50% TLT:30% GLD:5% BTC:5%)")
        print("-"*50)
        print(f"  Total Return:      {benchmark_total_return*100:>8.2f}%")
        print(f"  Sharpe Ratio:      {benchmark_sharpe:>8.4f}")
        print(f"  Max Drawdown:      {benchmark_max_drawdown*100:>8.2f}%")
        print(f"  Ann. Volatility:   {np.std(benchmark_rewards)*np.sqrt(252)*100:>8.2f}%")
        
        print("\n" + "-"*50)
        print("COMPARISON (A2C vs Benchmark)")
        print("-"*50)
        
        if show_statistics:
            mean_return_diff = (np.mean(all_returns) - benchmark_total_return) * 100
            mean_sharpe_diff = np.mean(all_sharpes) - benchmark_sharpe
            print(f"  Return Difference:   {mean_return_diff:>+8.2f}% (mean)")
            print(f"  Sharpe Difference:   {mean_sharpe_diff:>+8.4f} (mean)")
            
            if np.mean(all_returns) > benchmark_total_return:
                print(f"\n  ✓ A2C outperformed benchmark in {prob_beat_benchmark:.1f}% of runs")
                print(f"  ✓ Average outperformance: {mean_return_diff:.2f}%")
            else:
                print(f"\n  ⚠ A2C underperformed benchmark on average")
        else:
            return_diff = (a2c_total_return - benchmark_total_return) * 100
            sharpe_diff = a2c_sharpe - benchmark_sharpe
            print(f"  Return Difference:   {return_diff:>+8.2f}%")
            print(f"  Sharpe Difference:   {sharpe_diff:>+8.4f}")
            
            if a2c_total_return > benchmark_total_return:
                print(f"\n  ✓ A2C outperformed benchmark by {return_diff:.2f}%")
            else:
                print(f"\n  ⚠ A2C underperformed benchmark by {abs(return_diff):.2f}%")
        
        #####################
        ### VISUALIZATION ###
        #####################
        visualize = input("\nWould you like to visualize the results? (y/n): ").strip().lower()
        
        if visualize == 'y':
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Cumulative Returns
            ax1 = axes[0, 0]
            ax1.plot(a2c_cumulative, label='A2C Agent', linewidth=2, color='blue')
            ax1.plot(benchmark_cumulative, label='Benchmark', linewidth=2, linestyle='--', color='orange')
            if show_statistics:
                ax1.set_title(f'Cumulative Returns (Median of {n_runs} runs)', fontsize=12, fontweight='bold')
            else:
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
            
            # 3. Daily Returns Distribution (or Return Distribution if multiple runs)
            ax3 = axes[1, 0]
            if show_statistics:
                ax3.hist(all_returns*100, bins=min(30, n_runs), alpha=0.7, label=f'A2C (n={n_runs})', 
                        density=True, color='blue', edgecolor='black')
                ax3.axvline(np.mean(all_returns)*100, color='blue', linestyle='--', 
                           linewidth=2, label='A2C Mean')
                ax3.axvline(benchmark_total_return*100, color='orange', linestyle='--', 
                           linewidth=2, label='Benchmark')
                ax3.set_title('Total Return Distribution', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Total Return (%)')
            else:
                ax3.hist(a2c_rewards*100, bins=50, alpha=0.5, label='A2C', density=True, color='blue')
                ax3.hist(benchmark_rewards*100, bins=50, alpha=0.5, label='Benchmark', 
                        density=True, color='orange')
                ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Daily Return (%)')
            ax3.set_ylabel('Density')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Portfolio Weights Over Time
            ax4 = axes[1, 1]
            asset_names = ['VNQ', 'SPY', 'TLT', 'GLD', 'BTC', 'Cash']
            colors = ['brown', 'green', 'blue', 'gold', 'orange', 'gray']
            for i, (name, color) in enumerate(zip(asset_names, colors)):
                ax4.plot(a2c_actions[:, i], label=name, alpha=0.7, linewidth=1.5, color=color)
            if show_statistics:
                ax4.set_title(f'Portfolio Weights Over Time (Median run)', fontsize=12, fontweight='bold')
            else:
                ax4.set_title('Portfolio Weights Over Time', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Trading Days')
            ax4.set_ylabel('Weight')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
            
            plt.suptitle(f'Evaluation Results: {df_test.index[0].strftime("%Y-%m-%d")} to {df_test.index[-1].strftime("%Y-%m-%d")}', 
                        fontsize=14, fontweight='bold', y=0.995)
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
        plt.plot(episode_returns_pct, alpha=0.2, label='Raw Returns', linewidth=0.5, color='blue')
        plt.plot(smooth(episode_returns_pct), label='Smoothed Returns', linewidth=2, color='red')
        plt.title("Episode Returns During Training", fontsize=14, fontweight='bold')
        plt.xlabel("Episode Number", fontsize=12)
        plt.ylabel("Total Return (%)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("\n✓ Plot displayed!")
        
    elif choice == '3':
        print("\nExiting... Goodbye!")
        break
    
    else:
        print("\n  Invalid choice. Please enter 1, 2, or 3.")

print("="*50)